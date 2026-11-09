"""Cavity-haloscope sensitivity: signal power, receiver noise, radiometer SNR, scan rate, limits.

Beta.  Pure Python and independent of the
FEM solver: a cavity mode enters only through plain numbers (or callables of frequency)
``f``, ``C``, ``Q0``, ``beta`` (or ``Q_L``), ``V`` and ``B``, which the eigenmode solver
(per-mode ``f, C, Q0``) or a published table supplies.

Every formula, its source equation and the differences between papers are in
``docs/physics/haloscope_sensitivity.md``; the tags ``(H*n*)`` below refer to it.

Conventions
-----------
* SI internally.  Inputs carry their unit in the name: ``f_Hz``, ``B_T``, ``V_m3``, ``T_K``,
  ``g_GeV`` (|g_agg| in GeV^-1), ``rho_GeV_cm3``.
* ``beta = Q0/Q_ext`` (one receiver port), ``Q_L = Q0/(1+beta)``.
* Noise is counted in quanta per unit bandwidth, ``N``, with ``k_B T_sys = h f N``  (H3.1).
  ``N_A`` is the added noise of the receiver referred to its input (the cavity port, or the
  amplifier input when a post-cavity transmission ``eta_out`` < 1 is modelled), *excluding*
  the input zero-point half quantum, so the standard quantum limit is ``N_A = 1/2`` and
  ``N_sys = n_th + 1/2 + N_A >= 1`` (HAYSTAC convention, arXiv:1611.07123 Eq. 4).
* Axion line: normalised one-sided spectrum ``s(f)`` (1/Hz).  Its noise-equivalent width
  ``Delta f_eff = 1/int s^2 df = f_a/Q_a`` defines ``Q_a`` (the package's ``Q_axion``,
  docs/physics/scan_and_signal.md S2.6; SHM v0 = 220, v_obs = 232 km/s gives 6.78e5).
  **A keyword called ``Q_a`` always means this radiometer quantity.**  Other conventions have
  their own names: ``Q_a_fwhm`` (f/FWHM of a Lorentzian line, :func:`signal_power`) and
  ``Q_a_kim`` (Kim et al.'s bin quality factor ~1e6, :func:`scan_rate_kim2020`); doc Sec. 2.
* "Optimal" means an optimally weighted (matched-filter) combination of all spectra, the
  dense-step limit of a scan, and no tuning dead time.  Real analyses lose 10-30% in SNR to
  bin-boxcar filters, background fits and rescans; they enter through ``eta_snr``.

Reused from the package: constants and ``n_bose`` (:mod:`axionfem.circuit`), ``ALPHA``
(:mod:`axionfem.multipole`), ``YEAR``, ``freq_to_mass``/``mass_to_freq`` and the QCD-line
coupling ``g_qcd`` (:mod:`axionfem.scan`, C++ core; vectorised here).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Mapping

import numpy as np

from . import scan as _scan
from .circuit import EV_J, GEV_J, HBAR, KB, MU0, n_bose  # noqa: F401  (EV_J re-exported)
from .multipole import ALPHA as ALPHA_EM  # CODATA 2018
from .scan import YEAR  # noqa: F401  (re-exported)

H_PLANCK = 2.0 * math.pi * HBAR          # J s
C_LIGHT = 299792458.0                    # m/s
GEV_CM3_TO_J_M3 = GEV_J * 1e6            # 1 GeV/cm^3 in J/m^3
LITRE = 1e-3                             # m^3
V0_KMS, VOBS_KMS = 220.0, 232.0          # SHM defaults of the package (scan_and_signal.md S2.4)

# Haloscope coupling convention (H1.2): g_agg = g_gamma alpha m_a / (pi Lambda^2),
# Lambda = 77.6 MeV, g_gamma = -0.97 (KSVZ), 0.36 (DFSZ).  arXiv:1611.07123 Eq. 1 and text;
# arXiv:1706.08388 Sec. II.  Used by ADMX, HAYSTAC, CAPP, TASEH to draw their KSVZ/DFSZ lines.
LAMBDA_MEV = 77.6
G_GAMMA = {"KSVZ": -0.97, "DFSZ": 0.36}



# =============================================================================
# 1. Units and couplings
# =============================================================================
_freq_to_mass_v = np.vectorize(_scan.freq_to_mass, otypes=[float])
_mass_to_freq_v = np.vectorize(_scan.mass_to_freq, otypes=[float])


def freq_to_mass_eV(f_Hz):
    """m_a [eV] = h f / e  (vectorised :func:`axionfem.scan.freq_to_mass`)."""
    return _freq_to_mass_v(f_Hz)


def mass_to_freq_Hz(m_eV):
    """f [Hz] = m_a e / h  (vectorised :func:`axionfem.scan.mass_to_freq`)."""
    return _mass_to_freq_v(m_eV)


def g_gamma_to_gagg(m_eV, g_gamma: float, Lambda_MeV: float = LAMBDA_MEV):
    """|g_agg| [GeV^-1] = |g_gamma| alpha m_a/(pi Lambda^2)  (H1.2; 1611.07123 text after Eq. 1)."""
    m_GeV = np.asarray(m_eV, dtype=float) * 1e-9
    return abs(g_gamma) * ALPHA_EM * m_GeV / (math.pi * (Lambda_MeV * 1e-3) ** 2)


def g_axion(m_eV, model: str = "KSVZ", convention: str = "haloscope", C_gamma: float = 1.77):
    """|g_agg| [GeV^-1] of a benchmark QCD axion.

    convention="haloscope": g_gamma/Lambda form used by the experiments (KSVZ -0.97, DFSZ 0.36,
    Lambda = 77.6 MeV), so that published "x KSVZ" statements convert exactly.
    convention="qcd": the package's E/N - C_gamma form (:func:`axionfem.scan.g_qcd`, lattice
    C_gamma = 1.77, m_a f_a = 5.70e-3 GeV^2); KSVZ is ~5% below the haloscope convention.
    """
    if convention == "haloscope":
        if model not in G_GAMMA:
            raise ValueError(f"g_axion: model {model!r} unknown for the haloscope convention; "
                             f"use one of {sorted(G_GAMMA)} or convention='qcd'")
        return g_gamma_to_gagg(m_eV, G_GAMMA[model])
    if convention == "qcd":
        out = _scan.g_qcd(np.atleast_1d(m_eV), model, C_gamma)
        return out if np.ndim(m_eV) else float(out[0])
    raise ValueError(f"g_axion: convention must be 'haloscope' or 'qcd', got {convention!r}")


def g_GeV_to_SI(g_GeV):
    """g [J^-1] from g [GeV^-1]."""
    return np.asarray(g_GeV, dtype=float) / GEV_J


# =============================================================================
# 2. Axion line shape (SHM; same equations as the C++ core, scan_and_signal.md S2.4-S2.6)
# =============================================================================
def _f_shm(v, v0, vobs):
    """Lab-frame SHM speed distribution, v in units of c, int_0^inf f dv = 1 (S2.4)."""
    v = np.asarray(v, dtype=float)
    return v / (math.sqrt(math.pi) * v0 * vobs) * (np.exp(-(v - vobs) ** 2 / v0 ** 2)
                                                   - np.exp(-(v + vobs) ** 2 / v0 ** 2))


def shm_line_shape(f_Hz, f_a_Hz: float, v0_kms: float = V0_KMS, vobs_kms: float = VOBS_KMS):
    """Normalised SHM spectrum s(f) [1/Hz], f = f_a (1 + v^2/2), int s df = 1  (H2.1; S2.5)."""
    v0, vo = v0_kms * 1e3 / C_LIGHT, vobs_kms * 1e3 / C_LIGHT
    u = np.clip(np.asarray(f_Hz, dtype=float) / f_a_Hz - 1.0, 0.0, None)
    v = np.sqrt(2.0 * u)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(v > 0, _f_shm(v, v0, vo) / (f_a_Hz * np.where(v > 0, v, 1.0)), 0.0)


@lru_cache(maxsize=16)
def _leggauss(n: int):
    """Cached Gauss-Legendre rule (numpy's construction is O(n^3))."""
    return np.polynomial.legendre.leggauss(n)


def _v_nodes(v0_kms: float, vobs_kms: float, n: int = 400):
    """Gauss-Legendre nodes/weights in v (units of c) covering the SHM line (0 .. vobs + 8 v0)."""
    x, w = _leggauss(n)
    vmax = (vobs_kms + 8.0 * v0_kms) * 1e3 / C_LIGHT
    return 0.5 * vmax * (x + 1.0), 0.5 * vmax * w


def _v_nodes_cavity(f_a_Hz: float, f_c_Hz: float, Q_L: float, v0_kms: float, vobs_kms: float,
                    n_per: int = 48):
    """Composite Gauss-Legendre nodes in v for integrands s(f) x Lorentzian(f - f_c): panels break
    at the line scale and at f_c +- {0, 1, 3, 10, ...} half-widths, so a cavity much narrower than
    the axion line (Q_L >> Q_a) is resolved.  (A single 400-node rule over the line is wrong by a
    factor ~2 at Q_L = 1e9.)"""
    vmax = (vobs_kms + 8.0 * v0_kms) * 1e3 / C_LIGHT
    hw = f_c_Hz / (2.0 * Q_L)
    df = hw * np.array([0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4])
    fb = f_c_Hz + np.concatenate([-df, df])
    u = fb / f_a_Hz - 1.0
    vb = np.sqrt(2.0 * u[u > 0])
    br = np.unique(np.concatenate([np.linspace(0.0, vmax, 9), vb[vb < vmax]]))
    x, w = _leggauss(n_per)
    a, b = br[:-1, None], br[1:, None]
    return (0.5 * (b - a) * (x + 1.0) + a).ravel(), (0.5 * (b - a) * w).ravel()


def Q_a_shm(v0_kms: float = V0_KMS, vobs_kms: float = VOBS_KMS) -> float:
    """Q_a = f_a/Delta f_eff = f_a int s^2 df for the SHM (H2.2); = scan.Q_axion_shm (6.78e5)."""
    v, w = _v_nodes(v0_kms, vobs_kms)
    v0, vo = v0_kms * 1e3 / C_LIGHT, vobs_kms * 1e3 / C_LIGHT
    fv = _f_shm(v, v0, vo)
    return float(np.sum(w * fv ** 2 / v))


_Q_a_shm_cached = lru_cache(maxsize=8)(Q_a_shm)


def Q_a_lorentzian_fwhm(Q_a_fwhm: float) -> float:
    """Radiometer Q_a of a Lorentzian line with f/FWHM = Q_a_fwhm: Delta f_eff = pi FWHM (H2.3)."""
    return Q_a_fwhm / math.pi


# =============================================================================
# 3. Signal power
# =============================================================================
def power_coefficient(g_GeV, f_Hz, B_T, V_m3, C, rho_GeV_cm3: float = 0.45):
    """P_coef = g^2 (hbar c^3/(mu0 omega)) rho B^2 V C  [W] (H3.1); P = eta_c Q P_coef.

    Natural units: g^2 (rho/m_a) B^2 V C (Sikivie 1985; Kim et al. 2020 Eq. 1 without Q).
    """
    g_si = g_GeV_to_SI(g_GeV)
    omega = 2.0 * math.pi * np.asarray(f_Hz, dtype=float)
    rho = rho_GeV_cm3 * GEV_CM3_TO_J_M3
    return g_si ** 2 * HBAR * C_LIGHT ** 3 / (MU0 * omega) * rho * np.asarray(B_T) ** 2 * V_m3 * C


def lorentzian(delta_Hz, f_c_Hz, Q_L):
    """Cavity power response 1/(1 + (2 Q_L delta/f_c)^2)."""
    return 1.0 / (1.0 + (2.0 * Q_L * np.asarray(delta_Hz, dtype=float) / f_c_Hz) ** 2)


def _QL_beta(Q0=None, beta=None, Q_L=None):
    """Resolve (Q_L, beta, eta_c = beta/(1+beta)) from two of Q0, beta, Q_L."""
    if Q_L is None:
        if Q0 is None or beta is None:
            raise ValueError("give Q_L, or both Q0 and beta (Q_L = Q0/(1+beta))")
        Q_L = Q0 / (1.0 + beta)
    if beta is None:
        if Q0 is None:
            raise ValueError("beta unknown: give beta, or Q0 together with Q_L")
        beta = Q0 / Q_L - 1.0
    return float(Q_L), float(beta), float(beta / (1.0 + beta))


def signal_power(g_GeV, f_Hz, B_T, V_m3, C, *, Q0=None, beta=None, Q_L=None, Q_a_fwhm=None,
                 rho_GeV_cm3: float = 0.45, detuning_Hz: float = 0.0, eta: float = 1.0):
    """Axion power delivered to the receiver port [W]  (H3.2).

    P = eta * beta/(1+beta) * g^2 (rho/m_a) B^2 V C * Q_eff * L(detuning)
    with Q_eff = Q_L Q_a/(Q_L + Q_a) for a Lorentzian axion line of f/FWHM = ``Q_a_fwhm``
    (Kim et al. 2020, arXiv:2001.05605 Eq. 7 = Eq. 18) and Q_eff = Q_L if ``Q_a_fwhm`` is None
    (narrow line).  ``Q_a_fwhm`` is NOT the radiometer ``Q_a`` of :func:`scan_rate` (which is
    Q_a_fwhm/pi for a Lorentzian, H2.3).
    ``eta`` is any extra signal transmission efficiency (e.g. HAYSTAC eta_L = 10^-0.06).
    """
    Q_L, _, eta_c = _QL_beta(Q0, beta, Q_L)
    Q_eff = Q_L if Q_a_fwhm is None else Q_L * Q_a_fwhm / (Q_L + Q_a_fwhm)
    P = power_coefficient(g_GeV, f_Hz, B_T, V_m3, C, rho_GeV_cm3)
    return eta * eta_c * Q_eff * P * lorentzian(detuning_Hz, f_Hz, Q_L)


def signal_power_shm(g_GeV, f_c_Hz, f_a_Hz, B_T, V_m3, C, *, Q0=None, beta=None, Q_L=None,
                     rho_GeV_cm3: float = 0.45, v0_kms: float = V0_KMS, vobs_kms: float = VOBS_KMS,
                     eta: float = 1.0):
    """Total received power for an SHM axion of rest frequency f_a, cavity at f_c  (H3.3):
    Q_eff = Q_L int s(f) L(f - f_c) df."""
    Q_L, _, eta_c = _QL_beta(Q0, beta, Q_L)
    v, w = _v_nodes_cavity(f_a_Hz, f_c_Hz, Q_L, v0_kms, vobs_kms)
    v0, vo = v0_kms * 1e3 / C_LIGHT, vobs_kms * 1e3 / C_LIGHT
    f = f_a_Hz * (1.0 + 0.5 * v ** 2)
    overlap = float(np.sum(w * _f_shm(v, v0, vo) * lorentzian(f - f_c_Hz, f_c_Hz, Q_L)))
    return eta * eta_c * Q_L * overlap * power_coefficient(g_GeV, f_a_Hz, B_T, V_m3, C, rho_GeV_cm3)


# =============================================================================
# 4. Noise
# =============================================================================
_n_bose_v = np.vectorize(n_bose, otypes=[float])


def n_thermal(f_Hz, T_K):
    """Bose occupation 1/(exp(hf/kT) - 1); 0 where T <= 0 (vectorised :func:`axionfem.circuit.n_bose`)."""
    return _n_bose_v(2.0 * math.pi * np.asarray(f_Hz, dtype=float), np.asarray(T_K, dtype=float))


def T_sys_quantum(f_Hz, T_K, N_A):
    """k_B T_sys = h f (n_th + 1/2 + N_A)  (H4.1; 1611.07123 Eq. 4)."""
    return H_PLANCK * np.asarray(f_Hz) * (n_thermal(f_Hz, T_K) + 0.5 + N_A) / KB


def N_A_from_T(f_Hz, T_A_K):
    """Added noise in quanta of a classical amplifier noise temperature: N_A = k T_A/(h f)."""
    return KB * T_A_K / (H_PLANCK * np.asarray(f_Hz, dtype=float))


def squeezer_gain_dB(S_measured: float, eta_total: float, *, N_A: float = 0.0, n_in: float = 0.0) -> float:
    """Squeezer gain G_s [dB] (the ``Receiver.squeeze_dB`` convention) from an OUTPUT-referred
    measured squeezing  (H4.3).

    ``S_measured`` = sigma^2_on/sigma^2_off of the receiver output off cavity resonance, squeezer
    on vs off (2301.09721 Sec. II A).  That ratio contains the added noise N_A (single quadrature,
    vacuum = 1/2) and the bath occupation n_in:
        S_measured = (s (n_in + 1/2) + N_A)/((n_in + 1/2) + N_A),
        s = eta_total/G_s + 1 - eta_total     (2301.09721 Eq. 6, eta_total = squeezer -> amplifier).
    With ``N_A = 0`` this is the plain inversion of Eq. 6.  Raises if the measured value is below
    what eta_total allows (s >= 1 - eta_total): the inputs are then mutually inconsistent.
    """
    v = n_in + 0.5
    s = (S_measured * (v + N_A) - N_A) / v
    if not 0.0 < eta_total <= 1.0:
        raise ValueError(f"squeezer_gain_dB: eta_total must be in (0, 1], got {eta_total}")
    if s <= 1.0 - eta_total:
        raise ValueError(f"squeezer_gain_dB: measured S = {S_measured:.4g} with N_A = {N_A:g} needs "
                         f"s = {s:.4g} at the amplifier input, below the loss floor 1 - eta = "
                         f"{1.0 - eta_total:.4g}: inputs are inconsistent (N_A double-counted?)")
    return 10.0 * math.log10(eta_total / (s - 1.0 + eta_total))


@dataclass(frozen=True)
class Receiver:
    """Noise model of a single-port haloscope receiver.

    Exactly one way of fixing the noise level:

    * ``T_sys_K``: a total system noise temperature as quoted by the experiment, referred to the
      cavity port (flat in detuning; k_B T_sys per Hz).  Overrides the physical noise model;
      only ``eta_snr`` still applies (``eta_out`` must stay 1).
    * otherwise the physical model, in quanta per Hz referred to the amplifier input:

          N_sys(delta) = k [ eta_out N_port(delta) + (1 - eta_out)(n_in + 1/2) + N_A ],
          N_port = |chi|^2 S_cav (n_in + 1/2) + (1 - |chi|^2)(n_c + 1/2),

      |chi|^2 = ((beta - 1)^2 + x^2)/((beta + 1)^2 + x^2) the cavity reflection, x = 2 Q0 delta/f,
      n_c the cavity-bath occupation (``n_cav`` if given, else n_th(``T_cav_K``)), n_in = n_th(``T_in_K``)
      the occupation of the termination incident on the port AND of the transmission-loss baths
      (all at the fridge temperature; ``T_in_K`` defaults to ``T_cav_K``), S_cav the squeezing
      delivered to the cavity (1 without squeezing), k = 2 for a phase-sensitive readout, else 1.
      This is 2301.09721 Eq. A4 (with alpha = ``eta_out``, rho = ``squeeze_eff``, G_S = 1/G_s).
      With an isolator at the cavity temperature and no squeezing, N_sys = k(n_c + 1/2 + N_A)
      is flat and independent of beta.  ``T_in_K = 0`` mimics "no isolator, cold amplifier input"
      (Kim et al. 2020 Sec. 3).

    ``eta_out`` (HAYSTAC's alpha, 2301.09721 Table II): power transmission between the cavity port
    and the amplifier.  It attenuates the axion signal (scan_rate and the limits multiply P by it)
    and replaces a fraction 1 - eta_out of the cavity output by fresh noise at n_in.  ``N_A`` is
    then the added noise referred to the AMPLIFIER input (HAYSTAC's N_a).

    ``N_A`` (quanta, excludes the input 1/2: SQL = 0.5) or ``T_A_K`` (classical HEMT noise
    temperature, N_A = k T_A/hf).

    ``phase_sensitive=True`` (HAYSTAC phase II squeezed-state receiver, SSR): one quadrature is
    measured; the axion's random-phase signal puts half its power there, so the noise is reported
    in phase-insensitive-equivalent quanta N_sys = 2 N_quad, with N_quad counted so that vacuum
    is 1/2 (the convention of the corrected HAYSTAC analysis, arXiv:2301.09721 Sec. III, Eqs. A1,
    A3-A6).  An ideal phase-sensitive amplifier without squeezing therefore gives N_sys = 1, the
    same as the SQL phase-insensitive one (no free lunch).  ``N_A`` is then the single-quadrature
    added noise (HAYSTAC's N_a).

    Squeezing convention (Malnou et al. 2019, arXiv:1809.06470): ``squeeze_dB`` is the
    squeezer's own one-quadrature gain G_s = 10^(dB/10) (noise reduction at the squeezer OUTPUT,
    before any loss).  ``squeeze_eff`` is the squeezer -> cavity transmission eta_1 (HAYSTAC's
    rho = (alpha rho)/alpha), so S_cav = eta_1/G_s + 1 - eta_1 (2301.09721 Eq. 6 at the cavity).
    A squeezing MEASURED at the receiver output already contains N_A and eta_out; never enter it
    as ``squeeze_dB`` -- convert it with :func:`squeezer_gain_dB`, which removes N_A (otherwise the
    added noise is counted twice).  The squeezer and the measured quadrature are centred on the
    cavity frequency, so chi(delta) multiplies both sideband quadratures by the same factor
    (chi(-delta)* = chi(delta)): no rotation of the ellipse, anti-squeezing never enters.  This
    agrees with Malnou et al. Eq. 2.
    ``eta_snr`` multiplies the SNR (analysis efficiency, < 1), so the rate goes as eta_snr^2.
    """

    T_cav_K: float = 0.0
    N_A: float | None = None
    T_A_K: float | None = None
    T_sys_K: float | None = None
    T_in_K: float | None = None
    n_cav: float | None = None
    squeeze_dB: float = 0.0
    squeeze_eff: float = 1.0
    eta_out: float = 1.0
    phase_sensitive: bool = False
    eta_snr: float = 1.0
    label: str = ""

    def __post_init__(self):
        if self.T_sys_K is None and self.N_A is None and self.T_A_K is None:
            raise ValueError("Receiver: give T_sys_K, or the amplifier noise as N_A (quanta) or T_A_K")
        if self.N_A is not None and self.T_A_K is not None:
            raise ValueError("Receiver: give N_A or T_A_K, not both")
        if not 0.0 <= self.squeeze_eff <= 1.0:
            raise ValueError(f"Receiver: squeeze_eff must be in [0, 1], got {self.squeeze_eff}")
        if not 0.0 < self.eta_out <= 1.0:
            raise ValueError(f"Receiver: eta_out must be in (0, 1], got {self.eta_out}")
        if self.T_sys_K is not None and self.eta_out != 1.0:
            raise ValueError("Receiver: T_sys_K is referred to the cavity port; eta_out must be 1 with it")
        if self.squeeze_dB < 0:
            raise ValueError("Receiver: squeeze_dB is the squeezer gain in dB, >= 0")
        if self.squeeze_dB > 0 and not self.phase_sensitive:
            raise ValueError("Receiver: squeezing needs a phase-sensitive readout; set phase_sensitive=True")

    def amp_N(self, f_Hz):
        if self.T_A_K is not None:
            return N_A_from_T(f_Hz, self.T_A_K)
        return self.N_A

    def _occupations(self, f_Hz):
        """(n_c, n_in): cavity bath and incident/loss-bath occupations."""
        n_c = self.n_cav if self.n_cav is not None else float(n_thermal(f_Hz, self.T_cav_K))
        T_in = self.T_cav_K if self.T_in_K is None else self.T_in_K
        return n_c, float(n_thermal(f_Hz, T_in))

    @property
    def delivered_S(self) -> float:
        """Squeezing delivered to the cavity, S_cav = eta_1/G_s + 1 - eta_1 (2301.09721 Eq. 6 with
        eta = squeeze_eff); 1 without squeezing."""
        G = 10.0 ** (self.squeeze_dB / 10.0)
        return self.squeeze_eff / G + 1.0 - self.squeeze_eff

    @property
    def output_S(self) -> float:
        """Off-resonance squeezing at the amplifier input, s = 1 - eta + eta/G_s with
        eta = squeeze_eff * eta_out (2301.09721 Eq. 6, noiseless amplifier)."""
        return self.eta_out * self.delivered_S + 1.0 - self.eta_out

    def measured_S(self, f_Hz) -> float:
        """Output-referred off-resonance squeezing sigma^2_on/sigma^2_off INCLUDING N_A (what an
        experiment measures, 2301.09721 Sec. II A); equal temperatures assumed."""
        _, n_in = self._occupations(f_Hz)
        v, N_A = n_in + 0.5, float(self.amp_N(f_Hz))
        return (self.output_S * v + N_A) / (v + N_A)

    @property
    def flat(self) -> bool:
        """True when N_sys does not depend on detuning or beta."""
        if self.T_sys_K is not None:
            return True
        if self.squeeze_dB != 0.0:
            return False
        if self.n_cav is not None:
            return False           # resolved per frequency in N_sys (equal occupations still integrate exactly)
        T_in = self.T_cav_K if self.T_in_K is None else self.T_in_K
        return T_in == self.T_cav_K

    def N_sys(self, f_Hz, delta_Hz=0.0, Q0: float | None = None, beta: float | None = None):
        """System noise in quanta per Hz at cavity detuning delta = f - f_c, referred to the
        amplifier input (= the cavity port when eta_out = 1)."""
        delta = np.asarray(delta_Hz, dtype=float)
        if self.T_sys_K is not None:
            return np.broadcast_to(KB * self.T_sys_K / (H_PLANCK * f_Hz), delta.shape) * 1.0
        n_c, n_in = self._occupations(f_Hz)
        N_A = self.amp_N(f_Hz)
        k = 2.0 if self.phase_sensitive else 1.0
        a = self.eta_out
        if self.flat:
            return np.broadcast_to(k * (n_c + 0.5 + N_A), delta.shape) * 1.0
        if Q0 is None or beta is None:
            raise ValueError("Receiver.N_sys: this noise depends on detuning; give Q0 and beta")
        x = 2.0 * Q0 * delta / f_Hz
        r2 = ((beta - 1.0) ** 2 + x ** 2) / ((beta + 1.0) ** 2 + x ** 2)    # |chi|^2
        N_port = r2 * self.delivered_S * (n_in + 0.5) + (1.0 - r2) * (n_c + 0.5)
        return k * (a * N_port + (1.0 - a) * (n_in + 0.5) + N_A)

    def T_sys(self, f_Hz, delta_Hz=0.0, Q0=None, beta=None):
        """k_B T_sys = h f N_sys  [K] (amplifier-input referred; divide by eta_out for the cavity port)."""
        return H_PLANCK * f_Hz * self.N_sys(f_Hz, delta_Hz, Q0, beta) / KB


def hemt(T_cav_K: float, T_A_K: float, **kw) -> Receiver:
    """Phase-insensitive HEMT-limited receiver (CAPP, TASEH, QUAX, ORGAN style)."""
    return Receiver(T_cav_K=T_cav_K, T_A_K=T_A_K, label=kw.pop("label", "HEMT"), **kw)


def sql(T_cav_K: float, **kw) -> Receiver:
    """Quantum-limited phase-insensitive amplifier: N_A = 1/2, N_sys = n_th + 1."""
    return Receiver(T_cav_K=T_cav_K, N_A=0.5, label=kw.pop("label", "SQL"), **kw)


def squeezed(T_cav_K: float, squeeze_dB: float, squeeze_eff: float = 1.0, N_A: float = 0.0, **kw) -> Receiver:
    """Squeezed-state receiver (HAYSTAC phase II): squeezed input + phase-sensitive amplifier.
    ``squeeze_dB`` is the squeezer gain G_s (not a measured output squeezing; see
    :func:`squeezer_gain_dB`), ``squeeze_eff`` the squeezer -> cavity transmission, ``N_A`` the
    single-quadrature added noise; ``eta_out`` etc. pass through.  ``squeeze_dB = 0`` is a bare
    phase-sensitive amplifier."""
    return Receiver(T_cav_K=T_cav_K, N_A=N_A, squeeze_dB=squeeze_dB, squeeze_eff=squeeze_eff,
                    phase_sensitive=True, label=kw.pop("label", f"squeezed {squeeze_dB:g} dB"), **kw)


# =============================================================================
# 5. SNR and scan rate
# =============================================================================
def dicke_snr(P_W, T_sys_K, t_s, bandwidth_Hz):
    """Radiometer SNR = P/(k T_sys) sqrt(t/b)  (H5.1; Dicke 1946, 1611.07123 Eq. 3)."""
    return np.asarray(P_W) / (KB * np.asarray(T_sys_K)) * np.sqrt(np.asarray(t_s) / bandwidth_Hz)


def _resolve(x, f):
    return float(x(f)) if callable(x) else x


def _detuning_integral(f_Hz, Q0, beta, rx: Receiver, n: int = 2001):
    """I = int d delta [L(delta)/N_sys(delta)]^2  [Hz]  (H5.3).  Flat noise: pi f/(4 Q_L N^2)."""
    Q_L = Q0 / (1.0 + beta)
    if rx.flat:
        N = float(np.asarray(rx.N_sys(f_Hz)).ravel()[0])
        return math.pi * f_Hz / (4.0 * Q_L) / N ** 2
    # delta = (f/(2 Q_L)) tan(theta): L = cos^2 theta, d delta = (f/2Q_L) sec^2 theta d theta
    th, w = _leggauss(n)
    th = 0.5 * math.pi * th * (1 - 1e-12)
    w = 0.5 * math.pi * w
    delta = f_Hz / (2.0 * Q_L) * np.tan(th)
    N = rx.N_sys(f_Hz, delta, Q0, beta)
    return float(np.sum(w * f_Hz / (2.0 * Q_L) * np.cos(th) ** 2 / N ** 2))


@dataclass(frozen=True)
class Cavity:
    """One cavity mode as seen by the sensitivity layer.  Any field may be a callable of f [Hz]
    (e.g. an interpolant of the FEM per-mode outputs C(f), Q0(f) along a tuning branch)."""

    B_T: object
    V_m3: object
    C: object
    Q0: object
    beta: object = 2.0
    eta: object = 1.0       # signal transmission efficiency (power), e.g. 10^-0.06
    label: str = ""

    def at(self, f_Hz: float) -> dict:
        return {k: _resolve(getattr(self, k), f_Hz) for k in ("B_T", "V_m3", "C", "Q0", "beta", "eta")}

    @classmethod
    def from_QL(cls, B_T, V_m3, C, Q_L, beta, **kw) -> "Cavity":
        """Build from a loaded Q: Q0 = Q_L (1 + beta)."""
        if callable(Q_L) or callable(beta):
            Q0 = lambda f: _resolve(Q_L, f) * (1.0 + _resolve(beta, f))  # noqa: E731
        else:
            Q0 = Q_L * (1.0 + beta)
        return cls(B_T=B_T, V_m3=V_m3, C=C, Q0=Q0, beta=beta, **kw)


def scan_rate(f_Hz: float, cav: Cavity, rx: Receiver, *, g_GeV: float | None = None, snr: float = 5.0,
              Q_a: float | None = None, rho_GeV_cm3: float = 0.45, model: str = "DFSZ",
              prefactor: str = "optimal") -> float:
    """Scan rate df/dt [Hz/s] reaching ``snr`` on an axion of coupling g.

    Optimal weighting, dense steps:
        df/dt = eta_snr^2 int d delta [P(delta)/(h f N_sys(delta))]^2 / (SNR^2 Delta f_eff),
        Delta f_eff = f/Q_a,   P = eta eta_out beta/(1+beta) Q_L P_coef L(delta).
    ``Q_a`` is the radiometer quality factor (f/Delta f_eff); None -> SHM, 6.78e5.
    Flat noise: (pi/4) (eta_c P_coef)^2 Q_L Q_a / (SNR k T_sys)^2, i.e. ~ g^4 B^4 V^2 C^2 Q_L/T^2.
    The separation of line shape and cavity response is exact in the dense-scan limit for any
    Q_L/Q_a, so there is no Q_L Q_a/(Q_L+Q_a) saturation here (cf. :func:`scan_rate_kim2020`).

    prefactor="haystac": replace pi/4 by the 4/5 of 1611.07123 Eq. 5 (squared Lorentzians summed
    over steps of ~Delta nu_c/2, boxcar bins); use it with Q_a ~ 1e6.
    g_GeV None -> the benchmark ``model`` coupling (haloscope convention).
    """
    p = cav.at(f_Hz)
    if g_GeV is None:
        g_GeV = float(g_axion(freq_to_mass_eV(f_Hz), model))
    if Q_a is None:
        Q_a = _Q_a_shm_cached(V0_KMS, VOBS_KMS)
    eta_c = p["beta"] / (1.0 + p["beta"])
    Q_L = p["Q0"] / (1.0 + p["beta"])
    P_pk = (p["eta"] * rx.eta_out * eta_c * Q_L
            * power_coefficient(g_GeV, f_Hz, p["B_T"], p["V_m3"], p["C"], rho_GeV_cm3))
    I = _detuning_integral(f_Hz, p["Q0"], p["beta"], rx)
    if prefactor == "haystac":
        I *= (4.0 / 5.0) / (math.pi / 4.0)
    elif prefactor != "optimal":
        raise ValueError(f"scan_rate: prefactor must be 'optimal' or 'haystac', got {prefactor!r}")
    hf = H_PLANCK * f_Hz
    return float((P_pk / hf) ** 2 * I * rx.eta_snr ** 2 / (snr ** 2 * f_Hz / Q_a))


def scan_rate_kim2020(f_Hz: float, cav: Cavity, *, g_GeV: float, T_sys_K: float | None = None,
                      T_eff_K: float | None = None, T_add_K: float | None = None, snr: float = 5.0,
                      Q_a_kim: float = 1e6, rho_GeV_cm3: float = 0.45) -> float:
    """Kim et al. 2020 (arXiv:2001.05605) Eq. 19 (H5.6).

    df/dt = (1/SNR^2) (P0/k T_eff)^2 [(beta/(1+beta)) / (4 beta/(1+beta)^2 + lambda)]^2 Q_l Q_a^2/(Q_l + Q_a)
    with lambda = T_add/T_eff (Eq. 18).  Give EITHER
    * ``T_sys_K``: flat system noise (isolator); the bracket's denominator is T_sys/T_eff, i.e.
      (P0 beta/(1+beta)/k T_sys)^2 Q_l Q_a^2/(Q_l+Q_a); OR
    * ``T_eff_K`` and ``T_add_K``: Kim's no-isolator form, noise k(T_eff 4 beta/(1+beta)^2 + T_add).
    ``Q_a_kim`` is Kim's axion quality factor (~1e6, f over the bin/line width), not the
    radiometer ``Q_a`` of :func:`scan_rate`.
    """
    p = cav.at(f_Hz)
    eta_c = p["beta"] / (1.0 + p["beta"])
    Q_L = p["Q0"] / (1.0 + p["beta"])
    P0 = p["eta"] * power_coefficient(g_GeV, f_Hz, p["B_T"], p["V_m3"], p["C"], rho_GeV_cm3)
    if T_sys_K is not None and T_eff_K is None and T_add_K is None:
        noise = KB * T_sys_K
    elif T_sys_K is None and T_eff_K is not None and T_add_K is not None:
        noise = KB * (T_eff_K * 4 * p["beta"] / (1 + p["beta"]) ** 2 + T_add_K)
    else:
        raise ValueError("scan_rate_kim2020: give T_sys_K alone, or T_eff_K together with T_add_K")
    Q_a = Q_a_kim
    return float((P0 * eta_c / noise) ** 2 * Q_L * Q_a ** 2 / (Q_L + Q_a) / snr ** 2)


def beta_opt(f_Hz: float, cav: Cavity, rx: Receiver, **kw) -> float:
    """Coupling maximising :func:`scan_rate` at fixed Q0 (numerical; = 2 for flat noise)."""
    from scipy.optimize import minimize_scalar

    def neg(lb):
        return -scan_rate(f_Hz, replace(cav, beta=math.exp(lb)), rx, **kw)

    r = minimize_scalar(neg, bounds=(math.log(0.05), math.log(200.0)), method="bounded",
                        options={"xatol": 1e-7})
    return math.exp(r.x)


def beta_opt_kim2020(Q0: float, Q_a_kim: float) -> float:
    """Kim et al. 2020, amplifier-dominated (lambda >> 1) root of Eq. 20 (closed form in the text
    after it): (1 + sqrt(9 + 8 Q0/Q_a))/2 -> 2 for Q0 << Q_a."""
    return 0.5 * (1.0 + math.sqrt(9.0 + 8.0 * Q0 / Q_a_kim))


# =============================================================================
# 6. Limits and projections
# =============================================================================
def snr_required(threshold_sigma: float | None = None, confidence: float = 0.90,
                 snr_target: float | None = None) -> float:
    """SNR an axion needs to be excluded  (H6.1).

    Either ``snr_target`` directly, or a candidate threshold (in sigma of the Gaussian grand
    spectrum) plus the confidence that a true signal exceeds it: SNR = z_thr + Phi^-1(CL)
    (HAYSTAC: 3.455 + 1.645 = 5.1 at 95%; 1706.08388 Sec. VIII).
    """
    if snr_target is not None:
        return float(snr_target)
    if threshold_sigma is None:
        raise ValueError("snr_required: give snr_target, or threshold_sigma with confidence")
    from scipy.stats import norm
    return float(threshold_sigma + norm.ppf(confidence))


def g_limit_scan(f_Hz, cav: Cavity, rx: Receiver, *, rate_Hz_s=None, tau_step_s=None, df_step_Hz=None,
                 snr: float = 5.0, Q_a: float | None = None, rho_GeV_cm3: float = 0.45,
                 prefactor: str = "optimal"):
    """Excluded |g_agg| [GeV^-1] at f for a scan at rate r = df_step/tau_step.

    SNR scales as g^2 at fixed scan rate, so g_lim = g_ref (r/r_ref(g_ref))^(1/4) where r_ref is
    :func:`scan_rate` at g_ref for the target SNR.  ``f_Hz`` may be an array.
    """
    if rate_Hz_s is None:
        if tau_step_s is None or df_step_Hz is None:
            raise ValueError("g_limit_scan: give rate_Hz_s, or tau_step_s and df_step_Hz")
        rate_Hz_s = df_step_Hz / tau_step_s
    f = np.atleast_1d(np.asarray(f_Hz, dtype=float))
    out = np.empty_like(f)
    for i, fi in enumerate(f):
        r = _resolve(rate_Hz_s, fi)
        g_ref = 1e-14
        r_ref = scan_rate(fi, cav, rx, g_GeV=g_ref, snr=snr, Q_a=Q_a, rho_GeV_cm3=rho_GeV_cm3,
                          prefactor=prefactor)
        out[i] = g_ref * (r / r_ref) ** 0.25
    return out if np.ndim(f_Hz) else float(out[0])


def g_limit_dwell(f_a_Hz, f_c_Hz: float, cav: Cavity, rx: Receiver, *, t_s: float, snr: float = 5.0,
                  rho_GeV_cm3: float = 0.45, v0_kms: float = V0_KMS, vobs_kms: float = VOBS_KMS,
                  Q_a: float | None = None, n: int = 48):
    """Excluded |g_agg| for one fixed tuning f_c integrated for t_s (QUAX/ORGAN-like)  (H6.3).

    Optimal (matched-filter) SNR with the SHM line shape and the full cavity response:
        SNR^2 = t int df [P_pk s(f) L(f - f_c)/(h f_c N_sys(f - f_c))]^2,
    which reduces to the Dicke form P(delta)/(kT) sqrt(t Q_a/f) when the line is narrow.
    ``Q_a`` given (radiometer convention) -> narrow-line Dicke form with Delta f_eff = f_a/Q_a
    instead of the SHM.  ``n``: Gauss nodes per panel of :func:`_v_nodes_cavity`.
    """
    p = cav.at(f_c_Hz)
    eta_c = p["beta"] / (1.0 + p["beta"])
    Q_L = p["Q0"] / (1.0 + p["beta"])
    fa = np.atleast_1d(np.asarray(f_a_Hz, dtype=float))
    out = np.empty_like(fa)
    g_ref = 1e-12
    v0, vo = v0_kms * 1e3 / C_LIGHT, vobs_kms * 1e3 / C_LIGHT
    for i, fai in enumerate(fa):
        P_pk = (p["eta"] * rx.eta_out * eta_c * Q_L
                * power_coefficient(g_ref, fai, p["B_T"], p["V_m3"], p["C"], rho_GeV_cm3))
        hf = H_PLANCK * f_c_Hz
        if Q_a is not None:
            d = fai - f_c_Hz
            N = rx.N_sys(f_c_Hz, d, p["Q0"], p["beta"])
            snr2 = t_s * (P_pk * lorentzian(d, f_c_Hz, Q_L) / (hf * N)) ** 2 * Q_a / fai
        else:
            v, w = _v_nodes_cavity(fai, f_c_Hz, Q_L, v0_kms, vobs_kms, n)
            f = fai * (1.0 + 0.5 * v ** 2)
            s = _f_shm(v, v0, vo) / (fai * v)          # s(f) [1/Hz]
            d = f - f_c_Hz
            N = rx.N_sys(f_c_Hz, d, p["Q0"], p["beta"])
            integrand = (P_pk * s * lorentzian(d, f_c_Hz, Q_L) / (hf * N)) ** 2
            snr2 = t_s * float(np.sum(w * integrand * fai * v))   # df = f_a v dv
        snr_ref = math.sqrt(snr2) * rx.eta_snr
        out[i] = g_ref * math.sqrt(snr / snr_ref)
    return out if np.ndim(f_a_Hz) else float(out[0])


def time_to_cover(f1_Hz: float, f2_Hz: float, cav: Cavity, rx: Receiver, *, g: str | float = "DFSZ",
                  g_factor: float = 1.0, snr: float = 5.0, Q_a: float | None = None,
                  rho_GeV_cm3: float = 0.45, duty_cycle: float = 1.0, n: int = 200,
                  prefactor: str = "optimal") -> dict:
    """Live time to scan [f1, f2] at coupling g (model name or GeV^-1) x g_factor  (H6.4).

    t = int df / (df/dt), log-spaced quadrature; parameters may vary with f via callables.
    Returns {"t_s", "t_yr", "f_Hz", "rate_Hz_s"}.
    """
    f = np.geomspace(f1_Hz, f2_Hz, n)
    rate = np.empty_like(f)
    for i, fi in enumerate(f):
        gi = (float(g_axion(freq_to_mass_eV(fi), g)) if isinstance(g, str) else float(g)) * g_factor
        rate[i] = scan_rate(fi, cav, rx, g_GeV=gi, snr=snr, Q_a=Q_a, rho_GeV_cm3=rho_GeV_cm3,
                            prefactor=prefactor)
    rate *= duty_cycle
    t = float(np.trapezoid(1.0 / rate, f))
    return {"t_s": t, "t_yr": t / YEAR, "f_Hz": f, "rate_Hz_s": rate}


# =============================================================================
# 7. Presets
# =============================================================================
from .haloscope_presets import PRESETS  # noqa: E402  (data module, imports nothing from here)


def preset(name: str) -> dict:
    """A copy of a published-parameter preset (see :data:`PRESETS`)."""
    if name not in PRESETS:
        raise KeyError(f"unknown haloscope preset {name!r}; available: {sorted(PRESETS)}")
    return {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in PRESETS[name].items()}


def preset_models(name: str) -> tuple[Cavity, Receiver, dict]:
    """(Cavity, Receiver, analysis dict) built from a preset's numbers."""
    p = PRESETS[name]
    c = p["cavity"]
    if "Q0" in c and c.get("Q0") is not None:
        cav = Cavity(B_T=c["B_T"], V_m3=c["V_L"] * LITRE, C=c["C"], Q0=c["Q0"], beta=c["beta"],
                     eta=c.get("eta", 1.0), label=name)
    else:
        cav = Cavity.from_QL(c["B_T"], c["V_L"] * LITRE, c["C"], c["Q_L"], c["beta"],
                             eta=c.get("eta", 1.0), label=name)
    rx = Receiver(**p["receiver"], label=name)
    return cav, rx, dict(p["analysis"])


def published_g(name: str, f_Hz: float | None = None) -> float:
    """Published exclusion |g_agg| [GeV^-1] of a preset; an "x KSVZ" statement is converted with
    the paper's own Lambda (``published["Lambda_MeV"]``, default 77.6 MeV)."""
    pub = PRESETS[name]["published"]
    if pub.get("g_GeV") is not None:
        return float(pub["g_GeV"])
    f = pub["f_Hz"] if f_Hz is None else f_Hz
    lam = pub.get("Lambda_MeV", LAMBDA_MEV)
    return float(pub["x_KSVZ"] * g_gamma_to_gagg(freq_to_mass_eV(f), G_GAMMA["KSVZ"], lam))


def preset_limit(name: str, f_Hz: float | None = None) -> dict:
    """Reconstructed exclusion |g_agg| at the preset's reference frequency vs the published one."""
    p = PRESETS[name]
    cav, rx, a = preset_models(name)
    f = p["published"]["f_Hz"] if f_Hz is None else f_Hz
    snr = snr_required(a.get("threshold_sigma"), a.get("confidence", 0.90), a.get("snr_target"))
    rho = a.get("rho_GeV_cm3", 0.45)
    Q_a = a.get("Q_a")
    if a["mode"] == "scan":
        g = g_limit_scan(f, cav, rx, rate_Hz_s=a["rate_Hz_s"], snr=snr, Q_a=Q_a, rho_GeV_cm3=rho)
    elif a["mode"] == "dwell":
        g = g_limit_dwell(f, a.get("f_c_Hz", f), cav, rx, t_s=a["t_s"], snr=snr, rho_GeV_cm3=rho, Q_a=Q_a)
    else:
        raise ValueError(f"preset {name}: analysis mode must be 'scan' or 'dwell'")
    g_pub = published_g(name, f)
    return {"name": name, "f_Hz": f, "g_GeV": g, "g_published_GeV": g_pub,
            "ratio": g / g_pub if g_pub else None, "snr": snr,
            "x_KSVZ": g / float(g_axion(freq_to_mass_eV(f), "KSVZ"))}


def implied_rate(name: str, f_Hz: float | None = None) -> dict:
    """Scan rate (and live time over the preset's band) that the published limit implies, for
    runs whose step size or live time is not published  (H6.5)."""
    p = PRESETS[name]
    cav, rx, a = preset_models(name)
    f = p["published"]["f_Hz"] if f_Hz is None else f_Hz
    g_pub = published_g(name, f)
    snr = snr_required(a.get("threshold_sigma"), a.get("confidence", 0.90), a.get("snr_target"))
    r = scan_rate(f, cav, rx, g_GeV=g_pub, snr=snr, Q_a=a.get("Q_a"), rho_GeV_cm3=a.get("rho_GeV_cm3", 0.45))
    band = p["f_range_Hz"][1] - p["f_range_Hz"][0]
    return {"rate_Hz_s": r, "live_time_s": band / r, "live_time_days": band / r / 86400.0}
