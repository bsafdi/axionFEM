#!/usr/bin/env python3
"""Independent reference implementation of docs/physics/scan_and_signal.md.

Readout, signal/noise PSDs, Asimov statistics and optimal-scan formulas for
lumped-element axion detectors (ABRACADABRA / DMRadio class).  Every numbered
spec equation "(S<n>)" in docs/physics/scan_and_signal.md has a function here.

Conventions (see spec Sec. 1-3 for the full statement):

* Internally everything is in natural Heaviside-Lorentz (HL) units,
  hbar = c = k_B = eps0 = mu0 = 1, with energies in eV.  Function names carry a
  suffix when the arguments are NOT natural: ``_si`` (SI), ``_Hz``, ``_K`` ...
* Geometric detector inputs are given the way the FEM produces them, in the
  natural-HL geometric definition but expressed with B in tesla and lengths
  in metres:  Psi in T m^3, L_p in m.  SI values: Psi_SI = mu0*Psi [T H m^2],
  L_SI = mu0*L_p [H].
* PSDs use the Foster et al. (1711.10489) normalisation
  <X^2> = (1/pi) int_0^inf domega S_X(omega)   (= "two-sided" PSD in omega,
  per unit dnu, i.e. half the usual one-sided-in-nu PSD).
* eta_A = 2 sqrt(S_Phi S_J)/hbar (=1 at the SQL in this normalisation),
  xi = 2 R S_Phi / (hbar omega0 M^2) (= alpha_1 of 1803.01627 Eq. F41).
* Phi_a is the *amplitude* of the axion-induced flux, Phi_a = kappa_a Psi,
  kappa_a = g_agg sqrt(2 rho_DM).

Dependencies: numpy, scipy only.  Run ``python scan_reference.py`` to print
all check values and (re)write scan_reference_values.json next to this file.
Light: a few seconds on one core.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, special

# =============================================================================
# 1. Constants and unit conversions  (spec Sec. 1)
# =============================================================================
# CODATA 2018 / SI-2019 values (identical to scipy.constants >= 1.4).
HBAR_JS = 1.054571817e-34        # J s (exact)
C_MS = 299792458.0               # m/s (exact)
QE_C = 1.602176634e-19           # C; also J per eV (exact)
KB_JK = 1.380649e-23             # J/K (exact)
MU0_HM = 1.25663706212e-6        # H/m (CODATA 2018; 4 pi 1e-7 differs by 5.5e-10)
ALPHA_EM = 7.2973525693e-3       # fine-structure constant
YEAR_S = 365.25 * 86400.0        # Julian year, s ("yr")

HBARC_EVM = HBAR_JS * C_MS / QE_C        # eV m     = 1.973269804e-7
HBAR_EVS = HBAR_JS / QE_C                # eV s     = 6.582119569e-16
M_IN_INVEV = 1.0 / HBARC_EVM             # 1 m  = 5.0677307e6 eV^-1
S_IN_INVEV = 1.0 / HBAR_EVS              # 1 s  = 1.5192674e15 eV^-1
K_IN_EV = KB_JK / QE_C                   # 1 K  = 8.617333262e-5 eV
# HL: B_nat^2/2 = B_SI^2/(2 mu0) converted to eV^4  =>  1 T = 195.35 eV^2
T_IN_EV2 = math.sqrt(HBARC_EVM ** 3 / (MU0_HM * QE_C))
WB_IN_NAT = T_IN_EV2 * M_IN_INVEV ** 2   # 1 Wb = 1 T m^2 in natural HL (dimensionless)
GEV_CM3_IN_EV4 = 1e9 * (HBARC_EVM * 1e2) ** 3   # 1 GeV/cm^3 in eV^4
E_HL = math.sqrt(4.0 * math.pi * ALPHA_EM)      # electron charge, natural HL

# Composite conversions used for the FEM-facing quantities
PSI_T_M3_IN_INVEV = T_IN_EV2 * M_IN_INVEV ** 3          # Psi:   1 T m^3 -> eV^-1
L_M_IN_INVEV = M_IN_INVEV                               # L_p:   1 m     -> eV^-1
F_T4M10_IN_INVEV2 = T_IN_EV2 ** 4 * M_IN_INVEV ** 10    # F:     1 T^4 m^10 -> eV^-2
ENERGY_T2M3_IN_J = T_IN_EV2 ** 2 * M_IN_INVEV ** 3 * QE_C  # (1/2)int B^2 unit: = 1/mu0 J


def unit_table() -> dict:
    """All conversion factors of spec Table 1 (natural HL <-> SI)."""
    return {
        "hbar_J_s": HBAR_JS, "c_m_per_s": C_MS, "e_C": QE_C, "kB_J_per_K": KB_JK,
        "mu0_H_per_m": MU0_HM, "alpha_em": ALPHA_EM, "year_s": YEAR_S,
        "hbarc_eV_m": HBARC_EVM, "hbar_eV_s": HBAR_EVS,
        "one_m_in_inv_eV": M_IN_INVEV, "one_s_in_inv_eV": S_IN_INVEV,
        "one_K_in_eV": K_IN_EV, "one_T_in_eV2": T_IN_EV2,
        "one_Wb_natural": WB_IN_NAT, "one_GeV_per_cm3_in_eV4": GEV_CM3_IN_EV4,
        "e_HL_natural": E_HL,
        "one_T_m3_in_inv_eV (Psi)": PSI_T_M3_IN_INVEV,
        "one_T4_m10_in_inv_eV2 (F)": F_T4M10_IN_INVEV2,
        "one_T2_m3_in_J (energy, =1/mu0)": ENERGY_T2M3_IN_J,
        "one_T2_m3_in_MJ": ENERGY_T2M3_IN_J * 1e-6,
        "flux_quantum_natural_pi_over_e": math.pi / E_HL,
        "flux_quantum_SI_times_Wb_natural": math.pi * HBAR_JS / QE_C * WB_IN_NAT,
    }


def psi_si(psi_T_m3):
    """Psi in SI [T H m^2] from the natural-HL geometric value [T m^3]  (S1.3)."""
    return MU0_HM * np.asarray(psi_T_m3)


def L_si(L_m):
    """Inductance in henry from the natural-HL geometric value in metres (S1.3)."""
    return MU0_HM * np.asarray(L_m)


def F_si(F_T4m10):
    """F = Psi^4/L^2 in SI [T^4 H^2 m^8] from natural [T^4 m^10] (= mu0^2 F) (S1.4)."""
    return MU0_HM ** 2 * np.asarray(F_T4m10)


def omega_from_mass_eV(m_eV):
    """Angular frequency [rad/s] of an axion of mass m [eV]: omega = m c^2/hbar."""
    return np.asarray(m_eV) / HBAR_EVS


def freq_Hz_from_mass_eV(m_eV):
    """Frequency [Hz] for mass [eV]."""
    return omega_from_mass_eV(m_eV) / (2 * math.pi)


# =============================================================================
# 2. Axion inputs  (spec Sec. 2)
# =============================================================================
MAFA_GEV2 = 5.70e-3            # m_a f_a [GeV^2], 1511.02867
C_GAMMA = {"chiPT_1511.02867": 1.92, "lattice_Brandt": 1.77, "lattice_Lin": 1.965}
E_OVER_N = {"KSVZ": 0.0, "DFSZ-I": 8.0 / 3.0, "DFSZ-II": 2.0 / 3.0}
RHO_DM_GEV_CM3 = 0.3
V0_KMS, VOBS_KMS = 220.0, 232.0
C_KMS = C_MS / 1e3


def g_agg_GeV(m_eV, model="DFSZ-I", C_gamma=1.77, mafa_GeV2=MAFA_GEV2):
    """|g_agg| [GeV^-1] on a QCD-axion line  (S2.1).

    g = alpha/(2 pi f_a) * |E/N - C_gamma|, f_a = (m_a f_a)/m_a.
    m_eV: axion mass [eV]; model: key of E_OVER_N or a float E/N.
    """
    EN = E_OVER_N[model] if isinstance(model, str) else float(model)
    fa_GeV = mafa_GeV2 / (np.asarray(m_eV) * 1e-9)
    return ALPHA_EM / (2 * math.pi * fa_GeV) * abs(EN - C_gamma)


def rho_eV4(rho_GeV_cm3=RHO_DM_GEV_CM3):
    """DM density [eV^4] from GeV/cm^3."""
    return rho_GeV_cm3 * GEV_CM3_IN_EV4


def kappa_a_eV(g_GeV, rho_GeV_cm3=RHO_DM_GEV_CM3):
    """kappa_a = g sqrt(2 rho) [eV] (natural HL)  (S2.2)."""
    return np.asarray(g_GeV) * 1e-9 * math.sqrt(2 * rho_eV4(rho_GeV_cm3))


def kappa_a_si(g_GeV, rho_GeV_cm3=RHO_DM_GEV_CM3):
    """SI kappa_a [H^-1] with Phi_a[Wb] = kappa_SI * Psi_SI[T H m^2]  (S2.3).

    kappa_SI = g_SI sqrt(2 rho_SI hbar c)/mu0, g_SI = g[GeV^-1]/(1e9 e) [J^-1].
    Also returns kappa in m^-1 (= mu0 kappa_SI), the factor with
    J_eff = kappa[m^-1] B/mu0 [A/m^2].
    """
    g_si = np.asarray(g_GeV) / (1e9 * QE_C)
    rho_si = rho_GeV_cm3 * 1e9 * QE_C * 1e6
    k_inv_m = g_si * math.sqrt(2 * rho_si * HBAR_JS * C_MS)
    return k_inv_m / MU0_HM, k_inv_m


def f_shm(v, v0=V0_KMS / C_KMS, vobs=VOBS_KMS / C_KMS):
    """SHM lab-frame speed distribution f(v), v in units of c  (S2.4; 1711.10489 Eq. 9).

    Normalised: int_0^inf f dv = 1 (no escape-speed truncation).
    """
    v = np.asarray(v, dtype=float)
    pref = v / (math.sqrt(math.pi) * v0 * vobs)
    return pref * (np.exp(-(v - vobs) ** 2 / v0 ** 2) - np.exp(-(v + vobs) ** 2 / v0 ** 2))


def line_shape(omega, m, v0=V0_KMS / C_KMS, vobs=VOBS_KMS / C_KMS):
    """Normalised line shape in omega, F(omega) = f(v)/(m v), v = sqrt(2(omega/m - 1))  (S2.5).

    omega, m in the same units (any); returns 1/[omega units]; int F domega = 1.
    """
    omega = np.asarray(omega, dtype=float)
    u = np.clip(omega / m - 1.0, 0.0, None)
    v = np.sqrt(2 * u)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(v > 0, f_shm(v, v0, vobs) / (m * np.where(v > 0, v, 1.0)), 0.0)
    return out


def Q_axion(v0_kms=V0_KMS, vobs_kms=VOBS_KMS):
    """Q_axion = int_0^inf dv f(v)^2/v  (S2.6).  Dimensionless."""
    v0, vo = v0_kms / C_KMS, vobs_kms / C_KMS
    vmax = (vobs_kms + 9 * v0_kms) / C_KMS
    q = integrate.quad(lambda v: f_shm(v, v0, vo) ** 2 / v, 0, vmax,
                       epsabs=0, epsrel=1e-13, limit=500)[0]
    return q


def v_grid(n=256, v0=V0_KMS / C_KMS, vobs=VOBS_KMS / C_KMS):
    """Gauss-Legendre nodes/weights in v (units of c) covering the SHM line."""
    x, w = np.polynomial.legendre.leggauss(n)
    vmax = vobs + 7 * v0
    return 0.5 * vmax * (x + 1), 0.5 * vmax * w


# =============================================================================
# 3. Readout circuit and noise model  (spec Sec. 3)
# =============================================================================
def n_bose(omega, T):
    """Exact Bose-Einstein occupation n = 1/(exp(omega/T)-1); omega, T in eV  (S3.1)."""
    x = np.asarray(omega, dtype=float) / np.asarray(T, dtype=float)
    with np.errstate(over="ignore"):
        return 1.0 / np.expm1(x)


def xi_opt(n, eta):
    """Optimal imprecision parameter xi (1803.01627 F54)  (S3.9)."""
    b = 2 * np.asarray(n, dtype=float) + 1
    return 2 * eta ** 2 / (b + np.sqrt(b * b + 8 * eta ** 2))


def Gbar(xi, n, eta):
    """G-bar(xi) = xi/[xi^2 + 2(2n+1) xi + eta^2]^{3/2}  (S3.8; 1803.01627 F53)."""
    xi = np.asarray(xi, dtype=float)
    return xi / (xi * xi + 2 * (2 * np.asarray(n) + 1) * xi + eta ** 2) ** 1.5


def Gbar_opt(n, eta):
    """G-bar at xi_opt (exact in n, eta)  (S3.10)."""
    return Gbar(xi_opt(n, eta), n, eta)


def Gbar_highT(omega0, T, eta):
    """n >> 1, n >> eta limit: G-bar ~ omega0/(6 sqrt3 eta T)  (S3.11). omega0, T in eV."""
    return np.asarray(omega0) / (6 * math.sqrt(3) * eta * T)


def A_noise(xi, n, eta):
    """A = xi^2 + 2(2n+1) xi + eta^2 (the on-resonance noise number)  (S3.7)."""
    return xi * xi + 2 * (2 * n + 1) * xi + eta ** 2


def sensitivity_bandwidths(omega0, Q, n, eta, xi=None):
    """Three sensitivity-bandwidth definitions (full widths, same units as omega0)  (S3.12).

    fw_noise_doubling : |x| < sqrt(A)/xi   (1803.01627 F45-F46)
    fw_thermal_gt_squid: thermal > total SQUID noise
    equivalent_width  : int domega0 theta(omega0)/theta_max = (pi/4)(omega0/Q) sqrt(A)/xi
    """
    if xi is None:
        xi = xi_opt(n, eta)
    A = A_noise(xi, n, eta)
    lw = omega0 / Q
    arg = np.maximum(2 * xi * (2 * n + 1) - eta ** 2 - xi ** 2, 0.0)
    return {
        "resonator_linewidth": lw,
        "fw_noise_doubling": lw * np.sqrt(A) / xi,
        "fw_thermal_gt_squid": lw * np.sqrt(arg) / xi,
        "equivalent_width": 0.25 * math.pi * lw * np.sqrt(A) / xi,
    }


@dataclass
class Detector:
    """Lumped-element detector + readout. FEM-facing units.

    Psi_T_m3 : coupling Psi = (1/I) int A_I . B_lab d^3x, natural-HL geometry, B in T  [T m^3]
    Lp_m     : pickup inductance (1/I^2) int |B_I|^2, natural-HL geometry  [m]
               (L_SI = mu0 * Lp_m henry)
    Q        : intrinsic resonator quality factor omega0 L_p / R
    T_K      : loss-weighted physical temperature [K]
    eta_A    : amplifier noise, 2 sqrt(S_Phi S_J)/hbar (Foster-normalised PSDs)
    R_exp    : R(omega) = R(omega0) (omega/omega0)^R_exp across one tuning (0: constant)
    """
    Psi_T_m3: float
    Lp_m: float
    Q: float
    T_K: float
    eta_A: float
    R_exp: float = 0.0

    @property
    def F_T4m10(self):
        return self.Psi_T_m3 ** 4 / self.Lp_m ** 2


def psd_signal_voltage(omega, m, Phi_a, v0=V0_KMS / C_KMS, vobs=VOBS_KMS / C_KMS):
    """Mean axion EMF PSD S_{V,a}(omega) (Foster norm.)  (S3.2).

    S_{V,a} = (pi/2) Phi_a^2 (omega^4/m^2) F(omega);  natural units:
    omega, m [eV], Phi_a [dimensionless flux amplitude] -> [eV] (V^2 per rad/s).
    Check: (1/pi) int S_{V,a} domega = <V^2> = (m Phi_a)^2/2.
    """
    return 0.5 * math.pi * Phi_a ** 2 * np.asarray(omega) ** 4 / m ** 2 * line_shape(omega, m, v0, vobs)


def circuit_noise_terms(omega, omega0, L, Q, T, eta, xi, R_exp=0.0):
    """Noise PSDs (Foster norm., natural units) for one tuning  (S3.3-S3.6).

    Returns dict of arrays (all eV, i.e. V^2/(rad/s) or A^2/(rad/s) natural):
      Z2      : |Z(omega)|^2 = R(omega)^2 + (omega L - 1/(omega C))^2, C = 1/(omega0^2 L)
      SV_th   : R(omega) omega (2 n(omega,T) + 1)       thermal+zero-point voltage
      SV_ba   : omega^2 M^2 S_J  = eta^2 R0 omega^2 / (2 xi omega0)
      SI_imp  : S_Phi/M^2        = xi omega0 / (2 R0)
      NV      : total noise referred to a series EMF = SV_th + SV_ba + Z2 SI_imp
      lamB    : total current-noise PSD = (SV_th + SV_ba)/Z2 + SI_imp
    with R0 = omega0 L / Q and R(omega) = R0 (omega/omega0)^R_exp.
    """
    omega = np.asarray(omega, dtype=float)
    R0 = omega0 * L / Q
    R = R0 * (omega / omega0) ** R_exp
    X = L * (omega - omega0) * (omega + omega0) / omega
    Z2 = R * R + X * X
    SV_th = R * omega * (2 * n_bose(omega, T) + 1)
    SV_ba = eta ** 2 * R0 * omega ** 2 / (2 * xi * omega0)
    SI_imp = xi * omega0 / (2 * R0) * np.ones_like(omega)
    NV = SV_th + SV_ba + Z2 * SI_imp
    return dict(Z2=Z2, SV_th=SV_th, SV_ba=SV_ba, SI_imp=SI_imp, NV=NV,
                lamB=(SV_th + SV_ba) / Z2 + SI_imp)


def coupling_from_xi_si(xi, omega0_rad_s, R_ohm, S_Phi_si, S_J_si):
    """SI: mutual inductance M [H] realising xi = 2 R S_Phi/(hbar omega0 M^2)  (S3.5).

    S_Phi [Wb^2 s] and S_J [A^2 s] in Foster normalisation (= half the one-sided
    PSDs in Wb^2/Hz, A^2/Hz).  Returns (M, eta_A).
    """
    M = math.sqrt(2 * R_ohm * S_Phi_si / (HBAR_JS * omega0_rad_s * xi))
    return M, 2 * math.sqrt(S_Phi_si * S_J_si) / HBAR_JS


# =============================================================================
# 4. Statistics  (spec Sec. 4)
# =============================================================================
THETA95 = 2.71                                  # 1711.10489 value
THETA95_EXACT = float(special.ndtri(0.95) ** 2)  # 2.705543...


def h_weak(s):
    """Weak-signal Asimov TS density: s^2  (S4.2)."""
    return np.asarray(s) ** 2


def h_limit(s):
    """Exact Asimov TS density for the expected upper limit (null Asimov data)  (S4.3).

    -q per bin = 2[ln(1+s) - s/(1+s)];  -> s^2 for s << 1.
    """
    s = np.asarray(s, dtype=float)
    return 2 * (np.log1p(s) - s / (1 + s))


def h_discovery(s):
    """Exact Asimov TS density for discovery (signal Asimov data)  (S4.4).

    TS per bin = 2[s - ln(1+s)];  -> s^2 for s << 1.
    """
    s = np.asarray(s, dtype=float)
    return 2 * (s - np.log1p(s))


H_FUNCS = {"weak": h_weak, "limit": h_limit, "discovery": h_discovery}


def limit_band_q(nsigma):
    """|q| defining the +-N sigma expected-limit band  (S4.5; 1711.10489 Eq. 56)."""
    z = special.ndtri(0.95)
    return (z + nsigma) ** 2, (z - nsigma) ** 2


def n_independent_masses(f_min_Hz, f_max_Hz, v0_kms=V0_KMS, vobs_kms=VOBS_KMS, alpha_lee=0.75):
    """Look-elsewhere trials factor N_ma = ln(fmax/fmin)/(alpha v0 vobs)  (S4.6; 1711.10489 Eq. 62)."""
    return math.log(f_max_Hz / f_min_Hz) / (alpha_lee * (v0_kms / C_KMS) * (vobs_kms / C_KMS))


def ts_discovery_threshold(global_sigma, N_ma, sided=1):
    """TS_thresh = [Phi^-1(1 - p/N_ma)]^2  (S4.7; 1711.10489 Eqs. 58-60).

    sided=1 (default, as stated in 1711.10489 text and its N=1 example TS=9 at 3 sigma):
    p = 1 - Phi(global_sigma).  sided=2 reproduces the numerical LEE examples
    quoted in 1711.10489 Sec. III D (40.9 / 57.5), which use p = 2(1 - Phi).
    """
    p = sided * special.ndtr(-global_sigma)
    return float(special.ndtri(1 - p / N_ma) ** 2)


# =============================================================================
# 5. Scan rate, requirement, time allocation, reach  (spec Sec. 5)
# =============================================================================
def scan_rate_nat(Phi_a, L_nat, Q, G, Qa, Theta=THETA95):
    """Local optimal scan rate d omega0/dt [eV^2]  (S5.1).

    d omega0/dt = (pi^2/8) (Qa/Theta) Phi_a^4/L^2 Q G-bar  (natural units).
    """
    return math.pi ** 2 / 8 * Qa / Theta * Phi_a ** 4 / L_nat ** 2 * Q * G


def scan_rate_si(Phi_a_Wb, L_H, Q, G, Qa, Theta=THETA95):
    """Same in SI: d omega0/dt [rad/s^2] = (pi^2/8)(Qa/Theta) Phi_a^4/(hbar^2 L^2) Q G  (S5.2)."""
    return math.pi ** 2 / 8 * Qa / Theta * Phi_a_Wb ** 4 / (HBAR_JS ** 2 * L_H ** 2) * Q * G


def scan_rate_detector(m_eV, g_GeV, det: Detector, Qa, Theta=THETA95, rho=RHO_DM_GEV_CM3,
                       exact_G=True):
    """d omega0/dt in rad/s^2 for a Detector at tuning omega0 = m (both in SI units)."""
    w = np.asarray(m_eV, dtype=float)
    T = det.T_K * K_IN_EV
    G = Gbar_opt(n_bose(w, T), det.eta_A) if exact_G else Gbar_highT(w, T, det.eta_A)
    Phi = kappa_a_eV(g_GeV, rho) * det.Psi_T_m3 * PSI_T_M3_IN_INVEV
    rate_nat = scan_rate_nat(Phi, det.Lp_m * L_M_IN_INVEV, det.Q, G, Qa, Theta)
    return rate_nat * S_IN_INVEV ** 2       # eV^2 -> (rad/s)/s


def F_req_closed_form(eta, T_K, Q, Qa, g1_GeV, t_s, m1_eV, m2_eV, Theta=THETA95,
                      rho=RHO_DM_GEV_CM3):
    """Required geometric figure of merit F_req  (S5.4): returns F_req in eV^-2.

    F_req = (3 sqrt3/pi^2) Theta eta T / (Qa Q rho^2 g1^4 t) [1 - (omega1/omega2)^4]
    (n >> 1, n >> eta, g proportional to m, constant Q, T, eta).
    """
    T = T_K * K_IN_EV
    t = t_s * S_IN_INVEV
    g1 = g1_GeV * 1e-9
    r4 = (m1_eV / m2_eV) ** 4
    return (3 * math.sqrt(3) / math.pi ** 2 * Theta * eta * T
            / (Qa * Q * rho_eV4(rho) ** 2 * g1 ** 4 * t) * (1 - r4))


def F_req_closed_form_si(eta, T_K, Q, Qa, g1_GeV, t_s, m1_eV, m2_eV, Theta=THETA95,
                         rho=RHO_DM_GEV_CM3):
    """SI form of (S5.4): F_req [T^4 H^2 m^8]
    = (12 sqrt3/pi^2) Theta eta kB T hbar / (Qa Q kappa_SI(m1)^4 t) [1 - (w1/w2)^4],
    kappa_SI = g_SI sqrt(2 rho_SI hbar c)/mu0 [H^-1]."""
    k_si, _ = kappa_a_si(g1_GeV, rho)
    return (12 * math.sqrt(3) / math.pi ** 2 * Theta * eta * KB_JK * T_K * HBAR_JS
            / (Qa * Q * k_si ** 4 * t_s) * (1 - (m1_eV / m2_eV) ** 4))


def F_req_general(m1_eV, m2_eV, t_s, Q, T_K, eta, Qa, g_of_m=None, Theta=THETA95,
                  rho=RHO_DM_GEV_CM3, exact_G=True):
    """General requirement  (S5.3): F_req = 8 Theta/(pi^2 Qa t) int dw /(Q k(w)^4 G(w)).

    Q, T_K, eta may be callables of m_eV. g_of_m: callable m_eV -> g [GeV^-1]
    (default DFSZ-I with C=1.77).  Returns eV^-2.
    """
    if g_of_m is None:
        g_of_m = lambda m: g_agg_GeV(m, "DFSZ-I", 1.77)
    call = lambda p, m: p(m) if callable(p) else p

    def integrand(lnm):
        m = math.exp(lnm)
        T = call(T_K, m) * K_IN_EV
        e = call(eta, m)
        G = Gbar_opt(n_bose(m, T), e) if exact_G else Gbar_highT(m, T, e)
        return m / (call(Q, m) * kappa_a_eV(g_of_m(m), rho) ** 4 * G)

    I = integrate.quad(integrand, math.log(m1_eV), math.log(m2_eV), epsabs=0, epsrel=1e-12,
                       limit=200)[0]
    return 8 * Theta / (math.pi ** 2 * Qa * t_s * S_IN_INVEV) * I


def dwell_density_target(m_eV, det: Detector, Qa, g_of_m, Theta=THETA95, rho=RHO_DM_GEV_CM3):
    """Target-curve allocation tau(omega0) = 1/(d omega0/dt at g_t)  [s per rad/s]  (S5.6)."""
    return 1.0 / scan_rate_detector(m_eV, g_of_m(m_eV), det, Qa, Theta, rho)


def time_target_curve_s(m1_eV, m2_eV, det: Detector, Qa, g_of_m, Theta=THETA95,
                        rho=RHO_DM_GEV_CM3):
    """Total live time [s] to reach g_of_m everywhere in [m1, m2] (local approx.)  (S5.6)."""
    f = lambda lnm: math.exp(lnm) / HBAR_EVS * dwell_density_target(math.exp(lnm), det, Qa,
                                                                     g_of_m, Theta, rho)
    return integrate.quad(f, math.log(m1_eV), math.log(m2_eV), epsabs=0, epsrel=1e-11,
                          limit=200)[0]


def g95_local(m_eV, tau_s_per_rad_s, det: Detector, Qa, Theta=THETA95, rho=RHO_DM_GEV_CM3):
    """Local reach g_95(m) [GeV^-1] for dwell density tau(omega0=m) [s/(rad/s)]  (S5.8).

    g95^4 = 2 Theta / (pi^2 tau Qa rho^2 F Q G-bar)  (natural units).
    """
    m = np.asarray(m_eV, dtype=float)
    T = det.T_K * K_IN_EV
    G = Gbar_opt(n_bose(m, T), det.eta_A)
    tau_nat = np.asarray(tau_s_per_rad_s) * S_IN_INVEV ** 2
    F = det.F_T4m10 * F_T4M10_IN_INVEV2
    g4 = 2 * Theta / (math.pi ** 2 * tau_nat * Qa * rho_eV4(rho) ** 2 * F * det.Q * G)
    return g4 ** 0.25 * 1e9


def tau_log_uniform(m_eV, t_total_s, m1_eV, m2_eV):
    """Log-uniform allocation (S5.7; 1803.01627 Eq. 245): tau(omega0) = t/(omega0 ln(w2/w1)) [s/(rad/s)]."""
    return t_total_s / (omega_from_mass_eV(m_eV) * math.log(m2_eV / m1_eV))


def tau_target(m_eV, t_total_s, m1_eV, m2_eV, det: Detector, Qa, g_of_m, Theta=THETA95,
               rho=RHO_DM_GEV_CM3):
    """Target-curve allocation normalised to total time t_total  [s/(rad/s)]  (S5.6)."""
    t_req = time_target_curve_s(m1_eV, m2_eV, det, Qa, g_of_m, Theta, rho)
    return dwell_density_target(m_eV, det, Qa, g_of_m, Theta, rho) * t_total_s / t_req


# ---------------------------------------------------------------------------
# Exact bin-by-bin Asimov sum for a discrete list of tunings  (S5.9 / S6.1)
# ---------------------------------------------------------------------------
def theta_per_tuning(m_eV, g_GeV, omega0_rad_s, t_s, det: Detector, Qa=None, xi=None,
                     stat="weak", rho=RHO_DM_GEV_CM3, n_v=256, v0_kms=V0_KMS,
                     vobs_kms=VOBS_KMS, return_smax=False):
    """Asimov TS contribution of each tuning to an axion at mass m  (S4.1-S4.4, S6.1).

    Theta_i = (t_i/2pi) int domega h(s_i(omega)),  s_i = lam_a/lam_B = S_{V,a}/N_V,
    with the full series-RLC impedance, exact n(omega), backaction ~ omega^2 and
    the exact SHM line (integrated in v with Gauss-Legendre, omega = m(1+v^2/2)).
    Qa is unused (kept for signature symmetry).  xi: per-tuning xi (default xi_opt(n(omega0))).
    Returns array Theta_i (dimensionless).
    """
    v0, vo = v0_kms / C_KMS, vobs_kms / C_KMS
    m = float(m_eV)
    w0 = np.atleast_1d(np.asarray(omega0_rad_s, dtype=float)) * HBAR_EVS
    t = np.atleast_1d(np.asarray(t_s, dtype=float)) * S_IN_INVEV
    L = det.Lp_m * L_M_IN_INVEV
    T = det.T_K * K_IN_EV
    if xi is None:
        xi = xi_opt(n_bose(w0, T), det.eta_A)
    xi = np.broadcast_to(np.asarray(xi, dtype=float), w0.shape)
    Phi = float(kappa_a_eV(g_GeV, rho)) * det.Psi_T_m3 * PSI_T_M3_IN_INVEV
    v, wv = v_grid(n_v, v0, vo)
    dm = m * v * v / 2                            # omega - m, exact to roundoff
    omega = m + dm
    SVa = 0.5 * math.pi * Phi ** 2 * omega ** 4 / m ** 2 * f_shm(v, v0, vo) / (m * v)
    # noise for every (tuning, v) pair; compute omega - omega0 without cancellation
    d = (m - w0)[:, None] + dm[None, :]
    R0 = (w0 * L / det.Q)[:, None]
    R = R0 * (omega[None, :] / w0[:, None]) ** det.R_exp
    X = L * d * (omega[None, :] + w0[:, None]) / omega[None, :]
    Z2 = R * R + X * X
    SVth = R * omega[None, :] * (2 * n_bose(omega, T)[None, :] + 1)
    SVba = det.eta_A ** 2 * R0 * omega[None, :] ** 2 / (2 * xi[:, None] * w0[:, None])
    SIim = xi[:, None] * w0[:, None] / (2 * R0)
    s = SVa[None, :] / (SVth + SVba + Z2 * SIim)
    hs = H_FUNCS[stat](s)
    theta = t / (2 * math.pi) * np.sum(hs * (m * v * wv)[None, :], axis=1)
    if return_smax:
        return theta, float(np.max(s))
    return theta


def theta_discrete(m_eV, g_GeV, omega0_rad_s, t_s, det, **kw):
    """Total Asimov TS of a discrete scan: sum_i Theta_i  (S5.9)."""
    return float(np.sum(theta_per_tuning(m_eV, g_GeV, omega0_rad_s, t_s, det, **kw)))


def g95_discrete(m_eV, omega0_rad_s, t_s, det, stat="weak", Theta=THETA95, g_ref=1e-19, **kw):
    """Reach g_95(m) [GeV^-1] of a discrete scan by solving Theta(m, g) = Theta95  (S5.10)."""
    th = theta_discrete(m_eV, g_ref, omega0_rad_s, t_s, det, stat="weak", **kw)
    g_weak = g_ref * (Theta / th) ** 0.25
    if stat == "weak":
        return g_weak
    fn = lambda lg: theta_discrete(m_eV, math.exp(lg), omega0_rad_s, t_s, det, stat=stat,
                                   **kw) - Theta
    return math.exp(optimize.brentq(fn, math.log(g_weak) - 2, math.log(g_weak) + 2,
                                    xtol=1e-12))


def dense_tunings_around(m_eV, det: Detector, tau_fn, half_widths=40.0, steps_per_hw=6):
    """Discrete tunings covering +-half_widths noise-doubling half-widths around m.

    tau_fn(m_eV_array) -> dwell density [s/(rad/s)]. Dwell t_i = tau(omega_i) * step.
    Returns (omega0_rad_s, t_s).
    """
    w = float(omega_from_mass_eV(m_eV))
    T = det.T_K * K_IN_EV
    n = float(n_bose(m_eV, T))
    hw = 0.5 * float(sensitivity_bandwidths(w, det.Q, n, det.eta_A)["fw_noise_doubling"])
    N = int(2 * half_widths * steps_per_hw) + 1
    w0 = w + hw * np.linspace(-half_widths, half_widths, N)
    step = hw / steps_per_hw
    t = tau_fn(w0 * HBAR_EVS) * step
    return w0, t


# =============================================================================
# 6. Finite-band (broadband) generalisation  (spec Sec. 6)
# =============================================================================
def kernel_matrix(m_targets_eV, g_targets_GeV, settings_w0_rad_s, settings_xi, det,
                  n_v=96, **kw):
    """K[k, j] = Theta per second of setting j at target (m_k, g_k)  (S6.2).

    Full-impedance, full-line Asimov (weak) kernel.  Units: 1/s.
    """
    K = np.empty((len(m_targets_eV), len(settings_w0_rad_s)))
    for k, (m, g) in enumerate(zip(m_targets_eV, g_targets_GeV)):
        K[k] = theta_per_tuning(m, g, settings_w0_rad_s, 1.0, det, xi=settings_xi,
                                n_v=n_v, **kw)
    return K


def kernel_narrow_line(m_eV, g_GeV, w0_rad_s, xi, det, Qa, rho=RHO_DM_GEV_CM3):
    """Narrow-line kernel (pi/8) Qa m^3 Phi_a^4 / N_V(m)^2 per unit time [1/s]  (S6.3)."""
    m = float(m_eV)
    L = det.Lp_m * L_M_IN_INVEV
    T = det.T_K * K_IN_EV
    Phi = float(kappa_a_eV(g_GeV, rho)) * det.Psi_T_m3 * PSI_T_M3_IN_INVEV
    w0 = np.asarray(w0_rad_s) * HBAR_EVS
    NV = circuit_noise_terms(m, w0, L, det.Q, T, det.eta_A, np.asarray(xi), det.R_exp)["NV"]
    return math.pi / 8 * Qa * m ** 3 * Phi ** 4 / NV ** 2 * S_IN_INVEV


def lp_schedule(K, Theta=THETA95):
    """Minimum-total-time dwell schedule: min sum t_j s.t. K t >= Theta, t >= 0  (S6.4)."""
    K = np.asarray(K, dtype=float)
    best = K.max(axis=1)
    if np.any(best <= 0):
        raise ValueError("a target has no sensitive setting")
    t_unit = float(np.max(Theta / best))     # rescale so the LP is O(1) (HiGHS tolerances)
    res = optimize.linprog(np.ones(K.shape[1]), A_ub=-K * t_unit / Theta,
                           b_ub=-np.ones(K.shape[0]), bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    t = np.maximum(res.x, 0) * t_unit
    t *= max(1.0, float(Theta / np.min(K @ t)))   # enforce feasibility after round-off
    return t, float(np.sum(t))


# =============================================================================
# 7. Long-detector analytic factors (long solenoid, long toroid)
# =============================================================================
def g_solenoid(x):
    """g_sol = (1-x^2)^2/(4 ln(1/x))  (long solenoid)."""
    return (1 - x * x) ** 2 / (4 * math.log(1 / x))


def g_toroid(y):
    """g_tor = g_CORE = y^2 ln(1/y)  (long toroid)."""
    return y * y * math.log(1 / y)


# =============================================================================
# 8. Self-tests of identities used in the spec
# =============================================================================
def self_tests() -> dict:
    out = {}
    # (a) Lorentzian integral int dx/(A+xi^2 x^2)^2 = pi/(2 xi A^{3/2})
    A, xi = 7.3, 0.37
    num = integrate.quad(lambda x: 1 / (A + xi * xi * x * x) ** 2, -np.inf, np.inf, epsabs=0,
                         epsrel=1e-12)[0]
    out["lorentzian_identity_rel_err"] = num / (math.pi / (2 * xi * A ** 1.5)) - 1
    # (b) xi_opt maximises G-bar
    for n, eta in [(0.0, 1.0), (50.0, 3.0), (1e3, 0.1)]:
        r = optimize.minimize_scalar(lambda lx: -Gbar(math.exp(lx), n, eta),
                                     bounds=(-30, 10), method="bounded",
                                     options={"xatol": 1e-12})
        out[f"xi_opt_rel_err_n{n:g}_eta{eta:g}"] = math.exp(r.x) / xi_opt(n, eta) - 1
    # (c) PSD normalisation: (1/pi) int S_{V,a} domega = (m Phi)^2/2 <(omega/m)^4>
    #     (the exact EMF = -dPhi/dt carries omega^4/m^4 = 1 + O(v^2) ~ 1 + 2.8e-6)
    m, Phi = 1.0, 1.0
    v, w = v_grid(400)
    om = m * (1 + v * v / 2)
    S = psd_signal_voltage(om, m, Phi)
    mean_w4 = np.sum(line_shape(om, m) * (om / m) ** 4 * m * v * w)
    out["signal_psd_norm_rel_err"] = (np.sum(S * m * v * w) / math.pi) / (0.5 * m * m * Phi * Phi * mean_w4) - 1
    out["signal_psd_omega4_correction"] = mean_w4 - 1
    # (d) Q_axion via line shape: m int F^2 domega = Qa
    Fv = line_shape(om, m)
    out["Qa_lineshape_rel_err"] = m * np.sum(Fv ** 2 * m * v * w) / Q_axion() - 1
    # (e) natural vs SI scan rate for an arbitrary detector
    det = Detector(Psi_T_m3=3.0, Lp_m=0.7, Q=1e6, T_K=0.03, eta_A=10)
    mm, gg, Qa = 1e-9, 1e-19, 6.8e5
    r_nat = scan_rate_detector(mm, gg, det, Qa)
    k_si, _ = kappa_a_si(gg)
    T = det.T_K * K_IN_EV
    G = float(Gbar_opt(n_bose(mm, T), det.eta_A))
    r_si = scan_rate_si(k_si * psi_si(det.Psi_T_m3), L_si(det.Lp_m), det.Q, G, Qa)
    out["scan_rate_nat_vs_SI_rel_err"] = float(r_nat / r_si - 1)
    # (f) energy unit: 1 T^2 m^3 (HL) == 1/mu0 J
    out["energy_unit_rel_err"] = ENERGY_T2M3_IN_J * MU0_HM - 1
    # (g) flux quantum
    out["flux_quantum_rel_err"] = (math.pi * HBAR_JS / QE_C * WB_IN_NAT) / (math.pi / E_HL) - 1
    # (h) series of exact Asimov densities: h_lim = s^2 - 4s^3/3 + ..., h_disc = s^2 - 2s^3/3 + ...
    s = 1e-3
    out["h_limit_series_rel_err"] = float(h_limit(s) / (s * s - 4 * s ** 3 / 3 + 1.5 * s ** 4) - 1)
    out["h_discovery_series_rel_err"] = float(h_discovery(s) / (s * s - 2 * s ** 3 / 3 + 0.5 * s ** 4) - 1)
    return out


# =============================================================================
# main(): all check values
# =============================================================================
def rel(a, b):
    return float(a / b - 1)


def main(write_json=True) -> dict:
    res: dict = {"meta": {
        "spec": "docs/physics/scan_and_signal.md",
        "units": "keys carry units; natural HL: hbar=c=kB=eps0=mu0=1; "
                 "Psi in T m^3 and L_p in m are natural-HL geometric values",
        "psd_normalisation": "<X^2> = (1/pi) int_0^inf domega S_X (Foster et al. 1711.10489)",
        "eta_A": "2 sqrt(S_Phi S_J)/hbar (Foster-normalised), =1 at SQL",
        "xi": "2 R S_Phi/(hbar omega0 M^2) = alpha_1 of 1803.01627 Eq. F41",
        "Phi_a": "amplitude: Phi_a = kappa_a Psi, kappa_a = g sqrt(2 rho)",
        "Theta95": THETA95, "year_s": YEAR_S,
    }}
    checks = {}

    def chk(name, value, expected, tol):
        checks[name] = {"value": float(value), "expected": float(expected),
                        "rel_err": rel(value, expected), "tol": tol,
                        "pass": abs(rel(value, expected)) <= tol}

    # ---------------- units ----------------
    res["units_table"] = unit_table()

    # ---------------- axion inputs ----------------
    m1, m2 = 0.5e-9, 4.1e-9
    ax = {"m1_eV": m1, "m2_eV": m2,
          "f1_Hz": float(freq_Hz_from_mass_eV(m1)), "f2_Hz": float(freq_Hz_from_mass_eV(m2)),
          "rho_DM_eV4": rho_eV4()}
    for model in E_OVER_N:
        for cname, cval in C_GAMMA.items():
            ax[f"g_{model}_{cname}_at_0.5neV_GeV^-1"] = float(g_agg_GeV(m1, model, cval))
    ax["g_DFSZ-I_C1.77_at_1neV_GeV^-1"] = float(g_agg_GeV(1e-9))
    ax["g_DFSZ-I_C1.77_at_4.1neV_GeV^-1"] = float(g_agg_GeV(m2))
    k_si, k_inv_m = kappa_a_si(1e-19, 0.3)
    ax["kappa_a_g1e-19_rho0.3_eV"] = float(kappa_a_eV(1e-19, 0.3))
    ax["kappa_a_g1e-19_rho0.3_SI_H^-1"] = float(k_si)
    ax["kappa_a_g1e-19_rho0.3_m^-1"] = float(k_inv_m)
    # natural route: kappa[eV] -> kappa[m^-1] = kappa[eV] / (hbar c [eV m])
    ax["kappa_a_g1e-19_rho0.3_m^-1_via_natural"] = float(kappa_a_eV(1e-19, 0.3) * M_IN_INVEV)
    res["axion"] = ax
    chk("g_DFSZ(0.5 neV) rounded 9.1e-20", ax["g_DFSZ-I_lattice_Brandt_at_0.5neV_GeV^-1"], 9.1e-20, 6e-3)
    chk("f1 = 120.9 kHz", ax["f1_Hz"], 120.9e3, 5e-4)
    chk("kappa m^-1 SI vs natural route", k_inv_m, ax["kappa_a_g1e-19_rho0.3_m^-1_via_natural"], 1e-12)

    # ---------------- Q_axion ----------------
    Qa = Q_axion(V0_KMS, VOBS_KMS)
    qa = {"Qa_SHM_v0_220_vobs_232": Qa,
          "Qa_v0_220_vobs_220": Q_axion(220, 220),
          "Qa_v0_220_vobs_247(June max approx)": Q_axion(220, 247),
          "vobs_kms_giving_6.4e5_at_v0_220": optimize.brentq(lambda vo: Q_axion(220, vo) - 6.4e5,
                                                             232, 270, xtol=1e-6),
          "Qa_v0_sqrt3x220_vobs_235.2(1803.01627 B1 literal)": Q_axion(math.sqrt(3) * 220,
                                                                       220 + 29.8 * 0.51),
          "Qa_v0_235_vobs_250(SHM++-like)": Q_axion(235, 250)}
    res["Q_axion"] = qa
    chk("Q_axion rounded 6.8e5", Qa, 6.8e5, 5e-3)

    # ---------------- G-bar check grid ----------------
    grid = []
    for n, eta in [(0.0, 1.0), (1.0, 1.0), (10.0, 1.0), (100.0, 10.0), (1e3, 0.1), (1e3, 10.0),
                   (1e4, 20.0), (1739.5, 0.1)]:
        x = float(xi_opt(n, eta))
        grid.append({"n": n, "eta_A": eta, "xi_opt": x, "Gbar_opt": float(Gbar(x, n, eta)),
                     "A_opt": float(A_noise(x, n, eta)),
                     "Gbar_highT_1/(6sqrt3 eta (n+1/2))": 1 / (6 * math.sqrt(3) * eta * (n + 0.5)),
                     "Gbar_at_xi=1": float(Gbar(1.0, n, eta))})
    res["Gbar_grid"] = grid
    chk("Gbar(n=0,eta=1)=4/27", grid[0]["Gbar_opt"], 4 / 27, 1e-14)

    # ---------------- F_req benchmarks ----------------
    t3 = 3 * YEAR_S
    g1 = float(g_agg_GeV(m1))
    gdfsz = lambda m: g_agg_GeV(m, "DFSZ-I", 1.77)
    # rounded_*: the closed-form values rounded to two or three significant figures; they must agree within rounding
    bench = {"optimistic": dict(Q=2e7, T_K=0.010, eta=0.1, rounded_T4m10=1.04e4, rounded_SI=1.65e-8),
             "conservative_toroid_CORE": dict(Q=1e6, T_K=0.030, eta=10.0, rounded_T4m10=6.26e7,
                                              rounded_SI=9.9e-5),
             "conservative_solenoid": dict(Q=1e5, T_K=0.030, eta=10.0, rounded_T4m10=6.26e8,
                                           rounded_SI=9.9e-4)}
    Freq = {}
    for name, b in bench.items():
        Fc = F_req_closed_form(b["eta"], b["T_K"], b["Q"], 6.8e5, g1, t3, m1, m2)
        Fe = F_req_general(m1, m2, t3, b["Q"], b["T_K"], b["eta"], 6.8e5, gdfsz)
        Fe_Qa = F_req_general(m1, m2, t3, b["Q"], b["T_K"], b["eta"], Qa, gdfsz)
        Fe_th = F_req_general(m1, m2, t3, b["Q"], b["T_K"], b["eta"], 6.8e5, gdfsz,
                              Theta=THETA95_EXACT)
        Fc_1MHz = F_req_closed_form(b["eta"], b["T_K"], b["Q"], 6.8e5, g1, t3, m1,
                                    1e6 * 2 * math.pi * HBAR_EVS)
        d = {"F_req_closed_form_eV^-2": Fc,
             "F_req_closed_form_T4_m10": Fc / F_T4M10_IN_INVEV2,
             "F_req_closed_form_SI_T4_H2_m8": float(F_si(Fc / F_T4M10_IN_INVEV2)),
             "F_req_exactGbar_T4_m10": Fe / F_T4M10_IN_INVEV2,
             "exact_over_closed_minus1": Fe / Fc - 1,
             "F_req_exactGbar_Qa_exact_6.7795e5_T4_m10": Fe_Qa / F_T4M10_IN_INVEV2,
             "F_req_exactGbar_Theta95_2.7055_T4_m10": Fe_th / F_T4M10_IN_INVEV2,
             "F_req_closed_omega2_1MHz_T4_m10": Fc_1MHz / F_T4M10_IN_INVEV2,
             "sqrt_F_req_T2_m5": math.sqrt(Fc / F_T4M10_IN_INVEV2),
             "inputs": b}
        Freq[name] = d
        chk(f"F_req {name} (T^4 m^10)", d["F_req_closed_form_T4_m10"], b["rounded_T4m10"], 5e-3)
        chk(f"F_req {name} SI", d["F_req_closed_form_SI_T4_H2_m8"], b["rounded_SI"], 6e-3)
        d["F_req_SI_direct_formula_T4_H2_m8"] = F_req_closed_form_si(b["eta"], b["T_K"], b["Q"],
                                                                    6.8e5, g1, t3, m1, m2)
        chk(f"F_req {name} SI direct formula vs mu0^2*natural",
            d["F_req_SI_direct_formula_T4_H2_m8"], d["F_req_closed_form_SI_T4_H2_m8"], 1e-12)
    res["F_req"] = Freq
    chk("F_req optimistic eV^-2 (rounded 1.70e80)",
        Freq["optimistic"]["F_req_closed_form_eV^-2"], 1.70e80, 5e-3)

    # ---------------- long detectors ----------------
    xs = optimize.minimize_scalar(lambda x: -g_solenoid(x), bounds=(0.05, 0.95), method="bounded",
                                  options={"xatol": 1e-12})
    x_max, gsol = float(xs.x), g_solenoid(float(xs.x))
    longd = {"x_max": x_max, "g_sol_max": gsol, "g_sol_at_0.534": g_solenoid(0.534),
             "y_max_toroid": math.exp(-0.5), "g_tor_max": 1 / (2 * math.e)}
    for name in bench:
        sF = Freq[name]["sqrt_F_req_T2_m5"]
        WR2 = sF / gsol
        longd[f"{name}_solenoid_WmagR2_MJ_m2"] = WR2 * ENERGY_T2M3_IN_J * 1e-6
        longd[f"{name}_solenoid_B2HR4_T2_m5"] = 2 * WR2 / math.pi
        longd[f"{name}_toroid_WmagR2_MJ_m2"] = 2 * math.e * sF * ENERGY_T2M3_IN_J * 1e-6
        longd[f"{name}_toroid_Bin2HR4_T2_m5"] = 4 * math.e ** 2 * sF / math.pi
    res["long_detectors"] = longd
    chk("x_max 0.534", x_max, 0.534, 2e-3)
    chk("g_sol 0.204", gsol, 0.204, 3e-3)
    chk("W R^2 solenoid optimistic 399 MJ m^2", longd["optimistic_solenoid_WmagR2_MJ_m2"], 399, 3e-3)
    chk("B^2 H R^4 solenoid optimistic 319", longd["optimistic_solenoid_B2HR4_T2_m5"], 319, 3e-3)
    chk("W R^2 solenoid conservative 9.8e4", longd["conservative_solenoid_solenoid_WmagR2_MJ_m2"], 9.8e4, 6e-3)
    chk("W R^2 toroid optimistic 442", longd["optimistic_toroid_WmagR2_MJ_m2"], 442, 3e-3)
    chk("W R^2 toroid conservative 3.4e4", longd["conservative_toroid_CORE_toroid_WmagR2_MJ_m2"], 3.4e4, 1.5e-2)

    # ---------------- sensitivity bandwidth / dwell / tunings (optimistic benchmark) -------
    bw = {}
    for name, b in bench.items():
        det_b = Detector(1.0, 1.0, b["Q"], b["T_K"], b["eta"])
        for lab, m in [("f1", m1), ("f2", m2)]:
            w = float(omega_from_mass_eV(m))
            n = float(n_bose(m, b["T_K"] * K_IN_EV))
            sb = sensitivity_bandwidths(w, b["Q"], n, b["eta"])
            bw[f"{name}_{lab}"] = {"n": n, "xi_opt": float(xi_opt(n, b["eta"])),
                                   **{k + "_Hz": float(v) / (2 * math.pi) for k, v in sb.items()},
                                   "fw_noise_doubling_over_f0": float(sb["fw_noise_doubling"] / w),
                                   "ringup_amplitude_2Q_over_omega0_s": 2 * b["Q"] / w,
                                   "coherence_1/(f v0 vobs)_s": 1 / (w / (2 * math.pi) * V0_KMS * VOBS_KMS / C_KMS ** 2),
                                   "coherence_Qa/f_s": Qa / (w / (2 * math.pi))}
    res["bandwidths"] = bw

    # dwell & number of tunings for the optimistic benchmark at exactly F = F_req (target alloc.)
    b = bench["optimistic"]
    Fopt = Freq["optimistic"]["F_req_exactGbar_T4_m10"]
    detF = Detector(Psi_T_m3=Fopt ** 0.25, Lp_m=1.0, Q=b["Q"], T_K=b["T_K"], eta_A=b["eta"])
    sched = {}
    t_req = time_target_curve_s(m1, m2, detF, 6.8e5, gdfsz)
    sched["t_req_years_at_F=F_req(exact G)"] = t_req / YEAR_S
    steps_per_fw = 10
    def fw_of(m):
        n = n_bose(m, b["T_K"] * K_IN_EV)
        return sensitivity_bandwidths(omega_from_mass_eV(m), b["Q"], n, b["eta"])["fw_noise_doubling"]
    Ntun = integrate.quad(lambda lnm: steps_per_fw * float(omega_from_mass_eV(math.exp(lnm)))
                          / float(fw_of(math.exp(lnm))), math.log(m1), math.log(m2))[0]
    sched["N_tunings_step=FW/10"] = Ntun
    for lab, m in [("f1", m1), ("f2", m2)]:
        tau = dwell_density_target(m, detF, 6.8e5, gdfsz)
        sched[f"tau_{lab}_s_per_rad_s"] = float(tau)
        sched[f"dwell_per_tuning_{lab}_s"] = float(tau * fw_of(m) / steps_per_fw)
    res["schedule_optimistic"] = sched
    chk("t_req at F_req(exact) = 3 yr", sched["t_req_years_at_F=F_req(exact G)"], 3.0, 1e-9)

    # ---------------- toy detector scan rates ----------------
    toyA = Detector(Psi_T_m3=10.0, Lp_m=1.0, Q=2e7, T_K=0.010, eta_A=0.1)
    toyB = Detector(Psi_T_m3=100.0, Lp_m=1.0, Q=1e6, T_K=0.030, eta_A=10.0)
    toys = {"toyA": toyA, "toyB": toyB}
    sr = {"toyA": asdict(toyA), "toyB": asdict(toyB)}
    for tn, det in toys.items():
        for m in (m1, 1e-9, m2):
            g = float(gdfsz(m))
            r = scan_rate_detector(m, g, det, 6.8e5)
            sr[f"{tn}_m{m*1e9:g}neV_gDFSZ_domega0dt_rad_s2"] = float(r)
            sr[f"{tn}_m{m*1e9:g}neV_gDFSZ_dnu0dt_Hz_per_yr"] = float(r / (2 * math.pi) * YEAR_S)
        r = scan_rate_detector(1e-9, 1e-19, det, 6.8e5)
        sr[f"{tn}_m1neV_g1e-19_domega0dt_rad_s2"] = float(r)
        r_hi = scan_rate_detector(1e-9, 1e-19, det, 6.8e5, exact_G=False)
        sr[f"{tn}_m1neV_g1e-19_highT_over_exact_minus1"] = float(r_hi / r - 1)
    res["scan_rate_toy"] = sr

    # ---------------- DMRadio-GUT Eq. 3 anchor (41 kHz/yr) ----------------
    cPU, B0, V = 0.1, 16.0, 10.0
    F_gut = 4 * cPU ** 4 * B0 ** 4 * V ** (10 / 3)          # Psi^4/L^2 = 4 cPU^4 B^4 V^{10/3}
    detG = Detector(Psi_T_m3=F_gut ** 0.25, Lp_m=1.0, Q=2e7, T_K=0.010, eta_A=0.1)
    mG = 2 * math.pi * 1e5 * HBAR_EVS
    rG = scan_rate_detector(mG, 1e-19, detG, 6.4e5, Theta=9.0, rho=0.45, exact_G=False)
    res["DMRadio_GUT_anchor"] = {"F_T4_m10": F_gut, "dnu_dt_kHz_per_yr": rG / (2 * math.pi) * YEAR_S / 1e3,
                                 "note": "2203.11246 Eq. 3 quotes 41 kHz/yr (SNR=3, Qa=6.4e5, rho=0.45)"}
    chk("DMRadio-GUT Eq.3 41 kHz/yr", res["DMRadio_GUT_anchor"]["dnu_dt_kHz_per_yr"], 41.0, 1.5e-2)

    # ---------------- reach curves (local) ----------------
    reach = {}
    masses = [m1, 1e-9, 2e-9, m2]
    for tn, det in toys.items():
        for alloc in ("target_DFSZ", "log_uniform"):
            for m in masses:
                if alloc == "target_DFSZ":
                    tau = tau_target(m, t3, m1, m2, det, 6.8e5, gdfsz)
                else:
                    tau = tau_log_uniform(m, t3, m1, m2)
                g95 = float(g95_local(m, tau, det, 6.8e5))
                reach[f"{tn}_{alloc}_m{m*1e9:g}neV_g95_GeV^-1"] = g95
                reach[f"{tn}_{alloc}_m{m*1e9:g}neV_g95_over_gDFSZ"] = g95 / float(gdfsz(m))
    res["reach_local_3yr_0.5-4.1neV"] = reach
    FreqA = Freq["optimistic"]["F_req_exactGbar_T4_m10"]
    chk("toyA target reach / DFSZ = (F_req/F)^(1/4)",
        reach["toyA_target_DFSZ_m1neV_g95_over_gDFSZ"], (FreqA / toyA.F_T4m10) ** 0.25, 1e-9)

    # ---------------- discrete exact bin-by-bin check ----------------
    # Schedule: the target-curve allocation with the rounded Qa = 6.8e5.  The
    # discrete sum integrates the true SHM line, whose Qa = Q_axion() = 6.7795e5, so
    # the like-for-like local comparison uses g95_local(..., Qa=Q_axion()).
    disc = {}
    for tn, det in toys.items():
        for m in (m1 * 1.0001, 1e-9, 3e-9, m2 * 0.9999):
            taufn = lambda mm, det=det: tau_target(mm, t3, m1, m2, det, 6.8e5, gdfsz)
            w0, ts = dense_tunings_around(m, det, taufn, half_widths=40.0, steps_per_hw=6)
            g_loc_true = float(g95_local(m, taufn(m), det, Qa))
            g_loc_68 = float(g95_local(m, taufn(m), det, 6.8e5))
            g_w = g95_discrete(m, w0, ts, det, stat="weak")
            g_l = g95_discrete(m, w0, ts, det, stat="limit")
            _, smax = theta_per_tuning(m, g_w, w0, ts, det, return_smax=True)
            mw = float(omega_from_mass_eV(m))
            n_m = float(n_bose(m, det.T_K * K_IN_EV))
            eqw = float(sensitivity_bandwidths(mw, det.Q, n_m, det.eta_A)["equivalent_width"])
            disc[f"{tn}_m{m*1e9:.5g}neV"] = {
                "N_tunings": len(w0), "dwell_per_tuning_s": float(np.median(ts)),
                "ringup_2Q_over_omega0_s": 2 * det.Q / mw,
                "total_time_within_equiv_width_s": float(taufn(m) * eqw),
                "g95_local_Qa_true": g_loc_true, "g95_local_Qa_6.8e5": g_loc_68,
                "g95_discrete_weak": g_w, "g95_discrete_exactAsimov": g_l,
                "discrete_over_local(Qa_true)_minus1": g_w / g_loc_true - 1,
                "exact_over_weak_minus1": g_l / g_w - 1, "max_s_per_bin_at_g95": smax}
            chk(f"discrete vs local {tn} m={m*1e9:.4g} neV", g_w, g_loc_true, 5e-5)
    res["discrete_scan_check"] = disc

    # ---------------- statistics ----------------
    f1, f2 = ax["f1_Hz"], ax["f2_Hz"]
    Nma = n_independent_masses(f1, f2)
    st = {"Theta95": THETA95, "Theta95_exact_(Phi^-1(0.95))^2": THETA95_EXACT,
          "q_band_+1sigma": limit_band_q(1)[0], "q_band_-1sigma": limit_band_q(1)[1],
          "q_band_+2sigma": limit_band_q(2)[0], "q_band_-2sigma": limit_band_q(2)[1],
          "N_ma_scan": Nma,
          "TS_thresh_3sigma_global": ts_discovery_threshold(3, Nma),
          "TS_thresh_5sigma_global": ts_discovery_threshold(5, Nma),
          "TS_thresh_5sigma_local(N=1)": ts_discovery_threshold(5, 1),
          "Foster_example_N_ma_100Hz-100MHz": n_independent_masses(100, 1e8)}
    st["g_disc5sigma_over_g95"] = (st["TS_thresh_5sigma_global"] / THETA95) ** 0.25
    st["g_disc3sigma_over_g95"] = (st["TS_thresh_3sigma_global"] / THETA95) ** 0.25
    Nf = st["Foster_example_N_ma_100Hz-100MHz"]
    st["Foster_example_TS3sigma_one_sided"] = ts_discovery_threshold(3, Nf)
    st["Foster_example_TS5sigma_one_sided"] = ts_discovery_threshold(5, Nf)
    st["Foster_example_TS3sigma_two_sided"] = ts_discovery_threshold(3, Nf, sided=2)
    st["Foster_example_TS5sigma_two_sided"] = ts_discovery_threshold(5, Nf, sided=2)
    st["TS_thresh_3sigma_global_two_sided"] = ts_discovery_threshold(3, Nma, sided=2)
    st["TS_thresh_5sigma_global_two_sided"] = ts_discovery_threshold(5, Nma, sided=2)
    res["statistics"] = st
    chk("TS_thresh 5 sigma local = 25", st["TS_thresh_5sigma_local(N=1)"], 25.0, 1e-9)
    chk("TS_thresh 3 sigma local = 9 (Foster N=1 example)", ts_discovery_threshold(3, 1), 9.0, 1e-9)
    chk("Foster 3sigma LEE example 40.9 (two-sided p)", st["Foster_example_TS3sigma_two_sided"],
        40.9, 3e-3)
    chk("Foster 5sigma LEE example 57.5 (two-sided p)", st["Foster_example_TS5sigma_two_sided"],
        57.5, 3e-3)

    # ---------------- finite-band demo (broadband regime) ----------------
    # Low-Q (copper-like) detector: fw/f0 ~ O(0.1-1); compare LP schedule with local time.
    detC = Detector(Psi_T_m3=100.0, Lp_m=1.0, Q=1e4, T_K=0.030, eta_A=10.0)
    fa, fb = 120.9e3, 400e3
    ma, mb = 2 * math.pi * fa * HBAR_EVS, 2 * math.pi * fb * HBAR_EVS
    t_loc = time_target_curve_s(ma, mb, detC, Qa, gdfsz)      # like-for-like: true-line Qa
    m_t = np.geomspace(ma, mb, 120)
    g_t = gdfsz(m_t)
    w_set = np.geomspace(0.5 * ma, 1.6 * mb, 200) / HBAR_EVS
    n_set = n_bose(w_set * HBAR_EVS, detC.T_K * K_IN_EV)
    xi_base = xi_opt(n_set, detC.eta_A)
    w_all = np.concatenate([w_set] * 3)
    xi_all = np.concatenate([xi_base * f for f in (0.3, 1.0, 3.0)])
    K = kernel_matrix(m_t, g_t, w_all, xi_all, detC, n_v=64)
    tj, ttot = lp_schedule(K)
    m_chk = np.geomspace(ma, mb, 481)
    cov = kernel_matrix(m_chk, gdfsz(m_chk), w_all, xi_all, detC, n_v=64) @ tj / THETA95
    Knl = np.array([kernel_narrow_line(mm, gg, w_all, xi_all, detC, Qa) for mm, gg in
                    zip(m_t[::20], g_t[::20])])
    # coverage of the *local* (dense, Lorentzian-derived) schedule restricted to the band,
    # evaluated with the exact finite-band kernel (Qa of the true line in the local tau)
    w_loc = np.geomspace(ma, mb, 801) / HBAR_EVS
    dw = np.gradient(w_loc)
    t_locj = dwell_density_target(w_loc * HBAR_EVS, detC, Qa, gdfsz) * dw
    xi_loc = xi_opt(n_bose(w_loc * HBAR_EVS, detC.T_K * K_IN_EV), detC.eta_A)
    cov_loc = kernel_matrix(m_chk, gdfsz(m_chk), w_loc, xi_loc, detC, n_v=64) @ t_locj / THETA95
    fw_a = float(sensitivity_bandwidths(1.0, detC.Q, float(n_bose(ma, detC.T_K * K_IN_EV)),
                                        detC.eta_A)["fw_noise_doubling"])
    interior = (m_chk > ma * (1 + fw_a)) & (m_chk < mb * (1 - fw_a))
    res["finite_band_demo"] = {
        "local_schedule_coverage_min_all": float(cov_loc.min()),
        "local_schedule_coverage_interior_min": float(cov_loc[interior].min()),
        "local_schedule_coverage_interior_max": float(cov_loc[interior].max()),
        "local_schedule_time_s(Qa_true)": float(np.sum(t_locj)),
        "detector": asdict(detC), "band_Hz": [fa, fb],
        "fw_over_f0_at_fa": float(sensitivity_bandwidths(1.0, detC.Q, float(n_bose(ma, detC.T_K * K_IN_EV)),
                                                         detC.eta_A)["fw_noise_doubling"]),
        "t_local_formula_s": t_loc, "t_LP_finite_band_s": ttot, "LP_over_local": ttot / t_loc,
        "min_coverage_dense_grid_(Theta/Theta95)": float(np.min(cov)),
        "narrowline_over_fullline_kernel_max_abs_dev": float(np.max(np.abs(Knl / K[::20] - 1)[K[::20] > 1e-3 * K[::20].max()])),
        "n_settings_used": int(np.sum(tj > 1e-9 * ttot))}

    # ---------------- self tests ----------------
    st_ = self_tests()
    res["self_tests"] = st_
    for k, v in st_.items():
        if k.endswith("_err"):
            checks[f"selftest {k}"] = {"value": v, "expected": 0.0, "rel_err": v, "tol": 1e-6,
                                       "pass": abs(v) <= 1e-6}
    res["checks"] = checks
    res["all_checks_pass"] = all(c["pass"] for c in checks.values())

    # ---------------- print ----------------
    def show(d, ind=0):
        for k, v in d.items():
            if isinstance(v, dict):
                print(" " * ind + f"{k}:")
                show(v, ind + 2)
            elif isinstance(v, list):
                print(" " * ind + f"{k}:")
                for item in v:
                    print(" " * (ind + 2) + json.dumps(item, default=float))
            elif isinstance(v, float):
                print(" " * ind + f"{k}: {v:.10g}")
            else:
                print(" " * ind + f"{k}: {v}")
    show({k: v for k, v in res.items() if k != "checks"})
    print("\nCHECKS:")
    for k, c in checks.items():
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {k}: value={c['value']:.8g} "
              f"expected={c['expected']:.6g} rel_err={c['rel_err']:+.3e} (tol {c['tol']:g})")
    print("ALL CHECKS PASS" if res["all_checks_pass"] else "SOME CHECKS FAILED")
    if write_json:
        p = Path(__file__).with_name("scan_reference_values.json")
        p.write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(f"wrote {p}")
    return res


if __name__ == "__main__":
    out = main()
    sys.exit(0 if out["all_checks_pass"] else 1)
