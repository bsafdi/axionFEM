"""Transcription of the 2022 multipole-scan computation behind the projections of arXiv:2211.00008.

The source is a Python notebook ("[C]" below, as in docs/physics/multipole_scan.md; not distributed).  Every formula
below is transcribed from [C].  [C] relies on interpolation helpers and on a summation of the multipole statistic
that are not part of it; ``interp_q``/``Q_Bandwidth`` come from a disabled part of it.  The pieces reconstructed here
are flagged ASSUMPTION (A1-A8); the list is ``ASSUMPTIONS``.  They are inferred, not confirmed by the original code.
The algorithms follow arXiv:2211.00008 and the scan-rate rule of arXiv:2302.14084.

Units follow [C]: frequencies in MHz, impedances in ohm, natural-unit conversion constants verbatim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import interpolate, ndimage, stats

ASSUMPTIONS = {
    "A1": "multipole statistic TS(f_s) = sum_k t_k TuningTS(f_s, f_k) over the saved strategy (capacitive "
          "tunings start_freq..f_MQS, times hf_collection_time); TuningTS is per second",
    "A2": "mqs_resistance(f) = 2 pi f L_MQS / MQS_Q with MQS_Q = 1e6 and L_MQS = Im Z(Freq[0]) / (2 pi Freq[0]); "
          "mqs_inductance = L_MQS; mqs_impedance(f) = j 2 pi f L_MQS",
    "A3": "mqs_source_voltage(f) = |V_oc(Freq[0])| f/Freq[0] (MQS EMF ~ omega Psi_MQS), times the coupling scale",
    "A4": "source_voltage(f) = |V_oc(f)| x scale, V_oc = V_port (Z + 50)/50 (COMSOL voltage run loads the port with "
          "Zref = 50 ohm; a magnitude is needed since [C] raises it to the 4th power and stores a real TS); scale "
          "[V per table unit per GeV^-1] = |J_eff| for g = 1 GeV^-1 (B, rho), calibrated by the 1-yr endpoint",
    "A5": "interpolators: linear (np.interp) on the 1 kHz COMSOL grid for Z and |V_oc| ('cubic' = CubicSpline "
          "available); they raise outside the table band (never extrapolate)",
    "A6": "Q_Bandwidth = omega L_eff / (Re Z + mqs_resistance), L_eff = (X/omega + dX/domega)/2 (2302.14084 Eq. 6, "
          "the 2022 rule), dX/domega by np.gradient on the table; mask_locs = points with Q <= 0 or non-finite; "
          "interp_q = interp1d(clean_freqs, clean_q, fill_value='extrapolate') (disabled in [C])",
    "A7": "temperature: [C]'s thermal_noise_power default T = 10 (K, with kB in eV/K) is never overridden in "
          "[C]; the paper's figures need T = 20 mK (single-pole eta dependence, M11).  Default here "
          "T_K = 0.020; T_K = 10 reproduces [C] literally",
    "A8": "plotted multipole curve = ndimage.gaussian_filter(g(sensitivity_freqs), 101)[::100], i.e. the "
          "paper curve is a smoothed and decimated version of g on the sensitivity grid",
}

# ---------------------------------------------------------------- constants, verbatim from [C]
TOTAL_SCAN_TIME = 1 * 365 * 86400
MQS_Q = 1e6
Q_A = 1e6
SIG_WIDTH = 1 / Q_A
KSVZ, DFSZ = 1.92, 0.75
OMEGA_TO_NAT = 6.582119569e-16
OHM_TO_NAT = 0.00265442
KB_EV = 8.61733e-05
TO_CURRENT_UNITS = 4.25285e-22
F_MQS = 149.896
F_SENS_MAX = 466.40568517314
TARGET_TS = float(stats.chi2.ppf(.9, df=1))


def freq_to_ev(freq):
    return 4.13567e-9 * freq


def g_x(C_ag, m_a):
    return 2e-10 * C_ag * m_a


def thermal_noise_power(omega, R, Z, T=10.0):
    """Verbatim from [C] (T in K)."""
    omega_nat = OMEGA_TO_NAT * omega
    R_nat = OHM_TO_NAT * R
    Z_nat = OHM_TO_NAT * Z
    T_nat = KB_EV * T
    norm = 2 * omega_nat / np.pi * R_nat / Z_nat ** 2
    thermal_dist = 1 / np.expm1(omega_nat / T_nat) + 1 / 2
    return norm * thermal_dist * TO_CURRENT_UNITS


def readout_noise_power(omega, omegaR, Z, ZR, eta_a):
    """Verbatim from [C] (eta_a is a global there)."""
    omega_nat = OMEGA_TO_NAT * omega
    omegaR_nat = OMEGA_TO_NAT * omegaR
    Z_nat = OHM_TO_NAT * Z
    ZR_nat = OHM_TO_NAT * ZR
    norm = eta_a / 2 / np.pi * omegaR_nat / ZR_nat
    return norm * ((omega_nat / omegaR_nat) ** 2 + (ZR_nat / Z_nat) ** 2) * TO_CURRENT_UNITS


# ---------------------------------------------------------------- make_interpolators (ASSUMPTIONS A2-A5)
@dataclass
class Interp:
    Freq: np.ndarray                 # MHz
    Z: np.ndarray                    # complex, COMSOL exp(+jwt): Im > 0 inductive
    Voc: np.ndarray                  # |V_oc| per table unit
    scale: float = 1.0               # V per table unit per GeV^-1 (A4)
    kind: str = "linear"
    Q_mqs: float = MQS_Q
    _cs: dict = field(default_factory=dict)

    def _chk(self, f):
        f = np.asarray(f, float)
        if np.any(f < self.Freq[0] * (1 - 1e-12)) or np.any(f > self.Freq[-1] * (1 + 1e-12)):
            raise ValueError(f"jf2022: frequency outside the table [{self.Freq[0]}, {self.Freq[-1]}] MHz "
                             "(the interpolators never extrapolate)")
        return f

    def _ev(self, name, y, f):
        f = self._chk(f)
        if self.kind == "linear":
            return np.interp(f, self.Freq, y)
        if name not in self._cs:
            self._cs[name] = interpolate.CubicSpline(self.Freq, y)
        return self._cs[name](f)

    def source_impedance(self, f):
        return self._ev("R", self.Z.real, f) + 1j * self._ev("X", self.Z.imag, f)

    def source_voltage(self, f):
        return self.scale * self._ev("V", self.Voc, f)

    @property
    def mqs_inductance(self):
        return float(self.Z.imag[0] / (2 * np.pi * 1e6 * self.Freq[0]))

    def mqs_impedance(self, f):
        return 1j * 2 * np.pi * 1e6 * np.asarray(f, float) * self.mqs_inductance

    def mqs_source_voltage(self, f):
        return self.scale * self.Voc[0] * np.asarray(f, float) / self.Freq[0]

    def mqs_resistance(self, f):
        return 2 * np.pi * 1e6 * np.asarray(f, float) * self.mqs_inductance / self.Q_mqs


def load_tables(table_dir, port_load_ohm=50.0):
    """Freq [MHz], complex Z (COMSOL), |V_oc| from a 2022 COMSOL ``source_{impedance,voltage}.npy`` pair
    (as [C] reads it).
    The archived lossless toroid stores frequencies in GHz; it is converted to MHz."""
    from pathlib import Path
    d = Path(table_dir)
    Zr = np.load(d / "source_impedance.npy")
    Vr = np.load(d / "source_voltage.npy")
    fz, fv = Zr[0].copy(), Vr[0].copy()
    if fz[0] < 1.0:          # GHz
        fz, fv = fz * 1e3, fv * 1e3
    Z = Zr[1] + 1j * Zr[2]
    V = Vr[1] + 1j * Vr[2]
    if fv.size != fz.size or np.max(np.abs(fv[: fz.size] - fz)) > 1e-9:
        V = np.interp(fz, fv, V.real) + 1j * np.interp(fz, fv, V.imag)
    if not Z.imag[0] > 0:
        raise ValueError("jf2022.load_tables: Im Z(5 MHz) <= 0 -- not inductive; wrong sign convention")
    Voc = np.abs(V * (Z + port_load_ohm) / port_load_ohm) if port_load_ohm else np.abs(V)
    return fz, Z, Voc


def coupling_scale(B_T=10.0, rho_GeV_cm3=0.3):
    """|J_eff| [A/m^2] for g = 1 GeV^-1: the COMSOL tables are per unit Je (A4)."""
    # same constants as axionfem.multipole.J_eff_A_per_m2 (kept self-contained: no compiled core needed)
    alpha, hbar_eVs, hbarc_eVm = 7.2973525693e-3, 6.62607015e-34 / (2 * math.pi) / 1.602176634e-19, 1.973269804e-7
    e_nat = math.sqrt(4 * math.pi * alpha)
    amp_per_m2_to_eV3 = (e_nat / 1.602176634e-19) * hbar_eVs * hbarc_eVm ** 2
    gev_cm3_to_eV4 = 1e9 * (hbarc_eVm * 100.0) ** 3
    J = 1e-9 * math.sqrt(2 * rho_GeV_cm3 * gev_cm3_to_eV4) * B_T * 195.35277
    return J / amp_per_m2_to_eV3


# ---------------------------------------------------------------- Q_Bandwidth / interp_q (A6)
def q_bandwidth(it: Interp, rule="2022"):
    f = it.Freq
    w = 2 * np.pi * 1e6 * f
    X = it.Z.imag
    dXdw = np.gradient(X, w)
    Le = 0.5 * (X / w + dXdw) if rule == "2022" else 0.5 * (np.abs(X) / w + dXdw)
    Q = w * Le / (it.Z.real + it.mqs_resistance(f))
    mask_locs = np.flatnonzero(~np.isfinite(Q) | (Q <= 0))
    clean_freqs = np.delete(f, mask_locs)
    clean_q = np.delete(Q, mask_locs)
    return interpolate.interp1d(clean_freqs, clean_q, fill_value="extrapolate"), Q, mask_locs


# ---------------------------------------------------------------- test statistics ([C])
def OnResTS(it: Interp, freq, eta_a, T_K, mqs=False):
    omega = 2 * np.pi * 1e6 * freq
    resistance = it.mqs_resistance(freq)
    ZR = resistance if mqs else np.real(it.source_impedance(freq) + resistance)
    total = thermal_noise_power(omega, resistance, ZR, T_K) + readout_noise_power(omega, omega, ZR, ZR, eta_a)
    norm = 1 / (2 * np.pi) / (omega / Q_A)
    V = it.mqs_source_voltage(freq) if mqs else it.source_voltage(freq)
    return norm * V ** 4 / ZR ** 4 / total ** 2 * g_x(DFSZ, freq_to_ev(freq)) ** 4


def tuning_ts_sum(it: Interp, freq, res_freqs, weights, eta_a, T_K, *, chunk=2048, amp="jf2022"):
    """sum_k weights_k TuningTS(freq, res_freqs_k) ([C] vectorised; A1).  ``amp="jf2022"`` is [C]'s readout
    noise; ``amp="axf"`` swaps in axionFEM's M13 frequency scaling (a diagnostic)."""
    freq = np.atleast_1d(np.asarray(freq, float))
    rf = np.asarray(res_freqs, float)
    wts = np.asarray(weights, float)
    Zk = it.source_impedance(rf)
    Xres = -np.imag(Zk)                                   # resonant_reactance
    cap = Xres < 0
    rk = it.mqs_resistance(rf)
    ZR = np.abs(Zk + 1j * Xres + rk)                      # resonant_impedance(res_freq) = j Xres
    wk = 2 * np.pi * 1e6 * rf
    out = np.zeros(freq.size)
    Zs_all = it.source_impedance(freq)
    rs_all = it.mqs_resistance(freq)
    Vs_all = it.source_voltage(freq)
    for i0 in range(0, freq.size, chunk):
        f = freq[i0:i0 + chunk, None]
        w = 2 * np.pi * 1e6 * f
        res_imp = 1j * np.where(cap, Xres * rf / f, Xres * f / rf)
        Z = np.abs(Zs_all[i0:i0 + chunk, None] + res_imp + rs_all[i0:i0 + chunk, None])
        th = thermal_noise_power(w, rs_all[i0:i0 + chunk, None], Z, T_K)
        if amp == "jf2022":
            ro = readout_noise_power(w, wk, Z, ZR, eta_a)
        elif amp == "axf":   # M13: imprecision ~ w_k |Z|^2/R_k, backaction ~ R_k w^2/w_k (voltage); as current PSD
            ro = eta_a / 2 / np.pi * (OMEGA_TO_NAT * wk) / (OHM_TO_NAT * ZR) * (
                1.0 + (ZR / Z) ** 2 * (w / wk) ** 2) * TO_CURRENT_UNITS
        else:
            raise ValueError("amp must be 'jf2022' or 'axf'")
        simple_norm = 1 / (2 * np.pi) / (w * SIG_WIDTH)
        TS = simple_norm * Vs_all[i0:i0 + chunk, None] ** 4 / Z ** 4 / (th + ro) ** 2
        out[i0:i0 + chunk] = (TS * wts[None, :]).sum(axis=1) * g_x(DFSZ, freq_to_ev(freq[i0:i0 + chunk])) ** 4
    return out


# ---------------------------------------------------------------- schedule ([C])
def mqs_projection(it: Interp, eta_a, T_K, f0=None):
    f0 = it.Freq[0] if f0 is None else f0
    N_a = int(np.ceil(np.log(F_MQS / f0) / np.log(1 + SIG_WIDTH)))
    mqs_tuning_freqs = np.geomspace(5, F_MQS, N_a)
    one_second_ts = OnResTS(it, mqs_tuning_freqs, eta_a, T_K, mqs=True)
    mqs_time_to_dfsz = TARGET_TS / one_second_ts
    sort_order = np.argsort(mqs_time_to_dfsz)
    sorted_freqs = mqs_tuning_freqs[sort_order]
    sorted_time = mqs_time_to_dfsz[sort_order]
    sorted_time[np.searchsorted(np.cumsum(sorted_time), TOTAL_SCAN_TIME) + 1:] = np.nan
    collection_time = sorted_time[np.argsort(sorted_freqs)]
    collection_ts = one_second_ts * collection_time
    mqs_sens = (collection_ts / TARGET_TS) ** .25 * g_x(DFSZ, freq_to_ev(mqs_tuning_freqs))  # [C] as written
    return {"freqs": mqs_tuning_freqs, "collection_time": collection_time, "sensitivity": mqs_sens,
            "start_freq": float(mqs_tuning_freqs[np.flatnonzero(~np.isnan(mqs_sens))[0]])}


def step_grid(f_lo, f_hi, interp_q):
    """[C]: f_{n+1} = f_n (1 + max(sig_width, 1/interp_q(f_n))) while f_n < f_hi."""
    out = [f_lo]
    f = f_lo
    while f < f_hi:
        f = f * (1 + max(SIG_WIDTH, 1 / float(interp_q(f))))
        out.append(f)
    return np.array(out)


def hf_strategy(it: Interp, mqs, interp_q, eta_a, T_K):
    hf = step_grid(mqs["start_freq"], F_MQS, interp_q)
    t = np.interp(hf, mqs["freqs"], mqs["collection_time"])
    t = t * np.maximum(1, 1e6 / interp_q(hf))
    one = OnResTS(it, hf, eta_a, T_K)
    sens = (TARGET_TS / (one * t)) ** .25 * g_x(DFSZ, freq_to_ev(hf))
    nan_locs = np.flatnonzero(np.imag(it.source_impedance(hf)) < 0)
    sens[nan_locs] = np.nan
    strategy = np.delete(np.vstack((hf, t)), nan_locs, axis=1)
    return {"hf_freqs": hf, "hf_time": t, "single_pole": sens, "strategy": strategy}


def sensitivity_freqs(interp_q, f_min=5.0, f_max=F_SENS_MAX):
    return step_grid(f_min, f_max, interp_q)


def plotted_curve(sens_freqs, g, sigma=101, every=100):
    """[C] (A8): what the figure draws."""
    return sens_freqs[::every], ndimage.gaussian_filter(g, sigma)[::every]


# ---------------------------------------------------------------- fast, algebraically identical form of the A1 sum
def tuning_ts_sum_fast(it: Interp, freq, res_freqs, weights, eta_a, T_K, *, chunk=32, amp="jf2022",
                       thermal="bose"):
    """Same as :func:`tuning_ts_sum` (test_jf2022_fast_kernel_equals_verbatim), rewritten in real arithmetic:
    ``TS = c_s V^4 / (A_s + B_k ZR_k^2 + B_k q Z^2)^2`` with ``A_s = thermal current PSD x |Z|^2``,
    ``B_k = eta omega_k/(2 pi ZR_k)`` (current units), ``q = (omega/omega_k)^2`` ([C]); ``amp="axf"`` gives
    ``A_s + B_k Z^2 + B_k ZR_k^2 q`` (axionFEM M13 scaling).  ``thermal="classical"`` replaces ``n + 1/2`` by
    ``T/omega`` (axionFEM M13 uses ``2kT``)."""
    freq = np.atleast_1d(np.asarray(freq, float))
    rf = np.asarray(res_freqs, float)
    wts = np.asarray(weights, float)
    Zk = it.source_impedance(rf)
    Xres = -np.imag(Zk)
    cap = Xres < 0
    rk = it.mqs_resistance(rf)
    ZR = np.abs(Zk + 1j * Xres + rk)
    wk = 2 * np.pi * 1e6 * rf
    Bk = TO_CURRENT_UNITS * eta_a / 2 / np.pi * (OMEGA_TO_NAT * wk) / (OHM_TO_NAT * ZR)
    Ck = Bk * ZR ** 2
    Zs = it.source_impedance(freq)
    rs = it.mqs_resistance(freq)
    Vs = it.source_voltage(freq)
    ws = 2 * np.pi * 1e6 * freq
    wn = OMEGA_TO_NAT * ws
    dist = 1 / np.expm1(wn / (KB_EV * T_K)) + 0.5 if thermal == "bose" else KB_EV * T_K / wn
    A = TO_CURRENT_UNITS * 2 * wn / np.pi * (OHM_TO_NAT * rs) / OHM_TO_NAT ** 2 * dist
    cs = 1 / (2 * np.pi) / (ws * SIG_WIDTH) * Vs ** 4 * g_x(DFSZ, freq_to_ev(freq)) ** 4
    Rtot = Zs.real + rs
    fc, fi = rf[cap], rf[~cap]
    out = np.zeros(freq.size)
    for i0 in range(0, freq.size, chunk):
        sl = slice(i0, i0 + chunk)
        f = freq[sl, None]
        Xt = np.empty((f.shape[0], rf.size))
        Xt[:, cap] = Xres[cap] * fc / f
        Xt[:, ~cap] = Xres[~cap] * f / fi
        Xt += Zs.imag[sl, None]
        Z2 = Xt * Xt
        Z2 += Rtot[sl, None] ** 2
        q = (ws[sl, None] / wk) ** 2
        if amp == "jf2022":
            D = A[sl, None] + Ck + Bk * q * Z2
        elif amp == "axf":
            D = A[sl, None] + Bk * Z2 + Ck * q
        else:
            raise ValueError("amp must be 'jf2022' or 'axf'")
        out[sl] = cs[sl] * (wts / (D * D)).sum(axis=1)
    return out
