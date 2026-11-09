"""dc SQUID amplifier chains specified by device and circuit parameters.

Turns a first-stage Tesche-Clarke (TC) dc SQUID, a second-stage series array and a preamplifier
into the two numbers the optimal scan needs, the noise parameter ``eta(f)`` and the uncoupled
energy sensitivity ``eps_uc(f)``, plus the scan-optimal coupling and a correlated
:class:`axionfem.circuit.Squid` for the readout / multiport closures.

Source: [DMR-SQUID] V. Ankel et al., "Noise limits for dc SQUID readout of high-Q resonators
below 300 MHz", J. Appl. Phys. 138, 094505 (2025), arXiv:2504.20398 (equation numbers of arXiv v1).
Every formula is re-derived in ``docs/physics/squid_chain.md``, which also lists the paper's
inconsistencies.  Beta: TC theory with the paper's rounded response functions, and a
second stage scaled from one measured 20x1 prototype.

Conventions
-----------
* Internally all PSDs are ONE-SIDED per Hz, ``S^(1)``, as in the paper (Johnson noise
  ``4 k_B T R``).  The package's Foster PSDs are half of these, ``S = S^(1)/2``
  (``docs/physics/scan_and_signal.md`` S3.2); the conversion happens only in
  :meth:`Chain.squid`.  ``eta`` and ``eps_uc`` do not depend on the convention.
* Time dependence ``exp(-i omega t)`` (inductor ``Z = -i omega L``), as in :mod:`axionfem.circuit`.
* The real TC flux / circulating-current cross-spectrum carries a sign ``s = +-1``
  (``Chain.sign``).  It is NOT derived: [DMR-SQUID] Eq. 12 gives only ``|Im S_IV|``.  It shifts the
  sensitivity band by ``x0 = s c/xi`` linewidths and leaves ``eta``, ``G-bar`` and the scan
  optimum unchanged (``docs/physics/squid_chain.md`` Sec. 5).

Quick use::

    from axionfem import squid, scan
    ch = squid.chain("C1+low+RT")                 # C1 + 48x3 array at 1 K + XXF-1 / 300 MHz preamp
    ch.eta(1e6), ch.eps_uc_over_hbar(1e6)         # 9.71, 10.83
    ch.kappa_g2_scan(1e6, Q=1e6, T_K=0.03)        # Eq. 48 coupling
    scan.reach(Psi, L, squid.benchmark(ch, Q=1e6, T_K=0.03))   # eta(f) in the scan
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

from .circuit import HBAR, KB, PHI0, Squid, n_bose

F_MAX_HZ = 300e6          # [DMR-SQUID] title and Sec. IV B: the analysis covers f < 300 MHz

# TC regime: beta_L ~ 1, beta_C ~ 1, Gamma <~ 0.05 (Eq. 1).  Eq. 1 PRINTS beta_L = L I0/Phi0 and
# beta_C = pi I0 R^2 C/Phi0, half the Tesche-Clarke definitions 2 L I0/Phi0 and 2 pi I0 R^2 C/Phi0
# (I0 per junction); App. A's 1.6 I0 R_j/Phi0 = 0.8 R_j/L_sq needs 2 L I0/Phi0 = 1, so the TC
# convention is the one the paper's numbers use, so the TC values are checked.  "~ 1" is not quantified; a
# factor-2 window is our choice (ASSUMED), and only warns.
BETA_WINDOW = (0.5, 2.0)
F_JOSEPHSON_FRACTION = 0.1   # warn above 0.1 f_J (ASSUMED threshold for f << f_J, Sec. II)
T2_MIN_K = 1.0               # Eq. 34 linear T2 scaling is supported down to ~1 K (Sec. IV D)
R_MATCH_TOL = 0.2            # CryoRFAmp: |R_dyn/R_in - 1| allowed for the matched-noise formula (ASSUMED)
GAMMA_MAX = 0.05
GAMMA_EQ2 = 0.025         # Eq. 2 (dV/dPhi = R_j/L_sq) assumes Gamma < 0.025 (footnote 2, App. A)

# [DMR-SQUID] Table I: prototype 20x1 second-stage array, measured at T2 = 4 K.  "+ feedback
# slope" (used for all designs, Sec. IV D) and "- feedback slope".  dPhi/dI is tabulated as a
# magnitude (the triple-product rule makes (dPhi/dI)_V negative, App. B).
TABLE_I_PLUS = {"N_ser": 20, "N_par": 1, "T_K": 4.0, "Phi_n_SQ_Phi0_rtHz": 0.297e-6,
                "dPhi_dI_H": 95.5e-12, "dV_dI_ohm": 64.9, "dV_dPhi_Hz": 6.80e11,
                "M_IN2_H": 105e-12, "M_FB2_H": 44e-12, "P_W": 0.834e-9,
                "Phi_n2_total_Phi0_rtHz": 0.394e-6, "Phi_n_preamp_Phi0_rtHz": 0.259e-6}
TABLE_I_MINUS = {**TABLE_I_PLUS, "Phi_n_SQ_Phi0_rtHz": 0.247e-6, "dPhi_dI_H": 231e-12,
                 "dV_dI_ohm": 154.0, "dV_dPhi_Hz": 6.66e11,
                 "Phi_n2_total_Phi0_rtHz": 0.453e-6, "Phi_n_preamp_Phi0_rtHz": 0.380e-6}
# Table I measurement preamplifier (Magnicon), used only for the Table I cross-check.
TABLE_I_PREAMP = {"V_n_V_rtHz": 0.32e-9, "I_n_A_rtHz": 2.7e-12}


def _f_array(f_Hz, f_max: float = F_MAX_HZ) -> np.ndarray:
    f = np.asarray(f_Hz, dtype=float)
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("squid: frequencies must be positive and finite [Hz]")
    if np.any(f > f_max * (1 + 1e-12)):
        raise ValueError(f"squid: {float(np.max(f)):.6g} Hz is above {f_max / 1e6:g} MHz, the range the "
                         "[DMR-SQUID] chain analysis covers; it is not extrapolated")
    return f


def _out(x, like):
    if np.ndim(like) != 0:
        return np.asarray(x)
    return complex(x) if np.iscomplexobj(x) else float(x)


def _positive(value, name):
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"squid: {name} must be positive and finite (got {value!r})")
    return float(value)


# ===================================================================== first stage (Sec. II-III)
@dataclass(frozen=True)
class FirstStage:
    """A symmetric, weakly coupled, TC-optimized dc SQUID (Eqs. 1-13).

    ``V_Phi_coeff`` and ``R_dyn_coeff`` are the response functions in units of ``R_j/L_sq`` and
    ``R_j``: the paper rounds both to 1 (Eqs. 2-3; App. A finds 0.8-0.9 and 0.97).  ``C_j_F`` is
    needed only for ``beta_C``; the paper does not give it for C1.
    """
    L_sq_H: float
    I0_A: float
    R_j_ohm: float
    T_j_K: float
    C_j_F: float | None = None
    V_Phi_coeff: float = 1.0
    R_dyn_coeff: float = 1.0
    label: str = "first stage"

    def __post_init__(self):
        for nm in ("L_sq_H", "I0_A", "R_j_ohm", "T_j_K", "V_Phi_coeff", "R_dyn_coeff"):
            _positive(getattr(self, nm), nm)
        if self.C_j_F is not None:
            _positive(self.C_j_F, "C_j_F")
        for msg in self.validity()["warnings"]:
            warnings.warn(f"squid.FirstStage({self.label}): {msg}", stacklevel=3)

    # ---------------------------------------------------------------- TC regime (Eq. 1)
    @property
    def beta_L(self) -> float:
        """Tesche-Clarke ``2 L_sq I0/Phi0`` (``I0`` per junction; twice Eq. 1 as printed)."""
        return 2.0 * self.L_sq_H * self.I0_A / PHI0

    @property
    def beta_C(self) -> float:
        """Stewart-McCumber ``2 pi I0 R_j^2 C_j/Phi0`` (twice Eq. 1 as printed; nan without ``C_j_F``)."""
        return 2.0 * math.pi * self.I0_A * self.R_j_ohm ** 2 * self.C_j_F / PHI0 if self.C_j_F else float("nan")

    @property
    def Gamma(self) -> float:
        return 2.0 * math.pi * KB * self.T_j_K / (self.I0_A * PHI0)

    @property
    def f_josephson_Hz(self) -> float:
        """``V/Phi0`` at the App. A mean voltage ``0.38 I0 R_j`` (the TC average response)."""
        return 0.38 * self.I0_A * self.R_j_ohm / PHI0

    def validity(self) -> dict:
        """Eq. 1 parameters and the reasons, if any, the TC formulas may not apply."""
        w = []
        lo, hi = BETA_WINDOW
        if not lo <= self.beta_L <= hi:
            w.append(f"beta_L = 2 L I0/Phi0 = {self.beta_L:.3g} is outside [{lo}, {hi}] (TC optimum ~ 1)")
        if self.C_j_F is not None and not lo <= self.beta_C <= hi:
            w.append(f"beta_C = 2 pi I0 R^2 C/Phi0 = {self.beta_C:.3g} is outside [{lo}, {hi}] (TC optimum ~ 1)")
        if self.Gamma > GAMMA_MAX:
            w.append(f"Gamma = {self.Gamma:.3g} > {GAMMA_MAX}: the TC noise coefficients 16/11/12 "
                     "(Eqs. 5-7) assume Gamma <~ 0.05 (Eq. 1)")
        elif self.Gamma > GAMMA_EQ2 and self.V_Phi_coeff == 1.0:
            w.append(f"Gamma = {self.Gamma:.3g} > {GAMMA_EQ2}: dV/dPhi = R_j/L_sq (Eq. 2) assumes "
                     "Gamma < 0.025; App. A gives 0.8 R_j/L_sq at Gamma = 0.05 (set V_Phi_coeff)")
        return {"beta_L": self.beta_L, "beta_C": self.beta_C, "Gamma": self.Gamma,
                "beta_L_eq1_printed": self.beta_L / 2.0, "beta_C_eq1_printed": self.beta_C / 2.0,
                "f_josephson_Hz": self.f_josephson_Hz,
                "beta_C_checked": self.C_j_F is not None, "warnings": w}

    # ---------------------------------------------------------------- responses (Eqs. 2-4)
    @property
    def V_Phi_Hz(self) -> float:
        """``(dV/dPhi)_I = V_Phi_coeff R_j/L_sq`` [V/Wb = Hz]  (Eq. 2)."""
        return self.V_Phi_coeff * self.R_j_ohm / self.L_sq_H

    @property
    def R_dyn_ohm(self) -> float:
        """``(dV/dI)_Phi = R_dyn_coeff R_j``  (Eq. 3)."""
        return self.R_dyn_coeff * self.R_j_ohm

    @property
    def dPhi_dI_H(self) -> float:
        """``|(dPhi/dI)_V| = R_dyn/V_Phi`` [H] (Eq. 4 / B3: ``L_sq`` with the paper's rounding).

        The voltage-biased first stage drives its output current through the second-stage
        input coil; this is the flux referral of everything downstream (Eq. 35)."""
        return self.R_dyn_ohm / self.V_Phi_Hz

    # ---------------------------------------------------------------- TC noise (Eqs. 5-13)
    def one_sided(self) -> dict:
        """One-sided TC PSDs referred to the SQUID flux (input open), from Eqs. 5-7:

        ``S_Phi = 16 k T R/V_Phi^2`` [Wb^2/Hz], ``S_J = 11 k T/R`` [A^2/Hz],
        ``S_PhiJ = 12 k T/V_Phi`` [Wb A/Hz] (magnitude; its sign is ``Chain.sign``)."""
        kT, R = KB * self.T_j_K, self.R_j_ohm
        return {"S1_Phi_Wb2_Hz": 16.0 * kT * R / self.V_Phi_Hz ** 2,
                "S1_J_A2_Hz": 11.0 * kT / R,
                "S1_PhiJ_WbA_Hz": 12.0 * kT / self.V_Phi_Hz}

    @property
    def eps_uc_J_s(self) -> float:
        """``eps_uc = S_Phi/(2 L_sq)`` (Eq. 13; ``8 k T L_sq/R_j`` with the paper's rounding)."""
        return self.one_sided()["S1_Phi_Wb2_Hz"] / (2.0 * self.L_sq_H)

    @property
    def eta_TC(self) -> float:
        """``sqrt(S_Phi S_J - S_PhiJ^2)/hbar`` (Eqs. 26-29; ``4 sqrt2 k T L_sq/(hbar R_j)``)."""
        p = self.one_sided()
        return math.sqrt(p["S1_Phi_Wb2_Hz"] * p["S1_J_A2_Hz"] - p["S1_PhiJ_WbA_Hz"] ** 2) / HBAR

    @property
    def on_resonance_over_SQL(self) -> float:
        """``sqrt(S_VV S_II)/(hbar omega) = sqrt(S_Phi S_J)/hbar`` (Eq. 24; ``4 sqrt11 k T L/(hbar R)``)."""
        p = self.one_sided()
        return math.sqrt(p["S1_Phi_Wb2_Hz"] * p["S1_J_A2_Hz"]) / HBAR

    def at_eta(self, eta: float) -> "FirstStage":
        """The same device with ``T_j`` rescaled so that ``eta_TC = eta`` (TC ratios kept).

        This is how a benchmark ``eta`` (e.g. 10) maps onto a TC device; it is a bookkeeping
        device, not a physical shunt temperature."""
        return replace(self, T_j_K=self.T_j_K * _positive(eta, "eta") / self.eta_TC,
                       label=f"{self.label} at eta_TC = {eta:g}")


# ===================================================================== second stage (Sec. IV C-D)
@dataclass(frozen=True)
class SecondStage:
    """Series-array second stage scaled from the Table I prototype (Eqs. 30-34).

    ``N_ser`` SQUIDs per bank, ``N_par`` banks with outputs in parallel; all input coils in
    series, plus ``n_dummy`` open-junction dummy SQUIDs per bank (input inductance only)."""
    N_ser: int
    N_par: int
    T2_K: float = 1.0                 # Sec. V A: operated at 1 K (linear T scaling, Eq. 34)
    L_SQ2_H: float = 120e-12          # Sec. IV D: unit-cell input inductance (WRSpice)
    L_int_H: float = 2e-9             # Sec. IV D: wirebond/interconnect to the first stage
    n_dummy: int = 2                  # Sec. IV D item 1: one dummy at each end of every bank
    unit: Mapping[str, float] = field(default_factory=lambda: dict(TABLE_I_PLUS))
    label: str = "second stage"

    def __post_init__(self):
        if int(self.N_ser) != self.N_ser or int(self.N_par) != self.N_par or self.N_ser < 1 or self.N_par < 1:
            raise ValueError("squid.SecondStage: N_ser and N_par must be positive integers")
        for nm in ("T2_K", "L_SQ2_H"):
            _positive(getattr(self, nm), nm)
        if self.L_int_H < 0 or self.n_dummy < 0:
            raise ValueError("squid.SecondStage: L_int_H and n_dummy must be nonnegative")
        if self.T2_K < T2_MIN_K:
            warnings.warn(f"squid.SecondStage: T2 = {self.T2_K:g} K is below {T2_MIN_K:g} K, where the linear "
                          "T2 scaling of Eq. 34 is supported (Sec. IV D: self-heating); extrapolated", stacklevel=3)

    @property
    def N_SQ(self) -> int:
        return int(self.N_ser * self.N_par)

    @property
    def L2_H(self) -> float:
        """``L2 = N_par (N_ser + n_dummy) L_SQ2 + L_int`` (Sec. IV D items 1-3: 18, 68, 150 L_SQ2)."""
        return self.N_par * (self.N_ser + self.n_dummy) * self.L_SQ2_H + self.L_int_H

    def tau_s(self, R_dyn1_ohm: float) -> float:
        """``tau = L2/R_dyn1``, the first-to-second-stage single pole (Table II, Eq. 35)."""
        return self.L2_H / _positive(R_dyn1_ohm, "R_dyn1_ohm")

    @property
    def P_W(self) -> float:                      # Eq. 30
        return self.unit["P_W"] * self.N_SQ / (self.unit["N_ser"] * self.unit["N_par"])

    @property
    def dPhi_dI_H(self) -> float:                # Eq. 31 (magnitude)
        return self.unit["dPhi_dI_H"] * self.unit["N_par"] / self.N_par

    @property
    def R_dyn_ohm(self) -> float:                # Eq. 32
        u = self.unit
        return u["dV_dI_ohm"] * (self.N_ser / u["N_ser"]) * (u["N_par"] / self.N_par)

    @property
    def dV_dPhi_Hz(self) -> float:               # Eq. 33
        return self.unit["dV_dPhi_Hz"] * self.N_ser / self.unit["N_ser"]

    @property
    def M_IN2_H(self) -> float:                  # per-SQUID input mutual, N-independent (Sec. IV C)
        return self.unit["M_IN2_H"]

    def S1_Phi_squid(self) -> float:
        """Array imprecision flux PSD [Wb^2/Hz, one-sided], Eq. 34 / first term of Eq. 36:
        ``Phi_n,20x1^2 (20/(N_ser N_par)) (T2/4 K)``."""
        u = self.unit
        return ((u["Phi_n_SQ_Phi0_rtHz"] * PHI0) ** 2 * (u["N_ser"] * u["N_par"]) / self.N_SQ
                * self.T2_K / u["T_K"])


# ===================================================================== follow-on amplifiers (Sec. IV E-F, V B-C)
@dataclass(frozen=True)
class OpAmpPreamp:
    """Room-temperature "op-amp mode" SQUID preamplifier with voltage/current noise and leads."""
    V_n_V_rtHz: float
    I_n_A_rtHz: float
    R_lead_ohm: float = 1.0           # Sec. V B: Cu coax 4 K -> RT, frequency independent
    T_lead_K: float = 200.0
    f_min_Hz: float = 0.0
    f_max_Hz: float = F_MAX_HZ
    label: str = "op-amp preamp"

    def S1_Phi2(self, stage: SecondStage) -> dict:
        """Flux PSDs at the second-stage input [Wb^2/Hz, one-sided], Eqs. 37-39:
        ``(V_n^2 + 4 k T_lead R_lead)/(dV/dPhi)^2`` and ``(I_n dPhi/dI)^2`` of the scaled array."""
        v = (self.V_n_V_rtHz ** 2 + 4.0 * KB * self.T_lead_K * self.R_lead_ohm) / stage.dV_dPhi_Hz ** 2
        i = (self.I_n_A_rtHz * stage.dPhi_dI_H) ** 2
        return {"preamp_V_and_lead": v, "preamp_I": i}


@dataclass(frozen=True)
class CryoRFAmp:
    """Matched 50-ohm cryogenic "scattering mode" RF amplifier of noise temperature ``T_n``."""
    T_n_K: float = 2.0
    f_min_Hz: float = 5e6
    f_max_Hz: float = 500e6
    label: str = "cryogenic RF amplifier"
    R_in_ohm: float = 50.0            # Sec. IV F: 50-ohm input, matched to the arrays' ~50-ohm R_dyn

    def S1_Phi2(self, stage: SecondStage) -> dict:
        """Eq. C1 (= Eq. 40 without its chain-rule minus sign, see the docs):
        ``4 k T_n |dPhi/dI| |dPhi/dV|`` of the scaled array ``= 4 k T_n R_dyn/(dV/dPhi)^2``.

        A single noise temperature describes the amplifier only for a matched source, so the
        array's ``R_dyn`` must be within ``R_MATCH_TOL`` of ``R_in_ohm`` (presets: 51.9 ohm)."""
        if abs(stage.R_dyn_ohm / self.R_in_ohm - 1.0) > R_MATCH_TOL:
            raise ValueError(f"squid: {self.label} needs a matched source, but the {stage.N_ser}x{stage.N_par} "
                             f"array has R_dyn = {stage.R_dyn_ohm:.4g} ohm vs R_in = {self.R_in_ohm:g} ohm; "
                             "choose N_ser/N_par ~ 20/(64.9/R_in) (Eq. 32) or model the mismatch")
        return {"rf_amp": 4.0 * KB * self.T_n_K * stage.dPhi_dI_H / stage.dV_dPhi_Hz}


@dataclass(frozen=True)
class SwitchedPreamp:
    """Different amplifiers in different bands: ``((f_lo, f_hi, amp), ...)``, ``f_lo <= f < f_hi``."""
    bands: tuple
    label: str = "switched preamp"

    def at(self, f: float):
        for lo, hi, amp in self.bands:
            if lo <= f < hi or (f == hi and hi == self.bands[-1][1]):
                return amp
        raise ValueError(f"squid: {f:.6g} Hz is outside every band of {self.label!r}")


# [DMR-SQUID] Sec. IV E: Magnicon XXF-1, < 50 MHz (open-loop bandwidth), 0.33 nV/rtHz, 2.6 pA/rtHz.
XXF1 = OpAmpPreamp(0.33e-9, 2.6e-12, f_max_Hz=50e6, label="Magnicon XXF-1")
# Sec. IV E: ultra-high-speed RT electronics (Drung, refs. 38-39), 300 MHz, 0.3 nV/rtHz, 6 pA/rtHz.
RT300 = OpAmpPreamp(0.3e-9, 6e-12, f_max_Hz=300e6, label="300 MHz RT preamp")
# Sec. IV F / App. C: CMT CITLF2, T_n < 2 K over 5-500 MHz; superconducting coax, R_lead = 0.
CITLF2 = CryoRFAmp(2.0, 5e6, 500e6, label="CITLF2 cryogenic RF amp")
# Fig. 3 solid lines: XXF-1 below 50 MHz, the 300 MHz RT preamp above.
RT = SwitchedPreamp(((0.0, 50e6, XXF1), (50e6, F_MAX_HZ, RT300)), label="RT (XXF-1 < 50 MHz, 300 MHz preamp above)")

C1 = FirstStage(L_sq_H=200e-12, I0_A=6.3e-6, R_j_ohm=6.0, T_j_K=0.150,
                label="C1")    # Sec. III B (SQUID C1 of Wellstood et al. 1987; shunts self-heat to 150 mK)
SECOND_STAGES = {"low": (48, 3), "medium": (32, 2), "high": (16, 1)}   # Sec. IV D, Table II
AMPLIFIERS = {"RT": RT, "cryo": CITLF2, "XXF1": XXF1, "RT300": RT300, "CITLF2": CITLF2}


# ===================================================================== the chain (Sec. V-VII)
@dataclass(frozen=True)
class Chain:
    """First stage, optional second stage + amplifier, and the correlation sign ``s``.

    Later stages add uncorrelated imprecision only (their backaction on the first stage
    and on ``S_IV`` is neglected)."""
    first: FirstStage
    second: SecondStage | None = None
    amplifier: OpAmpPreamp | CryoRFAmp | SwitchedPreamp | None = None
    sign: float = 1.0                  # s = +-1, OPEN (not derived; docs Sec. 5)
    label: str = "chain"

    def __post_init__(self):
        if (self.second is None) != (self.amplifier is None):
            raise ValueError("squid.Chain: give both second= and amplifier=, or neither (first stage alone)")
        if self.sign not in (1.0, -1.0):
            raise ValueError("squid.Chain: sign must be +1 or -1 (the open TC correlation sign)")

    @property
    def tau_s(self) -> float:
        return self.second.tau_s(self.first.R_dyn_ohm) if self.second else 0.0

    def __call__(self, f_Hz):
        """``eta(f)``, so a chain can be passed wherever a callable ``eta_A(f)`` is accepted."""
        return self.eta(f_Hz)

    # ---------------------------------------------------------------- PSDs at the first-stage flux
    def _amp_parts(self, f: float) -> dict:
        amp = self.amplifier
        if isinstance(amp, SwitchedPreamp):
            amp = amp.at(f)
        if not amp.f_min_Hz <= f <= amp.f_max_Hz * (1 + 1e-12):
            raise ValueError(f"squid: {f:.6g} Hz is outside the specified band of {amp.label} "
                             f"[{amp.f_min_Hz:.3g}, {amp.f_max_Hz:.3g}] Hz")
        return amp.S1_Phi2(self.second)

    def one_sided(self, f_Hz) -> dict:
        """One-sided PSDs referred to the FIRST-STAGE flux [Wb^2/Hz, A^2/Hz, Wb A/Hz].

        ``S1_Phi_total = S1_Phi_TC + (dPhi1/dI |Phi_n2| /M_IN2)^2 (1 + omega^2 tau^2)``.
        The input-current PSD is multiplied by ``M^2`` to refer it to flux.  ``S1_PhiJ`` is
        signed (``sign * 12 k T/V_Phi``)."""
        f = _f_array(f_Hz)
        fj = self.first.f_josephson_Hz
        if np.any(f > F_JOSEPHSON_FRACTION * fj):
            warnings.warn(f"squid: {float(np.max(f)):.4g} Hz exceeds {F_JOSEPHSON_FRACTION:g} f_J = "
                          f"{F_JOSEPHSON_FRACTION * fj:.4g} Hz of {self.first.label}; the averaged TC response "
                          "(Eqs. 2-7) needs f << f_J", stacklevel=2)
        p = self.first.one_sided()
        out = {"f_Hz": f, "S1_Phi_TC_Wb2_Hz": np.full(f.shape, p["S1_Phi_Wb2_Hz"]),
               "S1_J_A2_Hz": np.full(f.shape, p["S1_J_A2_Hz"]),
               "S1_PhiJ_WbA_Hz": np.full(f.shape, self.sign * p["S1_PhiJ_WbA_Hz"])}
        ref = np.zeros(f.shape)
        parts: dict[str, np.ndarray] = {}
        if self.second is not None:
            g2 = (self.first.dPhi_dI_H / self.second.M_IN2_H) ** 2 * (1.0 + (2 * math.pi * f * self.tau_s) ** 2)
            sq = self.second.S1_Phi_squid()
            parts["second_stage_squid"] = g2 * sq
            for i, fi in np.ndenumerate(f):
                for k, v in self._amp_parts(float(fi)).items():
                    parts.setdefault(k, np.zeros(f.shape))[i] = g2[i] * v
            ref = sum(parts.values())
        out["S1_Phi_ref_Wb2_Hz"] = ref
        out["S1_Phi_ref_parts_Wb2_Hz"] = parts
        out["S1_Phi_Wb2_Hz"] = out["S1_Phi_TC_Wb2_Hz"] + ref
        return out

    # ---------------------------------------------------------------- the two scan numbers
    def eta(self, f_Hz):
        """``eta(omega) = sqrt(S_VV S_IIsys - (Im S_IV)^2)/(hbar omega)
        = sqrt(S_Phi,tot S_J - S_PhiJ^2)/hbar``  (independent of ``M``)."""
        p = self.one_sided(f_Hz)
        e = np.sqrt(p["S1_Phi_Wb2_Hz"] * p["S1_J_A2_Hz"] - p["S1_PhiJ_WbA_Hz"] ** 2) / HBAR
        return _out(e, f_Hz)

    def eta_uncorrelated(self, f_Hz):
        """``sqrt(S_Phi,tot S_J)/hbar``: what ``eta`` would be with the correlation dropped."""
        p = self.one_sided(f_Hz)
        return _out(np.sqrt(p["S1_Phi_Wb2_Hz"] * p["S1_J_A2_Hz"]) / HBAR, f_Hz)

    def eps_uc(self, f_Hz):
        """``eps_uc(omega) = M^2 S_IIsys/(2 L_sq) = S_Phi,tot/(2 L_sq)`` [J s]."""
        p = self.one_sided(f_Hz)
        return _out(p["S1_Phi_Wb2_Hz"] / (2.0 * self.first.L_sq_H), f_Hz)

    def eps_uc_over_hbar(self, f_Hz):
        return _out(np.asarray(self.eps_uc(f_Hz)) / HBAR, f_Hz)

    # ---------------------------------------------------------------- input-referred (Eqs. 9-12, 35, 41)
    def input_referred(self, f_Hz, M_H: float) -> dict:
        """One-sided PSDs referred to the input coil coupled by ``M``:
        ``S_IIsys = S_Phi,tot/M^2`` [A^2/Hz], ``S_VV = omega^2 M^2 S_J`` [V^2/Hz]
        and ``S_IV = <I_n V_ba*> = -i omega S_PhiJ`` [W/Hz], ``S_PhiJ`` signed
        (backaction EMF ``+i omega M J`` for ``exp(-i omega t)``)."""
        M = _positive(abs(M_H), "|M_H|")
        p = self.one_sided(f_Hz)
        w = 2 * math.pi * p["f_Hz"]
        out = {"S_II_A2_Hz": p["S1_Phi_TC_Wb2_Hz"] / M ** 2, "S_II_ref_A2_Hz": p["S1_Phi_ref_Wb2_Hz"] / M ** 2,
               "S_II_sys_A2_Hz": p["S1_Phi_Wb2_Hz"] / M ** 2, "S_VV_V2_Hz": w ** 2 * M ** 2 * p["S1_J_A2_Hz"],
               "S_IV_W_Hz": -1j * w * p["S1_PhiJ_WbA_Hz"]}
        return {k: _out(v, f_Hz) for k, v in out.items()}

    def noise_temperature(self, f_Hz, Z_ohm, M_H: float):
        """Amplifier noise temperature [K] for a source of loop impedance ``Z`` (package
        convention, ``Z = R - i X`` for an inductive loop), from the series-EMF referral
        ``4 k T_n Re Z = S_VV + |Z|^2 S_IIsys + 2 omega S_PhiJ Im Z`` (``S_PhiJ`` signed;
        the cross term is the correlation)."""
        r = self.input_referred(f_Hz, M_H)
        p = self.one_sided(f_Hz)
        Z = np.asarray(Z_ohm, dtype=complex)
        if np.any(Z.real <= 0):
            raise ValueError("squid.noise_temperature: Re Z must be > 0")
        w = 2 * math.pi * p["f_Hz"]
        Se = r["S_VV_V2_Hz"] + abs(Z) ** 2 * r["S_II_sys_A2_Hz"] + 2.0 * w * p["S1_PhiJ_WbA_Hz"] * Z.imag
        return _out(Se / (4.0 * KB * Z.real), f_Hz)

    def optimal_source_impedance(self, f_Hz, M_H: float):
        """Complex source impedance minimizing ``T_n`` (package convention):
        ``Im Z = -omega S_PhiJ/S_IIsys`` (``S_PhiJ`` signed), ``Re Z = sqrt(S_VV/S_IIsys - (Im Z)^2)``."""
        r = self.input_referred(f_Hz, M_H)
        p = self.one_sided(f_Hz)
        w = 2 * math.pi * p["f_Hz"]
        X = -w * p["S1_PhiJ_WbA_Hz"] / np.asarray(r["S_II_sys_A2_Hz"])
        R = np.sqrt(np.asarray(r["S_VV_V2_Hz"]) / np.asarray(r["S_II_sys_A2_Hz"]) - X ** 2)
        return _out(R + 1j * X, f_Hz)

    def T_min_K(self, f_Hz):
        """Minimum noise temperature at the optimal complex source: ``eta hbar omega/(2 k)``."""
        f = _f_array(f_Hz)
        return _out(np.asarray(self.eta(f)) * HBAR * 2 * math.pi * f / (2.0 * KB), f_Hz)

    def on_resonance(self, f_Hz, Q: float | None = None, M_H: float | None = None) -> dict:
        """Real-source (on-resonance) match: the correlation drops out.

        ``k T_min = (1/2) omega sqrt(S_Phi,tot S_J)`` and ``T_min/SQL = sqrt(S_Phi,tot S_J)/hbar``;
        with ``Q`` the matching coupling ``kappa_g^2 = sqrt(S_Phi,tot/S_J)/(L_sq Q)`` (``1.2/Q``
        for TC); with ``M`` the matching resistance ``R = omega M^2 sqrt(S_J/S_Phi,tot)``."""
        p = self.one_sided(f_Hz)
        w = 2 * math.pi * p["f_Hz"]
        root = np.sqrt(p["S1_Phi_Wb2_Hz"] * p["S1_J_A2_Hz"])
        out = {"T_min_K": 0.5 * w * root / KB, "T_min_over_SQL": root / HBAR}
        if Q is not None:
            out["kappa_g2_match"] = np.sqrt(p["S1_Phi_Wb2_Hz"] / p["S1_J_A2_Hz"]) / (self.first.L_sq_H * _positive(Q, "Q"))
        if M_H is not None:
            out["R_match_ohm"] = w * M_H ** 2 * np.sqrt(p["S1_J_A2_Hz"] / p["S1_Phi_Wb2_Hz"])
        return {k: _out(v, f_Hz) for k, v in out.items()}

    # ---------------------------------------------------------------- scan-optimal match (Sec. VII)
    def xi_opt(self, f_Hz, T_K: float):
        """``xi_opt = 2 eta^2/(2n+1+sqrt((2n+1)^2+8 eta^2))``, with Bose-Einstein occupation ``n``."""
        f = _f_array(f_Hz)
        n = np.array([n_bose(2 * math.pi * x, T_K) for x in np.atleast_1d(f)]).reshape(f.shape)
        e2 = np.asarray(self.eta(f)) ** 2
        s = 2 * n + 1
        return _out(2 * e2 / (s + np.sqrt(s * s + 8 * e2)), f_Hz)

    def kappa_g2_scan(self, f_Hz, Q: float, T_K: float):
        """Scan-optimal global coupling:
        ``kappa_g^2 = (1/Q)(eps_uc/hbar)[2n+1+sqrt((2n+1)^2+8 eta^2)]/eta^2 = 2 eps_uc/(hbar Q xi_opt)``,
        ``kappa_g^2 = M^2/(L_sq L_tot)``.  Unconstrained: values > 1 are not attainable
        (see :meth:`M_scan_H`)."""
        f = _f_array(f_Hz)
        return _out(2.0 * np.asarray(self.eps_uc(f)) / (HBAR * _positive(Q, "Q") * np.asarray(self.xi_opt(f, T_K))), f_Hz)

    def M_scan_H(self, f_Hz, Q: float, L_tot_H: float, T_K: float):
        """``M = sqrt(kappa_g^2 L_sq L_tot)`` at the scan-optimal coupling [H].

        This is an unconstrained optimum; ``kappa_g^2 = kappa^2 L_in/L_tot <= 1`` is required,
        so this raises where the optimum asks for more (e.g. low Q at low frequency).
        With a known input coil the tighter bound is ``kappa^2 L_in/L_tot``."""
        k2 = np.asarray(self.kappa_g2_scan(f_Hz, Q, T_K))
        if np.any(k2 > 1.0):
            raise ValueError(f"squid: Eq. 48 asks for kappa_g^2 = {float(np.max(k2)):.4g} > 1, which no input "
                             "transformer reaches (Eq. 21); the scan-optimal coupling is not attainable here")
        return _out(np.sqrt(k2 * self.first.L_sq_H * _positive(L_tot_H, "L_tot_H")), f_Hz)

    # ---------------------------------------------------------------- hand-off to the circuit closures
    def squid(self, f0_Hz: float, M_H: float, input_coil: str = "L_in") -> Squid:
        """A :class:`axionfem.circuit.Squid` with this chain's PSDs FROZEN at ``f0`` (Foster
        normalisation, ``S = S^(1)/2``), including ``S_PhiJ = s S^(1)_PhiJ/2``; ``eta_A`` is the
        correlated ``eta(f0)``.  Across a noise band of width ``dw`` the frozen ``1 + w^2 tau^2``
        is off by at most ``2 dw/w`` (relative), < 1e-6 for C1+low+RT at 1 MHz and Q = 1e6."""
        if np.ndim(f0_Hz) != 0:
            raise ValueError("squid.Chain.squid: give one tuning frequency")
        p = self.one_sided(float(f0_Hz))
        sq = Squid(M_H=float(M_H), eta_A=float(self.eta(float(f0_Hz))),
                   S_Phi=float(p["S1_Phi_Wb2_Hz"][()]) / 2.0, S_J=float(p["S1_J_A2_Hz"][()]) / 2.0,
                   input_coil=input_coil, S_PhiJ=float(p["S1_PhiJ_WbA_Hz"][()]) / 2.0)
        return sq

    def squid_for_resonator(self, f0_Hz: float, Q: float, L_tot_H: float, T_K: float,
                            input_coil: str = "L_in") -> Squid:
        """:meth:`squid` at the scan-optimal mutual inductance of a resonator ``(Q, L_tot, T)``."""
        return self.squid(f0_Hz, float(self.M_scan_H(float(f0_Hz), Q, L_tot_H, T_K)), input_coil)

    def draw(self, f_Hz: float | None = None, annotate: bool = True, title: str | None = None):
        """Block schematic of the chain (``schemdraw``; install ``axionfem[draw]``): input coil,
        TC first stage, second-stage array and preamplifier, as present in this chain.  With
        ``annotate`` each stage shows its parameters (``L_sq, I0, R_j, T_j``, ``V_Phi``, ``eta_TC``;
        ``N_ser x N_par``, ``T_2``, ``L_2``, ``tau``, ``M_IN2``, flux gain; the amplifier's noise),
        and with ``f_Hz`` also ``eta(f)``, ``eps_uc/hbar`` and each stage's referred flux noise.
        Returns a ``schemdraw.Drawing``."""
        from .draw import draw_chain
        return draw_chain(self, f_Hz=f_Hz, annotate=annotate, title=title)

    def summary(self, f_Hz, Q: float | None = None, T_K: float | None = None) -> dict:
        """``eta``, ``eps_uc/hbar``, the referred-noise breakdown and (with ``Q``, ``T``) the scan-optimal
        coupling, on a frequency grid; plain lists, ready for JSON."""
        f = np.atleast_1d(_f_array(f_Hz))
        p = self.one_sided(f)
        out = {"label": self.label, "sign": self.sign, "f_Hz": f.tolist(),
               "eta": np.atleast_1d(self.eta(f)).tolist(),
               "eta_uncorrelated": np.atleast_1d(self.eta_uncorrelated(f)).tolist(),
               "eps_uc_over_hbar": np.atleast_1d(self.eps_uc_over_hbar(f)).tolist(),
               "S1_Phi_TC_Wb2_Hz": p["S1_Phi_TC_Wb2_Hz"].tolist(),
               "S1_Phi_ref_parts_Wb2_Hz": {k: v.tolist() for k, v in p["S1_Phi_ref_parts_Wb2_Hz"].items()},
               "first_stage": {**{k: v for k, v in self.first.validity().items()},
                               "eta_TC": self.first.eta_TC, "eps_uc_over_hbar": self.first.eps_uc_J_s / HBAR},
               "tau_s": self.tau_s, "L2_H": self.second.L2_H if self.second else None}
        if Q is not None and T_K is not None:
            out["Q"], out["T_K"] = Q, T_K
            out["kappa_g2_scan"] = np.atleast_1d(self.kappa_g2_scan(f, Q, T_K)).tolist()
            out["xi_opt"] = np.atleast_1d(self.xi_opt(f, T_K)).tolist()
        return out


# ===================================================================== presets and the user-facing API
def presets() -> list[str]:
    """Named chains: ``"C1"`` (first stage alone, a lower bound) and ``"C1+<low|medium|high>+<RT|cryo>"``."""
    return ["C1"] + [f"C1+{bw}+{amp}" for bw in SECOND_STAGES for amp in ("RT", "cryo")]


def chain(preset: str = "C1+low+RT", *, first: FirstStage | None = None, T2_K: float = 1.0,
          sign: float = 1.0, amplifier=None) -> Chain:
    """A named chain from Ankel et al. (arXiv:2504.20398).

    ``"C1"``: the TC first stage alone (no later stages; eta = 3.70, a lower bound).
    ``"C1+low|medium|high+RT|cryo"``: C1 + a 48x3 / 32x2 / 16x1 array at ``T2_K`` (Table II)
    + the RT preamps (XXF-1 below 50 MHz, the 300 MHz RT preamp above; Fig. 3 solid) or the
    CITLF2 cryogenic RF amplifier (specified for 5-500 MHz only; Fig. 3 dashed).
    ``first`` replaces C1 (e.g. ``squid.C1.at_eta(10)``); ``amplifier`` replaces the preset's.
    ``sign`` is the open correlation sign ``s``."""
    parts = preset.split("+")
    fs = first if first is not None else C1
    if parts == ["C1"]:
        if amplifier is not None:
            raise ValueError("squid.chain: preset 'C1' has no later stages; use 'C1+<bw>+<amp>'")
        return Chain(fs, sign=sign, label=f"{fs.label} alone")
    if len(parts) != 3 or parts[0] != "C1" or parts[1] not in SECOND_STAGES or parts[2] not in ("RT", "cryo"):
        raise ValueError(f"squid.chain: unknown preset {preset!r}; choose one of {presets()}")
    ns, npar = SECOND_STAGES[parts[1]]
    st = SecondStage(ns, npar, T2_K=T2_K, label=f"{parts[1]} BW {ns}x{npar}")
    amp = amplifier if amplifier is not None else AMPLIFIERS[parts[2]]
    return Chain(fs, st, amp, sign=sign, label=f"{fs.label} + {st.label} + {getattr(amp, 'label', parts[2])}")


def eta_curve(preset: str, f_Hz, **kw):
    """``eta(f)`` of a named chain (see :func:`chain`)."""
    return chain(preset, **kw).eta(f_Hz)


def benchmark(ch: Chain | str, Q: float, T_K: float) -> dict:
    """A noise benchmark for :func:`axionfem.scan.reach` with ``eta_A = eta(f)`` of the chain:
    ``scan.reach(Psi, L, squid.benchmark("C1+low+RT", Q=1e6, T_K=0.03))``."""
    ch = chain(ch) if isinstance(ch, str) else ch
    return {"Q": _positive(Q, "Q"), "T": float(T_K), "eta_A": ch, "label": ch.label}


def table_I_check(column: Mapping[str, float] = TABLE_I_PLUS,
                  preamp: Mapping[str, float] = TABLE_I_PREAMP) -> dict:
    """Recompute Table I's referred preamp noise and its quadrature split from Eqs. 38-39
    (20x1 array, no lead term) [uPhi0/rtHz]: an internal-consistency check of the table."""
    st = SecondStage(20, 1, T2_K=4.0, unit=dict(column))
    amp = OpAmpPreamp(preamp["V_n_V_rtHz"], preamp["I_n_A_rtHz"], R_lead_ohm=0.0)
    pre = math.sqrt(sum(amp.S1_Phi2(st).values())) / PHI0 * 1e6
    tot = column["Phi_n2_total_Phi0_rtHz"] * 1e6
    return {"preamp_referred": pre, "preamp_referred_table": column["Phi_n_preamp_Phi0_rtHz"] * 1e6,
            "squid_by_quadrature": math.sqrt(tot ** 2 - pre ** 2),
            "squid_table": column["Phi_n_SQ_Phi0_rtHz"] * 1e6,
            "chain_rule_dV_dI_ohm": column["dV_dPhi_Hz"] * column["dPhi_dI_H"], "dV_dI_table_ohm": column["dV_dI_ohm"]}


__all__ = ["AMPLIFIERS", "C1", "CITLF2", "Chain", "CryoRFAmp", "FirstStage", "OpAmpPreamp", "RT", "RT300",
           "SECOND_STAGES", "SecondStage", "SwitchedPreamp", "TABLE_I_MINUS", "TABLE_I_PLUS", "XXF1",
           "benchmark", "chain", "eta_curve", "presets", "table_I_check"]
