"""Energy and component reductions for reciprocal exp(-i omega t) multiports.

These exact circuit identities do not qualify a meshed component, its source
quadrature, or a detector's continuous frequency response. Energies use peak
phasors: a current vector I stores the time-average energy I^dagger G I.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import HBAR, Squid
from .multiport import _psd_matrix


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class QuarterSymmetry:
    """One y-odd/z-even sector with paired modules and a bisected readout.

    Full order is (upper modules, lower modules, centered readout). Physical
    reduced currents are pair-total module currents and actual readout current.
    The full-current lift is H; quarter currents are half the physical currents.
    The component loads are C_eff=2*C_module and unchanged readout impedance.
    """

    pairs: int

    def __post_init__(self):
        if int(self.pairs) != self.pairs or self.pairs < 1:
            raise ValueError("QuarterSymmetry.pairs must be a positive integer")

    @property
    def current_lift(self):
        h = np.zeros((2 * self.pairs + 1, self.pairs + 1))
        h[:self.pairs, :self.pairs] = np.eye(self.pairs) / 2
        h[self.pairs:2*self.pairs, :self.pairs] = np.eye(self.pairs) / 2
        h[-1, -1] = 1
        return h

    def physical_full_to_reduced(self, Z, Psi, G=None):
        """Power-conserving projection; this also projects an FDT covariance.

        Z_eff=H.T Z_full H, Psi_eff=H.T Psi_full. G_eff=H.T G_full H.
        Independent full thermal sources become their projected covariance;
        they do not acquire the coherent axion voltage's image factor squared.
        """
        h = self.current_lift
        z, psi = np.asarray(Z, complex), np.asarray(Psi, complex)
        if z.shape != (h.shape[0], h.shape[0]) or psi.shape != (h.shape[0],):
            raise ValueError("full Z/Psi do not match paired-module/readout order")
        result = (h.T @ z @ h, h.T @ psi)
        if G is not None:
            g = np.asarray(G, complex)
            if g.shape != z.shape:
                raise ValueError("full energy matrix G must have the shape of Z")
            result += (h.T @ g @ h,)
        return result

    def quarter_to_physical(self, Z, Psi, G=None):
        """Z_eff=Z_q, Psi_eff=2*Psi_q, G_eff=G_q in physical currents.

        At fixed *quarter* currents the full energy is four times the quarter
        energy, but physical current is twice quarter current. The two factors
        cancel in G_eff. This is not a whole-matrix doubling prescription.
        """
        n = self.pairs + 1
        z, psi = np.asarray(Z, complex), np.asarray(Psi, complex)
        if z.shape != (n, n) or psi.shape != (n,):
            raise ValueError("quarter Z/Psi do not match module/readout order")
        result = (z.copy(), 2 * psi)
        if G is not None:
            g = np.asarray(G, complex)
            if g.shape != z.shape:
                raise ValueError("quarter energy matrix G must have the shape of Z")
            result += (g.copy(),)
        return result


@dataclass(frozen=True)
class LosslessTransmissionLine:
    """Exact uniform line; both terminal currents enter the line.

    Differential terminal voltages use the same polarity at both ends. Stamp
    impedance() with MultiportCircuit.multiport; the return conductor belongs
    to these differential voltages. L and C are the entire line's distributed
    totals, not independent lumped series/shunt corrections.
    """

    length_m: float
    inductance_H: float
    capacitance_F: float

    def __post_init__(self):
        for key in ("length_m", "inductance_H", "capacitance_F"):
            _positive(getattr(self, key), key)

    @property
    def impedance_ohm(self):
        return np.sqrt(self.inductance_H / self.capacitance_F)

    @property
    def delay_s(self):
        return np.sqrt(self.inductance_H * self.capacitance_F)

    def _angle(self, omega):
        angle = _positive(omega, "omega") * self.delay_s
        if abs(np.sin(angle)) < 1e-14:
            raise ValueError("line impedance has an open-port pole; use a different port representation")
        return angle

    def impedance(self, omega):
        a = self._angle(omega)
        return 1j * self.impedance_ohm / np.sin(a) * np.array(
            [[np.cos(a), 1.0], [1.0, np.cos(a)]])

    def energy_matrix(self, omega):
        """G=-d Im(Z)/d omega /4, in joules per ampere squared."""
        a = self._angle(omega)
        # Eigenmode form avoids subtracting nearly equal cot/csc derivatives.
        even = self.impedance_ohm * self.delay_s / (8 * np.sin(a/2)**2)
        odd = self.impedance_ohm * self.delay_s / (8 * np.cos(a/2)**2)
        return np.array([[even+odd, even-odd], [even-odd, even+odd]]) / 2

    def fields(self, omega, currents, x_m):
        """Return V(x), +x-directed I(x) for direct energy integration."""
        a = self._angle(omega)
        currents = np.asarray(currents, complex)
        if currents.shape != (2,):
            raise ValueError("line currents must contain the two entering currents")
        x = np.asarray(x_m, float)
        if np.any((x < 0) | (x > self.length_m)):
            raise ValueError("line x coordinates must lie between its reference planes")
        voltage0 = (self.impedance(omega) @ currents)[0]
        ax = a * x / self.length_m
        voltage = np.cos(ax) * voltage0 + 1j*self.impedance_ohm*np.sin(ax)*currents[0]
        current = 1j/self.impedance_ohm*np.sin(ax)*voltage0 + np.cos(ax)*currents[0]
        return voltage, current

    def loaded_impedance(self, omega, load):
        """Input impedance with physical load at the far end; inf means open."""
        a = self._angle(omega)
        z0 = self.impedance_ohm
        if np.isinf(load):
            return 1j*z0/np.tan(a)
        load = complex(load)
        if not np.isfinite(load):
            raise ValueError("line load must be finite or positive infinity for open")
        return z0 * (load-1j*z0*np.tan(a)) / (z0-1j*load*np.tan(a))


@dataclass(frozen=True)
class ModalLossCalibration:
    """A fixed positive loss realization at one tuning, including its bath."""
    resistance_ohm: np.ndarray
    total_energy_J: float
    dissipated_power_W: float
    omega0: float
    Q: float
    T_K: float
    energy_hermiticity_relative_error: float


def calibrate_modal_loss(omega0, currents, G, *, Q=1e6, T_K=0.03,
                         extra_energy_J=0.0, participation=None):
    """Calibrate R so omega0*U/(I^dagger R I/2)=Q.

    G is the *physical-coordinate* field-energy matrix. extra_energy_J must
    include omitted capacitors/ESL, line and input-coil energies in the same
    actual mode. The returned R and its temperature remain fixed over this
    tuning's noise bandwidth. Recalibrating at off-center frequencies changes
    the physical model and is incorrect. Existing loss must be accounted for
    by the caller before assigning an additional positive reservoir.
    """
    omega0, Q = _positive(omega0, "omega0"), _positive(Q, "Q")
    if not np.isfinite(T_K) or T_K < 0:
        raise ValueError("T_K must be finite and nonnegative")
    i = np.asarray(currents, complex)
    if i.ndim != 1 or not np.all(np.isfinite(i)):
        raise ValueError("mode currents must be a finite vector")
    g = np.asarray(G, complex)
    if g.shape != (len(i), len(i)):
        raise ValueError("energy matrix G must match the mode currents")
    if not np.all(np.isfinite(g)):
        raise ValueError("energy matrix G must be finite")
    gscale = max(float(np.linalg.norm(g, 2)), np.finfo(float).tiny)
    hermiticity_error = float(np.linalg.norm(g-g.conj().T, 2)/gscale)
    # Independently assembled FEM cross terms can differ at solver roundoff.
    # Taking the Hermitian part here is exactly taking Re(I^dagger G I), not
    # repairing the extracted impedance or its physical bath covariance.
    if hermiticity_error > 1e-9:
        raise ValueError("energy matrix G fails Hermiticity at1e-9; improve field convergence")
    g = _psd_matrix((g+g.conj().T)/2, "modal energy G")
    if not np.isfinite(extra_energy_J) or extra_energy_J < 0:
        raise ValueError("extra component/line/input energy must be finite and nonnegative")
    energy = _positive(np.vdot(i, g @ i).real + extra_energy_J, "total mode energy")
    rshape = np.eye(len(i)) if participation is None else np.asarray(participation, complex)
    if rshape.shape != g.shape:
        raise ValueError("loss participation must match energy matrix G")
    rshape = _psd_matrix(rshape, "loss participation")
    denominator = _positive(np.vdot(i, rshape @ i).real, "loss participation in mode")
    power = omega0 * energy / Q
    return ModalLossCalibration(rshape * (2*power/denominator), float(energy),
                                float(power), omega0, Q, float(T_K), hermiticity_error)


def fixed_squid(*, flux_noise_one_sided_Phi0=1e-6, eta_A=10.0,
                M_H=1e-9, input_coil="L_in"):
    """Positive, frequency-independent SQUID PSDs with S_PhiJ=0.

    The caller freezes these PSDs and the physical coupling throughout each
    tuning's noise-band integration. eta_A alone does not define nonzero PSDs.
    """
    _positive(flux_noise_one_sided_Phi0, "one-sided flux noise")
    _positive(eta_A, "eta_A")
    _positive(abs(M_H), "absolute SQUID mutual inductance")
    squid = Squid.from_flux_noise(flux_noise_one_sided_Phi0, eta_A=eta_A, M_H=M_H)
    squid.input_coil = input_coil
    if not np.isclose(2*np.sqrt(squid.S_Phi*squid.S_J)/HBAR, eta_A, rtol=1e-14):
        raise ValueError("SQUID PSD normalization failed")
    return squid


def series_capacitor_impedance(omega, capacitance_F, ESL_H=0.0, resistance_ohm=0.0):
    omega = _positive(omega, "omega")
    capacitance_F = _positive(capacitance_F, "capacitance_F")
    if not np.isfinite(ESL_H+resistance_ohm) or min(ESL_H, resistance_ohm) < 0:
        raise ValueError("ESL and series resistance must be finite and nonnegative")
    return resistance_ohm + 1j/(omega*capacitance_F) - 1j*omega*ESL_H


def series_capacitor_energy(omega, current, capacitance_F, ESL_H=0.0):
    """Whole omitted C+ESL component's time-average electric+magnetic energy."""
    series_capacitor_impedance(omega, capacitance_F, ESL_H)
    return float(abs(current)**2 * (ESL_H + 1/(omega*omega*capacitance_F))/4)


def fit_series_capacitor(omega, impedances):
    """Extract C and ESL from sampled *de-embedded* component impedance.

    The reported fit residual is a model check, not an internal-mode search.
    At least three points are needed for any meaningful residual; held-out
    frequencies and three actual capacitor states still require direct solves.
    """
    w, z = np.asarray(omega, float), np.asarray(impedances, complex)
    if w.ndim != 1 or z.shape != w.shape or len(w) < 3 or np.any(w <= 0):
        raise ValueError("component extraction requires at least three positive frequency samples")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(z)):
        raise ValueError("component samples must be finite")
    if len(np.unique(w)) != len(w):
        raise ValueError("component frequencies must be distinct")
    wscale = np.max(w)
    coefficients = np.linalg.lstsq(np.column_stack((np.ones_like(w), -(w/wscale)**2)),
                                  w*z.imag, rcond=None)[0]
    inverse_C, ESL = coefficients[0], coefficients[1]/wscale**2
    if inverse_C <= 0 or ESL < 0:
        raise ValueError("component C+ESL fit is nonphysical; retain the extracted multiport/poles")
    fitted = 1j/(w/inverse_C) - 1j*w*ESL
    scale = max(float(np.max(abs(z))), np.finfo(float).tiny)
    return {"C_F": float(1/inverse_C), "ESL_H": float(ESL),
            "max_absolute_residual_ohm": float(np.max(abs(z-fitted))),
            "relative_residual": float(np.max(abs(z-fitted))/scale),
            "sample_count": len(w)}


def eliminate_component(A_gg, A_gc, A_cc, q_g, q_c, *, A_cg=None):
    """Schur-reduce fields AND coherent source, using transpose reciprocity.

    q_eff=q_g-A_gc solve(A_cc,q_c). Dropping the second term is not justified
    by removing PEC plates: the omitted vacuum/dielectric can carry overlap.
    """
    gg, gc, cc = (np.asarray(a, complex) for a in (A_gg, A_gc, A_cc))
    cg = gc.T if A_cg is None else np.asarray(A_cg, complex)
    qg, qc = np.asarray(q_g, complex), np.asarray(q_c, complex)
    solved = np.linalg.solve(cc, np.column_stack((cg, qc)))
    return gg-gc@solved[:, :-1], qg-gc@solved[:, -1]


def omitted_overlap_bound(B0_squared_integral_T2_m3, electric_energy_J,
                          permittivity_F_m=8.8541878128e-12):
    """Cauchy bound on |int_omitted E.B0 dV| for vacuum/uniform dielectric.

    Requires the actual component-drive electric energy and the actual local
    integral of B0 squared. This is not a bound from volume or PEC occupancy.
    """
    eps = _positive(permittivity_F_m, "permittivity")
    if min(B0_squared_integral_T2_m3, electric_energy_J) < 0 or not np.isfinite(
            B0_squared_integral_T2_m3+electric_energy_J):
        raise ValueError("B0 norm and component electric energy must be finite and nonnegative")
    return float(np.sqrt(4*electric_energy_J/eps * B0_squared_integral_T2_m3))
