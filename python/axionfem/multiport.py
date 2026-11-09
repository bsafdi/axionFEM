"""Reciprocal FEM multiports in axionFEM's readout-circuit convention.

Time dependence is exp(-i*omega*t). Each port is a branch with the constitutive
law V_p - V_m = sum(Z_jk I_k) - emf_j. Port sources are coherent; retain their
complex phases. This module closes an already extracted Z matrix. It does not
generate a mesh, fit frequency data, assign a modal Q, or certify passivity
between samples. See docs/physics/multiport.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .circuit import Circuit, HBAR, Squid, kappa_a_SI, n_bose


def _matrix(value, omega: float, n: int, label: str) -> np.ndarray:
    out = np.asarray(value(omega) if callable(value) else value, dtype=complex)
    if out.shape != (n, n) or not np.all(np.isfinite(out)):
        raise ValueError(f"{label}: expected a finite {n} by {n} matrix")
    return out


def _psd_matrix(value: np.ndarray, label: str) -> np.ndarray:
    scale = max(float(np.linalg.norm(value, 2)), np.finfo(float).tiny)
    # A fit/FEM error tolerance is not safe here: a tiny negative resistance
    # can give arbitrarily negative noise near an otherwise lossless pole.
    # Allow only eigensolver roundoff relative to the LOSS matrix itself.
    tol = 8 * np.finfo(float).eps * scale
    if np.max(np.abs(value - value.conj().T)) > tol:
        raise ValueError(f"{label}: loss matrix must be Hermitian")
    herm = (value + value.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(herm)
    if eigenvalues[0] < -tol:
        raise ValueError(f"{label}: negative dissipated power; loss matrix is not positive semidefinite")
    if eigenvalues[0] < 0:
        # Roundoff only, not a passivity repair for extracted data.
        herm = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.conj().T
    return herm


@dataclass
class LossBath:
    """Contribution to (Z+Z^dagger)/2 in ohms at one temperature.

The sum over baths must equal the multiport's full dissipative matrix. In
particular, individual diagonal resistances do not describe correlated loss.
"""
    name: str
    resistance: object
    T_K: float


@dataclass
class _Block:
    name: str
    ports: tuple[str, ...]
    impedance: object
    T_K: float
    baths: tuple[LossBath, ...] | None

    def at(self, omega: float) -> tuple[np.ndarray, list[tuple[str, np.ndarray, float]]]:
        z = _matrix(self.impedance, omega, len(self.ports), self.name)
        scale = max(float(np.linalg.norm(z, 2)), np.finfo(float).tiny)
        if np.max(np.abs(z - z.T)) > 1e-9 * scale:
            raise ValueError(f"{self.name}: Z is not reciprocal (Z != Z.T); check port orientations and FEM convergence")
        # Roundoff tolerance is relative to impedance; never replace the actual
        # matrix by a symmetrized or projected one for the circuit solve.
        loss = _psd_matrix((z + z.conj().T) / 2, self.name)
        if self.baths is None:
            return z, [(self.name, loss, self.T_K)]
        parts = [(f"{self.name}/{b.name}",
                  _psd_matrix(_matrix(b.resistance, omega, len(self.ports), b.name), b.name),
                  b.T_K) for b in self.baths]
        total = sum((r for _, r, _ in parts), np.zeros_like(z))
        difference = total - loss
        loss_scale = max(float(np.linalg.norm(loss, 2)), np.finfo(float).tiny)
        mismatch = np.linalg.norm(difference, 2) > 1e-9 * loss_scale
        # Test weak dissipative directions too: a small loss channel must not
        # vanish merely because another port has much larger resistance.
        if np.any(difference):
            values, vectors = np.linalg.eigh(loss)
            directional_scale = np.maximum(values, 8*np.finfo(float).eps*loss_scale)
            relative = (vectors.conj().T @ difference @ vectors)
            relative /= np.sqrt(directional_scale[:, None] * directional_scale[None, :])
            mismatch = mismatch or np.linalg.norm(relative, 2) > 1e-9
        if mismatch:
            raise ValueError(f"{self.name}: bath resistances must sum to (Z + Z.conj().T)/2")
        return z, parts


class MultiportCircuit(Circuit):
    """Existing MNA circuit with full reciprocal impedance blocks and FDT noise."""

    def __init__(self, ground: str = "gnd"):
        super().__init__(ground)
        self._multiports: list[_Block] = []

    def multiport(self, name: str, ports: Sequence[tuple[str, str, str]], Z,
                  *, T_K: float = 0.0, baths: Sequence[LossBath] | None = None) -> tuple[str, ...]:
        """Add (branch_name, positive_node, negative_node) for each FEM port.

        Z is a complex matrix or callable Z(omega), in the identical port order.
        For the FEM convention v_j=-integral E.j_j, orient p->m along the gap
        current profile. Circuit branch voltage is -v_j and branch current is
        minus the current entering the FEM device. Then emf=V_a=-i*omega*Psi
        gives the correct external load voltage; an open branch has Vp-Vm=-V_a.
        A uniform bath uses the full Hermitian part, including cross correlations.
        Multiple temperatures require explicit positive-semidefinite loss matrices
        whose sum equals that Hermitian part. No thermal participation is inferred
        from the DC field or a stated total Q.
        """
        ports = tuple(ports)
        if not ports or any(len(p) != 3 for p in ports):
            raise ValueError("multiport: give at least one (name, p, m) port")
        names = tuple(p[0] for p in ports)
        if len(set(names)) != len(names) or any(self._used(n) for n in names):
            raise ValueError(f"multiport {name!r}: port branch names must be unique in the circuit")
        if name in {b.name for b in self._multiports}:
            raise ValueError(f"multiport: duplicate block name {name!r}")
        if any(p == m for _, p, m in ports):
            raise ValueError(f"multiport {name!r}: each port needs two different nodes")
        if not np.isfinite(T_K) or T_K < 0:
            raise ValueError("multiport: T_K must be finite and nonnegative")
        baths = tuple(baths) if baths is not None else None
        if baths is not None and (len({b.name for b in baths}) != len(baths)
                                  or any(not np.isfinite(b.T_K) or b.T_K < 0 for b in baths)):
            raise ValueError("multiport: bath names must be unique and temperatures finite/nonnegative")
        for nm, p, m in ports:
            # No diagonal Johnson noise here: it belongs to the full matrix.
            self.impedance(nm, p, m, 0j, noise=False)
        self._multiports.append(_Block(name, names, Z, float(T_K), baths))
        return names

    def draw(self, annotate: bool = True, *, omega: float | None = None, compact: bool | None = None,
             title: str | None = None):
        """Block diagram of the closure (``schemdraw``; install ``axionfem[draw]``): each multiport
        is a box (``Z(omega)`` N x N, its bath temperature), one row per port with the port's
        external load drawn as a loop from its two terminals; other elements as in
        :meth:`Circuit.draw`.  ``compact`` folds consecutive identical rows into one (default:
        when there are more than ``axionfem.draw.COMPACT_PORTS`` ports).  Returns a
        ``schemdraw.Drawing``."""
        from .draw import draw_network
        return draw_network(self, annotate=annotate, omega=omega, compact=compact, title=title)

    def _assemble(self, w):
        matrix, nodes, names = super()._assemble(w)
        for block in self._multiports:
            z, _ = block.at(w)
            indices = [len(nodes) + names.index(n) for n in block.ports]
            matrix[np.ix_(indices, indices)] -= z
        return matrix, nodes, names

    def _groups(self, omega, temperature_free=False):
        groups = []
        for name, resistance, temperature in self.noisy_branches(omega):
            factor = 1 if temperature_free else HBAR * omega * (2 * n_bose(omega, temperature) + 1)
            groups.append((name, (name,), np.array([[resistance * factor]], complex)))
        for block in self._multiports:
            _, parts = block.at(omega)
            for label, resistance, temperature in parts:
                factor = 1 if temperature_free else HBAR * omega * (2 * n_bose(omega, temperature) + 1)
                groups.append((label, block.ports, resistance * factor))
        return groups

    def _quadratic_noise(self, omega, readout, temperature_free=False):
        groups = self._groups(omega, temperature_free)
        names = list(dict.fromkeys(n for _, ns, _ in groups for n in ns))
        if not names:
            return {"total": 0.0, "by_branch": {}, "transfer2": {}}
        solutions = self.solve_many(omega, [{n: 1} for n in names], replace=True)
        h = dict(zip(names, (readout(s) for s in solutions)))
        parts = {}
        for label, ns, covariance in groups:
            hs = np.array([h[n] for n in ns])
            # Row transfer h: S_out = h S_emf h^dagger (not h^dagger S h).
            values, vectors = np.linalg.eigh(covariance)
            # The covariance was validated above. Factor form avoids a tiny
            # negative dot-product cancellation at an exactly dark loss mode.
            contribution = float(np.sum(np.maximum(values, 0) * abs(hs @ vectors) ** 2))
            if label in parts:
                raise ValueError(f"multiport noise: duplicate bath/branch label {label!r}")
            parts[label] = contribution
        return {"total": sum(parts.values()), "by_branch": parts,
                "transfer2": {n: abs(v) ** 2 for n, v in h.items()}}

    def thermal_noise(self, omega, readout):
        """Foster-normalized PSD including correlated multiport noise.

        ``by_branch`` keeps the existing result schema; a matrix loss contributes
        one entry per physical bath, since assigning its cross terms to individual
        ports would not be unique.
        """
        result = self._quadratic_noise(omega, readout)
        result["loss_diagnostics"] = self.loss_diagnostics(omega)
        return result

    def loss_diagnostics(self, omega):
        """Record raw dissipative spectra and any covariance roundoff projection.

        No circuit impedance is changed. Negative eigenvalues beyond eight
        floating-point epsilons of the loss norm are rejected, not repaired.
        """
        result = {}
        for block in self._multiports:
            z, _ = block.at(omega)
            raw = (z + z.conj().T) / 2
            matrices = [(block.name, raw)]
            if block.baths is not None:
                matrices += [(f"{block.name}/{b.name}", _matrix(b.resistance, omega, len(block.ports), b.name))
                             for b in block.baths]
            for name, matrix in matrices:
                projected = _psd_matrix(matrix, name)
                result[name] = {
                    "raw_min_eigenvalue_ohm": float(np.linalg.eigvalsh(matrix)[0]),
                    "covariance_roundoff_projection_norm_ohm": float(np.linalg.norm(projected-matrix, 2))}
        return result

    def nyquist_check(self, omega, branch):
        zeff = self.driving_point_impedance(omega, branch)
        if zeff.real <= 0:
            raise ValueError("multiport.nyquist_check: needs a dissipative network")
        readout = lambda s: s.I(branch)
        h = self.transfer(omega, branch, readout)
        resistance = self._quadratic_noise(omega, readout, True)["total"] / abs(h) ** 2
        return abs(resistance - zeff.real) / zeff.real


@dataclass
class MultiportReadout:
    """Coherent axion source vector closed with Circuit and the existing Squid.

    ``emf_per_kappa`` maps branch names to V_a/kappa [V H], or is a callable of
    omega returning that mapping. FEM conventions give V_a/kappa=-i omega Psi.
    At nonzero signal the PSD can be referred to any nonzero reference Psi
    for the existing kernel_row_noise API. Signal zeros are rejected by that
    finite-noise referral; inverse_reference_noise returns zero information.
    output_snr_rate integrates the output PSD directly, including coherent
    zeros, without referring the noise through a vanishing source. No
    single-mode reduction is assumed here.
    """
    circuit: MultiportCircuit
    squid: Squid
    emf_per_kappa: Mapping[str, complex] | Callable[[float], Mapping[str, complex]]

    def draw(self, annotate: bool = True, *, omega: float | None = None, compact: bool | None = None,
             title: str | None = None):
        """:meth:`MultiportCircuit.draw` plus the SQUID on its input coil and the ports that carry
        an axion EMF (a callable ``emf_per_kappa`` is evaluated at ``omega`` to find them)."""
        from .draw import draw_multiport_readout
        return draw_multiport_readout(self, annotate=annotate, omega=omega, compact=compact, title=title)

    def flux(self, solution):
        return self.squid.M_H * solution.I(self.squid.input_coil)

    def flux_per_kappa(self, omega):
        pattern = self.emf_per_kappa(omega) if callable(self.emf_per_kappa) else self.emf_per_kappa
        if any(not np.isfinite(v) for v in pattern.values()):
            raise ValueError("multiport readout: source EMFs must be finite")
        return self.flux(self.circuit.solve(omega, pattern, replace=True))

    def signal(self, omega, g_GeV=1e-19, rho_GeV_cm3=0.3):
        return kappa_a_SI(g_GeV, rho_GeV_cm3) * self.flux_per_kappa(omega)

    def noise_psd(self, omega):
        th = self.circuit.thermal_noise(omega, self.flux)
        h_in = self.circuit.transfer(omega, self.squid.input_coil, self.flux)
        backaction = abs(h_in * omega * self.squid.M_H) ** 2 * self.squid.S_J
        imprecision = self.squid.S_Phi
        if min(backaction, imprecision) < 0 or not np.isfinite(backaction + imprecision):
            raise ValueError("multiport readout: amplifier PSDs must be finite and nonnegative")
        # Imprecision-backaction cross term 2 Re[conj(i omega M h_in) S_PhiJ]; 0 unless the Squid
        # carries a correlation (docs/physics/squid_chain.md Sec. 4).
        correlation = self.squid.correlation_term(omega, h_in)
        return {"S_Phi_thermal_Wb2_s": th["total"],
                "S_Phi_backaction_Wb2_s": backaction,
                "S_Phi_imprecision_Wb2_s": imprecision,
                "S_Phi_correlation_Wb2_s": correlation,
                "S_Phi_total_Wb2_s": th["total"] + backaction + imprecision + correlation,
                "thermal_by_branch_Wb2_s": th["by_branch"]}

    def reference_noise(self, omega, Psi_ref_SI):
        """N_V [V^2 s] to pair with Psi_ref_SI in kernel_row_noise.

        This is an exact referral, including complex interference and frequency
        variation, only where the signal is nonzero. The C++ NoiseCurve requires
        finite noise: a signal zero raises rather than silently passing infinity.
        Use inverse_reference_noise for an exact zero-information diagnostic;
        scanning through zeros requires a separate zero-aware kernel adapter.
        """
        if not np.isfinite(Psi_ref_SI) or abs(Psi_ref_SI) == 0:
            raise ValueError("multiport readout: give a finite nonzero reference Psi")
        h2 = abs(self.flux_per_kappa(omega)) ** 2
        if h2 == 0:
            raise ValueError("multiport readout: signal zero cannot be represented by a finite NoiseCurve; use a zero-aware kernel")
        noise = self.noise_psd(omega)["S_Phi_total_Wb2_s"] * abs(omega * Psi_ref_SI) ** 2 / h2
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("multiport readout: non-finite or zero referred noise cannot enter NoiseCurve")
        return noise

    def inverse_reference_noise(self, omega, Psi_ref_SI):
        """1/N_V [1/(V^2 s)], with zero at a coherent signal null.

        The current C++ noise-table API does not accept inverse noise. Use
        output_snr_rate for direct, zero-aware output integration.
        """
        if not np.isfinite(Psi_ref_SI) or abs(Psi_ref_SI) == 0:
            raise ValueError("multiport readout: give a finite nonzero reference Psi")
        noise = self.noise_psd(omega)["S_Phi_total_Wb2_s"]
        if not np.isfinite(noise) or noise <= 0:
            raise ValueError("multiport readout: a positive finite total output noise is required")
        return abs(self.flux_per_kappa(omega)) ** 2 / (noise * abs(omega * Psi_ref_SI) ** 2)

    def output_snr_rate(self, omega, weights_domega, *, kappa_a, axion_omega,
                        line_shape_per_rad_s):
        """Weak-signal Theta/time [s^-1], directly at the SQUID output.

        Supply positive angular frequencies, nonnegative quadrature weights
        in rad/s, and a nonnegative axion line shape L(omega), normalized to
        integral L d omega=1. The source convention gives
        S_signal=(pi/2)*kappa_a^2*(omega/axion_omega)^2*|Phi/kappa|^2*L.
        Then Theta/time=integral(S_signal/S_noise)^2 d omega/(2*pi).

        A coherent zero contributes exactly zero, including an interval with
        vanishing signal. This performs no interpolation and certifies no
        frequency coverage: the caller must resolve the line shape, circuit
        bandwidth and zeros, and freeze capacitor settings, SQUID PSDs and
        the calibrated loss realization across this entire integration.
        """
        w, weights, line = (np.asarray(v, float) for v in
                            (omega, weights_domega, line_shape_per_rad_s))
        if w.ndim != 1 or not len(w) or weights.shape != w.shape or line.shape != w.shape:
            raise ValueError("output SNR: frequencies, quadrature weights and line shape must be matching vectors")
        if not all(np.all(np.isfinite(v)) for v in (w, weights, line)) or np.any(w <= 0):
            raise ValueError("output SNR: frequencies must be positive and all quadrature inputs finite")
        if np.any(weights < 0) or np.any(line < 0):
            raise ValueError("output SNR: weights and line shape must be nonnegative")
        if not np.isfinite(kappa_a) or kappa_a < 0 or not np.isfinite(axion_omega) or axion_omega <= 0:
            raise ValueError("output SNR: require nonnegative finite kappa_a and positive axion_omega")
        h2 = np.array([abs(self.flux_per_kappa(x))**2 for x in w])
        noise = np.array([self.noise_psd(x)["S_Phi_total_Wb2_s"] for x in w])
        if np.any(noise <= 0) or not np.all(np.isfinite(noise)):
            raise ValueError("output SNR: a positive finite total output noise is required")
        signal = np.pi/2 * kappa_a**2 * (w/axion_omega)**2 * h2 * line
        snr = signal/noise
        if not np.all(np.isfinite(snr)):
            raise ValueError("output SNR: non-finite signal/noise; check quadrature and source normalization")
        return {"theta_per_s": float(np.dot(weights, snr**2)/(2*np.pi)),
                "line_normalization": float(np.dot(weights, line)),
                "zero_signal_samples": int(np.count_nonzero(h2 == 0)),
                "sample_count": len(w), "max_signal_to_noise_psd": float(np.max(snr))}
