"""Multi-branch torsional waveguide: the m = 0 mobility of an infinite uniform tube (beta).

The physics is summarised below and in docs/physics/torsion_highfreq.md (part A); validation:
docs/validation/torsion_highfreq.md.

An infinite tube (inner radius a, outer radius b = a + t; a = 0 is a solid rod) of an isotropic
material (shear modulus mu, density rho, c_T = sqrt(mu/rho)) carries the axisymmetric torsional
displacement u_phi(r, z).  Its torsional branches T(0, n), n = 0, 1, 2, ... are exact:

    u_phi = psi_n(r) e^{i k z},     omega^2 = c_T^2 (k^2 + kappa_n^2),

    psi_0 = r (kappa_0 = 0, the non-dispersive T(0,1) branch),
    psi_n = Y_2(kappa a) J_1(kappa r) - J_2(kappa a) Y_1(kappa r),   n >= 1,
    with J_2(kappa a) Y_2(kappa b) = J_2(kappa b) Y_2(kappa a)          (traction-free walls)

(a solid rod: psi_n = J_1(kappa r) with J_2(kappa b) = 0).  The psi_n are orthogonal with weight
r and normalised here to int_a^b psi_n^2 r dr = 1.  An azimuthal load, body force g(r, z) [N/m^3]
plus surface tractions tau_a(z), tau_b(z) [Pa] on the inner and outer wall, projects on branch n
as G_n(z) = int g psi_n r dr + a tau_a psi_n(a) + b tau_b psi_n(b), and the generalised
receptance and mobility of the pair (load scale, q = int g u_phi dV + int tau u_phi dA) are, with
a structural loss factor eta (mu -> mu (1 - i eta)) and time dependence e^{-i omega t}:

    q = sum_n  i pi/(k_n mu (1 - i eta))  int int G_n(z) G_n(z') e^{i k_n |z - z'|} dz dz',
    k_n^2 = omega^2/(c_T^2 (1 - i eta)) - kappa_n^2,   Im k_n > 0,      Y = -i omega q.

In the lossless limit only the propagating branches (f above the cutoff f_n = c_T kappa_n/2 pi)
contribute to Re Y:

    Re Y = sum_{n: f > f_n}  pi |G_n^(k_n)|^2/(rho v_g,n),   v_g,n = c_T^2 k_n/omega,

with G^ the axial Fourier transform (symmetrised over +-k).  v_g -> 0 at a cutoff, so Re Y has an
integrable 1/sqrt(f - f_n) singularity there; the damped formula regularises it.

The finite free-free tube of length L (traction-free ends) has the exact separable modes
psi_n(r) cos(m pi (z + L/2)/L); `finite_tube_mobility` sums them (the finite-length reference).

Units: SI.  Every quantity is per unit load scale: the load is (load scale) x (the profiles given).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = ["Tube", "Gaussian", "Rect", "Ring", "Sampled", "TubeLoad", "WaveguideMobility"]


# --------------------------------------------------------------------------- axial profiles
class _Axial:
    """An axial load profile p(z) [1]: the load is (radial part) x p(z)."""

    def transform(self, k):
        """P^(k) = int p(z) e^{-i k z} dz for real k."""
        raise NotImplementedError

    def kernel(self, k):
        """I(k) = int int p(z) p(z') e^{i k |z - z'|} dz dz' for complex k, Im k >= 0, or None."""
        return None

    def cosine(self, beta, z_left):
        """int p(z) cos(beta (z - z_left)) dz (the finite tube's axial modes), or None."""
        return None

    def samples(self, z):
        raise NotImplementedError

    def extent(self):
        """(z_min, z_max) of the support, and the shortest length scale of the profile."""
        raise NotImplementedError


@dataclass
class Gaussian(_Axial):
    """p(z) = exp(-(z - z0)^2/(2 sigma^2)) (peak 1)."""

    sigma: float
    z0: float = 0.0

    def __post_init__(self):
        if not self.sigma > 0:
            raise ValueError(f"Gaussian: sigma must be positive (got {self.sigma})")

    def transform(self, k):
        k = np.asarray(k, dtype=float)
        return math.sqrt(2 * math.pi) * self.sigma * np.exp(-0.5 * (k * self.sigma) ** 2 - 1j * k * self.z0)

    def kernel(self, k):
        from scipy.special import wofz
        # int int ... = (1/2 pi) int |P^|^2 2 i k/(k^2 - beta^2) d beta = 2 pi sigma^2 w(sigma k)
        return 2 * math.pi * self.sigma ** 2 * wofz(self.sigma * np.asarray(k, dtype=complex))

    def cosine(self, beta, z_left):
        # the Gaussian is assumed to lie inside the tube (checked by the caller through extent())
        beta = np.asarray(beta, dtype=float)
        return (math.sqrt(2 * math.pi) * self.sigma * np.exp(-0.5 * (beta * self.sigma) ** 2)
                * np.cos(beta * (self.z0 - z_left)))

    def samples(self, z):
        return np.exp(-0.5 * ((np.asarray(z) - self.z0) / self.sigma) ** 2)

    def extent(self):
        return (self.z0 - 8 * self.sigma, self.z0 + 8 * self.sigma), self.sigma


@dataclass
class Rect(_Axial):
    """p(z) = 1 for |z - z0| <= width/2, else 0."""

    width: float
    z0: float = 0.0

    def __post_init__(self):
        if not self.width > 0:
            raise ValueError(f"Rect: width must be positive (got {self.width})")

    def transform(self, k):
        k = np.asarray(k, dtype=float)
        return self.width * np.sinc(k * self.width / (2 * math.pi)) * np.exp(-1j * k * self.z0)

    def kernel(self, k):
        k = np.asarray(k, dtype=complex)
        W = self.width
        al = 1j * k
        small = np.abs(al * W) < 1e-4
        al_safe = np.where(small, 1.0, al)
        # 2 int_0^W (W - s) e^{al s} ds = 2 (-W/al + (e^{al W} - 1)/al^2)
        exact = 2 * (-W / al_safe + np.expm1(al_safe * W) / al_safe ** 2)
        series = W ** 2 + al * W ** 3 / 3 + al ** 2 * W ** 4 / 12
        return np.where(small, series, exact)

    def cosine(self, beta, z_left):
        beta = np.asarray(beta, dtype=float)
        lo = self.z0 - self.width / 2 - z_left
        hi = self.z0 + self.width / 2 - z_left
        safe = np.where(beta == 0, 1.0, beta)
        return np.where(beta == 0, hi - lo, (np.sin(safe * hi) - np.sin(safe * lo)) / safe)

    def samples(self, z):
        return (np.abs(np.asarray(z) - self.z0) <= self.width / 2).astype(float)

    def extent(self):
        return (self.z0 - self.width / 2, self.z0 + self.width / 2), self.width


@dataclass
class Ring(_Axial):
    """p(z) = delta(z - z0): a line (ring) load at one axial position, [1/m]."""

    z0: float = 0.0

    def transform(self, k):
        k = np.asarray(k, dtype=float)
        return np.exp(-1j * k * self.z0)

    def kernel(self, k):
        return np.ones_like(np.asarray(k, dtype=complex))

    def cosine(self, beta, z_left):
        return np.cos(np.asarray(beta, dtype=float) * (self.z0 - z_left))

    def samples(self, z):
        raise ValueError("Ring: a delta profile cannot be sampled; combine it with other terms only through "
                         "a narrow Gaussian, or use it alone")

    def extent(self):
        return (self.z0, self.z0), 0.0


@dataclass
class Sampled(_Axial):
    """p(z) given on a uniform grid z (piecewise linear, zero outside)."""

    z: Sequence[float]
    values: Sequence[float]

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.z.ndim != 1 or self.z.size < 3 or self.z.shape != self.values.shape:
            raise ValueError("Sampled: z and values must be 1D arrays of the same length (>= 3)")
        h = np.diff(self.z)
        if not (np.all(h > 0) and np.allclose(h, h[0], rtol=1e-9, atol=0)):
            raise ValueError("Sampled: z must be uniformly spaced and increasing")

    def transform(self, k):
        k = np.atleast_1d(np.asarray(k, dtype=float))
        w = _trap_weights(self.z)
        return np.array([np.sum(w * self.values * np.exp(-1j * kk * self.z)) for kk in k])

    def cosine(self, beta, z_left):
        beta = np.atleast_1d(np.asarray(beta, dtype=float))
        w = _trap_weights(self.z) * self.values
        return np.array([np.sum(w * np.cos(b * (self.z - z_left))) for b in beta])

    def samples(self, z):
        return np.interp(np.asarray(z), self.z, self.values, left=0.0, right=0.0)

    def extent(self):
        """Support, and the scale 2/beta_c, beta_c the wavenumber beyond which the profile's spectrum holds
        less than 1e-10 of its energy (at least the grid step): the grid step itself would make the finite
        tube's (branches x modes) arrays grow as 1/h^2."""
        h = float(self.z[1] - self.z[0])
        nfft = 4 * self.z.size
        P = np.abs(np.fft.rfft(self.values, n=nfft)) ** 2
        beta = 2 * math.pi * np.fft.rfftfreq(nfft, d=h)
        tot = P.sum()
        if tot > 0:
            cum = np.cumsum(P) / tot
            bc = beta[min(int(np.searchsorted(cum, 1 - 1e-10)), beta.size - 1)]
            scale = max(h, 2.0 / bc) if bc > 0 else h
        else:
            scale = h
        return (float(self.z[0]), float(self.z[-1])), scale


def _end_values(axial, z_ends):
    """|p| at the tube ends relative to the profile's peak (Ring: inf if outside, else 0)."""
    if isinstance(axial, Ring):
        return 0.0 if z_ends[0] <= axial.z0 <= z_ends[1] else math.inf
    if isinstance(axial, Sampled):
        peak = float(np.max(np.abs(axial.values))) or 1.0
        out = np.abs(axial.values[(axial.z <= z_ends[0]) | (axial.z >= z_ends[1])])
        return float(out.max()) / peak if out.size else 0.0
    if isinstance(axial, Rect):
        lo, hi = axial.z0 - axial.width / 2, axial.z0 + axial.width / 2
        return 0.0 if (lo >= z_ends[0] - 1e-12 and hi <= z_ends[1] + 1e-12) else 1.0
    if not (z_ends[0] <= axial.z0 <= z_ends[1]):
        return math.inf
    return float(np.max(np.abs(axial.samples(np.array(z_ends)))))


_GL16 = np.polynomial.legendre.leggauss(16)


def _branch_tail(kap, Y_br):
    """Sum over the branches beyond the last one kept, from Y_n ~ c_n/kappa_n^2 with kappa_n ~ kappa_N + (n - N) D.

    c_n is the mean of the last two branches' Y kappa^2: for tractions on both walls R_n alternates with the parity
    of n (psi_n(b)/psi_n(a) ~ (-1)^n sqrt(a/b)), so one branch alone doubles or drops the tail.  D is
    the mean root spacing of the last four intervals; sum_{m>N} 1/kappa_m^2 ~ 1/(D (kappa_N + D/2)).
    """
    c = 0.5 * (Y_br[-1] * kap[-1] ** 2 + Y_br[-2] * kap[-2] ** 2)
    D = (kap[-1] - kap[-5]) / 4.0
    return c / (D * (kap[-1] + 0.5 * D))


def _trap_weights(z):
    h = z[1] - z[0]
    w = np.full(z.size, h)
    w[0] = w[-1] = h / 2
    return w


# --------------------------------------------------------------------------- loads
@dataclass
class TubeLoad:
    """One separable term of an azimuthal load: (radial part) x axial(z).

    volume: body-force density w(r) [N/m^3 at p = 1]: a number (uniform through the wall), a
            callable w(r), or None.
    inner, outer: azimuthal surface traction [Pa at p = 1] on r = a and r = b (positive along
            +phi), or 0.
    axial: Gaussian, Rect, Ring or Sampled.
    """

    axial: _Axial
    volume: float | Callable | None = None
    inner: float = 0.0
    outer: float = 0.0


@dataclass
class WaveguideMobility:
    f_Hz: np.ndarray
    Y: np.ndarray                    # complex mobility, Y = -i omega q (e^{-i omega t})
    Y_branch: np.ndarray             # (n_branches, n_f) complex, per branch
    kappa: np.ndarray                # radial wavenumbers of the branches used [1/m]
    f_cut_Hz: np.ndarray             # their cutoffs
    projection: np.ndarray           # (n_terms, n_branches): R_cn = int w psi r dr + a s_a psi(a) + b s_b psi(b)
    eta: float
    notes: list = field(default_factory=list)
    truncation: np.ndarray | None = None   # |Re Y of the last branch|/|Re Y| per frequency
    tail: np.ndarray | None = None         # analytic estimate of the branches beyond the last (included in Y)

    @property
    def Re(self):
        return self.Y.real


# --------------------------------------------------------------------------- the tube
class Tube:
    """An infinite uniform tube (a = inner radius, t = wall; a = 0 for a solid rod of radius t).

    Material: give mu and rho, or E, nu and rho.
    """

    def __init__(self, a: float, t: float, rho: float, mu: float | None = None, E: float | None = None,
                 nu: float | None = None):
        if not (a >= 0 and t > 0 and rho > 0):
            raise ValueError(f"Tube: need a >= 0, t > 0 and rho > 0 (got a = {a}, t = {t}, rho = {rho})")
        if (mu is None) == (E is None):
            raise ValueError("Tube: give either mu (shear modulus) or E (with nu), not both or neither")
        if E is not None:
            if nu is None:
                raise ValueError("Tube: E needs nu")
            mu = E / (2 * (1 + nu))
        if not mu > 0:
            raise ValueError(f"Tube: the shear modulus must be positive (got {mu})")
        self.a, self.t, self.b = float(a), float(t), float(a + t)
        self.mu, self.rho = float(mu), float(rho)
        self.c_T = math.sqrt(self.mu / self.rho)
        self._kappa: list[float] = [0.0]
        self._scan_to = 0.0

    # ---- radial eigenproblem
    def _disp(self, x):
        from scipy.special import jv, yv
        if self.a == 0:
            return jv(2, x * self.b)
        return jv(2, x * self.a) * yv(2, x * self.b) - jv(2, x * self.b) * yv(2, x * self.a)

    def _extend(self, kappa_max: float):
        """Find every root of the dispersion function up to kappa_max (bracketing scan + brentq)."""
        from scipy.optimize import brentq
        if kappa_max <= self._scan_to:
            return
        step = math.pi / max(self.t, self.b if self.a == 0 else self.t) / 64
        step = min(step, math.pi / self.b / 64)
        x0 = max(self._scan_to, 1e-9 / self.b)
        xs = np.arange(x0, kappa_max + step, step)
        d = self._disp(xs)
        for i in range(len(xs) - 1):
            if d[i] == 0.0:
                self._kappa.append(float(xs[i]))
            elif d[i] * d[i + 1] < 0:
                self._kappa.append(brentq(self._disp, xs[i], xs[i + 1], xtol=1e-14 * xs[i + 1], rtol=1e-15))
        self._scan_to = float(xs[-1])

    def kappa(self, n_branches: int | None = None, kappa_max: float | None = None) -> np.ndarray:
        """Radial wavenumbers kappa_n of the first n_branches branches (or all up to kappa_max)."""
        if n_branches is not None:
            guess = (n_branches + 2) * math.pi / self.t if self.a > 0 else (n_branches + 2) * math.pi / self.b
            while len(self._kappa) < n_branches:
                self._extend(guess)
                guess *= 1.5
            return np.array(self._kappa[:n_branches])
        self._extend(kappa_max)
        return np.array([k for k in self._kappa if k <= kappa_max])

    def cutoffs_Hz(self, n_branches: int) -> np.ndarray:
        """Cutoff frequencies f_n = c_T kappa_n/(2 pi) of T(0, n + 1), n = 0 .. n_branches - 1."""
        return self.c_T * self.kappa(n_branches) / (2 * math.pi)

    def _raw(self, kap, r):
        from scipy.special import jv, yv
        r = np.asarray(r, dtype=float)
        if kap == 0.0:
            return r.copy()
        if self.a == 0:
            return jv(1, kap * r)
        return yv(2, kap * self.a) * jv(1, kap * r) - jv(2, kap * self.a) * yv(1, kap * r)

    def _norm2(self, kap):
        """int_a^b psi^2 r dr in closed form (Lommel; with psi' = psi/r at both walls it is r^2 psi^2/2)."""
        if kap == 0.0:
            return (self.b ** 4 - self.a ** 4) / 4
        pa = self._raw(kap, self.a) if self.a > 0 else 0.0
        pb = self._raw(kap, self.b)
        return (self.b ** 2 * pb ** 2 - self.a ** 2 * pa ** 2) / 2

    def mode_shape(self, n: int, r) -> np.ndarray:
        """psi_n(r), normalised to int_a^b psi_n^2 r dr = 1 [1/m]."""
        kap = self.kappa(n + 1)[n]
        return self._raw(kap, r) / math.sqrt(self._norm2(kap))

    def _projection(self, load: TubeLoad, kappas) -> np.ndarray:
        if self.a == 0 and load.inner != 0:
            raise ValueError("Tube: a solid rod (a = 0) has no inner wall; set inner = 0")
        out = np.zeros(len(kappas))
        x16, w16 = _GL16
        for i, kap in enumerate(kappas):
            nrm = math.sqrt(self._norm2(kap))
            v = 0.0
            if load.volume is not None:
                # composite 16-point Gauss-Legendre: one panel per half-wave of psi_n, at least 4 (a single
                # leggauss rule of 4 kappa t/pi points costs O(nq^3) and minutes per call)
                npan = max(4, int(math.ceil(kap * self.t / math.pi)) + 1)
                edges = np.linspace(self.a, self.b, npan + 1)
                half = 0.5 * np.diff(edges)
                r = (0.5 * (edges[:-1] + edges[1:])[:, None] + half[:, None] * x16[None, :]).ravel()
                wr = (half[:, None] * w16[None, :]).ravel()
                wv = load.volume(r) if callable(load.volume) else np.full_like(r, float(load.volume))
                v += np.sum(wr * wv * self._raw(kap, r) * r) / nrm
            if load.inner:
                v += self.a * load.inner * self._raw(kap, self.a) / nrm
            if load.outer:
                v += self.b * load.outer * self._raw(kap, self.b) / nrm
            out[i] = v
        return out

    # ---- mobility of the infinite tube
    def _branches_for(self, f_max, n_branches, n_evanescent, terms=(), notes=None, max_branches=20000):
        """Branches: the first n_branches, or the largest of
        - every propagating one plus n_evanescent;
        - every branch up to kappa = 20 k_T(f_max): the analytic tail is an asymptotic (kappa >> k_T) form, and on
          thick walls at high frequency (26 propagating branches on a 4 cm wall at 1 MHz) n_prop + 100 is too
          few;
        - for terms with a surface component, enough that kappa_N s >= 5 (Gaussian, sampled) or 50 (top hat, whose
          edges act as ring loads: a 1/(kappa W) correction per branch), s the axial scale.  A
          volume load's projections decay fast, so it needs no extra branches."""
        k_top = 2 * math.pi * f_max / self.c_T
        if n_branches is None:
            prop = self.kappa(kappa_max=k_top)
            need = [len(prop) + n_evanescent, len(self.kappa(kappa_max=20.0 * k_top)) + 2]
            for t_ in terms:
                if isinstance(t_.axial, Ring) or not (t_.inner or t_.outer):
                    continue
                sc = t_.axial.extent()[1]
                if sc > 0:
                    mult = 50.0 if isinstance(t_.axial, Rect) else 5.0
                    need.append(len(self.kappa(kappa_max=mult / sc)) + 2)
            n_branches = max(need)
            if n_branches > max_branches:
                if notes is not None:
                    notes.append(f"the load's axial scale needs {n_branches} branches; capped at {max_branches}: "
                                 f"the tail may be low")
                n_branches = max_branches
        return self.kappa(n_branches)

    def mobility(self, f_Hz, loads: TubeLoad | Sequence[TubeLoad], eta: float, n_branches: int | None = None,
                 n_evanescent: int = 100, z_step: float | None = None, tail: bool = True) -> WaveguideMobility:
        """Complex mobility Y(f) = -i omega q of the infinite tube for the load (damped, exact per branch).

        Branches: every propagating one at max(f_Hz) plus `n_evanescent`, or the first `n_branches`.
        Far evanescent branches add Re Y = O(eta) each, ~ R_n^2/kappa_n^2: for a surface traction R_n does
        not decay and the sum converges only as 1/N, so `tail` adds the analytic remainder
        Y_N (kappa_N/(kappa_N - kappa_{N-1}) - 1/2) (sum_{m>N} 1/m^2 ~ 1/(N + 1/2)).  A ring (delta) axial
        profile on a surface has a logarithmically divergent Re Y (the line load's strain energy diverges);
        no tail is added there and a note says so.
        A single term with a Gaussian, Rect or Ring profile uses the closed-form axial kernel; several
        terms (or a Sampled profile) use the trapezoidal double integral on a uniform grid (step
        `z_step`, default a 64th of the shortest wavelength or profile scale) with one Richardson step.
        """
        f = np.atleast_1d(np.asarray(f_Hz, dtype=float))
        if not np.all(f > 0):
            raise ValueError("mobility: frequencies must be positive")
        if eta < 0:
            raise ValueError(f"mobility: eta must be >= 0 (got {eta})")
        terms = [loads] if isinstance(loads, TubeLoad) else list(loads)
        if not terms:
            raise ValueError("mobility: no load terms")
        notes = []
        kap = self._branches_for(f.max(), n_branches, n_evanescent, terms, notes)
        R = np.array([self._projection(t_, kap) for t_ in terms])        # (n_terms, n_br)
        w = 2 * math.pi * f
        mup = self.mu * (1 - 1j * eta)
        k2 = (w[None, :] ** 2) * self.rho / mup - kap[:, None] ** 2           # (n_br, n_f)
        k = np.sqrt(k2.astype(complex))
        k = np.where(k.imag < 0, -k, k)
        # eta = 0: a propagating branch has k real > 0 (the outgoing wave); keep Re k >= 0 there
        k = np.where((k.imag == 0) & (k.real < 0), -k, k)
        if len(terms) == 1 and terms[0].axial.kernel(np.array([1.0 + 0j])) is not None:
            I = terms[0].axial.kernel(k) * (R[0][:, None] ** 2)
        else:
            I, est = _sampled_kernel(terms, R, k, w, self.c_T, z_step)
            notes.append(f"sampled axial kernel: Richardson correction up to {est:.2e} of |q| per branch")
        q_br = 1j * math.pi / (k * mup) * I
        Y_br = -1j * w[None, :] * q_br
        Y = Y_br.sum(axis=0)
        trunc = np.abs(Y_br[-1].real) / np.maximum(np.abs(Y.real), 1e-300)
        Ytail = np.zeros_like(Y)
        ring_surface = any(isinstance(t_.axial, Ring) and (t_.inner or t_.outer) for t_ in terms)
        last_evanescent = len(kap) > 5 and kap[-2] > 2 * math.pi * f.max() / self.c_T
        if ring_surface:
            notes.append("ring (delta) load on a surface: Re Y grows logarithmically with the branch count "
                         "(the line load's strain energy diverges); no tail added, use a finite axial width")
        elif tail and last_evanescent:
            Ytail = _branch_tail(kap, Y_br)
            Y = Y + Ytail
        f_cut = self.c_T * kap / (2 * math.pi)
        if eta == 0:
            near = np.any(np.abs(f[None, :] / np.maximum(f_cut[1:, None], 1e-300) - 1) < 1e-9)
            if near:
                notes.append("a frequency sits on a cutoff: the lossless Re Y diverges there")
        return WaveguideMobility(f_Hz=f, Y=Y, Y_branch=Y_br, kappa=kap, f_cut_Hz=f_cut, projection=R,
                                 eta=float(eta), notes=notes, truncation=trunc, tail=Ytail)

    def mobility_lossless(self, f_Hz, loads: TubeLoad | Sequence[TubeLoad]) -> tuple[np.ndarray, np.ndarray]:
        """Re Y = sum over propagating branches of pi |G_n^(k_n)|^2/(rho v_g,n) (eta -> 0).

        Returns (Re Y, Re Y per branch (n_br, n_f)); diverges as 1/sqrt(f - f_n) at each cutoff.
        """
        f = np.atleast_1d(np.asarray(f_Hz, dtype=float))
        terms = [loads] if isinstance(loads, TubeLoad) else list(loads)
        kap = self.kappa(kappa_max=2 * math.pi * f.max() / self.c_T)
        R = np.array([self._projection(t_, kap) for t_ in terms])
        w = 2 * math.pi * f
        out = np.zeros((len(kap), f.size))
        for n, kn in enumerate(kap):
            kz2 = (w / self.c_T) ** 2 - kn ** 2
            ok = kz2 > 0
            if not np.any(ok):
                continue
            kz = np.sqrt(kz2[ok])
            Gp = sum(R[c, n] * terms[c].axial.transform(kz) for c in range(len(terms)))
            Gm = sum(R[c, n] * terms[c].axial.transform(-kz) for c in range(len(terms)))
            G2 = 0.5 * (np.abs(Gp) ** 2 + np.abs(Gm) ** 2)
            vg = self.c_T ** 2 * kz / w[ok]
            out[n, ok] = math.pi * G2 / (self.rho * vg)
        return out.sum(axis=0), out

    # ---- the finite free-free tube (exact modal sum)
    def finite_tube_mobility(self, f_Hz, loads: TubeLoad | Sequence[TubeLoad], eta: float, L: float,
                             beta_max: float | None = None, n_branches: int | None = None,
                             branch_tail: bool = True, end_tolerance: float = 1e-6,
                             max_modes: int = 50_000_000) -> np.ndarray:
        """Exact Y(f) of the free-free tube z in [-L/2, L/2] (traction-free ends): the double modal sum.

        Modes psi_n(r) cos(m pi (z + L/2)/L), lambda = c_T^2 (kappa_n^2 + (m pi/L)^2), unit modal mass;
        q = sum c^2/(lambda (1 - i eta) - omega^2).  Axial modes up to beta_max (default 4 k_T(f_max)
        plus 40/(shortest profile scale)), branches with kappa_n <= beta_max; with `branch_tail` the branches
        beyond (evanescent, local) are added as in the infinite tube (`mobility`, with its tail).
        """
        f = np.atleast_1d(np.asarray(f_Hz, dtype=float))
        terms = [loads] if isinstance(loads, TubeLoad) else list(loads)
        zl = -L / 2
        for t_ in terms:
            # the axial transforms are those of the profile on the whole line: the profile must be negligible
            # at the ends (relative value <= end_tolerance), not entirely inside the tube
            ev = _end_values(t_.axial, (-L / 2, L / 2))
            if ev > end_tolerance:
                raise ValueError(f"finite_tube_mobility: a load profile has {ev:.3g} of its peak at a tube end "
                                 f"(tube [-{L / 2}, {L / 2}], tolerance {end_tolerance:g}); move it inside or "
                                 f"raise end_tolerance")
        kT = 2 * math.pi * f.max() / self.c_T
        if beta_max is None:
            scales = [t_.axial.extent()[1] for t_ in terms]
            sc = min([s for s in scales if s > 0], default=L)
            beta_max = 4 * kT + 40 / sc
        kap = self.kappa(n_branches) if n_branches else self.kappa(kappa_max=beta_max)
        R = np.array([self._projection(t_, kap) for t_ in terms])        # (n_terms, n_br)
        m = np.arange(0, int(beta_max * L / math.pi) + 1)
        if len(kap) * m.size > max_modes:
            raise ValueError(f"finite_tube_mobility: {len(kap)} branches x {m.size} axial modes exceed max_modes = "
                             f"{max_modes} (beta_max = {beta_max:.3g}/m from the load's axial scale); give a "
                             f"smaller beta_max or raise max_modes")
        beta = m * math.pi / L
        ell = np.where(m == 0, L, L / 2)
        C = np.array([t_.axial.cosine(beta, zl) for t_ in terms])        # (n_terms, n_m)
        # c_nm = 2 pi sum_c R_cn C_cm / sqrt(2 pi rho ell_m)
        c = 2 * math.pi * np.einsum("cn,cm->nm", R, C) / np.sqrt(2 * math.pi * self.rho * ell)[None, :]
        lam = self.c_T ** 2 * (kap[:, None] ** 2 + beta[None, :] ** 2)
        c2 = (c ** 2).ravel()
        lam = lam.ravel()
        w = 2 * math.pi * f
        q = np.array([np.sum(c2 / (lam * (1 - 1j * eta) - ww * ww)) for ww in w])
        Y = -1j * w * q
        if branch_tail and kap[-1] > kT:
            # branches beyond kappa_max are evanescent at every frequency and local (decay length 1/kappa_n,
            # far shorter than the distance to the ends), so they contribute as in the infinite tube: add
            # the infinite tube's branches len(kap) .. len(kap) + 100 and its analytic tail
            nb = len(kap)
            full = self.mobility(f, terms, eta, n_branches=nb + 100)
            Y = Y + full.Y_branch[nb:].sum(axis=0) + full.tail
        return Y


def _sampled_kernel(terms, R, k, w, c_T, z_step):
    """sum_cc' R_cn R_c'n int int p_c p_c' e^{i k |z - z'|}: trapezoid on a uniform grid, one Richardson step.

    The double integral is evaluated in O(N) per (branch, frequency) by the first-order recursion
    A_j = e^{i k h} A_{j-1} + x_j (scipy.signal.lfilter), with the kink of the kernel on grid nodes.
    """
    from scipy.signal import lfilter
    lo, hi, scale = np.inf, -np.inf, np.inf
    for t_ in terms:
        (z0, z1), s = t_.axial.extent()
        lo, hi = min(lo, z0), max(hi, z1)
        if s > 0:
            scale = min(scale, s)
    lam_min = 2 * math.pi * c_T / w.max()
    h = z_step or min(lam_min, scale) / 64
    n = int(math.ceil((hi - lo) / (2 * h))) * 2 + 1
    z = np.linspace(lo, hi, n)
    h = z[1] - z[0]
    P = np.array([t_.axial.samples(z) for t_ in terms])                  # (n_terms, n)

    def S(zz, PP, hh):
        wts = _trap_weights(zz)
        out = np.zeros(k.shape, dtype=complex)
        for nb in range(k.shape[0]):
            G = (R[:, nb][:, None] * PP).sum(axis=0) * wts
            if not np.any(G):
                continue
            for j in range(k.shape[1]):
                E = np.exp(1j * k[nb, j] * hh)
                A = lfilter([1.0], [1.0, -E], G.astype(complex))
                out[nb, j] = 2 * np.dot(G, A) - np.dot(G, G)
        return out

    fine = S(z, P, h)
    coarse = S(z[::2], P[:, ::2], 2 * h)
    rich = fine + (fine - coarse) / 3
    est = float(np.max(np.abs(rich - fine) / np.maximum(np.abs(rich), 1e-300)))
    return rich, est
