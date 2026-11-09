"""F13a closed forms and series (case definition, out of band): Hartmann channel, Shercliff and Hunt ducts.

Derived from the PDE of the inductionless MHD formulation, NOT copied from the papers.
Dimensional, constant mu, sigma; imposed B = B0 e_y; Hartmann walls at y = +-a (perpendicular to B), side walls at
x = +-L (parallel to B); G = -dp/dz > 0 drives u = w(x, y) e_z (duct) or u = U(y) e_x (in-plane channel).
    J = sigma (-grad phi + E + u x B),   mu Lap u + G e + J x B = 0,   div J = 0.
m = B0 sqrt(sigma/mu) (Hartmann number Ha = m a).

Hartmann (exact):  U = K (1 - cosh(m y)/cosh(m a)),  K = G_eff/(sigma B0^2),  G_eff = G - sigma B0 E_z (in-plane) or
                   G + sigma B0 E_x (duct mode with electrodes, E_x = -d phi/dx).
Shercliff (insulating walls): w = (A(x,y) + A(x,-y))/2, H = sqrt(mu sigma) (A(x,y) - A(x,-y))/2 with J = (dH/dy, -dH/dx),
                   Lap A + m dA/dy = -G/mu, A = 0 on the walls; A = sum_n a_n(y) cos(beta_n x), beta_n = (n+1/2) pi/L.
Hunt case I with perfectly conducting Hartmann walls (phi = 0 at y = +-a) and insulating side walls (J.n = 0):
                   w = sum_k [v_p + 2 Re(C' cosh(l x)/cosh(l L))] cos(alpha_k y),
                   phi = s sum_k 2 Re(C' (i l/alpha) sinh(l x)/cosh(l L)) cos(alpha_k y),
                   alpha_k = (k+1/2) pi/a, l = sqrt(alpha^2 + i m alpha), v_p = c_k/(alpha^2 + m^2),
                   C' = -(v_p/2)(1 - i m/alpha), s = sqrt(mu/sigma), c_k = (G/mu) 2 (-1)^k/(alpha_k a).
Every evaluation is overflow-free (exponentials of non-positive real part only).

Numpy only (the check module imports the field evaluators); mpmath is used by f13a_reference.py, not here.
"""
from __future__ import annotations

import math

import numpy as np


# ============================================================================================== Hartmann
def _one_minus_cosh_ratio(m, y, a):
    """1 - cosh(m y)/cosh(m a), stable for every m >= 0 and |y| <= a (2 sinh(A) sinh(B)/cosh(A+B))."""
    A = 0.5 * m * (a + y)
    B = 0.5 * m * (a - y)
    return (-np.expm1(-2 * A)) * (-np.expm1(-2 * B)) / (1 + np.exp(-2 * (A + B)))


def hartmann_F(m, a):
    """F = int_{-a}^{a} (1 - cosh(m y)/cosh(m a)) dy = 2a - 2 tanh(m a)/m (series for small m a)."""
    x = m * a
    if x < 1e-3:
        return 2 * a * (x * x / 3 - 2 * x ** 4 / 15 + 17 * x ** 6 / 315)
    return 2 * a - 2 * math.tanh(x) / m


def hartmann_profile_factor(m, y, a):
    """(1 - cosh(m y)/cosh(m a)) / m^2, stable down to m = 0 (limit (a^2 - y^2)/2)."""
    y = np.asarray(y, float)
    if m * a < 1e-3:
        return 0.5 * (a * a - y * y) * (1 - m * m * (5 * a * a - y * y) / 12)
    return _one_minus_cosh_ratio(m, y, a) / (m * m)


def hartmann_profile_factor_dy(m, y, a):
    """d/dy of hartmann_profile_factor = -sinh(m y)/(m cosh(m a))."""
    y = np.asarray(y, float)
    if m * a < 1e-3:
        return -y * (1 - m * m * (5 * a * a - y * y) / 12) + 0.5 * (a * a - y * y) * (m * m * y / 6)
    ay = np.abs(y)
    return -np.sign(y) * np.exp(m * (ay - a)) * (-np.expm1(-2 * m * ay)) / (1 + np.exp(-2 * m * a)) / m


class Hartmann:
    """Fully developed Hartmann flow between walls y = +-a, B = (0, B0, 0).

    mode = 'inplane': u = (U(y), 0, 0), J = (0, 0, sigma (E_z + B0 U)), force_x = -J_z B0, phi = const;
                      G_eff = G - sigma B0 E_z.
    mode = 'duct'   : u = (0, 0, W(y)), phi = -E_x x, J = (sigma (E_x - B0 W), 0, 0), force_z = J_x B0;
                      G_eff = G + sigma B0 E_x.
    circuit = 'short' (E = 0) or 'open' (E chosen so that the net current across the channel vanishes), or a float
    (the prescribed E_z / E_x)."""

    def __init__(self, mu, sigma, B0, G, a=1.0, mode="inplane", circuit="short"):
        self.mu, self.sigma, self.B0, self.G, self.a, self.mode = mu, sigma, B0, G, a, mode
        self.m = B0 * math.sqrt(sigma / mu)
        self.Ha = self.m * a
        F = hartmann_F(self.m, a)
        if circuit == "short":
            E = 0.0
        elif circuit == "open":
            if B0 == 0:
                E = 0.0
            elif mode == "inplane":
                E = -F * G / (sigma * B0 * (2 * a - F))
            else:
                E = F * G / (sigma * B0 * (2 * a - F))
        else:
            E = float(circuit)
        self.E = E
        self.Geff = G - sigma * B0 * E if mode == "inplane" else G + sigma * B0 * E
        self.Q = self.Geff / mu * (F / self.m ** 2 if self.m * a >= 1e-3 else
                                   2 * a ** 3 / 3 * (1 - 2 * (self.m * a) ** 2 / 5 + 17 * (self.m * a) ** 4 / 105))

    def U(self, y):
        return self.Geff / self.mu * hartmann_profile_factor(self.m, y, self.a)

    def dU(self, y):
        return self.Geff / self.mu * hartmann_profile_factor_dy(self.m, y, self.a)

    def current(self, y):
        """The non-zero current component: J_z (inplane) or J_x (duct)."""
        if self.mode == "inplane":
            return self.sigma * (self.E + self.B0 * self.U(y))
        return self.sigma * (self.E - self.B0 * self.U(y))

    def net_current(self):
        s = 1.0 if self.mode == "inplane" else -1.0
        return self.sigma * (2 * self.a * self.E + s * self.B0 * self.Q)

    def joule(self):
        """int J^2/sigma dy (per unit length) by 64-point Gauss on each half (smooth integrand, layer-resolved with a
        geometric split at the wall)."""
        return _layer_quad(lambda y: self.current(y) ** 2 / self.sigma, self.a, self.m)

    def viscous(self):
        return _layer_quad(lambda y: self.mu * self.dU(y) ** 2, self.a, self.m)


def _layer_quad(f, a, m, n=64):
    """int_{-a}^{a} f by Gauss-Legendre on panels geometrically refined towards both walls (layer width 1/m)."""
    d = 1.0 / max(m, 1.0)
    edges = [0.0]
    t = min(a, 0.5 * d)
    while t < a:
        edges.append(t)
        t *= 2.0
    edges.append(a)
    edges = sorted(set(edges))
    g, w = np.polynomial.legendre.leggauss(n)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # distance from the wall s in [lo, hi] -> y = a - s and y = -a + s
        s = 0.5 * (hi - lo) * g + 0.5 * (hi + lo)
        tot += 0.5 * (hi - lo) * float(np.sum(w * (f(a - s) + f(-a + s))))
    return tot


# ============================================================================================== Shercliff
class Shercliff:
    """Rectangular duct |x| < L, |y| < a, all walls insulating (and no-slip), B = B0 e_y, pressure gradient G."""

    def __init__(self, mu, sigma, B0, G, L=1.0, a=1.0, K=200000):
        self.mu, self.sigma, self.B0, self.G, self.L, self.a = mu, sigma, B0, G, L, a
        self.m = B0 * math.sqrt(sigma / mu)
        n = np.arange(K, dtype=float)
        self.beta = (n + 0.5) * math.pi / L
        self.sgn = np.where(np.arange(K) % 2 == 0, 1.0, -1.0)
        self.c = (G / mu) * 2 * self.sgn / (self.beta * L)
        m, b = self.m, self.beta
        root = np.sqrt(m * m + 4 * b * b)
        self.rp = 2 * b * b / (m + root)              # = (-m + root)/2 > 0, cancellation-free
        self.rm = -0.5 * (m + root)                   # < 0
        ea = np.exp(2 * a * self.rm)                  # e^{2 a r-}
        eb = np.exp(-2 * a * self.rp)                 # e^{-2 a r+}
        D = 1 - ea * eb
        self.P = (1 - ea) / D
        self.R = (1 - eb) / D
        self.amp = self.c / b ** 2

    def _terms_A(self, y):
        """[1 - P e^{r+(y-a)} - R e^{r-(y+a)}] for every mode (y scalar or array -> [..., K])."""
        y = np.asarray(y, float)[..., None]
        return 1 - self.P * np.exp(self.rp * (y - self.a)) - self.R * np.exp(self.rm * (y + self.a))

    def _terms_dA(self, y):
        y = np.asarray(y, float)[..., None]
        return -self.P * self.rp * np.exp(self.rp * (y - self.a)) - self.R * self.rm * np.exp(self.rm * (y + self.a))

    def w(self, x, y, K=None):
        K = K or len(self.beta)
        x = np.asarray(x, float)[..., None]
        t = 0.5 * (self._terms_A(y) + self._terms_A(-np.asarray(y, float)))
        return np.sum((self.amp * t * np.cos(self.beta * x))[..., :K], axis=-1)

    def H(self, x, y):
        x = np.asarray(x, float)[..., None]
        t = 0.5 * (self._terms_A(y) - self._terms_A(-np.asarray(y, float)))
        return math.sqrt(self.mu * self.sigma) * np.sum(self.amp * t * np.cos(self.beta * x), axis=-1)

    def J(self, x, y):
        """(J_x, J_y) = (dH/dy, -dH/dx)."""
        y = np.asarray(y, float)
        xs = np.asarray(x, float)[..., None]
        k = math.sqrt(self.mu * self.sigma)
        dt = 0.5 * (self._terms_dA(y) + self._terms_dA(-y))      # d/dy of the odd combination
        t = 0.5 * (self._terms_A(y) - self._terms_A(-y))
        Jx = k * np.sum(self.amp * dt * np.cos(self.beta * xs), axis=-1)
        Jy = k * np.sum(self.amp * t * self.beta * np.sin(self.beta * xs), axis=-1)
        return Jx, Jy

    def Q(self, K=None):
        K = K or len(self.beta)
        a, rp, rm = self.a, self.rp, self.rm
        Iy = 2 * a - self.P * (-np.expm1(-2 * a * rp)) / rp - self.R * (-np.expm1(2 * a * rm)) / (-rm)
        terms = self.amp * Iy * 2 * self.sgn / self.beta
        return float(np.sum(terms[:K]))

    def w_center(self, K=None):
        K = K or len(self.beta)
        t = 1 - self.P * np.exp(-self.rp * self.a) - self.R * np.exp(self.rm * self.a)
        return float(np.sum((self.amp * t)[:K]))

    def dphi_side(self, K=None):
        """phi(L, 0) - phi(-L, 0) = 2 int_0^L (-J_x/sigma - B0 w)(x, 0) dx."""
        K = K or len(self.beta)
        a = self.a
        k = math.sqrt(self.mu * self.sigma)
        Jx0 = -k * self.amp * (self.P * self.rp * np.exp(-a * self.rp) + self.R * self.rm * np.exp(a * self.rm))
        w0 = self.amp * (1 - self.P * np.exp(-a * self.rp) - self.R * np.exp(a * self.rm))
        integ = self.sgn / self.beta
        return float(2 * np.sum(((-Jx0 / self.sigma - self.B0 * w0) * integ)[:K]))


# ============================================================================================== Hunt
class Hunt:
    """Rectangular duct |x| < L, |y| < a; Hartmann walls y = +-a perfectly conducting (phi = 0), side walls x = +-L
    insulating; no-slip everywhere; B = B0 e_y (Hunt's case I with non-conducting side walls)."""

    def __init__(self, mu, sigma, B0, G, L=1.0, a=1.0, K=200000):
        self.mu, self.sigma, self.B0, self.G, self.L, self.a = mu, sigma, B0, G, L, a
        self.m = B0 * math.sqrt(sigma / mu)
        self.s = math.sqrt(mu / sigma)
        k = np.arange(K)
        self.alpha = (k + 0.5) * math.pi / a
        self.sgn = np.where(k % 2 == 0, 1.0, -1.0)
        self.c = (G / mu) * 2 * self.sgn / (self.alpha * a)
        al, m = self.alpha, self.m
        self.lam = np.sqrt(al * al + 1j * m * al)          # principal root, Re > 0
        self.vp = self.c / (al * al + m * m)
        self.C = -(self.vp / 2) * (1 - 1j * m / al)
        e2 = np.exp(-2 * self.lam * L)
        self.tanhL = (1 - e2) / (1 + e2)
        self.sechL = 2 * np.exp(-self.lam * L) / (1 + e2)

    def _ch(self, x):
        """cosh(l x)/cosh(l L) and sinh(l x)/cosh(l L), |x| <= L, overflow-free."""
        x = np.asarray(x, float)[..., None]
        ax = np.abs(x)
        L = self.L
        den = 1 + np.exp(-2 * self.lam * L)
        base = np.exp(self.lam * (ax - L))
        ch = base * (1 + np.exp(-2 * self.lam * ax)) / den
        sh = np.sign(x) * base * (1 - np.exp(-2 * self.lam * ax)) / den
        return ch, sh

    def w(self, x, y):
        ch, _ = self._ch(x)
        cy = np.cos(self.alpha * np.asarray(y, float)[..., None])
        return np.sum((self.vp + 2 * np.real(self.C * ch)) * cy, axis=-1)

    def phi(self, x, y):
        _, sh = self._ch(x)
        cy = np.cos(self.alpha * np.asarray(y, float)[..., None])
        return self.s * np.sum(2 * np.real(self.C * (1j * self.lam / self.alpha) * sh) * cy, axis=-1)

    def J(self, x, y):
        """J_x = sigma(-dphi/dx - B0 w), J_y = -sigma dphi/dy."""
        ch, sh = self._ch(x)
        Y = np.asarray(y, float)[..., None]
        cy, sy = np.cos(self.alpha * Y), np.sin(self.alpha * Y)
        dphidx = self.s * np.sum(2 * np.real(self.C * (1j * self.lam ** 2 / self.alpha) * ch) * cy, axis=-1)
        dphidy = -self.s * np.sum(2 * np.real(self.C * (1j * self.lam / self.alpha) * sh) * self.alpha * sy, axis=-1)
        w = np.sum((self.vp + 2 * np.real(self.C * ch)) * cy, axis=-1)
        return self.sigma * (-dphidx - self.B0 * w), -self.sigma * dphidy

    def Q(self, K=None):
        K = K or len(self.alpha)
        terms = (2 * self.sgn / self.alpha) * (2 * self.L * self.vp + 4 * np.real(self.C * self.tanhL / self.lam))
        return float(np.sum(terms[:K]))

    def w_center(self, K=None):
        K = K or len(self.alpha)
        return float(np.sum((self.vp + 2 * np.real(self.C * self.sechL))[:K]))

    def phi_side(self, K=None):
        """phi(L, 0) (side-wall mid-point; phi(-L, 0) = -phi(L, 0))."""
        K = K or len(self.alpha)
        return float(self.s * np.sum((2 * np.real(self.C * (1j * self.lam / self.alpha) * self.tanhL))[:K]))

    def wall_current_moment(self, K=None):
        """M = int_{-L}^{L} J.n (x/L) dx on the top Hartmann wall y = a (outward n = +e_y): the first moment of the
        current that leaves the fluid into the wall."""
        K = K or len(self.alpha)
        L = self.L
        terms = (4 * self.sigma * self.s / L) * self.sgn * np.real(1j * self.C * (L - self.tanhL / self.lam))
        return float(np.sum(terms[:K]))


# ============================================================================================== core current (fast)
def hunt_J_fast(h: "Hunt", x, y, K=400):
    """Hunt current at interior points with the Hartmann part in closed form (sum_k v_p cos(alpha_k y) is the
    short-circuit Hartmann profile), so only the exponentially decaying C' terms are summed (K terms).  Accurate to
    round-off at points a fixed distance from the side walls (the core measure of)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    C, lam, al = h.C[:K], h.lam[:K], h.alpha[:K]
    ax = np.abs(x)[..., None]
    den = 1 + np.exp(-2 * lam * h.L)
    base = np.exp(lam * (ax - h.L))
    ch = base * (1 + np.exp(-2 * lam * ax)) / den
    sh = np.sign(x)[..., None] * base * (1 - np.exp(-2 * lam * ax)) / den
    Y = y[..., None]
    cy, sy = np.cos(al * Y), np.sin(al * Y)
    wH = h.G / h.mu * hartmann_profile_factor(h.m, y, h.a)
    w = wH + np.sum(2 * np.real(C * ch) * cy, axis=-1)
    dphidx = h.s * np.sum(2 * np.real(C * (1j * lam ** 2 / al) * ch) * cy, axis=-1)
    dphidy = -h.s * np.sum(2 * np.real(C * (1j * lam / al) * sh) * al * sy, axis=-1)
    return h.sigma * (-dphidx - h.B0 * w), -h.sigma * dphidy


def shercliff_J_fast(s: "Shercliff", x, y, K=None, chunk=4096):
    """Shercliff current (dH/dy, -dH/dx) at interior points; in the odd combination the slowly converging constant
    part of a_n cancels, so the terms decay like exp(-r+ (a - |y|)) with r+ ~ beta for beta > m: K = 4 m a/pi + 400
    terms suffice for |y| <= a/2 (default); evaluated in chunks of points (memory)."""
    if K is None:
        K = min(len(s.beta), int(4 * s.m * s.a / math.pi) + 400)
    xs = np.asarray(x, float)
    ys = np.asarray(y, float)
    shp = np.broadcast(xs, ys).shape
    xf, yf = np.broadcast_to(xs, shp).ravel(), np.broadcast_to(ys, shp).ravel()
    outx, outy = np.empty(xf.size), np.empty(yf.size)
    for i in range(0, xf.size, chunk):
        jx, jy = _shercliff_J_block(s, xf[i:i + chunk], yf[i:i + chunk], K)
        outx[i:i + chunk], outy[i:i + chunk] = jx, jy
    return outx.reshape(shp), outy.reshape(shp)


def _shercliff_J_block(s, x, y, K):
    y = np.asarray(y, float)
    X = np.asarray(x, float)[..., None]
    Y = y[..., None]
    k = math.sqrt(s.mu * s.sigma)
    P, Rr, rp, rm, a = s.P[:K], s.R[:K], s.rp[:K], s.rm[:K], s.a
    amp, beta = s.amp[:K], s.beta[:K]
    def A(yy):
        return 1 - P * np.exp(rp * (yy - a)) - Rr * np.exp(rm * (yy + a))
    def dA(yy):
        return -P * rp * np.exp(rp * (yy - a)) - Rr * rm * np.exp(rm * (yy + a))
    t = 0.5 * (A(Y) - A(-Y))
    dt = 0.5 * (dA(Y) + dA(-Y))
    Jx = k * np.sum(amp * dt * np.cos(beta * X), axis=-1)
    Jy = k * np.sum(amp * t * beta * np.sin(beta * X), axis=-1)
    return Jx, Jy


# ============================================================================================== two-layer Hartmann
class HartmannTwoLayer:
    """In-plane Hartmann channel with piecewise-constant conductivity: sigma = s1 for -a < y < 0, s2 for 0 < y < a
    (interface on a mesh line), short circuit (E_z = 0), B = (0, B0, 0), constant mu.  On each layer
        mu U'' - s_i B0^2 U = -G,   U = G/(s_i B0^2) + A_i cosh(m_i y) + C_i sinh(m_i y),  m_i = B0 sqrt(s_i/mu),
    with U(-a) = U(a) = 0 and U, mu U' continuous at y = 0 (4 x 4 linear system, solved once).  J_z = s_i B0 U."""

    def __init__(self, mu, s1, s2, B0, G, a=1.0):
        self.mu, self.s, self.B0, self.G, self.a = mu, (s1, s2), B0, G, a
        self.m = tuple(B0 * math.sqrt(s / mu) for s in (s1, s2))
        self.P = tuple(G / (s * B0 ** 2) for s in (s1, s2))
        m1, m2 = self.m
        # unknowns A1, C1, A2, C2 ; scaled hyperbolics are fine here (m a <= ~50 in the fixed cases)
        M = np.array([[math.cosh(m1 * a), -math.sinh(m1 * a), 0, 0],
                      [0, 0, math.cosh(m2 * a), math.sinh(m2 * a)],
                      [1, 0, -1, 0],
                      [0, m1, 0, -m2]])
        rhs = np.array([-self.P[0], -self.P[1], self.P[1] - self.P[0], 0.0])
        self.c = np.linalg.solve(M, rhs)

    def U(self, y):
        y = np.asarray(y, float)
        A1, C1, A2, C2 = self.c
        m1, m2 = self.m
        lo = self.P[0] + A1 * np.cosh(m1 * y) + C1 * np.sinh(m1 * y)
        hi = self.P[1] + A2 * np.cosh(m2 * y) + C2 * np.sinh(m2 * y)
        return np.where(y < 0, lo, hi)

    def dU(self, y):
        y = np.asarray(y, float)
        A1, C1, A2, C2 = self.c
        m1, m2 = self.m
        lo = m1 * (A1 * np.sinh(m1 * y) + C1 * np.cosh(m1 * y))
        hi = m2 * (A2 * np.sinh(m2 * y) + C2 * np.cosh(m2 * y))
        return np.where(y < 0, lo, hi)

    def sigma(self, y):
        return np.where(np.asarray(y, float) < 0, self.s[0], self.s[1])

    def Q(self):
        A1, C1, A2, C2 = self.c
        m1, m2 = self.m
        a = self.a
        q1 = self.P[0] * a + A1 * math.sinh(m1 * a) / m1 - C1 * (math.cosh(m1 * a) - 1) / m1
        q2 = self.P[1] * a + A2 * math.sinh(m2 * a) / m2 + C2 * (math.cosh(m2 * a) - 1) / m2
        return q1 + q2
