"""MHD variable-field annulus (case analytic_mhd_annulus_variable_b; group F13a): the check module's
closed forms, mesh family and error computation (runner.check_module "f13a_annulus_check").

It imports f13a_mhd_check (and through it f1a_stokes_check) and f13a_mhd_series, and changes none of them.
Nothing here calls the solver.

Geometry: annulus r1 = 1 < r < r2 = 2, ISOPARAMETRIC P2 elements (6-node triangles; boundary-edge midpoints on the
circle at the bisecting angle, every other edge straight: the F1a annulus convention).  The Evaluator protocol is
that of f13a_mhd_check, except that ev.geometry(ref) returns the CURVED map x(ref) = sum_i N_i(ref) X6_i and its
Jacobian determinant (iso_map below), and gradients are physical gradients on the curved element.

Fields (derived from the continuum equations and re-derived symbolically by
tools/fluids_mms/f13a_annulus_fields.py), mu = 1/2, sigma = 2, G = 1:
  radial    B = (C/r) e_r = (C x/r^2, C y/r^2, 0), m = C sqrt(sigma/mu), n = sqrt(1 + m^2):
            w(r)   = A [r^2 - f(r)],   A = G/(mu (m^2 - 4)),  f = a (r/r2)^m + b (r1/r)^m,  f(r1) = r1^2, f(r2) = r2^2
            u_t(r) = D [r - g(r)],     D = E_z / C,           g = a'(r/r2)^n + b'(r1/r)^n, g(r1) = r1, g(r2) = r2
            phi = 0, p = const (rho = 0); J = sigma(w C/r e_theta + (E_z - u_t C/r) e_z); J.n = 0 on every circle,
            so the insulating and the grounded potential walls have the same solution.
  azimuthal B = (C/r) e_theta = (-C y/r^2, C x/r^2, 0); u = w(r) e_z; J_r = K/r (closes through the walls);
            w = -G r^2/(4 mu) - (K C/(2 mu)) ln^2 r + a1 ln r + a2,  phi = -(K/sigma) ln r - C int_1^r w/s ds,
            p = -sigma E_z C ln r;  w(1) = w(2) = 0, phi(1) = 0, phi(2) = V fix a1, a2, K.
            Variant azimuthal_slide_ins: insulating walls (J_r = 0 there, so K = 0) and an inner wall sliding
            axially, w(1) = W, w(2) = 0: then (u x B).n = -W C/r1 != 0 on the insulating inner wall, and
            phi = -C int_1^r w/s ds (up to a constant) is non-trivial.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

import f13a_mhd_check as M
import f1a_stokes_check as F1

QUAD_DEGREE = 10
PARAMS = {"mu": 0.5, "sigma": 2.0, "G": 1.0, "r1": 1.0, "r2": 2.0}
NR0, NT0 = 8, 32                # base radial and angular intervals of the nested polar family
LEVELS = (0, 1, 2, 3, 4)

VARIANTS = {
    "radial_ins_m5": {"field": "radial", "C": 2.5, "E_z": 0.5, "potential": "insulating"},
    "radial_gnd_m5": {"field": "radial", "C": 2.5, "E_z": 0.5, "potential": "grounded"},
    "radial_ins_m20": {"field": "radial", "C": 10.0, "E_z": 0.5, "potential": "insulating"},
    "azimuthal_cond": {"field": "azimuthal", "C": 2.0, "E_z": 0.5, "V": 0.25, "potential": "dirichlet"},
    "azimuthal_slide_ins": {"field": "azimuthal", "C": 2.0, "E_z": 0.5, "W": 0.5, "potential": "insulating"},
}


def m_of(C: float) -> float:
    return C * math.sqrt(PARAMS["sigma"] / PARAMS["mu"])


# ============================================================================================== exact fields
class _Base:
    mu, sigma_v, G = PARAMS["mu"], PARAMS["sigma"], PARAMS["G"]

    @staticmethod
    def _r(X, Y):
        return np.hypot(X, Y)

    def sigma(self, X, Y):
        return np.full(np.shape(X), self.sigma_v)

    def J(self, X, Y):
        return M.current_h(self.u(X, Y), self.grad_phi(X, Y), self.sigma(X, Y), self.B(X, Y), self.Ez)


class RadialExact(_Base):
    """Radial field B = (C/r) e_r; see the module docstring.  Stable forms: every power is <= 1 on [r1, r2]."""

    def __init__(self, C: float, Ez: float):
        r1, r2 = PARAMS["r1"], PARAMS["r2"]
        self.C, self.Ez = float(C), float(Ez)
        self.m = m_of(C)
        self.n = math.sqrt(1.0 + self.m ** 2)
        if abs(self.m - 2.0) < 1e-6 or abs(self.n - 3.0) < 1e-6:
            raise ValueError("resonant exponent (m = 2 or n = 3)")
        self.A = self.G / (self.mu * (self.m ** 2 - 4.0))
        self.D = self.Ez / self.C
        # f = a (r/r2)^m + b (r1/r)^m with f(r1) = r1^2, f(r2) = r2^2;  e = (r1/r2)^m
        e = (r1 / r2) ** self.m
        self.a = (r2 ** 2 - e * r1 ** 2) / (1 - e * e)
        self.b = (r1 ** 2 - e * r2 ** 2) / (1 - e * e)
        en = (r1 / r2) ** self.n
        self.a_ = (r2 - en * r1) / (1 - en * en)
        self.b_ = (r1 - en * r2) / (1 - en * en)
        self.e, self.en = e, en
        m, n = self.m, self.n
        # Q = 2 pi int w r dr, L_theta = 2 pi int u_t r^2 dr (closed form; r1 = 1, r2 = 2 assumed below)
        assert (r1, r2) == (1.0, 2.0)
        self.Q = 2 * math.pi * self.A * (15 / 4 - self.a * (4 - e) / (m + 2) - self.b * (1 - 4 * e) / (m - 2))
        self.L_theta = 2 * math.pi * self.D * (15 / 4 - self.a_ * (8 - en) / (n + 3) - self.b_ * (1 - 8 * en) / (n - 3))

    def _f(self, r, k=0):
        r1, r2, m = PARAMS["r1"], PARAMS["r2"], self.m
        P, Q = (r / r2) ** m, (r1 / r) ** m
        if k == 0:
            return self.a * P + self.b * Q
        return m * (self.a * P - self.b * Q) / r

    def _g(self, r, k=0):
        r1, r2, n = PARAMS["r1"], PARAMS["r2"], self.n
        P, Q = (r / r2) ** n, (r1 / r) ** n
        if k == 0:
            return self.a_ * P + self.b_ * Q
        return n * (self.a_ * P - self.b_ * Q) / r

    def w_r(self, r):
        r = np.asarray(r, float)
        return self.A * (r ** 2 - self._f(r))

    def dw_r(self, r):
        r = np.asarray(r, float)
        return self.A * (2 * r - self._f(r, 1))

    def ut_r(self, r):
        r = np.asarray(r, float)
        return self.D * (r - self._g(r))

    def dut_r(self, r):
        r = np.asarray(r, float)
        return self.D * (1.0 - self._g(r, 1))

    def u(self, X, Y):
        r = self._r(X, Y)
        h = self.ut_r(r) / r
        return np.stack([-h * Y, h * X, self.w_r(r)], axis=-1)

    def grad_u(self, X, Y):
        r = self._r(X, Y)
        h = self.ut_r(r) / r
        dh = (self.dut_r(r) - h) / r                    # d(u_t/r)/dr
        dw = self.dw_r(r)
        g = np.zeros(np.shape(X) + (3, 2))
        g[..., 0, 0] = -dh * X * Y / r
        g[..., 0, 1] = -h - dh * Y * Y / r
        g[..., 1, 0] = h + dh * X * X / r
        g[..., 1, 1] = dh * X * Y / r
        g[..., 2, 0] = dw * X / r
        g[..., 2, 1] = dw * Y / r
        return g

    def p(self, X, Y):
        return np.zeros(np.shape(X))

    def phi(self, X, Y):
        return np.zeros(np.shape(X))

    def grad_phi(self, X, Y):
        return np.zeros(np.shape(X) + (2,))

    def B(self, X, Y):
        r2 = X * X + Y * Y
        return np.stack([self.C * X / r2, self.C * Y / r2, np.zeros(np.shape(X))], axis=-1)


class AzimuthalExact(_Base):
    """Azimuthal field B = (C/r) e_theta, conducting walls phi(r1) = 0, phi(r2) = V; see the module docstring."""

    def __init__(self, C: float, Ez: float, V: float = 0.0, W: float | None = None):
        """W is None: conducting walls phi(1) = 0, phi(2) = V, no slip.  W given: insulating walls (K = 0), w(1) = W,
        w(2) = 0, potential up to a constant (fixed here by phi(1) = 0; compared mean-free)."""
        assert (PARAMS["r1"], PARAMS["r2"]) == (1.0, 2.0)
        self.C, self.Ez, self.V = float(C), float(Ez), float(V)
        mu, s, G, L = self.mu, self.sigma_v, self.G, math.log(2.0)
        if W is None:
            # phi(2) = -K L/sigma - C Iw(2),  Iw(2) = int_1^2 w/r dr = G (5L - 3)/(8 mu) + K C L^3/(12 mu)
            self.K = -(V + C * G * (5 * L - 3) / (8 * mu)) / (L / s + C * C * L ** 3 / (12 * mu))
            self.a2 = G / (4 * mu)
            self.a1 = (3 * G / (4 * mu) + self.K * C * L * L / (2 * mu)) / L
        else:
            self.K = 0.0
            self.a2 = float(W) + G / (4 * mu)                 # w(1) = -G/(4 mu) + a2 = W
            self.a1 = (G / mu - self.a2) / L                   # w(2) = -G/mu + a1 L + a2 = 0
        K, a1, a2 = self.K, self.a1, self.a2
        self.Q = 2 * math.pi * (-G / (4 * mu) * 15 / 4 - K * C / (2 * mu) * (2 * L * L - 2 * L + 0.75)
                                + a1 * (2 * L - 0.75) + a2 * 1.5)
        self.I_wall = 2 * math.pi * K               # int_{r = r2} J.n ds (outward); -I_wall through r = r1

    def w_r(self, r):
        r = np.asarray(r, float)
        lr = np.log(r)
        return -self.G * r ** 2 / (4 * self.mu) - self.K * self.C / (2 * self.mu) * lr ** 2 + self.a1 * lr + self.a2

    def dw_r(self, r):
        r = np.asarray(r, float)
        lr = np.log(r)
        return -self.G * r / (2 * self.mu) - self.K * self.C / self.mu * lr / r + self.a1 / r

    def phi_r(self, r):
        r = np.asarray(r, float)
        lr = np.log(r)
        Iw = (-self.G * (r ** 2 - 1) / (8 * self.mu) - self.K * self.C / (6 * self.mu) * lr ** 3
              + self.a1 * lr ** 2 / 2 + self.a2 * lr)
        return -self.K / self.sigma_v * lr - self.C * Iw

    def dphi_r(self, r):
        r = np.asarray(r, float)
        return -self.K / (self.sigma_v * r) - self.C * self.w_r(r) / r

    def u(self, X, Y):
        r = self._r(X, Y)
        z = np.zeros(np.shape(X))
        return np.stack([z, z, self.w_r(r)], axis=-1)

    def grad_u(self, X, Y):
        r = self._r(X, Y)
        dw = self.dw_r(r)
        g = np.zeros(np.shape(X) + (3, 2))
        g[..., 2, 0] = dw * X / r
        g[..., 2, 1] = dw * Y / r
        return g

    def p(self, X, Y):
        return -self.sigma_v * self.Ez * self.C * np.log(self._r(X, Y))

    def phi(self, X, Y):
        return self.phi_r(self._r(X, Y))

    def grad_phi(self, X, Y):
        r = self._r(X, Y)
        d = self.dphi_r(r)
        return np.stack([d * X / r, d * Y / r], axis=-1)

    def B(self, X, Y):
        r2 = X * X + Y * Y
        return np.stack([-self.C * Y / r2, self.C * X / r2, np.zeros(np.shape(X))], axis=-1)


def exact(variant: str):
    v = VARIANTS[variant]
    if v["field"] == "radial":
        return RadialExact(v["C"], v["E_z"])
    return AzimuthalExact(v["C"], v["E_z"], v.get("V", 0.0), v.get("W"))


# ============================================================================================== mesh family
def radial_base_nodes(variant: str) -> np.ndarray:
    """NR0 base intervals on [r1, r2]: for the radial field graded to the layer thickness r1/m at both walls
    (f13a_mhd_check.graded_nodes; uniform when the layer is thicker than a uniform interval), uniform otherwise."""
    v = VARIANTS[variant]
    r1, r2 = PARAMS["r1"], PARAMS["r2"]
    a = 0.5 * (r2 - r1)
    delta = r1 / m_of(v["C"]) if v["field"] == "radial" else math.inf
    nodes = M.graded_nodes(NR0, a, delta) + 0.5 * (r1 + r2)
    nodes[0], nodes[-1] = r1, r2
    return nodes


def annulus_mesh(variant: str, k: int):
    """Level k of the nested polar family: every base radial interval split into 2^k equal parts (in r), NT0 2^k
    equal angles; each polar quadrilateral split along its (i, j)-(i+1, j+1) diagonal.  This equals k red
    refinements in the (r, theta) parameter space (nested there; the map to (x, y) is smooth).  Returns V, T, X6
    (X6[ne, 6, 2]: vertices, then the midpoints of edges 01, 12, 20; boundary-edge midpoints on the circle at the
    bisecting angle, other edges straight)."""
    base = radial_base_nodes(variant)
    f = 2 ** k
    rs = np.concatenate([np.linspace(base[i], base[i + 1], f + 1)[:-1] for i in range(len(base) - 1)] + [base[-1:]])
    Nt = NT0 * f
    ts = 2 * np.pi * np.arange(Nt) / Nt
    V = np.array([[r * math.cos(t), r * math.sin(t)] for r in rs for t in ts])
    idx = lambda i, j: i * Nt + (j % Nt)  # noqa: E731
    T = []
    for i in range(len(rs) - 1):
        for j in range(Nt):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            T += [[a, b, c], [a, c, d]]
    T = np.array(T, dtype=np.int64)
    rad = np.repeat(rs, Nt)
    X6 = np.empty((len(T), 6, 2))
    X6[:, :3] = V[T]
    r1, r2 = PARAMS["r1"], PARAMS["r2"]
    for kk, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        A_, B_ = V[T[:, i]], V[T[:, j]]
        ra, rb = rad[T[:, i]], rad[T[:, j]]
        mid = 0.5 * (A_ + B_)
        on = ((np.abs(ra - r1) < 1e-12) & (np.abs(rb - r1) < 1e-12)) | ((np.abs(ra - r2) < 1e-12) & (np.abs(rb - r2) < 1e-12))
        ang = np.arctan2(A_[:, 1] + B_[:, 1], A_[:, 0] + B_[:, 0])
        mid[on] = np.c_[ra[on] * np.cos(ang[on]), ra[on] * np.sin(ang[on])]
        X6[:, 3 + kk] = mid
    return V, T, X6


def h_nominal(k: int) -> float:
    return 1.0 / (NR0 * 2 ** k)


def mesh_record(variant: str, k: int) -> dict:
    V, T, X6 = annulus_mesh(variant, k)
    return {"n_vertices": int(len(V)), "n_elements": int(len(T)), "mesh_hash": F1.mesh_hash(V, T),
            "p2_nodes_hash": F1.p2_nodes_hash(X6)}


# ============================================================================================== isoparametric map
def p2_shape(ref):
    s, t = ref[:, 0], ref[:, 1]
    l0 = 1 - s - t
    return np.c_[l0 * (2 * l0 - 1), s * (2 * s - 1), t * (2 * t - 1), 4 * s * l0, 4 * s * t, 4 * t * l0]


def iso_map(X6, ref):
    """x[ne, nq, 2], dx/dref[ne, nq, 2(x,y), 2(s,t)], detJ[ne, nq] of the P2 isoparametric element map."""
    ref = np.asarray(ref, float)
    N = p2_shape(ref)                                   # [nq, 6]
    dN = M._p2_grad_ref(ref)                            # [nq, 6, 2]
    x = np.einsum("qk,ekx->eqx", N, X6)
    Jac = np.einsum("qks,ekx->eqxs", dN, X6)
    det = Jac[..., 0, 0] * Jac[..., 1, 1] - Jac[..., 0, 1] * Jac[..., 1, 0]
    return x, Jac, det


# ============================================================================================== functionals
def _wq(ev, deg=QUAD_DEGREE):
    return M._weights(ev, deg)


def flow_rate(ev, deg=QUAD_DEGREE) -> float:
    """Q_h = int u_h,z dA (axial flow rate) on the curved elements."""
    q, xq, W = _wq(ev, deg)
    u, _ = ev.velocity(q)
    return float((W * u[..., 2]).sum())


def angular_momentum(ev, deg=QUAD_DEGREE) -> float:
    """L_h = int (x u_h,y - y u_h,x) dA (= 2 pi int u_t r^2 dr for the exact swirl)."""
    q, xq, W = _wq(ev, deg)
    u, _ = ev.velocity(q)
    return float((W * (xq[..., 0] * u[..., 1] - xq[..., 1] * u[..., 0])).sum())


def wall_current_variational(ev, ex, deg=QUAD_DEGREE) -> float:
    """I_var = int J_h . grad chi dA, chi = (r - r1)/(r2 - r1): for a divergence-free J equal to the outward current
    through r = r2 (chi = 1 there, 0 on r = r1).  J_h from the check module's Ohm's law (exact sigma, B, E_z)."""
    q, xq, W = _wq(ev, deg)
    u, _ = ev.velocity(q)
    _, gphi = ev.potential(q)
    X, Y = xq[..., 0], xq[..., 1]
    J = M.current_h(u, gphi, ex.sigma(X, Y), ex.B(X, Y), ex.Ez)
    r = np.hypot(X, Y)
    dchi = 1.0 / (PARAMS["r2"] - PARAMS["r1"])
    return float((W * (J[..., 0] * X + J[..., 1] * Y) / r * dchi).sum())


def wall_current_direct(ev, ex, X6, n_gauss=10) -> float:
    """I_direct = int_{r = r2} J_h . n ds by Gauss quadrature on the curved boundary edges (one-sided traces of
    grad phi_h; n the outward normal of the discrete curved edge): the 'surface' current, O(h^2)."""
    gp, gw = np.polynomial.legendre.leggauss(n_gauss)
    s = 0.5 * (gp + 1)
    r2 = PARAMS["r2"]
    rad = np.hypot(X6[..., 0], X6[..., 1])
    tot = 0.0
    for (i, j, mid) in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
        on = (np.abs(rad[:, i] - r2) < 1e-12) & (np.abs(rad[:, j] - r2) < 1e-12)
        if not np.any(on):
            continue
        Ri, Rj = M._P2_REF[i], M._P2_REF[j]
        ref = Ri[None, :] + s[:, None] * (Rj - Ri)[None, :]
        x, Jac, _ = iso_map(X6, ref)
        tan = np.einsum("eqxs,s->eqx", Jac, Rj - Ri)           # dx/ds along the edge
        u, _ = ev.velocity(ref)
        _, gphi = ev.potential(ref)
        X, Y = x[..., 0], x[..., 1]
        J = M.current_h(u, gphi, ex.sigma(X, Y), ex.B(X, Y), ex.Ez)
        nrm = np.stack([tan[..., 1], -tan[..., 0]], axis=-1)   # |nrm| = |tan|: ds = |tan| ds_ref
        sgn = np.sign((nrm * x).sum(-1))
        jn_ds = sgn * (J[..., 0] * nrm[..., 0] + J[..., 1] * nrm[..., 1])
        tot += float(0.5 * (jn_ds[on] * gw[None, :]).sum())
    return tot


def potential_spread(ev, deg=6) -> dict:
    return M.potential_spread(ev, deg)


def geometry_area_error(ev, deg=QUAD_DEGREE) -> float:
    """|area_h - 3 pi| / (3 pi): the isoparametric geometry (affine elements give O(h^2), P2 boundaries O(h^4))."""
    _, _, W = _wq(ev, deg)
    a = math.pi * (PARAMS["r2"] ** 2 - PARAMS["r1"] ** 2)
    return abs(float(W.sum()) - a) / a


def data_sha256(obj) -> str:
    return hashlib.sha256(repr(obj).encode()).hexdigest()
