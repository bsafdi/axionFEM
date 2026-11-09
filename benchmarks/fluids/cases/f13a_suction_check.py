"""MHD Hartmann channel with wall suction (case analytic_hartmann_suction_2d; group F13a):
the non-MMS case of the convection row rho (u.grad)u, including the transport of u_z (runner.check_module
"f13a_suction_check").

It imports f13a_mhd_check (and through it f1a_stokes_check) and f13a_mhd_series, and changes none of them.
Nothing here calls the solver.  Geometry: the fixed Hartmann strip family f13a_mhd_check.hartmann_strip_mesh
(affine, nested, graded to the Hartmann layer).

Porous walls y = -+1 with a uniform cross-flow u_y = V (injection at y = -1, suction at y = +1), B = (0, B0, 0),
rho > 0, mu, sigma uniform; m = B0 sqrt(sigma/mu).  With u = (U(y), V, 0) (mode 'inplane', traction-driven, E_z = 0,
insulating) or u = (0, V, W(y)) (mode 'duct', G_z-driven, potential 0 on x = 0 and x = LX), the inductionless MHD
equations reduce to
    mu F'' - rho V F' - sigma B0^2 F = -G,       F(-1) = F(1) = 0,
since (u x B) = F B0 e_z (in-plane: J_z = sigma B0 U, force -J_z B0 e_x) or -F B0 e_x (duct: J_x = -sigma B0 W,
force J_x B0 e_z), and the convective term rho (u.grad)F = rho V F' is the only one left (u_y = V is constant,
nothing depends on x).  Solution F = (G/(sigma B0^2)) [1 - a e^{l+ (y - 1)} - b e^{l- (y + 1)}],
l+- = (rho V +- sqrt(rho^2 V^2 + 4 mu sigma B0^2))/(2 mu); every exponent is <= 0 on [-1, 1].  Pressure: in-plane
p = G (LX - x) (the y-momentum row is 0 = -p_y), duct p = const.
"""
from __future__ import annotations

import math

import numpy as np

import f13a_mhd_check as M

PARAMS = {"mu": 0.5, "sigma": 2.0, "G": 1.0, "rho": 1.0, "a": 1.0}
LEVELS = (0, 1, 2, 3, 4)
VARIANTS = {
    "inplane_Ha3_V2": {"mode": "inplane", "Ha": 3.0, "V": 2.0},
    "inplane_Ha10_V4": {"mode": "inplane", "Ha": 10.0, "V": 4.0},
    "duct_Ha3_V2": {"mode": "duct", "Ha": 3.0, "V": 2.0},
    "duct_Ha10_V4": {"mode": "duct", "Ha": 10.0, "V": 4.0},
}

class SuctionExact:
    def __init__(self, mode: str, Ha: float, V: float):
        mu, s, G, rho = PARAMS["mu"], PARAMS["sigma"], PARAMS["G"], PARAMS["rho"]
        self.mode, self.Ha, self.V = mode, float(Ha), float(V)
        self.mu, self.sigma_v, self.G, self.rho = mu, s, G, rho
        self.B0 = Ha / (PARAMS["a"] * math.sqrt(s / mu))
        self.Ez = 0.0
        disc = math.sqrt(rho * rho * V * V + 4 * mu * s * self.B0 ** 2)
        self.lp = (rho * V + disc) / (2 * mu)
        self.lm = (rho * V - disc) / (2 * mu)
        self.K = G / (s * self.B0 ** 2)
        ep, em = math.exp(-2 * self.lp), math.exp(2 * self.lm)     # both < 1
        # F(1) = 0: 1 - a - b em = 0;  F(-1) = 0: 1 - a ep - b = 0
        det = 1 - ep * em
        self.a = (1 - em) / det
        self.b = (1 - ep) / det
        lp, lm = self.lp, self.lm
        # Q = int_{-1}^{1} F dy (per unit width)
        self.Q = self.K * (2 - self.a * (1 - ep) / lp - self.b * (1 - em) / (-lm))

    def F(self, y):
        y = np.asarray(y, float)
        return self.K * (1 - self.a * np.exp(self.lp * (y - 1)) - self.b * np.exp(self.lm * (y + 1)))

    def dF(self, y):
        y = np.asarray(y, float)
        return -self.K * (self.a * self.lp * np.exp(self.lp * (y - 1)) + self.b * self.lm * np.exp(self.lm * (y + 1)))

    def u(self, X, Y):
        F = self.F(Y)
        z = np.zeros_like(F)
        Vv = np.full_like(F, self.V)
        return np.stack([F, Vv, z] if self.mode == "inplane" else [z, Vv, F], axis=-1)

    def grad_u(self, X, Y):
        g = np.zeros(np.shape(Y) + (3, 2))
        g[..., 0 if self.mode == "inplane" else 2, 1] = self.dF(Y)
        return g

    def p(self, X, Y):
        return self.G * (M.LX_STRIP - X) if self.mode == "inplane" else np.zeros(np.shape(X))

    def phi(self, X, Y):
        return np.zeros(np.shape(X))

    def grad_phi(self, X, Y):
        return np.zeros(np.shape(X) + (2,))

    def sigma(self, X, Y):
        return np.full(np.shape(X), self.sigma_v)

    def B(self, X, Y):
        Bv = np.zeros(np.shape(X) + (3,))
        Bv[..., 1] = self.B0
        return Bv

    def J(self, X, Y):
        return M.current_h(self.u(X, Y), self.grad_phi(X, Y), self.sigma(X, Y), self.B(X, Y), self.Ez)


def exact(variant: str) -> SuctionExact:
    v = VARIANTS[variant]
    return SuctionExact(v["mode"], v["Ha"], v["V"])


def mesh(variant: str, k: int):
    return M.hartmann_strip_mesh(VARIANTS[variant]["Ha"], k)


def flow_rate(ev, mode: str, deg=M.QUAD_DEGREE) -> float:
    """Q_h = (1/LX) int u_h,x dA (in-plane) or (1/LX) int u_h,z dA (duct): flow rate per unit width."""
    q, xq, W = M._weights(ev, deg)
    u, _ = ev.velocity(q)
    return float((W * u[..., 0 if mode == "inplane" else 2]).sum()) / M.LX_STRIP
