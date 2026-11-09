"""Duct Poiseuille flow (3D Stokes): out-of-band series reference.

    python3 -I -B tools/fluids_mms/duct_series_reference.py OUT.json

Pure standard library (decimal, 50 significant digits): no package of the environment enters the numbers.

Problem (P1): fully developed Stokes flow along z in the duct
[0, W] x [0, H] x [0, L], W = 1, H = 1/2, L = 2, mu = 1, pressure gradient -dp/dz = G = 2:
    -mu (w_xx + w_yy) = G,  w = 0 on the four walls;   u = (0, 0, w(x, y)),  p = G (L - z).
With a = W/2, b = H/2 and centred coordinates X = x - a, Y = y - b, separation of variables (derived here, not taken from a source) gives the x-series
    w = G/(2 mu) (a^2 - X^2) - (16 G a^2 / (mu pi^3)) sum_{k>=0} (-1)^k n^-3 cos(n pi X / 2a) cosh(n pi Y / 2a) /
        cosh(n pi b / 2a),     n = 2k + 1,
and, integrating term by term (int cos = (4a/n pi)(-1)^k, int cosh = (4a/n pi) sinh), the flow rate
    Q = 4 G a^3 b / (3 mu) - (256 G a^4 / (mu pi^5)) sum_k tanh(n pi b / 2a) / n^5.
The y-series is the same with (a, X) <-> (b, Y).  Both are evaluated independently; they must agree.

Truncation bounds (stated in the output): for Q, 0 <= tanh <= 1 gives |tail after n_K| <= (256 G a^4/(mu pi^5))
sum_{n > n_K, odd} n^-5 <= (256 G a^4/(mu pi^5)) / (8 n_K^4) (integral bound); for the centre velocity the k-th term
is bounded by amp n^-3 / cosh(n pi b/2a) <= 2 amp n^-3 exp(-n pi b/2a), summed geometrically.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal as D, getcontext

getcontext().prec = 60


def pi_dec():
    """pi by Machin's formula, 60 digits."""
    def arctan_inv(x):
        x = D(x)
        x2 = x * x
        term = D(1) / x
        s, n, sign = term, 1, 1
        while True:
            term /= x2
            n += 2
            sign = -sign
            t = term / n
            if t < D(10) ** -70:
                break
            s += sign * t
        return s
    return 16 * arctan_inv(5) - 4 * arctan_inv(239)


PI = pi_dec()


def tanh(x):
    e = (-2 * x).exp()
    return (1 - e) / (1 + e)


def sech(x):
    e = (-x).exp()
    return 2 * e / (1 + e * e)


def flow_rate(a, b, G, mu, kmax):
    amp = 256 * G * a ** 4 / (mu * PI ** 5)
    s = D(0)
    for k in range(kmax):
        n = 2 * k + 1
        s += tanh(n * PI * b / (2 * a)) / D(n) ** 5
    nK = 2 * kmax - 1
    tail = amp / (8 * D(nK) ** 4)
    return 4 * G * a ** 3 * b / (3 * mu) - amp * s, tail


def centre_velocity(a, b, G, mu, kmax):
    amp = 16 * G * a * a / (mu * PI ** 3)
    s = D(0)
    for k in range(kmax):
        n = 2 * k + 1
        s += (-1) ** k * sech(n * PI * b / (2 * a)) / D(n) ** 3
    n1 = 2 * kmax + 1
    r = (-PI * b / a).exp()
    tail = 2 * amp * (-n1 * PI * b / (2 * a)).exp() / D(n1) ** 3 / (1 - r)
    return G / (2 * mu) * a * a - amp * s, tail


def main(out):
    W, H, L, G, mu = D(1), D(1) / 2, D(2), D(2), D(1)
    a, b = W / 2, H / 2
    kq = 20000
    Qx, tqx = flow_rate(a, b, G, mu, kq)
    Qy, tqy = flow_rate(b, a, G, mu, kq)
    wx, twx = centre_velocity(a, b, G, mu, 200)
    wy, twy = centre_velocity(b, a, G, mu, 400)
    agree_Q = abs(Qx - Qy)
    agree_w = abs(wx - wy)
    if agree_Q > D(10) ** -15 * Qx or agree_w > D(10) ** -15 * wx:
        raise SystemExit(f"x- and y-series disagree: Q {agree_Q}, w {agree_w}")
    res = {
        "problem": {"W": "1", "H": "1/2", "L": "2", "G": "2", "mu": "1", "axis": "z",
                    "pde": "-mu (w_xx + w_yy) = G, w = 0 on the walls; u = (0, 0, w), p = G (L - z)"},
        "method": "x-series and y-series of the separated solution, Python decimal at 60 "
                  "digits, pi by Machin's formula; the two series are independent evaluations and must agree",
        "quantities": {
            "Q": {"value": float(Qx), "value_str": str(+Qx)[:40],
                  "truncation_bound_abs": float(max(tqx, tqy)), "x_vs_y_series_abs": float(agree_Q),
                  "definition": "flow rate int_section w dA (= (1/L) int_Omega u_z dV for the exact solution)"},
            "w_centre": {"value": float(wx), "value_str": str(+wx)[:40],
                         "truncation_bound_abs": float(max(twx, twy)), "x_vs_y_series_abs": float(agree_w),
                         "definition": "w at the duct centre (x, y) = (1/2, 1/4)"},
            "Q_over_volume_mean": {"value": float(Qx / (W * H)), "definition": "mean axial velocity Q / (W H)"},
        },
        "terms": {"Q": kq, "w_centre_x": 200, "w_centre_y": 400},
    }
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(json.dumps(res["quantities"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
