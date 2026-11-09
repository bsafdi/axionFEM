"""F13a analytic reference values (out of band).

Computes the stage-1 reference quantities of analytic_hartmann_2d, analytic_shercliff_duct_2d and
analytic_hunt_duct_2d from the closed forms / series of f13a_series.py, with
  * truncation: partial sums at K = 2^14 ... 2^21 terms; the tail beyond K_max is bounded by 10x the last observed
    increment |S(K_max) - S(K_max/2)| (terms decay algebraically like K^-3 or faster, so the true tail is below the
    last increment; the factor 10 is a safety margin), reported per quantity as `truncation_bound`;
  * floating point: the same partial sum at K = 4096 recomputed in mpmath (40 digits) from the same formulas
    (catches cancellation/overflow in the numpy forms);
  * derivation check, independent of the series: a Chebyshev collocation solve of the PDE in the phi formulation
    (w, phi unknowns, NOT the H formulation used for Shercliff's series) at Ha = 1, 10, 30 on N = 32, 40, 48 points
    per direction, compared with the series (Q, w(0,0), phi or Delta phi);
  * limits: L -> infinity reproduces the Hartmann closed form (Shercliff: open circuit; Hunt: short circuit).

    python3 -B tools/fluids_mms/f13a_reference.py tools/fluids_mms/f13a_reference/analytic_reference.json
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "benchmarks" / "fluids" / "cases"))
import f13a_mhd_series as S  # noqa: E402

# Physical parameters of every F13a analytic case: deliberately mu != 1, sigma != 1,
# so that Ha = B0 a sqrt(sigma/mu) differs from B0, B0 sqrt(sigma), B0/sqrt(mu) (catches misplaced sigma or mu).
MU, SIGMA, G, A, L = 0.5, 2.0, 1.0, 1.0, 1.0
HA_HARTMANN = [0.0, 1.0, 10.0, 100.0, 1000.0]
HA_DUCT = [0.0, 10.0, 100.0, 1000.0]          # Ha = 0 only for Shercliff (the Poiseuille duct limit)


def B0_of(Ha):
    return Ha / (A * math.sqrt(SIGMA / MU))


def partial_sums(fn, kmax_pow=21):
    Ks = [2 ** p for p in range(14, kmax_pow + 1)]
    vals = [fn(K) for K in Ks]
    inc = abs(vals[-1] - vals[-2])
    return vals[-1], 10 * inc, dict(zip(Ks, vals))


# ---------------------------------------------------------------------------------------------- mpmath re-check
def mp_check(kind, Ha, K=4096):
    import mpmath as mp
    mp.mp.dps = 40
    mu, sig, G_, a, L_ = map(mp.mpf, (MU, SIGMA, G, A, L))
    B0 = mp.mpf(Ha) / (a * mp.sqrt(sig / mu))
    m = B0 * mp.sqrt(sig / mu)
    out = {}
    if kind == "hunt":
        s = mp.sqrt(mu / sig)
        Q = wc = ph = M = mp.mpf(0)
        for k in range(K):
            al = (k + mp.mpf(1) / 2) * mp.pi / a
            sg = 1 if k % 2 == 0 else -1
            c = (G_ / mu) * 2 * sg / (al * a)
            lam = mp.sqrt(al * al + 1j * m * al)
            vp = c / (al * al + m * m)
            C = -(vp / 2) * (1 - 1j * m / al)
            th, sech = mp.tanh(lam * L_), 1 / mp.cosh(lam * L_)
            Q += (2 * sg / al) * (2 * L_ * vp + 4 * mp.re(C * th / lam))
            wc += vp + 2 * mp.re(C * sech)
            ph += s * 2 * mp.re(C * (1j * lam / al) * th)
            M += (4 * sig * s / L_) * sg * mp.re(1j * C * (L_ - th / lam))
        out = {"Q": Q, "w_center": wc, "phi_side": ph, "wall_current_moment": M}
    else:
        Q = wc = dp = mp.mpf(0)
        kk = mp.sqrt(mu * sig)
        for n in range(K):
            b = (n + mp.mpf(1) / 2) * mp.pi / L_
            sg = 1 if n % 2 == 0 else -1
            c = (G_ / mu) * 2 * sg / (b * L_)
            root = mp.sqrt(m * m + 4 * b * b)
            rp, rm = (-m + root) / 2, (-m - root) / 2
            ea, eb = mp.e ** (2 * a * rm), mp.e ** (-2 * a * rp)
            D = 1 - ea * eb
            P, R = (1 - ea) / D, (1 - eb) / D
            amp = c / b ** 2
            Iy = 2 * a - P * (1 - eb) / rp - R * (1 - ea) / (-rm)
            Q += amp * Iy * 2 * sg / b
            wc += amp * (1 - P * mp.e ** (-rp * a) - R * mp.e ** (rm * a))
            Jx0 = -kk * amp * (P * rp * mp.e ** (-a * rp) + R * rm * mp.e ** (a * rm))
            w0 = amp * (1 - P * mp.e ** (-a * rp) - R * mp.e ** (a * rm))
            dp += 2 * (-Jx0 / sig - B0 * w0) * sg / b
        out = {"Q": Q, "w_center": wc, "dphi_side": dp}
    return {k: float(v) for k, v in out.items()}


# ---------------------------------------------------------------------------------------------- collocation
def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.r_[2, np.ones(N - 1), 2] * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def collocation(kind, Ha, N):
    """Solve mu Lap w - sigma B0 phi_x - sigma B0^2 w + G = 0, Lap phi + B0 w_x = 0 on [-L,L]x[-a,a] (Chebyshev)."""
    B0 = B0_of(Ha)
    D, x = cheb(N)
    Dx, Dy = D / L, D / A
    xs, ys = L * x, A * x
    I = np.eye(N + 1)
    DX, DY = np.kron(Dx, I), np.kron(I, Dy)           # index (i, j) -> i*(N+1)+j, x_i, y_j
    LAP = DX @ DX + DY @ DY
    n = (N + 1) ** 2
    Z = np.zeros((n, n))
    Id = np.eye(n)
    Aw = np.block([[MU * LAP - SIGMA * B0 ** 2 * Id, -SIGMA * B0 * DX], [B0 * DX, LAP]])
    rhs = np.r_[-G * np.ones(n), np.zeros(n)]
    Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
    Xf, Yf = Xg.ravel(), Yg.ravel()
    onx = np.isclose(np.abs(Xf), L)
    ony = np.isclose(np.abs(Yf), A)
    bnd = onx | ony
    for k in np.where(bnd)[0]:                         # w = 0
        Aw[k] = 0
        Aw[k, k] = 1
        rhs[k] = 0
    for k in np.where(bnd)[0]:
        r = n + k
        Aw[r] = 0
        rhs[r] = 0
        if kind == "hunt" and ony[k]:
            Aw[r, n + k] = 1                           # phi = 0 on the Hartmann walls (corners included)
        elif onx[k]:
            Aw[r, n:] = DX[k]                          # J_x = 0 with w = 0: phi_x = 0
        else:
            Aw[r, n:] = DY[k]                          # J_y = 0: phi_y = 0
    if kind == "shercliff":                            # fix the constant: replace one corner row by phi(corner) = 0
        k = int(np.where(onx & ony)[0][0])
        Aw[n + k] = 0
        Aw[n + k, n + k] = 1
    sol = np.linalg.solve(Aw, rhs)
    w, ph = sol[:n].reshape(N + 1, N + 1), sol[n:].reshape(N + 1, N + 1)
    # Clenshaw-Curtis weights for Q
    wcc = clenshaw_curtis(N)
    Q = float(L * A * wcc @ w @ wcc)
    i0 = N // 2
    out = {"Q": Q, "w_center": float(w[i0, i0])}
    if kind == "hunt":
        out["phi_side"] = float(ph[0, i0])             # x_0 = +L
        # moment of J_y = -sigma phi_y on y = +a (j = 0): int J_y x/L dx
        phiy = (ph @ Dy.T)[:, 0]
        out["wall_current_moment"] = float(L * wcc @ (-SIGMA * phiy * xs / L))
    else:
        out["dphi_side"] = float(ph[0, i0] - ph[N, i0])
    return out


def clenshaw_curtis(N):
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1 / (N ** 2 - 1)
        for k in range(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
        v -= np.cos(N * theta[1:-1]) / (N ** 2 - 1)
    else:
        w[0] = w[N] = 1 / N ** 2
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
    w[1:-1] = 2 * v / N
    return w


# ---------------------------------------------------------------------------------------------- main
def main():
    out_path = Path(sys.argv[1])
    res = {"format": "axionfem.fluids.f13a.analytic_reference/1",
           "parameters": {"mu": MU, "sigma": SIGMA, "G": G, "a": A, "L": L},
           "hartmann": {}, "shercliff": {}, "hunt": {}, "checks": {}}
    for Ha in HA_HARTMANN:
        B0 = B0_of(Ha)
        for mode in ("inplane", "duct"):
            for circ in ("short", "open"):
                h = S.Hartmann(MU, SIGMA, B0, G, A, mode, circ)
                res["hartmann"][f"{mode}_{circ}_Ha{Ha:g}"] = {
                    "Ha": Ha, "B0": B0, "E": h.E, "Q": h.Q, "U_center": float(h.U(0.0)),
                    "wall_shear": float(h.mu * h.dU(-A)), "net_current": h.net_current(),
                    "joule": h.joule(), "viscous": h.viscous(), "power_in": G * h.Q}
    for kind, cls, qs in (("shercliff", S.Shercliff, ("Q", "w_center", "dphi_side")),
                          ("hunt", S.Hunt, ("Q", "w_center", "phi_side", "wall_current_moment"))):
        for Ha in HA_DUCT:
            if kind == "hunt" and Ha == 0:
                continue
            obj = cls(MU, SIGMA, B0_of(Ha), G, L, A, K=2 ** 21)
            ent = {"Ha": Ha, "B0": B0_of(Ha)}
            for q in qs:
                fn = {"Q": obj.Q, "w_center": obj.w_center}.get(q) or getattr(obj, q)
                v, tb, sums = partial_sums(fn)
                ent[q] = v
                ent[q + "_truncation_bound"] = tb
                ent[q + "_partial_sums"] = {str(k): s for k, s in sums.items()}
            mpv = mp_check(kind, Ha)
            ent["mpmath_K4096"] = mpv
            ent["numpy_K4096"] = {q: ({"Q": obj.Q, "w_center": obj.w_center}.get(q) or getattr(obj, q))(4096)
                                  for q in qs}
            ent["mp_vs_numpy_rel_max"] = max(abs(mpv[q] - ent["numpy_K4096"][q]) / max(abs(mpv[q]), 1e-12)
                                             for q in qs)
            res[kind][f"Ha{Ha:g}"] = ent
            print(kind, Ha, {q: ent[q] for q in qs}, "mp rel", ent["mp_vs_numpy_rel_max"], flush=True)
    # independent derivation check (collocation, phi formulation)
    coll = {}
    for kind in ("shercliff", "hunt"):
        for Ha in (1.0, 10.0, 30.0):
            obj = (S.Shercliff if kind == "shercliff" else S.Hunt)(MU, SIGMA, B0_of(Ha), G, L, A, K=2 ** 20)
            ser = {"Q": obj.Q(), "w_center": obj.w_center()}
            ser.update({"dphi_side": obj.dphi_side()} if kind == "shercliff" else
                       {"phi_side": obj.phi_side(), "wall_current_moment": obj.wall_current_moment()})
            rows = {}
            for N in (32, 40, 48):
                c = collocation(kind, Ha, N)
                rows[N] = {q: c[q] for q in ser}
            dev = {q: abs(rows[48][q] - ser[q]) / abs(ser[q]) for q in ser}
            coll[f"{kind}_Ha{Ha:g}"] = {"series": ser, "collocation": rows, "rel_dev_N48": dev}
            print("collocation", kind, Ha, dev, flush=True)
    res["checks"]["collocation"] = coll
    lim = {}
    for Ha in (1.0, 10.0, 100.0):
        B0 = B0_of(Ha)
        sh = S.Shercliff(MU, SIGMA, B0, G, L=40.0, a=A, K=2 ** 20).w_center()
        hu = S.Hunt(MU, SIGMA, B0, G, L=40.0, a=A, K=2 ** 20).w_center()
        lim[f"Ha{Ha:g}"] = {"shercliff_L40_vs_hartmann_duct_open": abs(sh / S.Hartmann(MU, SIGMA, B0, G, A, "duct", "open").U(0.0) - 1),
                            "hunt_L40_vs_hartmann_duct_short": abs(hu / S.Hartmann(MU, SIGMA, B0, G, A, "duct", "short").U(0.0) - 1)}
    res["checks"]["wide_duct_limit_rel"] = lim
    print("wide-duct limits", lim)
    import mpmath
    res["provenance"] = {"script": "tools/fluids_mms/f13a_reference.py",
                         "script_sha256": hashlib.sha256((HERE / "f13a_reference.py").read_bytes()).hexdigest(),
                         "series_sha256": hashlib.sha256((HERE.parents[1] / "benchmarks/fluids/cases/f13a_mhd_series.py").read_bytes()).hexdigest(),
                         "python": platform.python_version(), "numpy": np.__version__, "mpmath": mpmath.__version__,
                         "produced_by": "tools/fluids_mms/f13a_reference.py (mpmath), out of band"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
