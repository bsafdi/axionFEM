"""Signal, readout noise and optimal scan (C++ implementation).

Conventions: Foster, Rodd & Safdi PSD normalization; Chaudhuri et al. optimal
resonant scan; the package's benchmark readout parameters and couplings.  See
docs/physics/scan_and_signal.md for every formula.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from . import _core

YEAR = 365.25 * 86400.0
# Illustrative noise benchmarks (resonator Q, temperature T [K], amplifier noise eta_A); a study replaces them with
# the parameters of its own readout (a design's "benchmarks" entry may be a dict {name, Q, T, eta_A}).
BENCHMARKS = {
    "optimistic": {"Q": 2e7, "T": 0.010, "eta_A": 0.1},
    "conservative": {"Q": 1e6, "T": 0.030, "eta_A": 10.0},
    "conservative_solenoid": {"Q": 1e5, "T": 0.030, "eta_A": 10.0},
}
MODELS = {"DFSZ": 8.0 / 3.0, "DFSZ-II": 2.0 / 3.0, "KSVZ": 0.0}


def g_qcd(m_eV: Iterable[float], model: str = "DFSZ", C_gamma: float = 1.77) -> np.ndarray:
    """|g_agg| [GeV^-1] on the QCD line (E/N from model, lattice C_gamma = 1.77 by default)."""
    return np.asarray(_core.g_qcd_GeV(list(np.atleast_1d(m_eV).astype(float)), MODELS[model], C_gamma))


def mass_to_freq(m_eV: float) -> float:
    return _core.mass_to_freq(m_eV)


def freq_to_mass(f_Hz: float) -> float:
    return _core.freq_to_mass(f_Hz)


def reach(Psi_SI: float, L_H: float, benchmark: str | dict = "optimistic", live_time_yr: float = 3.0,
          f_min: float = 120.9e3, f_max: float = 1.0e6, allocation: str = "target", n: int = 200,
          model: str = "DFSZ", C_gamma: float = 1.77, Q_axion: float = 6.8e5, Theta: float = 2.71,
          rho: float = 0.3) -> dict:
    """Expected 95% upper-limit reach g_95(m_a) of a detector (Psi [H T m^2], L_p [H]) for a noise
    benchmark, with the optimal resonant scan over [f_min, f_max] in the live time. The default band, live time
    and model (DFSZ over 0.5-4.1 neV in 3 years) and the benchmarks are illustrative: set them for your study.
    allocation: 'target' (time distributed to follow the QCD line) or 'log_uniform'.

    The result carries a ``"band"`` dict that echoes what was scanned -- mass and frequency
    range, number of mass points, allocation, live time, the benchmark (Q, T, eta_A), the
    detector (Psi, L, F) and the scan conventions (Q_axion, Theta, rho, v0, vobs, the QCD
    line) -- so that an output document is self-describing.

    ``benchmark["eta_A"]`` may also be a callable ``eta_A(f_Hz)``, e.g. an
    :class:`axionfem.squid.Chain` (``squid.benchmark(chain, Q, T_K)``).  The scan then runs
    :func:`reach_dispersive` with constant ``Psi``, ``L``, ``Q``, ``T`` and the frequency-dependent
    ``eta_A``.
    That route also applies reach_dispersive's local-validity guards (noise bandwidth < 0.1 f, slow
    variation), which the scalar route does not, so a very low ``Q`` raises there; and its ``band``
    entries for Q, T, eta_A, L, Psi are [f_min, f_max] endpoint values rather than scalars."""
    b = BENCHMARKS[benchmark] if isinstance(benchmark, str) else benchmark
    if callable(b["eta_A"]):
        out = reach_dispersive([f_min, f_max], Psi_SI, L_H, Q=b["Q"], T_K=b["T"], eta_A=b["eta_A"],
                               live_time_yr=live_time_yr, f_min=f_min, f_max=f_max, allocation=allocation,
                               n=n, model=model, C_gamma=C_gamma, Q_axion=Q_axion, Theta=Theta, rho=rho,
                               label=str(b.get("label", getattr(b["eta_A"], "label", "eta_A(f)"))))
        out["band"].update(benchmark=benchmark if isinstance(benchmark, str) else "custom",
                           eta_A_source=out["band"]["label"])
        if allocation == "log_uniform":
            # reach_dispersive stores its dimensionless normalisation there; the scalar route
            # reports the live time for log-uniform
            out["band"]["t_req_s"] = live_time_yr * YEAR
        return out
    m1, m2 = freq_to_mass(f_min), freq_to_mass(f_max)
    out = dict(_core.reach_local(Psi_SI, L_H, b["Q"], b["T"], b["eta_A"], m1, m2, live_time_yr * YEAR, allocation, n,
                                 MODELS[model], C_gamma, 5.70e-3, Q_axion, Theta, rho))
    out["band"] = dict(out["band"], benchmark=benchmark if isinstance(benchmark, str) else "custom",
                       model=model, kernel="local")
    return out


def psd(Psi_SI: float, L_H: float, f0_Hz: float, benchmark: str | dict = "optimistic", g_GeV: float | None = None,
        m_a_eV: float | None = None, span: float = 20.0, n: int = 2001, rho: float = 0.3) -> dict:
    """Signal and noise PSDs (Foster normalization) around a tuning f0 for an axion of mass m_a
    (default: on resonance) and coupling g (default: DFSZ).  span: half-width in units of the
    sensitivity bandwidth estimate f0/Q * 2T/(eta hf0) (clipped to f0/4).  A callable
    ``eta_A(f)`` (e.g. a squid.Chain) is evaluated at f0."""
    b = BENCHMARKS[benchmark] if isinstance(benchmark, str) else benchmark
    if callable(b["eta_A"]):
        b = dict(b, eta_A=float(b["eta_A"](f0_Hz)))
    m_a = m_a_eV if m_a_eV is not None else freq_to_mass(f0_Hz)
    g = g_GeV if g_GeV is not None else float(g_qcd([m_a])[0])
    h, kB = 6.62607015e-34, 1.380649e-23
    bw = f0_Hz / b["Q"] * max(1.0, 2 * kB * b["T"] / (b["eta_A"] * h * f0_Hz))
    half = min(span * bw, f0_Hz / 4)
    f = np.linspace(f0_Hz - half, f0_Hz + half, n)
    # resolve the axion line (width ~ f_a / Q_axion, one-sided above f_a)
    fa = mass_to_freq(m_a)
    f = np.unique(np.concatenate([f, fa * (1 + np.linspace(0, 8e-6, 801))]))
    return dict(_core.psd_at_tuning(Psi_SI, L_H, b["Q"], b["T"], b["eta_A"], f0_Hz, m_a, g, list(f), rho))


def Q_axion_shm(v0_kms: float = 220.0, vobs_kms: float = 232.0) -> float:
    """Exact SHM Q_axion = int f(v)^2/v dv (commonly rounded to 6.8e5)."""
    return _core.Q_axion_shm(v0_kms, vobs_kms)


# ===================================================================== finite band (spec Sec. 6)
def _log_interp(f_new: np.ndarray, f: np.ndarray, y: np.ndarray, *, log_y: bool = True) -> np.ndarray:
    """Interpolate y(f) at f_new, linear in log f and (by default) in log y, with the end
    slopes continued outside the table.

    ``log_y`` is honoured only when every sample is positive and finite; a table that changes
    sign or contains an infinity (``L(omega)`` above a self-resonance, a radiation-free
    ``Q = omega L/R`` with ``R = 0``) is interpolated linearly instead."""
    lf, lx = np.log(f), np.log(f_new)
    if log_y and not bool(np.all((y > 0) & np.isfinite(y))):
        log_y = False
    ly = np.log(y) if log_y else y
    i = np.clip(np.searchsorted(lf, lx) - 1, 0, lf.size - 2)
    t = (lx - lf[i]) / (lf[i + 1] - lf[i])
    v = ly[i] + t * (ly[i + 1] - ly[i])
    return np.exp(v) if log_y else v


def _as_curve(value, f_grid: np.ndarray, f_table: np.ndarray, name: str, *,
              log_y: bool = True) -> np.ndarray:
    """Evaluate a detector property on ``f_grid``.

    ``value`` may be a number (constant), a callable of ``f`` [Hz], a 1-D array on the table's
    own grid ``f_table``, or a ``(f_Hz, values)`` pair with its own grid.  Interpolation is
    linear in ``log f`` and, unless ``log_y=False``, in ``log y``."""
    if callable(value):
        return np.array([float(value(float(x))) for x in f_grid])
    v = np.asarray(value, dtype=float)
    if v.ndim == 0:
        return np.full(f_grid.shape, float(v))
    if v.ndim == 1:
        if v.size != f_table.size:
            raise ValueError(f"{name}: a 1-D table has {v.size} values but f_Hz has {f_table.size}; "
                             "give a (f_Hz, values) pair to use a different grid")
        return _log_interp(f_grid, f_table, v, log_y=log_y)
    if v.ndim == 2 and v.shape[0] == 2:
        return _log_interp(f_grid, np.asarray(v[0], float), np.asarray(v[1], float), log_y=log_y)
    raise ValueError(f"{name}: give a number, a callable of f [Hz], a 1-D table on f_Hz, "
                     "or a (f_Hz, values) pair")


def finite_band(readout, f_min_Hz: float, f_max_Hz: float, *, n_settings: int = 40,
                n_targets: int = 60, xi_factors: Sequence[float] = (1.0,), match_xi: bool = True,
                model: str = "DFSZ", C_gamma: float = 1.77, Theta: float = 2.71, rho: float = 0.3,
                Q_axion: float | None = None, stat: str = "weak", n_v: int = 96,
                half_widths: float = 60.0, decades: float = 3.0, points_per_decade: float = 48.0,
                band_points: int = 200, band_margin: float = 4.0,
                reduced: bool = True, reduced_R_exp: float = 0.0, narrow_line: bool = True,
                local_schedule: bool = True, convergence: bool = False,
                live_time_yr: float | None = None) -> dict:
    """Plan a scan with the **exact** circuit noise N_V(omega) instead of the reduced pole.

    Driven by ``axionfem.circuit``: for each *setting* -- a
    tuning capacitance (hence an ``omega_0``) and a coupling ``xi`` -- the readout network is
    tuned, ``N_V(omega)`` is tabulated exactly
    (:meth:`~axionfem.circuit.Readout.noise_curve`), and the C++ kernel

        ``K_j(m) = (1/2pi) int domega h(S_Va(omega)/N_{V,j}(omega))``   [1/s]

    is built for every target mass.  The minimum-total-time schedule is the LP, and the
    reach follows from ``sum_j t_j K_j(m) = Theta``.  The pickup's ``Psi`` is taken from the
    readout at each tuning, so a frequency-dependent ``Psi(omega)`` is used if the readout has
    one.

    With ``reduced=True`` the same settings are also evaluated through the single-pole
    reduction of each setting (``L_eff``, ``Q``, ``T_eff``, ``eta_A_eff``, ``xi``), which is what
    the rest of the package uses; ``comparison`` then quantifies the error of that reduction.
    ``reduced_R_exp`` gives the reduced pole a power-law resistance
    ``R(omega) = R_0 (omega/omega_0)^{R_exp}`` within a tuning (0 = constant, 1/2 for a
    normal-metal skin effect, 2 for a superconductor), which is the one piece of dispersion the
    single pole can carry.  ``narrow_line=True`` adds the narrow-line kernel, evaluated with the exact
    ``N_V(omega_a)``.  ``convergence=True`` repeats the kernel with half the ``N_V`` grid density
    and half the line nodes, so the numerical error of the exact kernel is reported rather than
    assumed.

    Parameters
    ----------
    readout
        an :class:`axionfem.circuit.Readout` with a tuning capacitor (``tuner``) -- it is tuned in
        place, setting by setting, and left at the last setting -- **or** a callable
        ``f0_Hz -> Readout`` that builds the network for each tuning.  The factory form is the one
        to use when the network itself depends on the tuning (a surface resistance quoted at the
        operating point, a full-wave ``Z_p(omega)`` and ``Psi(omega)``, a different input coil per
        octave); ``Psi(omega)`` is read from the first instance, the pickup being the same.
    n_settings, xi_factors
        ``n_settings`` log-spaced tunings over [f_min, f_max] times one setting per entry of
        ``xi_factors`` (multiples of ``xi_opt``; ``match_xi`` rescales ``M`` accordingly).
    n_targets
        log-spaced target masses over the same band; the LP covers these exactly and the
        returned reach is evaluated on them.
    live_time_yr
        if given, the LP schedule is also reported rescaled to this live time
        (``coverage_at_live_time`` = the Theta reached per target, in units of ``Theta``).

    Notes
    -----
    ``local_schedule=True`` also discretises the *local* optimal-scan dwell density onto the
    settings and scores it with the exact kernel: a
    plan made from the Lorentzian scan rate covers at ``~Theta`` only while the sensitivity band
    is narrow.

    Returns a dict with ``band`` (the echo), ``settings``, ``targets``, ``K_per_s``,
    ``schedule``, ``reach``, and -- when asked for -- ``reduced``/``narrow_line``/
    ``local_schedule``/``convergence`` and ``comparison``.
    """
    from . import circuit as ck

    factory = readout if callable(readout) else None
    if factory is not None:
        readout = factory(float(f_min_Hz))
        if not isinstance(readout, ck.Readout):
            raise TypeError("scan.finite_band: the readout factory must return a "
                            f"circuit.Readout, got {type(readout).__name__}")
    if readout.tuner is None:
        raise ValueError("scan.finite_band: the readout has no tuning capacitor, so it has no "
                         "settings to schedule; build it with circuit.series_readout(...)")
    if not (0 < f_min_Hz < f_max_Hz):
        raise ValueError(f"scan.finite_band: need 0 < f_min < f_max (got {f_min_Hz}, {f_max_Hz})")
    if n_settings < 1 or n_targets < 1:
        raise ValueError("scan.finite_band: need n_settings >= 1 and n_targets >= 1")
    if not len(xi_factors):
        raise ValueError("scan.finite_band: xi_factors must hold at least one factor")
    Qa = Q_axion_shm() if Q_axion is None else float(Q_axion)   # exact line <-> exact kernel

    f_set = np.geomspace(f_min_Hz, f_max_Hz, n_settings) if n_settings > 1 else np.array([f_min_Hz])
    m_t = np.geomspace(freq_to_mass(f_min_Hz), freq_to_mass(f_max_Hz), n_targets)
    g_t = g_qcd(m_t, model=model, C_gamma=C_gamma)

    S: dict[str, list] = {k: [] for k in ("f0_Hz", "omega0_rad_s", "xi", "xi_opt", "L_eff_H", "Q_loaded",
                                          "T_eff_K", "eta_A_eff", "N_V_at_omega0_V2_s", "Psi_SI_H_T_m2",
                                          "C_tuned_F", "participation", "xi_factor")}
    omegas: list[list[float]] = []
    NVs: list[list[float]] = []
    omegas_c: list[list[float]] = []
    NVs_c: list[list[float]] = []
    omegas_b: list[list[float]] = []
    NVs_b: list[list[float]] = []
    # a band-wide grid merged into every setting's noise curve: it resolves structure that sits
    # at a FIXED frequency (a self-resonance of Z_p) rather than at a fixed detuning
    w_band = (np.geomspace(2.0 * math.pi * f_min_Hz / band_margin,
                           2.0 * math.pi * f_max_Hz * band_margin, band_points)
              if band_points > 1 else np.array([]))
    r0 = readout                  # the pickup is the same in every instance: Psi(omega) comes here
    M_0 = readout.squid.M_H
    for fac in xi_factors:
        for f0 in f_set:
            if factory is not None:
                readout = factory(float(f0))
                M_0 = readout.squid.M_H
            else:
                readout.squid.M_H = M_0
            C = readout.tune(float(f0))
            w0 = 2.0 * math.pi * float(f0)
            if match_xi:
                readout.match_xi(w0, xi=float(fac) * readout.equivalent(w0)["xi_opt"])
            eq = readout.equivalent(w0)
            nc = readout.noise_curve(w0, half_widths=half_widths, decades=decades,
                                     points_per_decade=points_per_decade, extra_omega=w_band)
            if convergence:
                ncc = readout.noise_curve(w0, half_widths=half_widths, decades=decades,
                                          points_per_decade=points_per_decade / 2,
                                          extra_omega=w_band)
                omegas_c.append([float(x) for x in ncc["omega_rad_s"]])
                NVs_c.append([float(x) for x in ncc["N_V_V2_s"]])
                ncb = readout.noise_curve(w0, half_widths=half_widths, decades=decades,
                                          points_per_decade=points_per_decade,
                                          extra_omega=w_band[::2] if w_band.size else None)
                omegas_b.append([float(x) for x in ncb["omega_rad_s"]])
                NVs_b.append([float(x) for x in ncb["N_V_V2_s"]])
            omegas.append([float(x) for x in nc["omega_rad_s"]])
            NVs.append([float(x) for x in nc["N_V_V2_s"]])
            S["f0_Hz"].append(float(f0))
            S["omega0_rad_s"].append(w0)
            S["xi"].append(eq["xi"])
            S["xi_opt"].append(eq["xi_opt"])
            S["xi_factor"].append(float(fac))
            S["L_eff_H"].append(eq["L_eff_H"])
            S["Q_loaded"].append(eq["Q_loaded"])
            S["T_eff_K"].append(eq["T_eff_K"])
            S["eta_A_eff"].append(eq["eta_A_eff"])
            S["participation"].append(eq["participation"])
            S["N_V_at_omega0_V2_s"].append(float(nc["N_V_at_omega0_V2_s"]))
            S["Psi_SI_H_T_m2"].append(readout.Psi_at(w0))
            S["C_tuned_F"].append(C)
    if factory is None:
        readout.squid.M_H = M_0
    S = {k: np.asarray(v) for k, v in S.items()}       # type: ignore[assignment]
    psi_t = np.array([r0.Psi_at(2.0 * math.pi * mass_to_freq(float(m))) for m in m_t])

    K = np.array([_core.kernel_row_noise(float(psi_t[k]), float(m_t[k]), float(g_t[k]), omegas, NVs,
                                         stat, n_v, Qa, Theta, rho) for k, m in enumerate(m_t)])
    sched = dict(_core.lp_schedule([[float(x) for x in row] for row in K], Theta))
    sched["t_s"] = np.asarray(sched["t_s"])
    reach_out = dict(_core.reach_band([float(x) for x in psi_t], [float(x) for x in m_t], omegas, NVs,
                                      [float(x) for x in sched["t_s"]], stat, n_v, Qa, Theta, rho))

    band = dict(reach_out["band"])
    band.update(allocation="lp (S6.4)", kernel="exact N_V (C4.5)", statistic=stat,
                n_settings=int(len(omegas)), n_targets=int(n_targets), model=model, C_gamma=C_gamma,
                xi_factors=[float(x) for x in xi_factors], n_v=n_v,
                noise_grid={"half_widths": half_widths, "decades": decades,
                            "points_per_decade": points_per_decade, "band_points": band_points,
                            "band_margin": band_margin, "points": int(len(omegas[0]))},
                live_time_s=float(sched["t_total_s"]),
                live_time_yr=float(sched["t_total_s"]) / YEAR,
                readout=r0.label, readout_per_setting=factory is not None,
                Psi_frequency_dependent=bool(callable(r0.Psi_SI)))
    out: dict = {"band": band, "settings": S,
                 "targets": {"m_eV": m_t, "f_Hz": np.array([mass_to_freq(float(m)) for m in m_t]),
                             "g_target_GeV": g_t},
                 "K_per_s": K, "schedule": sched, "reach": reach_out}
    if live_time_yr is not None:
        scale = live_time_yr * YEAR / sched["t_total_s"]
        out["coverage_at_live_time"] = (K @ sched["t_s"]) * scale / Theta
        band["requested_live_time_yr"] = float(live_time_yr)

    if reduced:
        Kr = np.empty_like(K)
        for j in range(len(omegas)):
            for k in range(n_targets):
                Kr[k, j] = _core.kernel_row(float(S["Psi_SI_H_T_m2"][j]), float(S["L_eff_H"][j]),
                                            float(S["Q_loaded"][j]), float(S["T_eff_K"][j]),
                                            float(S["eta_A_eff"][j]), float(m_t[k]), float(g_t[k]),
                                            [float(S["omega0_rad_s"][j])], [float(S["xi"][j])],
                                            stat, n_v, reduced_R_exp, Qa, Theta, rho)[0]
        sched_r = dict(_core.lp_schedule([[float(x) for x in row] for row in Kr], Theta))
        sched_r["t_s"] = np.asarray(sched_r["t_s"])
        # the reduced schedule, scored with the EXACT kernel: what the single-pole plan really buys
        cov_r_exact = (K @ sched_r["t_s"]) / Theta
        cov_exact = (K @ sched["t_s"]) / Theta                 # = 1 at the binding targets
        cov_r_same = (Kr @ sched["t_s"]) / Theta               # the pole's estimate of the same plan
        g95_r = np.array([_g95_from_coverage(Kr[k] @ sched_r["t_s"], float(g_t[k]), Theta)
                          for k in range(n_targets)])
        # exactly on resonance: evaluate both kernels at m = freq_to_mass(f0_j) of each setting
        # (the nearest *target* can be many linewidths away for a high-Q resonator, where the two
        # noise models legitimately differ while K is negligible)
        on_exact, on_red = np.empty(len(omegas)), np.empty(len(omegas))
        for j in range(len(omegas)):
            mj = freq_to_mass(float(S["f0_Hz"][j]))
            gj = float(g_qcd([mj], model=model, C_gamma=C_gamma)[0])
            pj = float(S["Psi_SI_H_T_m2"][j])
            on_exact[j] = _core.kernel_row_noise(pj, mj, gj, [omegas[j]], [NVs[j]], stat, n_v,
                                                 Qa, Theta, rho)[0]
            on_red[j] = _core.kernel_row(pj, float(S["L_eff_H"][j]), float(S["Q_loaded"][j]),
                                         float(S["T_eff_K"][j]), float(S["eta_A_eff"][j]), mj, gj,
                                         [float(S["omega0_rad_s"][j])], [float(S["xi"][j])], stat,
                                         n_v, reduced_R_exp, Qa, Theta, rho)[0]
        out["reduced"] = {"K_per_s": Kr, "schedule": sched_r, "g95_GeV": g95_r,
                          "coverage_with_exact_kernel": cov_r_exact,
                          "coverage_of_exact_schedule": cov_r_same,
                          "K_on_resonance_per_s": on_red,
                          "K_on_resonance_exact_per_s": on_exact}
        sig = K > 1e-3 * K.max(axis=1, keepdims=True)   # significant *within each target row*
        out["comparison"] = {
            "kernel_max_rel_dev": float(np.max(np.abs(Kr[sig] / K[sig] - 1.0))) if sig.any() else 0.0,
            "kernel_rms_rel_dev": float(np.sqrt(np.mean((Kr[sig] / K[sig] - 1.0) ** 2))) if sig.any() else 0.0,
            "kernel_on_resonance_max_rel_dev": float(np.max(np.abs(on_red / on_exact - 1.0))),
            "t_total_ratio_reduced_over_exact": float(sched_r["t_total_s"] / sched["t_total_s"]),
            "min_coverage_of_reduced_schedule": float(cov_r_exact.min()),
            # same schedule, two noise models: the reach the pole would claim over the true one
            "g95_same_schedule_max_rel_dev": float(np.max(np.abs(
                (cov_exact / np.maximum(cov_r_same, 1e-300)) ** 0.25 - 1.0))),
            "g95_own_schedule_max_rel_dev": float(np.max(np.abs(g95_r / reach_out["g95_GeV"] - 1.0))),
            "reduced_R_exp": float(reduced_R_exp),
            "note": "reduced = the single-pole detector of readout_circuit.md Sec. 5, re-derived at "
                    "each setting; exact = N_V(omega) of (C4.5).  g95_same_schedule compares the two "
                    "noise models on the exact LP plan (the reach the pole would claim vs the true "
                    "one); min_coverage_of_reduced_schedule is the penalty of *planning* with the "
                    "pole (the LP enforces coverage = Theta for its own kernel, which is why "
                    "g95_own_schedule is trivially small).",
        }
    if local_schedule and n_settings > 2:
        # The local optimal-scan schedule (S5.5-S5.6) discretised onto the first xi family:
        # t_j = Delta omega_j / (domega_0/dt)_j with the reduced detector of setting j.  Scoring
        # it with the exact kernel is the spec Sec. 6 demonstration -- it must cover at ~Theta
        # for a detector whose sensitivity band is narrow, and fails at the band edges and
        # wherever tau varies strongly across the band.
        # the family closest to xi_opt: the local theory assumes the optimal coupling
        fac0 = float(min(xi_factors, key=lambda x: abs(math.log(float(x)))))
        sel = np.nonzero(S["xi_factor"] == fac0)[0]
        w0s = S["omega0_rad_s"][sel]
        edges = np.concatenate([[w0s[0]], np.sqrt(w0s[1:] * w0s[:-1]), [w0s[-1]]])
        dw = np.diff(edges)
        ms = np.array([freq_to_mass(float(x) / (2.0 * math.pi)) for x in w0s])
        gs = g_qcd(ms, model=model, C_gamma=C_gamma)
        rate = _core.scan_rate_curve(
            [float(x) for x in S["Psi_SI_H_T_m2"][sel]], [float(x) for x in S["L_eff_H"][sel]],
            [float(x) for x in S["Q_loaded"][sel]], [float(x) for x in S["T_eff_K"][sel]],
            [float(x) for x in S["eta_A_eff"][sel]], [float(x) for x in ms],
            [float(x) for x in gs], Qa, Theta, rho, True)
        t_loc = np.zeros(K.shape[1])
        t_loc[sel] = dw / np.asarray(rate)
        cov_loc = (K @ t_loc) / Theta
        fw = np.array([_core.bandwidths(float(w), float(q), float(t), float(e))["noise_doubling"] / w
                       for w, q, t, e in zip(w0s, S["Q_loaded"][sel], S["T_eff_K"][sel],
                                             S["eta_A_eff"][sel])])
        ft = np.array([mass_to_freq(float(x)) for x in m_t])
        inner = (ft > ft[0] * (1 + fw[0])) & (ft < ft[-1] * (1 - fw[-1]))
        out["local_schedule"] = {
            "t_s": t_loc, "t_total_s": float(t_loc.sum()),
            "coverage": cov_loc,
            "coverage_min": float(cov_loc.min()), "coverage_max": float(cov_loc.max()),
            "coverage_interior_min": float(cov_loc[inner].min()) if inner.any() else float("nan"),
            "coverage_interior_max": float(cov_loc[inner].max()) if inner.any() else float("nan"),
            "lp_over_local_time": float(sched["t_total_s"] / t_loc.sum()),
            "noise_doubling_over_f0": fw, "xi_factor": fac0,
            # how well the setting grid samples the sensitivity band: below ~2 the local
            # schedule leaves holes between settings and under-covers for a reason that has
            # nothing to do with the band edges (the LP still covers, by reallocating)
            "spacing_over_noise_doubling_width": (dw / w0s) / fw,
            "note": "coverage = (sum_j t_j K_j(m))/Theta with the EXACT kernel and the local "
                    "(Lorentzian) dwell density of the reduced detector; 'interior' excludes one "
                    "noise-doubling width at each band edge, where the local schedule necessarily "
                    "under-covers because the schedule stops but the band does not.  Check "
                    "spacing_over_noise_doubling_width: where it exceeds ~1/2 the settings are "
                    "spaced wider than the sensitivity band and the interior coverage dips between "
                    "them, which is under-sampling, not physics.",
        }
    if convergence:
        Kc = np.array([_core.kernel_row_noise(float(psi_t[k]), float(m_t[k]), float(g_t[k]), omegas_c,
                                             NVs_c, stat, n_v, Qa, Theta, rho)
                       for k in range(n_targets)])
        sig = K > 1e-3 * K.max(axis=1, keepdims=True)
        Kc2 = np.array([_core.kernel_row_noise(float(psi_t[k]), float(m_t[k]), float(g_t[k]), omegas,
                                              NVs, stat, max(n_v // 2, 8), Qa, Theta, rho)
                        for k in range(n_targets)])
        Kb = np.array([_core.kernel_row_noise(float(psi_t[k]), float(m_t[k]), float(g_t[k]), omegas_b,
                                             NVs_b, stat, n_v, Qa, Theta, rho)
                       for k in range(n_targets)])
        out["convergence"] = {
            "noise_grid_halved_max_rel_dev": float(np.max(np.abs(Kc[sig] / K[sig] - 1.0))),
            "band_grid_halved_max_rel_dev": float(np.max(np.abs(Kb[sig] / K[sig] - 1.0))),
            "n_v_halved_max_rel_dev": float(np.max(np.abs(Kc2[sig] / K[sig] - 1.0))),
            "note": "max |dK/K| over the significant kernel entries (K > 1e-3 of each target's "
                    "largest) when the detuning grid density, the band-wide grid, resp. the number "
                    "of line-integration nodes is halved.  The error of the quoted K is about a "
                    "third of a second-order number and far below the n_v one (Gauss-Legendre). A "
                    "large band_grid number means the readout has structure (a self-resonance) that "
                    "needs more band_points.",
        }
    if narrow_line:
        Knl = np.array([_core.kernel_narrow_line_noise(
            [float(psi_t[k])] * len(omegas), [float(m_t[k])] * len(omegas), [float(g_t[k])] * len(omegas),
            [float(_interp_loglog(omegas[j], NVs[j], 2.0 * math.pi * mass_to_freq(float(m_t[k]))))
             for j in range(len(omegas))], Qa, rho) for k in range(n_targets)])
        sig = K > 1e-3 * K.max()
        out["narrow_line"] = {
            "K_per_s": Knl,
            "max_rel_dev": float(np.max(np.abs(Knl[sig] / K[sig] - 1.0))) if sig.any() else 0.0,
        }
    return out


def _interp_loglog(x: Sequence[float], y: Sequence[float], xq: float) -> float:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    return float(np.exp(np.interp(math.log(xq), np.log(xa), np.log(ya))))


def _g95_from_coverage(Theta_at_g: float, g: float, Theta: float) -> float:
    """Weak-signal inversion Theta ~ g^4 of a coverage computed at coupling g."""
    return float("inf") if Theta_at_g <= 0 else g * (Theta / Theta_at_g) ** 0.25


# ============================================ beyond MQS / dispersive readout
def _pole_inductance(f, L_H, ft, explicit=None):
    """Series-tuned pole slope L_eff=L+(1/2)dL/dln(f), with a table-resolution check."""
    L = _as_curve(L_H, f, ft, "L_H", log_y=False)
    if explicit is not None:
        return _as_curve(explicit, f, ft, "L_eff_H", log_y=False)
    if callable(L_H):
        h = 1e-4
        d1 = (_as_curve(L_H, f*np.exp(h), ft, "L_H", log_y=False) -
              _as_curve(L_H, f*np.exp(-h), ft, "L_H", log_y=False))/(2*h)
        d2 = (_as_curve(L_H, f*np.exp(h/2), ft, "L_H", log_y=False) -
              _as_curve(L_H, f*np.exp(-h/2), ft, "L_H", log_y=False))/h
        if not np.allclose(L+d1/2, L+d2/2, rtol=1e-3, atol=1e-30):
            raise ValueError("pole slope is unresolved; refine the impedance or use finite_band")
        return L+d2/2
    if np.ndim(L_H) == 0:
        return L
    pair = np.asarray(L_H, float)
    grid, vals = (pair if pair.ndim == 2 and pair.shape[0] == 2 else (ft, pair))
    grid, vals = np.asarray(grid, float), np.asarray(vals, float)
    if np.all(vals == vals[0]):
        return L
    if len(grid) < 4 or np.any(np.diff(grid) <= 0):
        raise ValueError("pole slope needs at least four increasing sweep frequencies; refine the sweep")
    from scipy.interpolate import CubicSpline, PchipInterpolator
    x, xq = np.log(grid), np.log(f)
    cubic = CubicSpline(x, vals)
    shape = PchipInterpolator(x, vals)
    le = cubic(xq) + cubic(xq, 1)/2
    check = shape(xq) + shape(xq, 1)/2
    if np.any(np.abs(le-check) > .05*np.maximum(np.abs(le), 1e-30)):
        raise ValueError("pole slope changes by more than 5% between interpolants; refine the full-wave sweep or use finite_band")
    return le


def _exact_pole_inductance(f_rows, L_rows, L_eff_rows):
    """The tuned-pole inductance between sweep rows that carry the exact L_eff (3D: from dZ/domega), as a callable of f.

    At each row dL/dln(f) = 2 (L_eff - L) is exact, so a cubic Hermite interpolant of L in ln f with those slopes gives
    L_eff = L + (1/2) dL/dln(f) between the rows, equal to the rows' L_eff at the rows.  As the guard of the other path
    (spline against PCHIP within 5 %), it must agree with the linear interpolation of the rows' L_eff within 5 % at
    every frequency it is asked for, or the rows are too sparse for the pole slope."""
    from scipy.interpolate import CubicHermiteSpline

    x = np.log(np.asarray(f_rows, float))
    L = np.asarray(L_rows, float)
    Le = np.asarray(L_eff_rows, float)
    if x.size < 2 or np.any(np.diff(x) <= 0) or not (np.all(np.isfinite(L)) and np.all(np.isfinite(Le))):
        raise ValueError("pole slope needs at least two increasing sweep frequencies with a finite L_eff_H")
    cubic = CubicHermiteSpline(x, L, 2.0 * (Le - L))

    def L_eff(f: float) -> float:
        xq = math.log(f)
        value = float(cubic(xq) + 0.5 * cubic(xq, 1))
        linear = float(np.interp(xq, x, Le))
        if abs(value - linear) > 0.05 * max(abs(value), 1e-30):
            raise ValueError(f"pole slope changes by more than 5% between interpolants of the rows' exact L_eff at "
                             f"{f:.8g} Hz (cubic Hermite of L with the exact slopes: {value:.6g} H; linear L_eff: "
                             f"{linear:.6g} H); refine the full-wave sweep or use finite_band")
        return value

    return L_eff


def reach_dispersive(f_Hz, Psi_SI, L_H, *, Q=None, R_ohm=None, T_K: float = 0.02,
                     eta_A: float = 1.0, live_time_yr: float = 3.0, f_min: float | None = None,
                     f_max: float | None = None, allocation: str = "target", n: int = 200,
                     model: str = "DFSZ", C_gamma: float = 1.77, Q_axion: float = 6.8e5,
                     Theta: float = 2.71, rho: float = 0.3, n_quad: int = 2001,
                     label: str = "dispersive", L_eff_H=None) -> dict:
    """Reach of a detector whose Psi, L, Q, T and eta_A all depend on the tuning frequency.

    This is the local optimal scan (spec S5.1-S5.8) with the *constant* detector replaced by
    tabulated curves -- the beyond-MQS case, where a full-wave sweep gives
    ``Psi(omega)``, ``L(omega) = -Im Z/omega`` and ``R(omega) = R_rad + R_loss`` instead of the
    MQS ``Psi``, ``L_p`` and an assumed ``Q``.  At each tuning the scan rate (S5.2) and the
    dwell density are evaluated with the local values,

        ``tau(omega_0) propto 1/(domega_0/dt)|_{g_t}``  (target allocation, S5.5-S5.6),
        ``tau = t/(omega_0 ln(omega_2/omega_1))``       (log-uniform, S5.7),

    normalised so that ``int tau domega_0 = live_time``; then ``g_95`` is (S5.8) with
    ``F(omega) = |Psi(omega)|^4/L_eff(omega)^2``. Input ``L_H`` is reactive inductance;
    ``L_eff_H`` optionally supplies the tuned pole slope, otherwise computed as
    ``L + 0.5*dL/dln(f)``. Results name both ``L_reactive_H`` and ``L_eff_H``; the
    existing result ``L_H`` is an alias of the effective scan inductance. In the constant limit it reproduces
    :func:`reach` to round-off (``tests/test_finite_band.py``).

    Each of ``Psi_SI``, ``L_H``, ``Q``, ``R_ohm``, ``T_K``, ``eta_A`` may be a number, a callable
    of ``f`` [Hz], or a ``(f_Hz, values)`` pair; the tables are interpolated log-log in ``f``
    (the full-wave columns are smooth that way).  ``f_Hz`` is the table's own frequency grid and
    sets the default band.  Give ``Q`` **or** ``R_ohm`` (then ``Q = omega L_eff/R``, which is how a
    full-wave sweep's radiation and surface resistance enter).

    The optimized noise-doubling bandwidth must be below 0.1*f, and logarithmic
    changes of L_eff, Psi, Q, T and eta across it must be below 5%. Spline/PCHIP
    slope agreement is a consistency check, not an h/p or frequency-convergence proof.
    Otherwise use finite_band with a resolved tuned circuit.

    Frequencies where ``L(omega) <= 0`` are **above the readout's first self-resonance**: a
    series tuning capacitor cannot resonate the loop there ([MQS] Sec. I), so ``g95`` is
    reported as ``inf`` and they are listed in ``band["self_resonant_f_Hz"]``.
    """
    ft = np.asarray(f_Hz, dtype=float).ravel()
    if ft.size < 1 or np.any(ft <= 0):
        raise ValueError("reach_dispersive: f_Hz must hold positive frequencies")
    if np.any(~np.isfinite(ft)) or np.any(np.diff(ft) <= 0):
        raise ValueError("reach_dispersive: frequencies must be finite and strictly increasing")
    if (Q is None) == (R_ohm is None):
        raise ValueError("reach_dispersive: give exactly one of Q= and R_ohm= "
                         "(R_ohm -> Q = omega L/R, the full-wave route)")
    if allocation not in ("target", "log_uniform"):
        raise ValueError("reach_dispersive: allocation must be 'target' or 'log_uniform'")
    f1 = float(f_min if f_min is not None else ft[0])
    f2 = float(f_max if f_max is not None else ft[-1])
    if not (0 < f1 < f2):
        raise ValueError(f"reach_dispersive: need 0 < f_min < f_max (got {f1}, {f2})")

    if f1 < ft[0]*(1-1e-12) or f2 > ft[-1]*(1+1e-12):
        raise ValueError("requested band is outside the computed sweep; extend/refine the sweep")

    def detector_at(f: np.ndarray) -> dict[str, np.ndarray]:
        # L is interpolated linearly (not log-log): it changes sign at a self-resonance
        psi = _as_curve(Psi_SI, f, ft, "Psi_SI")
        Lreact = _as_curve(L_H, f, ft, "L_H", log_y=False)
        L = _pole_inductance(f, L_H, ft, L_eff_H)
        invalid_slope = (Lreact > 0) & ((L <= 0) | ~np.isfinite(L))
        if np.any(invalid_slope):
            i = np.flatnonzero(invalid_slope)[0]
            raise ValueError(f"nonpositive/nonfinite tuned-pole inductance at {f[i]:.8g} Hz; "
                             "refine the impedance sweep or use scan.finite_band")
        L = np.where(Lreact > 0, L, -np.abs(L))
        if Q is not None:
            Qv = _as_curve(Q, f, ft, "Q")
        else:
            R = _as_curve(R_ohm, f, ft, "R_ohm")
            with np.errstate(divide="ignore", invalid="ignore"):
                Qv = 2.0 * math.pi * f * L / R
        T = _as_curve(T_K, f, ft, "T_K", log_y=False)
        e = _as_curve(eta_A, f, ft, "eta_A")
        return {"Psi_SI": psi, "L_H": L, "Q": Qv, "T_K": T, "eta_A": e}

    # --- dwell density on a dense grid, normalised to the live time
    fq = np.geomspace(f1, f2, n_quad)
    wq = 2.0 * math.pi * fq
    dq = detector_at(fq)
    # Use the optimally noise-matched sensitivity width, not the resonator linewidth.
    valid = (dq["Q"] > 0) & np.isfinite(dq["Q"]) & (dq["L_H"] > 0) & (dq["Psi_SI"] > 0)
    widths = np.zeros(fq.size)
    for i in np.flatnonzero(valid):
        widths[i] = _core.bandwidths(float(wq[i]), float(dq["Q"][i]), float(dq["T_K"][i]),
                                      float(dq["eta_A"][i]))["noise_doubling"] / wq[i]
    variation = np.zeros(fq.size)
    for name in ("L_H", "Psi_SI", "Q", "T_K", "eta_A"):
        value = np.abs(dq[name])
        if np.all(value == 0): continue
        variation = np.maximum(variation, np.abs(np.gradient(np.log(np.maximum(value,1e-300)), np.log(fq))))
    broad = valid & (widths > .1)
    varying = valid & (variation*widths > .05)
    if np.any(broad) or np.any(varying):
        i = int(np.argmax(np.where(valid, np.maximum(widths/.1,variation*widths/.05), -1)))
        raise ValueError(f"local noise bandwidth/variation guard failed at {fq[i]:.8g} Hz: "
                         f"width/f={widths[i]:.6g} (limit 0.1), logarithmic variation="
                         f"{variation[i]*widths[i]:.6g} (limit 0.05); "
                         "use scan.finite_band with the tuned circuit")
    ok_q = (dq["L_H"] > 0) & (dq["Q"] > 0) & np.isfinite(dq["Q"]) & (dq["Psi_SI"] > 0)
    if not ok_q.any():
        raise ValueError(
            "reach_dispersive: the readout is above its self-resonance (L <= 0) or has no loss "
            f"everywhere in [{f1:.6g}, {f2:.6g}] Hz, so no tuning exists.  Narrow the band.")
    t_total = live_time_yr * YEAR
    if allocation == "target":
        mq = np.array([freq_to_mass(float(x)) for x in fq])
        gq = g_qcd(mq, model=model, C_gamma=C_gamma)
        rate = np.zeros(fq.size)
        idx = np.nonzero(ok_q)[0]
        rate[idx] = _core.scan_rate_curve(
            [float(x) for x in dq["Psi_SI"][idx]], [float(x) for x in dq["L_H"][idx]],
            [float(x) for x in dq["Q"][idx]], [float(x) for x in dq["T_K"][idx]],
            [float(x) for x in dq["eta_A"][idx]], [float(x) for x in mq[idx]],
            [float(x) for x in gq[idx]], Q_axion, Theta, rho, True)
        shape = np.zeros(fq.size)
        shape[idx] = 1.0 / rate[idx]
    else:
        shape = np.where(ok_q, 1.0 / (wq * math.log(wq[-1] / wq[0])), 0.0)
    # int tau domega = int (tau omega) dln(omega): Simpson in ln omega (error h^4) when the
    # whole band is tunable, trapezoid when part of it is above the self-resonance (shape = 0
    # there, and Simpson across that step would be wrong).
    u = np.log(wq)
    G = shape * wq
    if ok_q.all() and fq.size % 2 == 1:
        h = (u[-1] - u[0]) / (fq.size - 1)
        t_req = float(h / 3.0 * (G[0] + G[-1] + 4.0 * G[1:-1:2].sum() + 2.0 * G[2:-2:2].sum()))
        quad = f"Simpson in ln(omega), {fq.size} nodes"
    else:
        t_req = float(np.trapezoid(G, u)) if hasattr(np, "trapezoid") else float(np.trapz(G, u))
        quad = f"trapezoid in ln(omega), {fq.size} nodes" + ("" if ok_q.all() else
                                                             " (band partly above self-resonance)")
    if not (t_req > 0):
        raise ValueError("reach_dispersive: the dwell density integrates to zero; check the tables")
    tau_q = shape * (t_total / t_req)

    # --- reach on the requested mass grid
    m = np.geomspace(freq_to_mass(f1), freq_to_mass(f2), n)
    fm = np.array([mass_to_freq(float(x)) for x in m])
    dm = detector_at(fm)
    # tau at the targets is evaluated from its own formula (not interpolated from the
    # quadrature grid): only the normalisation t_total/t_req comes from the quadrature
    wm = 2.0 * math.pi * fm
    ok_m = (dm["L_H"] > 0) & (dm["Q"] > 0) & np.isfinite(dm["Q"]) & (dm["Psi_SI"] > 0)
    tau = np.zeros(fm.size)
    if allocation == "target":
        gm = g_qcd(m, model=model, C_gamma=C_gamma)
        im = np.nonzero(ok_m)[0]
        if im.size:
            rm = _core.scan_rate_curve(
                [float(x) for x in dm["Psi_SI"][im]], [float(x) for x in dm["L_H"][im]],
                [float(x) for x in dm["Q"][im]], [float(x) for x in dm["T_K"][im]],
                [float(x) for x in dm["eta_A"][im]], [float(x) for x in m[im]],
                [float(x) for x in gm[im]], Q_axion, Theta, rho, True)
            tau[im] = (t_total / t_req) / np.asarray(rm)
    else:
        tau = np.where(ok_m, (t_total / t_req) / (wm * math.log(wq[-1] / wq[0])), 0.0)
    ok = ok_m & (tau > 0)
    g95 = np.full(m.size, np.inf)
    idx = np.nonzero(ok)[0]
    g95[idx] = _core.g95_curve(
        [float(x) for x in dm["Psi_SI"][idx]], [float(x) for x in dm["L_H"][idx]],
        [float(x) for x in dm["Q"][idx]], [float(x) for x in dm["T_K"][idx]],
        [float(x) for x in dm["eta_A"][idx]], [float(x) for x in m[idx]],
        [float(x) for x in tau[idx]], Q_axion, Theta, rho, True)
    F_SI = dm["Psi_SI"] ** 4 / dm["L_H"] ** 2
    band = {
        "m1_eV": float(m[0]), "m2_eV": float(m[-1]), "f_min_Hz": f1, "f_max_Hz": f2,
        "live_time_s": t_total, "live_time_yr": float(live_time_yr),
        "t_req_s": float(t_req), "allocation": allocation, "kernel": "local, dispersive",
        "n_settings": 0, "n_targets": int(n), "statistic": "weak",
        "Psi_SI_H_T_m2": [float(dm["Psi_SI"][0]), float(dm["Psi_SI"][-1])],
        "L_H": [float(dm["L_H"][0]), float(dm["L_H"][-1])],
        "L_eff_H": [float(dm["L_H"][0]),float(dm["L_H"][-1])],
        "L_reactive_H": [float(x) for x in _as_curve(L_H, fm[[0,-1]], ft, "L_H", log_y=False)],
        "Q": [float(dm["Q"][0]), float(dm["Q"][-1])],
        "T_K": [float(dm["T_K"][0]), float(dm["T_K"][-1])],
        "eta_A": [float(dm["eta_A"][0]), float(dm["eta_A"][-1])],
        "F_SI_T4_H2_m8": [float(F_SI[0]), float(F_SI[-1])],
        "Q_axion": Q_axion, "Theta": Theta, "rho_GeV_cm3": rho, "v0_kms": 220.0, "vobs_kms": 232.0,
        "exact_Gbar": True, "model": model, "C_gamma": C_gamma, "label": label,
        "allocation_quadrature": quad,
        "noise_bandwidth_over_f_max": float(np.max(widths)),
        "local_variation_bound": float(np.max(variation*widths)),
        "table_f_Hz": [float(ft[0]), float(ft[-1])], "table_points": int(ft.size),
        "self_resonant_f_Hz": [float(x) for x in fm[~ok]],
        "note": "per-tuning Psi(omega), tuned-pole L_eff, Q = omega L_eff/R, T and eta_A; "
                "the band entries for those quantities are the values at the two band edges",
    }
    return {"m_eV": m, "f_Hz": fm, "g95_GeV": g95,
            "g_target_GeV": g_qcd(m, model=model, C_gamma=C_gamma),
            "tau_s_per_rad_s": tau, "Psi_SI": dm["Psi_SI"], "L_H": dm["L_H"],
            "L_eff_H": dm["L_H"], "L_reactive_H": _as_curve(L_H, fm, ft, "L_H", log_y=False), "Q": dm["Q"],
            "T_K": dm["T_K"], "eta_A": dm["eta_A"], "F_SI_T4_H2_m8": F_SI, "band": band}


def reach_fullwave(results, benchmark: str | dict = "optimistic", *, amplitude: str = "as_designed",
                   T_K: float | None = None, eta_A: float | None = None, Q_floor: float | None = None,
                   compare_mqs: bool = True, T_radiation_K: float | None = None, **kw) -> dict:
    """Local beyond-MQS reach with the series-tuned slope inductance.

    Independent resistances add: R_total=R_rad+R_material+omega*L_p/Q_other.
    The Q_floor argument (or the benchmark's Q) is the intrinsic MQS Q_other,
    referenced to the static L_p; it is not a minimum-Q ceiling.
    The MQS comparison retains material and intrinsic losses and removes radiation.
    Material/intrinsic loss is at T_K. Radiation uses T_radiation_K; omitting
    it explicitly assumes the same bath temperature and records that assumption.
    Both the tabulated slope and local noise bandwidth must pass the guards in
    reach_dispersive. Requested bands must lie in the contiguous positive-L sweep.
    For broad or rapidly dispersive response use finite_band with a tuned circuit.

    The sweep's R_ohm holds the surface loss R_loss that the engines compute from the conductors'
    rf blocks (2D and 3D), so Q_other must exclude that loss: a benchmark Q that
    stands for the total intrinsic Q counts the conductor loss twice. When every row carries the
    exact tuned-pole L_eff_H (3D designs whose conductors are meshed as volumes: from dZ/domega),
    the full-wave scan takes L_eff from a cubic Hermite interpolant of L with those exact slopes,
    which must agree with the linear interpolation of the rows' L_eff within 5 % (else the rows are
    too sparse and the reach refuses), instead of L + 0.5 dL/dln(f) of the interpolated L (which
    needs four rows and spline/PCHIP agreement within 5 %). The MQS comparison always uses L_p.
    """
    from . import circuit as ck

    data = getattr(results, "data", results)
    t = ck.fullwave_tables(data)
    b = BENCHMARKS[benchmark] if isinstance(benchmark, str) else dict(benchmark)
    T = b.get("T", 0.02) if T_K is None else T_K
    eta = b.get("eta_A", 1.0) if eta_A is None else eta_A
    cap = b.get("Q") if Q_floor is None else Q_floor
    psi = t["Psi_SI"].copy()
    scale = 1.0
    if amplitude == "required":
        if not isinstance(benchmark, str):
            raise ValueError("reach_fullwave: amplitude='required' needs a benchmark name")
        for bb in data.get("benchmarks", []):
            if bb["name"] == benchmark:
                scale = bb["amplitude_scale"]
                break
        else:
            raise KeyError(f"reach_fullwave: no benchmark {benchmark!r} in the results")
        psi = psi * scale
    elif amplitude != "as_designed":
        raise ValueError("reach_fullwave: amplitude must be 'as_designed' or 'required'")

    # Trim the table to the first contiguous run of L(omega) > 0: above the first
    # self-resonance L and hence Q change sign, and keeping those rows would (a) poison the
    # log-log interpolation of Q at the band edge and (b) force Q onto a linear interpolation
    # over five decades.  f_self_resonance_Hz records where the run ends.
    keep = t["L_H"] > 0
    n_keep = int(np.argmin(keep)) if not keep.all() else keep.size
    if n_keep < 2:
        raise ValueError(
            "reach_fullwave: the sweep has fewer than two frequencies below the readout's first "
            f"self-resonance (L(omega) <= 0 from {t['f_Hz'][max(n_keep - 1, 0)]:.6g} Hz on), so no "
            "series-tuned scan exists; extend the sweep downwards.")
    f_sr = float(t["f_Hz"][n_keep]) if n_keep < keep.size else None
    tt = {k: v[:n_keep] for k, v in t.items()}
    psi = psi[:n_keep]
    w = 2.0 * math.pi * tt["f_Hz"]
    for key, bound, lower in (("f_min", tt["f_Hz"][0], True), ("f_max", tt["f_Hz"][-1], False)):
        value = kw.get(key)
        if value is not None and ((value < bound if lower else value > bound)):
            raise ValueError("requested band is outside the contiguous positive-L sweep below self-resonance; extend/refine the sweep")
    if cap is not None and not (math.isfinite(cap) and cap > 0):
        raise ValueError("Q_floor (intrinsic MQS Q) must be finite and positive")
    Lref = float(data.get("coupling", {}).get("L_p_H", tt["L_H"][0]))
    if not (math.isfinite(Lref) and Lref > 0):
        raise ValueError("reach_fullwave: intrinsic-loss reference L_p must be finite and positive")
    Rother = w*Lref/cap if cap is not None else np.zeros_like(w)
    Rtotal = tt["R_ohm"] + Rother
    if np.any(Rtotal <= 0):
        raise ValueError("reach_fullwave: loss-free rows need Q_floor or a benchmark intrinsic Q")
    Trad = T if T_radiation_K is None else T_radiation_K
    if T < 0 or Trad < 0 or not (math.isfinite(T) and math.isfinite(Trad)):
        raise ValueError("bath temperatures must be finite and nonnegative")
    # Independent reservoirs add noise occupations weighted by their resistances.
    Teff = np.array([ck.T_from_occupation(float(wi),
                     ((tt["R_loss_ohm"][i]+Rother[i])*ck.n_bose(float(wi),T) +
                      tt["R_rad_ohm"][i]*ck.n_bose(float(wi),Trad))/Rtotal[i]) for i,wi in enumerate(w)])
    if f_sr is not None:
        kw.setdefault("f_max", float(tt["f_Hz"][-1]))
    # the rows' exact tuned-pole inductance (3D: from dZ/domega), when every row has it, between the rows with a
    # 5 % guard; otherwise reach_dispersive takes L + 0.5 dL/dln(f) of the interpolated L. Only the full-wave scan
    # gets it: the MQS comparison below keeps L_p
    caller_slope = "L_eff_H" in kw
    exact_slope = "L_eff_H" in tt and not caller_slope
    fw_kw = dict(kw)
    if exact_slope:
        fw_kw["L_eff_H"] = _exact_pole_inductance(tt["f_Hz"], tt["L_H"], tt["L_eff_H"])
    out = reach_dispersive(tt["f_Hz"], psi, tt["L_H"], R_ohm=Rtotal, T_K=Teff, eta_A=eta,
                           label=f"{data.get('name', 'design')} full wave", **fw_kw)
    R_edges = _as_curve(tt["R_ohm"], out["f_Hz"][[0,-1]], tt["f_Hz"], "R_ohm")
    Qfw = 2*math.pi*out["f_Hz"][[0,-1]]*out["L_eff_H"][[0,-1]]/np.maximum(R_edges,1e-300)
    out["band"].update(benchmark=benchmark if isinstance(benchmark, str) else "custom",
                       amplitude=amplitude, amplitude_scale=scale, Q_cap=cap,
                       Q_radiation_limited=[float(Qfw[0]), float(Qfw[-1])],
                       k_R=[float(tt["k_R"][0]), float(tt["k_R"][-1])],
                       f_self_resonance_Hz=f_sr,
                       sweep_rows_used=[int(n_keep), int(t["f_Hz"].size)])
    if caller_slope:
        pole_reduction = "L_eff supplied by the caller (L_eff_H); local bandwidth check"
    elif exact_slope:
        pole_reduction = ("L_eff from the rows' exact dZ/domega (L_eff_H): cubic Hermite interpolant of L with the "
                          "exact slopes; agreement with the linear interpolation of the rows' L_eff within 5%; local "
                          "bandwidth check")
    else:
        pole_reduction = "L_eff = L + 0.5 dL/dln(f); interpolant slope agreement within 5%; local bandwidth check"
    out["band"]["pole_reduction"] = pole_reduction
    out["band"]["Q_radiation_limited_note"] = (
        "omega L_eff / R_ohm with the sweep's R_ohm = Re Z + R_loss: radiation, bulk loss and the surface loss the "
        "engines compute from the rf blocks (2D and 3D); without Q_other")
    out["band"]["Q_cap_meaning"] = "Q_cap is Q_other, an independent intrinsic MQS loss; not a minimum-Q cap"
    out["band"]["intrinsic_loss_model"] = "R_other = omega L_p / Q_other; independent resistances add"
    out["band"].update(Q_other=cap, L_loss_reference_H=Lref, T_material_K=T, T_radiation_K=Trad,
                       radiation_temperature_assumed_equal=(T_radiation_K is None))
    out["fullwave"] = {k: t[k] for k in t}
    out["pole_reduction"] = {"f_Hz":out["f_Hz"], "L_eff_H":out["L_eff_H"]}
    if compare_mqs:
        Psi_mqs = data.get("coupling", {}).get("Psi_SI_H_T_m2", 0.0) * scale
        L_mqs = data.get("coupling", {}).get("L_p_H", 0.0)
        if not (Psi_mqs > 0 and L_mqs > 0):
            raise ValueError("reach_fullwave: the results have no coupling/Psi_SI_H_T_m2 and L_p_H "
                             "to compare the MQS scan with; pass compare_mqs=False")
        Rm = tt["R_loss_ohm"] + Rother
        if np.any(Rm <= 0):
            raise ValueError("MQS comparison has zero material/intrinsic loss; supply Q_floor or compare_mqs=False")
        # the MQS scan's tuned-pole inductance is L_p: no full-wave L_eff_H, the rows' or the caller's
        mqs_kw = {k: v for k, v in kw.items() if k != "L_eff_H"}
        mqs = reach_dispersive(tt["f_Hz"], np.full(tt["f_Hz"].size, Psi_mqs),
                              np.full(tt["f_Hz"].size, L_mqs), R_ohm=Rm, T_K=T, eta_A=eta,
                              label="MQS", **mqs_kw)
        out["mqs"] = mqs
        out["ratio_g95_fullwave_over_mqs"] = out["g95_GeV"] / mqs["g95_GeV"]
        out["ratio_F_fullwave_over_mqs"] = out["F_SI_T4_H2_m8"] / (Psi_mqs ** 4 / L_mqs ** 2)
    return out


# ============================================ multi-band / multipole scans (docs/physics/multipole_scan.md)
def multipole_scan(table, **kw) -> dict:
    """Beta. The multipole (multi-band) scan strategy (arXiv:2211.00008) on a full-wave ``Z(omega), V(omega)``
    table (``axionfem.multipole.SourceTable``, or axionFEM results with a ``fullwave`` sweep).  Thin
    wrapper of :func:`axionfem.multipole.multiband_scan`.  The scan of arXiv:2211.00008 as defined there
    is :func:`reach_2211`.  ``reach_fullwave`` is the single-band local scan below the first
    self-resonance."""
    from . import multipole as _mp
    if not isinstance(table, _mp.SourceTable):
        table = _mp.table_from_results(table)
    return _mp.multiband_scan(table, **kw)


reach_multiband = multipole_scan


def reach_2211(table, **kw) -> dict:
    """Beta. Single-pole and multipole reach of arXiv:2211.00008 (:func:`axionfem.multipole.reach_2211`)."""
    from . import multipole as _mp
    if not isinstance(table, _mp.SourceTable):
        table = _mp.table_from_results(table)
    return _mp.reach_2211(table, **kw)
