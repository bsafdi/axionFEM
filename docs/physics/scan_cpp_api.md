# C++ API: signal, readout noise, optimal scan and statistics

Here *the spec* is [`scan_and_signal.md`](scan_and_signal.md), the specification of the signal and scan model; "Sec. n"
and the tags "(S*)" refer to it.

Headers `cpp/include/axionfem/signal.hpp` (spec Secs. 1–4) and
`cpp/include/axionfem/scan.hpp` (Secs. 5–7; includes `signal.hpp`); namespace
`axf::scan`. Implementation in `cpp/src/signal.cpp`, `cpp/src/scan.cpp`. Physics and
equation tags "(S*)": [`scan_and_signal.md`](scan_and_signal.md). Test:
`cpp/tests/test_scan.cpp` (487 checks against `tests/reference/scan_reference_values.json`).

## Units at the interface

| quantity | unit |
|---|---|
| angular frequency `omega`, tuning `omega0` | rad/s |
| frequency `f` | Hz |
| time, dwell `t` | s; dwell density `tau` in s per (rad/s) |
| temperature | K |
| axion mass `m_a` | eV (`omega = m_a eV/hbar`) |
| coupling `g` | GeV⁻¹ |
| DM density `rho` | GeV/cm³ |
| halo speeds `v0`, `vobs` | km/s (`f_shm` takes `v` in units of c) |
| `Psi`, `L_p` (FEM geometry) | natural HL: T m³ and m; SI `Psi_SI = mu0 Psi` [T H m²], `L = mu0 L_p` [H] |
| `F = Psi⁴/L_p²` | `F_nat` T⁴ m¹⁰; `F_SI = mu0² F_nat` T⁴ H² m⁸; `F_nat_eV` eV⁻² |
| flux amplitude `Phi_a` | Wb |
| PSDs (Foster normalisation, ⟨X²⟩ = (1/π)∫₀^∞ S dω) | voltage V² s, current A² s |
| scan rate dω₀/dt | rad/s² (× year/2π for Hz/yr) |

Internally SI with explicit ħ, k_B (spec Sec. 10). Constants come from `units.hpp`.

## Defaults

`kQaxionDefault = 6.8e5` (exact SHM: `Q_axion_shm() = 6.779483e5`), `kTheta95 = 2.71`
(exact: `theta95_exact() = 2.705543`), `kRhoDM_GeV_cm3 = 0.3`, `kV0_kms = 220`,
`kVobs_kms = 232`, `kMaFa_GeV2 = 5.70e-3`, `kCgammaLatticeBrandt = 1.77`
(also `kCgammaChiPT`, `kCgammaLatticeLin`), `kEoverN_{KSVZ,DFSZ_I,DFSZ_II}`, `kAlphaLEE = 0.75`.

`ScanParams {Q_axion, Theta, rho_GeV_cm3, v0_kms, vobs_kms, exact_Gbar}` carries these
conventions into Sec. 5 functions. `Q_axion` is used by the *local* formulas; the exact
discrete sums integrate the SHM line (`v0_kms`, `vobs_kms`) and so implicitly use
`Q_axion_shm(v0, vobs)`. Compare like with like (spec Sec. 9 item 2).

## Detector

```cpp
struct Detector { double psi_nat_T_m3, L_p_m, Q, T_K, eta_A, R_exp = 0; };
Detector::from_natural(psi_T_m3, L_p_m, Q, T_K, eta_A);
Detector::from_SI(psi_SI_T_H_m2, L_H, Q, T_K, eta_A);
d.psi_SI(); d.L_H(); d.F_nat(); d.F_SI();
```
`eta_A = 2 sqrt(S_Phi S_J)/hbar` (=1 at the SQL); `R_exp` gives `R(omega) = R0 (omega/omega0)^R_exp`
within a tuning in the exact sums (0 = constant).

## Sec. 1–2: axion inputs and halo (`signal.hpp`)

| function | returns | spec |
|---|---|---|
| `omega_from_mass(m)`, `mass_from_omega(w)`; `mass_to_freq(m)`, `freq_to_mass(f)` | rad/s, eV, Hz, eV | — |
| `g_qcd_GeV(m, E_over_N=8/3, C_gamma=1.77, ma_fa=5.7e-3)`; `qcd_line(...)` (a `MassFunction`) | GeV⁻¹ | S2.1 |
| `kappa_a_eV(g, rho)`, `kappa_a_inv_m(g, rho)`, `kappa_a_SI(g, rho)` | eV, 1/m, 1/H | S2.2–S2.3 |
| `flux_amplitude_Wb(g, Psi_nat, rho)` = κ[1/m] Ψ | Wb | S2.3 |
| `f_shm(v_over_c, v0, vobs)` | per unit v/c | S2.4 |
| `line_shape(omega, omega_a, v0, vobs)`, `line_shape_detuned(omega-omega_a, omega_a, ...)` | s | S2.5 |
| `Q_axion_shm(v0, vobs)` (adaptive Gauss–Kronrod in v, rel. tol 1e-14) | — | S2.6 |
| `coherence_time_s(f, v0, vobs)` = 1/(f v0 vobs) | s | Sec. 2.3 |
| `line_v_grid(n_v, v0, vobs, v, w)`: Gauss–Legendre on [0, vobs + 7 v0] | — | S5.9 |

`g_agg_GeV(m, ScanSpec)` (original API) is the same as `g_qcd_GeV` with the spec's fields.

## Sec. 3: circuit, noise, coupling

| function | returns | spec |
|---|---|---|
| `n_bose(omega, T_K)` (expm1; 0 for T ≤ 0); `occupation(f_Hz, T_K)` | — | S3.1 |
| `A_noise(xi, n, eta)` | — | S3.7 |
| `G_bar_xi(xi, n, eta)`; `xi_opt(n, eta)`; `G_bar(n, eta)` = Ḡ(ξ_opt) | — | S3.8–S3.10 |
| `G_bar_highT(n, eta)` = 1/(3√3 η(2n+1)); `G_bar_classical(w0, T, eta)` = ħω₀/(6√3 η k_BT) | — | S3.11 |
| `G_bar_at(det, omega0, exact=true)` | — | S3.10/S3.11 |
| `sensitivity_bandwidths(omega0, Q, n, eta, xi=-1)` → `{linewidth, noise_doubling, thermal_gt_squid, equivalent, xi}` | rad/s (full widths) | S3.12 |
| `ringup_time_s(omega0, Q)` = 2Q/ω₀ | s | Sec. 5.5 |
| `squid_coupling_from_xi(xi, omega0, R0, S_Phi, S_J)` → `{M_H, eta_A}` (S_Φ in Wb² s, S_J in A² s, Foster-normalised) | H | S3.5 |
| `noise_psd(omega, omega0, det, xi)`, `noise_psd_detuned(omega, omega-omega0, omega0, det, xi)` → `{Z2, SV_th, SV_ba, SI_imp, N_V, lambda_B}` | Ω², V² s, A² s | S3.3–S3.6 |
| `signal_psd_V(omega, omega_a, Phi_a_Wb, v0, vobs)` | V² s | S3.2 |
| `psd_at_tuning(det, f0_Hz, m_a, g, f_Hz, xi=-1, rho, v0, vobs)` → `PSDCurve` (S_Va, N_V, λ_a, λ_B, noise components, s per frequency) | V² s, A² s | S3.2–S3.6 |

`xi <= 0` means ξ_opt(n(ω₀)). Full |Z|², exact n(ω), ω² backaction are used everywhere
except in the explicitly local formulas.

## Sec. 4: statistics

| function | returns | spec |
|---|---|---|
| `h_weak(s)`, `h_limit(s)`, `h_discovery(s)`, `h_stat(Stat, s)`; `enum class Stat {Weak, Limit, Discovery}` | — | S4.2–S4.4 |
| `norm_cdf(x)`, `norm_ppf(p)`, `theta95_exact()` | — | S4.5 |
| `limit_band_q(n_sigma)` → {(z+N)², (z−N)²} | — | S4.5 |
| `n_independent_masses(f_min, f_max, v0, vobs, alpha=0.75)` | — | S4.6 |
| `ts_discovery_threshold(sigma, N_ma, sided=1)` | — | S4.7 |
| `g_disc_over_g95_weak(TS, Theta)` = (TS/Θ)^¼ | — | Sec. 4.2 |

`ts_discovery_threshold` evaluates Φ⁻¹ in the tail (q = p/N_ma), so it is accurate
for q ~ 1e-14; see "Discrepancies" below.

## Sec. 5: scan rate, requirement, allocation, reach (`scan.hpp`)

| function | returns | spec |
|---|---|---|
| `scan_rate_SI(Phi_a_Wb, L_H, Q, Gbar, Q_axion, Theta)` | rad/s² | S5.2 |
| `scan_rate(det, m_a, g, params)` (tuned at ω₀ = ω_a) | rad/s² | S5.1 |
| `F_req_closed_form(eta, T_K, Q, g1, t_s, m1, m2, params)` → `Requirement {F_nat_eV, F_nat, F_SI, g_min_GeV}` | eV⁻², T⁴m¹⁰, T⁴H²m⁸ | S5.4 |
| `F_req_general(m1, m2, t_s, g_target, Q(m), T(m), eta(m), params)`; overload with constant Q, T, η | same | S5.3 |
| `F_required(Benchmark, ScanSpec)` (original API, closed form from the design spec) | same | S5.4 |
| `time_target_curve_s(det, m1, m2, g_target, params)` | s | S5.6 |
| `allocation_target(det, m1, m2, t_total, g_target, params)`; `allocation_log_uniform(m1, m2, t_total)` → `Allocation {kind, m1_eV, m2_eV, t_total_s, t_req_s, tau(omega0)}` | s/(rad/s) | S5.5–S5.7 |
| `tau_log_uniform(omega0, t, omega1, omega2)` | s/(rad/s) | S5.7 |
| `g95_local(det, m_a, tau, params)` | GeV⁻¹ | S5.8 |
| `reach_local(det, alloc, masses \| n_points, params)`; `reach_local(det, m1, m2, live_time_s, Allocation::Kind, n_points, g_target, params)` → `ReachCurve {m_eV, f_Hz, g95_GeV, tau_s_per_rad_s, s_max}` | GeV⁻¹ | S5.8 |

`MassFunction = std::function<double(double m_a_eV)>`. `Allocation::tau` is the formula,
not truncated at the band edges (the schedule is zero outside [m1, m2]).

### Exact discrete sums

| function | returns | spec |
|---|---|---|
| `Tunings {omega0_rad_s, t_s, xi}` (xi empty → ξ_opt(n(ω₀,i))) | — | — |
| `theta_per_tuning(det, m_a, g, tunings, stat, params, n_v=256, &s_max)` | Θ_i | S4.1, S5.9 |
| `theta_discrete(...)` = Σ Θ_i | — | S5.9 |
| `g95_discrete(det, m_a, tunings, stat=Limit, params, n_v, &s_max)` (weak: closed form; limit/discovery: Brent in ln g) | GeV⁻¹ | S5.10 |
| `reach_discrete(det, tunings, masses, stat, params, window_half_widths=40, n_v)` (warns if s_max > 0.05) | `ReachCurve` | S5.10 |
| `dense_tunings_around(det, m_a, alloc, half_widths=40, steps_per_hw=6)` | `Tunings` | Sec. 8 check |
| `discretise_allocation(det, alloc, steps_per_fw=10)` (mid-point steps Δω₂ₓ/N_s) | `Tunings` | Sec. 5.5 |
| `n_tunings(det, m1, m2, steps_per_fw)` | — | Sec. 5.5 |

The sums use the full |Z|², n(ω), ω² backaction, R(ω), the exact SHM line integrated in
v (no sqrt edge), and ω − ω₀ = (ω_a − ω₀) + ω_a v²/2.

## Sec. 6: finite band

| function | returns | spec |
|---|---|---|
| `kernel_row(det, m_a, g, omega0[], xi[], stat=Weak, params, n_v=96)` | 1/s | S6.1–S6.2 |
| `kernel_narrow_line(det, m_a, g, omega0, xi, params)` | 1/s | S6.3 |
| `lp_schedule(K[target][setting], Theta)` → `{t_s, t_total_s, min_coverage, pivots}` | s | S6.4 |

`lp_schedule` is a dense tableau simplex on the dual LP (max Θ Σy, Kᵀy ≤ 1; feasible at
y = 0, so no phase I), K rescaled per target to max 1, Dantzig pricing with a Bland
fallback; t are the dual multipliers, rescaled so min_k (Kt)_k ≥ Θ. Adequate for
O(10²) targets × O(10³) settings; on the spec's demo (120 × 600) it matches HiGHS to 1.4e-9.

### Sec. 6 with an externally supplied N_V(ω) (the exact readout network)

The kernels above take their noise from the single-pole `Detector` (S3.6). A real network is
reduced to that pole only to second order in the detuning (`readout_circuit.md` limitation 5),
so the exact `N_V(ω)` of its (C4.5) can be passed instead. Everything the kernel needs from
the readout is `N_V(ω)` per setting plus the pickup's `Psi_SI`, since (S6.1) is
K = (1/2π)∫dω h(S_Va/N_V) and S_Va depends only on Φ_a = κ_a(g, ρ) Ψ_SI and the line shape.

| function | returns | spec |
|---|---|---|
| `NoiseCurve {omega_rad_s, N_V_V2_s}`; `.at(omega)`, `.validate(where)` | V² s | — |
| `kernel_row_noise(psi_SI, m_a, g, noise[], stat=Weak, params, n_v=96)` (overloads: `vector<NoiseCurve>`, `vector<NoiseFunction>`) | 1/s | S6.1–S6.2 |
| `kernel_narrow_line_noise(psi_SI, m_a, g, N_V(omega_a), params)` | 1/s | S6.3 |
| `g95_band(psi_SI, m_a, noise[], t_s[], stat=Limit, params, n_v, &s_max)` | GeV⁻¹ | S6.1/S6.4 |
| `reach_band(psi_SI[], m_eV[], noise[], t_s[], stat=Limit, params, n_v)` → `ReachCurve` | GeV⁻¹ | S6.1/S6.4 |

`NoiseCurve.at` interpolates linearly in (ln ω, ln N_V) — every term of (C4.5) is a power law
in ω or in the detuning (|Z|² ~ R² + (2L δω)²), so log-log is exact in each limit and smooth
through the resonance — and continues the end slope outside the table. `psi_SI` in `reach_band`
may hold one value per target mass, which is how a frequency-dependent coupling Ψ(ω) enters
(`readout_circuit.md` limitation 7). `validate` refuses a non-ascending grid, a non-positive
N_V, fewer than two samples, or a length mismatch, each with a specific message.

### The band echo (`ScanBand`)

`ReachCurve::band` describes the call that produced the curve, so an output document is
self-describing: `m1_eV`, `m2_eV`, `f_min_Hz`, `f_max_Hz`, `live_time_s`, `t_req_s`,
`allocation` ("target" | "log_uniform" | "discrete" | "lp"), `n_settings`, `n_targets`, `stat`,
the detector (`Psi_SI_T_H_m2`, `L_H`, `Q`, `T_K`, `eta_A`, `F_SI_T4_H2_m8`) and the conventions
actually used (`Q_axion`, `Theta`, `rho_GeV_cm3`, `v0_kms`, `vobs_kms`, `exact_Gbar`).
`reach_local`, `reach_discrete` and `reach_band` all fill it.

### Python bindings (`axionfem._core`)

`reach_local` (now with `band`), `psd_at_tuning`, `kernel_row`, `kernel_narrow_line`,
`kernel_row_noise`, `kernel_narrow_line_noise` (vectorised over targets), `lp_schedule`,
`reach_band`, `scan_rate_curve` and `g95_curve` (per-mass detector curves: the dispersive /
beyond-MQS reach), `bandwidths`, `xi_opt`, `G_bar_opt`, `g_qcd_GeV`, `Q_axion_shm`,
`mass_to_freq`, `freq_to_mass`. Noise curves cross as lists of lists (one ω grid and one N_V
grid per setting). The Python wrappers are `axionfem.scan.reach`, `.finite_band`,
`.reach_dispersive`, `.reach_fullwave`, `.psd`, `.g_qcd`, `.Q_axion_shm`.

## Sec. 7: long detectors

`g_solenoid_long(x)`, `g_toroid_long(y)`, `solenoid_long_optimum()` (root of
1 − x² + 4x² ln x = 0: x = 0.5335430, g = 0.2036322), `toroid_long_optimum()` (y = e^{−½}, g = 1/2e).
`W_mag R² = sqrt(F_req)/g` [T² m⁵ in HL; ×1/μ₀ for J m²].

## Notes on the implementation

* `g_agg_GeV` returns |E/N − C_γ| as in (S2.1), positive for KSVZ too (F_req ∝ g⁴ does not
  depend on the sign).
* `units.hpp` `hbarc_eVm = 1.973269804e-7` (CODATA-rounded) differs from
  ħc/e = 1.9732698034e-7 built from its own `hbar`, `c`, `e` by 3.1e-10; quantities in eV
  units (`F_nat_eV`, `rho_eV4`, `kappa_a_eV`) inherit ≈1e-9. SI-route quantities do not.
* LEE thresholds: the reference evaluates `ndtri(1 - p/N)`, rounding 1 − p/N to one ulp.
  Accurate values: TS₅σ (0.5–4.1 neV scan, one-sided) = 55.0745 (spec: 55.073), two-sided 53.7124
  (53.713); [F] example one-sided 5σ = 58.775 (spec: 58.77), two-sided 57.412 (57.41).
  3σ values agree to < 3e-8. The test reproduces the reference path to 1e-12 as well.
* The reference self-test value ⟨(ω/m)⁴⟩ − 1 = 2.8133109e-6 carries a 7e-12 absolute
  cancellation error; the analytic value 2⟨v²⟩ + 1.5⟨v⁴⟩ = 2.8133180e-6 is reproduced to 2e-15.
