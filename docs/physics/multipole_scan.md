# Multi-band and multipole scans: specification

!!! warning "Beta"
    `axionfem.multipole` (`scan.multipole_scan`, `scan.reach_multiband`, `scan.reach_2211`) is beta.

This page specifies the methods that `axionfem.multipole` implements: the multi-band scan with the local scan rate of
arXiv:2302.14084 (Secs. 1 and 4), and the single-pole and multipole accounting of arXiv:2211.00008 (Secs. 2, 3 and 5).
Every equation has a tag M0–M14 that the code cites.

Implementation: `python/axionfem/multipole.py` (wrappers `scan.multipole_scan` = `scan.reach_multiband`,
`scan.reach_2211`; M14 in Sec. 5 is `jf2022_schedule`, `jf2022_theta`, `jf2022_plot_curve`). Tests:
`tests/test_multipole_scan.py` (a synthetic lossless Foster-form source table with two poles between 5 and 200 MHz,
closed forms, the C++ scan core; the 2211.00008 tables when they are present).

Sources:

- **[R]** A separate Python implementation of the multi-band projection (a notebook, not distributed with axionFEM).
  The options that follow its conventions are named `notebook*` in the code; where those conventions are not
  physical, the default differs and the page says why.
- **[P]** arXiv:2211.00008 (Benabou, Foster, Kahn, Safdi, Salemi) and its COMSOL tables (not distributed): lossless
  shielded toroid 5–500 MHz, TACHYON-lined toroid 5–650 MHz (the lossy toroid of Fig. S3: ≈ 20 % in the paper, 19.0 % in the COMSOL model), 1 kHz step; impedance with the port driven, voltage
  with `Je = 1 A/m²` and the port loaded by `Zref = 50 Ω`. The tests that need them run when `AXF_COMSOL_2211_DIR`
  names their directory.
- **[S]** `docs/physics/scan_and_signal.md` (S1–S10).
- **[C]** The 2022 computation behind [P]'s projections (Figs. 2, S2, S3), transcribed in
  `benchmarks/multipole_scan/reference_jf2022/jf2022.py`. Some of its inputs are not available and are reconstructed
  (Sec. 5).

Units in [R]: Ω, H, V, MHz, mK, T; this module is SI throughout.

## 0. Conventions

**M0 (coupling, source).** [R] and [P]'s figures use `g = 2e-10 C_ag (m_a/eV)` GeV⁻¹, `C_ag` = 0.75 (DFSZ), 1.92
(KSVZ), not axionFEM's `scan.g_qcd` (E/N, C_γ = 1.77). COMSOL tables are per unit source current density; the axion
current amplitude is `|J_eff| = g B √(2ρ)`. [R]'s `10.6049` A/m² per (GeV⁻¹ T) is exactly this with
**ρ = 0.45 GeV/cm³** (`J_eff_A_per_m2`, test to 1e-5); its `3.30227` (V→eV) and `4.03278e12` (H→eV⁻¹) are the
Heaviside–Lorentz conversions to 6 digits (1.6e-5 in the rate; `rounded_constants`). axionFEM tables carry
`V = ω|Ψ_SI|` per unit `κ_a` (`table_from_results`; κ from [S] Sec. 8).

**M0b (sign, port load).** COMSOL uses `exp(+jωt)` (inductor `+jωL`), axionFEM `exp(−iωt)`
(`Z_axf = conj(Z_COMSOL)`, `L_H = −Im Z_axf/ω`). This module uses the reactance `X` with **inductive > 0**:
`X = Im Z_COMSOL = −Im Z_axf = ω L_H`. `source_table(convention=…)` converts and refuses a table that is not inductive
at its lowest frequency, and refuses one with `Re Z < 0` at most points (a passive port has `Re Z ≥ 0`; isolated
negative points, e.g. near poles of a lossy sweep, are kept with a warning; `test_wrong_sign_convention_is_refused`).
A table that stores **−Z** (Re Z < 0 at every point, Im Z < 0 wherever the source is inductive) is read as
`source_table(f, −Z, V, convention="exp(+jωt)")`; [R] negates only Im Z (it ignores Re Z). A COMSOL voltage
run with excitation off loads the port with `Zref`; the open-circuit (Thévenin) source is
`V_oc = V_port (Z + Z_load)/Z_load` (`port_load_ohm=50`). [R] uses `|V_port|` directly; with a 50-Ω loaded
voltage run its rates are low by `|Z+50|⁴/50⁴`, a factor that grows large near poles. [P] needs `V_oc` (Sec. 2).

## 1. The multi-band scan ([R]; `multiband_scan`)

**M1 (tuning element).** `C = 1/(ωX)` where `X > 0` (source inductive → capacitor), `L_t = −X/ω` where `X < 0`
(source capacitive → inductor).

**M2 (bands).** A frequency is tunable if `30 pF < C < 300 pF` or `10 nH < L_t < 500 nH` (open intervals). Contiguous
tunable runs with `f_last − f_first ≥ 5 MHz` are kept and shrunk by 2.5 MHz at both ends
(`test_bands_and_greedy_allocation_on_a_synthetic_table`). Unlike [P], inductor tunings are allowed (`allow_inductive=False` gives [P]'s
rule).

**M3 (slope inductance).** [R] uses `L_eff = (X + f dX/df)/(4πf) = ½(X/ω + dX/dω)` (arXiv:2302.14084 Eq. 6), with a
cubic spline of `X(f)`. The tuned port's total reactance is `X − 1/(ωC)` (capacitor) or `X + ωL_t` (inductor); at its
zero the half slope is `L_eff = ½(dX/dω + |X|/ω)` (`rule="tuned"`, default). The two agree for `X > 0`. **For inductor
tuning the rule of [R] (`rule="notebook"`) is lower by `|X|/ω`** (`test_tuned_slope_differs_from_notebook_only_for_inductor_tuning`),
so its rate there is too high by the square of the ratio of the two slopes.
axionFEM's `_pole_inductance` (`L + ½ dL/d ln f`, `L = X/ω`) is the same as M3 for `X > 0`; `reach_fullwave` stops at
the first self-resonance.

**M4 (Ḡ)** = [S] (S3.8)–(S3.10) exactly (`α_bar` = ξ_opt; exact Bose `n`; test to 1e-12 against the C++). [R] always
uses **η_A = 20** in Ḡ.

**M5 (scan rate).** `dν/dt = (π/16)(Q_a/Θ) V⁴ Q Ḡ/(m_a⁴ L_eff²)` (natural units; `V` the peak EMF), with
`Q_a = 6.4e5`, `Θ = χ²₁.ppf(0.9) = 2.7055`, a **fixed** `Q` (1e5 in [R]), T = 20 mK, B = 5 T. This is [S] (S5.1)
divided by 2π with `Φ_a = V/m_a` (identical to the C++ `scan_rate` to 1e-8 once units agree). [R] converts the rate
from eV² to MHz/yr by dividing by `h[eV/MHz]·h[eV yr]`; the correct conversion is by `ħ²` (ν_SI = ν_nat/ħ,
t_SI = ħ t_nat), so [R]'s MHz/yr are **(2π)² = 39.48× too small** (`test_gbar_and_rate_equal_the_cpp_scan_core`,
`test_notebook_rate_is_low_by_two_pi_squared`). `scan_rate` is physical; `notebook_dnu_dt(notebook_units=True)` and
`multiband_scan(notebook_units=True)` follow [R]'s units.

**M6 (time split: the multi-band strategy).** Rate at the DFSZ coupling (`C_ag = 0.75`, Q = 1e5, B = 5 T, T = 20 mK)
on all bands; each band cut into `round(width/0.1 MHz)` chunks; chunk cost `∫df/max(rate, 1e-10)` (spline); chunks
taken **cheapest first** while the cumulative time stays below 1 yr; kept chunks merged. This maximises the
DFSZ-covered bandwidth for the budget across all bands at once (pole-to-pole bands compete for time).
`merge="notebook"` merges the kept chunks as [R] does: edges rounded to 10 kHz, which can split a covered interval
where the chunk widths differ from 0.1 MHz (a rounding artefact); `merge="adjacent"` joins chunks that share an edge
(`test_bands_and_greedy_allocation_on_a_synthetic_table`). With the physical rate (M5, `notebook_units=False`), the
tuned slope (M3) and `V_oc` (M0b, `port_load_ohm=50`) the same allocation gives a different, physical coverage.

**M7 (uniform dwell).** `g = g_DFSZ [Δν/(T_tot R_DFSZ(ν))]^{1/4}` with Δν the summed band width (`uniform_dwell_reach`).
`target_allocation_reach` adds the optimal allocation of
[S] (S5.5)–(S5.6) restricted to the bands; with one band it equals `reach_fullwave` (t_req to 1e-4,
`test_single_band_reduces_to_reach_fullwave`).

[R] has no quality factor `Q_B`, no source loss, no off-resonance or multi-pole response and no `g/g_MQS`; it allows
inductor tunings and uses Ḡ-optimal (arXiv:2302.14084) noise. It therefore cannot by itself produce [P]'s "Single
Pole"/"Multipole" curves; Secs. 2–3 reconstruct those from [P]'s text and figures, and Sec. 5 from [C].

## 2. [P]'s single-pole sensitivity and Q_B (`single_pole_ratio`)

**M8 (MQS reference).** `L_MQS = X/ω` and `Ψ_MQS = V_oc/ω` at the lowest table frequency (5 MHz); `V_MQS = ωΨ_MQS`;
intrinsic resistance `r = ωL_MQS/Q_MQS`, `Q_MQS = 1e6` ([P]: "intrinsic frequency-dependent resistance such that
Q_B = 10⁶ at all frequencies in the MQS approximation").

**M9 (Q_B).** `Q_B = ωL_eff/R_t`, `R_t = r + R` (R = Re Z_s), `L_eff` from M3 (tuned). Lossless:
`Q_B = 10⁶ (dX/dω + |X|/ω)/(2L_MQS)`.

**M10 ([P]'s noise model).** On resonance [P] writes `S_II^a = S_VV/R²`, `S_II^noise ≈ Q(2T + η_A ω)/(π m_a L_p)`,
`Θ = t/2π ∫(S_II^a/S_II^noise)² dω`. [P]'s `S_VV` is [S]'s `S_V,a`. Compared with [S]'s per-bin normalisation the
noise carries an extra 1/π; `noise_norm="paper2211"` keeps it (the reproduction) and `"foster"` uses [S]'s
normalisation. The normalisation of the computation behind the figures, from [C], is derived in Sec. 5 (per-bin SNR
2× [S]'s, Θ 4×, g √2 optimistic). Ratios (g/g_MQS, M11) are unaffected. Single-pole accounting: each tuning's dwell
is the MQS dwell scaled by `Q_MQS/Q_B` ("scan more coarsely when Q_B < 10⁶; only a fraction amplified when
Q_B > 10⁶").

**M11 (g/g_MQS: the lossy formula).**

    g/g_MQS = (V_MQS/|V_oc|) (Q_B/Q_MQS)^{1/4} [ (2kT r + η_A ħω R_t) / (r (2kT + η_A ħω)) ]^{1/2}

i.e. **Johnson noise from the intrinsic r only** (the COMSOL material loss R = Re Z_s enters `Q_B` and the amplifier
matching but radiates no noise) and **amplifier noise matched to R_t** (current noise `η_A ħω/(πR_t)`). This is
`noise="paper2211"`, the reproduction. Lossless (`R = 0`): `(V_MQS/V)(Q_B/Q_MQS)^{1/4}`, independent of η_A
(`test_2211_single_pole_limits`). The alternative consistent with the fluctuation–dissipation theorem is
`noise="fdt"`: the material loss R radiates Johnson noise at T as well, which gives
`(V_MQS/V)(Q_B/Q_MQS)^{1/4}(R_t/r)^{1/2}`, independent of η_A. It is larger than the `paper2211` ratio by the factor
`[1 + 2kT R/(2kT r + η_A ħω R_t)]^{1/2}` whenever R > 0 (same test). axionFEM's `reach_fullwave` adds all
reservoirs' noise.

## 3. [P]'s schedule and multipole sum (`mqs_schedule`, `multipole_theta`, `reach_2211`)

**M12 (MQS schedule).** Tunings at `δf/f = 10⁻⁶` from `f_MQS = 150 MHz` downwards; each dwells until the MQS
on-resonance Θ at the target reaches Θ₉₅: `t_i = Θ₉₅/K_MQS(f_i)`, `K = (1/2π)∫s² dω = c_s² π Q_a V⁴/(8 ω R² E²)`
(`∫L² dω = Q_a/ω`, `E = 2kT + η_A ħω`, `c_s = π` for [P]'s normalisation, 1 for `noise_norm="foster"`). Only
capacitive tunings (X > 0) are made, and only they take time (`test_schedule_spends_no_time_on_inductive_tunings`).
The defaults are `Q_a = 6.8e5`, ρ = 0.3, Θ = 2.7055, B = 10 T; the scan stops when the live time is spent.
`reach_2211(f_end_Hz=…)` imposes [P]'s endpoint; normalisation constants cancel in `g/g_target` otherwise.

**M13 (multipole sum; superseded for reproducing [P] by M14, whose amplifier-noise frequency weights are the
reverse).** For tuning `C_i` (`1/C_i = ω_i X(ω_i)`) the circuit is `Z_i(ω) = R_t(ω) + j(X(ω) − 1/(ωC_i))`, resonant at
every ω with `ωX(ω) = ω_iX(ω_i)`, one per capacitive band. Per-bin SNR `s_i = c_s S_VV/N_i`,
`N_i = 2kT r(ω) + (η_A ħ/2)(ω_i|Z_i|²/R_t(ω_i) + R_t(ω_i) ω²/ω_i)` (noise matched at ω_i, [S] (S3.5) with ξ = η_A;
`amp_model="flat"` alternative). The statistic at f sums all tunings: `Θ(f)/Θ₉₅ = ∫ dln f_i (1/δ) K_i(f)/K_MQS(f_i)`
(continuum limit; targets weighted with `g_target`), integrated on asinh grids centred on every resonance root
(piecewise-linear `u = ωX`) merged with all table points (the kernel also varies on the table scale through R(f₀) near
poles). `check_convergence=True` repeats the integral with four times the asinh nodes and returns the relative change
(the test requires |ΔΘ/Θ| < 1e-4). On an MQS table this gives exactly `π/(4√a)`, `a = η_Aħω/(2(2kT + η_Aħω))`
(`test_multipole_continuum_sum_on_an_mqs_table`, 2e-3; at 20 MHz and 20 mK, 15 % in g at η_A = 20 and 39 % at
η_A = 1): the sum over neighbouring tunings exceeds [P]'s single-pole accounting by that Lorentzian factor. For a lossy
table with R_t ≫ r, a → ½ and the factor is 3 % in g. M14 differs from M13 mainly in the frequency weights of the two
amplifier-noise terms (Sec. 5).

## 4. Relation to [S] (axionFEM scan spec)

| item | [R] (M1–M7) | [P] (M8–M14) | [S] / axionFEM |
|---|---|---|---|
| rate | S5.1/2π, `Φ = V/m`; SI with h (×1/(2π)²) | per-tuning Θ, no rate | S5.1–S5.2 |
| noise | Ḡ_opt (S3.8–S3.10), η_A = 20 fixed | additive `2T + η_Aω`, per-bin SNR 2× [S] (Θ 4×, M14), noise-matched; material loss noiseless | Ḡ_opt; all reservoirs (S3.3) |
| Q | fixed 1e5 | `Q_B = ωL_eff/(R + r)` | `ωL_eff/R_total`, intrinsic `Q_other` |
| Q_a, Θ, ρ | 6.4e5, 2.7055, 0.45 | 10⁶ ([C]), 2.7055, 0.45 with B = 10 T (inferred from the endpoints; M14) | 6.8e5, 2.71, 0.3 |
| source V | 50-Ω loaded V_port (not converted) | V_oc | ω\|Ψ\| κ_a |
| g target | 2e-10 C m | same | `g_qcd` (E/N, C_γ) |
| L_eff | ½(X/ω + dX/dω) | ½(dX/dω + \|X\|/ω) | `L + ½ dL/dln f` (X > 0 only) |
| bands | C 30–300 pF ∪ L 10–500 nH, ±2.5 MHz | capacitive only, f ≤ f_MQS | below first self-resonance |
| T, η_A | 20 mK, 20 | 20 mK, 20/1 | benchmark |

## 5. The 2022 computation behind [P] (M14)

[C] defines the noise, the per-tuning statistic, the MQS allocation, the strategy and the sensitivity grid. Some of its
inputs are not available (helper modules, the saved multipole statistic, the summation script), so parts are
reconstructed; the list is `ASSUMPTIONS` in `jf2022.py`: A1 `TS(f_s) = Σ_k t_k TuningTS(f_s, f_k)` over the saved
strategy; A2 `r(f) = ωL_MQS/10⁶`, `L_MQS = X(5 MHz)/ω`; A3 `V_MQS = |V_oc(5 MHz)| f/5 MHz`; A4
`source_voltage = |V_oc|` (50-Ω port load removed) times `|J_eff|(B, ρ)` per GeV⁻¹; A5 linear interpolation on the
1 kHz table, no extrapolation; A6 `Q_B = ωL_eff/(R + r)`, `L_eff = (X/ω + dX/dω)/2` (np.gradient), points with `Q ≤ 0`
dropped, `interp1d(…, fill_value="extrapolate")`; A7 T = 20 mK (inferred, see Calibration; [C]'s own default is
`T = 10` K, which `T_K = 10` reproduces); A8 the plotted curve is `gaussian_filter(g, 101)[::100]` on the
sensitivity grid.

**M14.** With `E_th = ħω(2n+1)` (Bose; `2kT` changes nothing visible) and the current PSDs as [C] writes them

    S_th = E_th r(f)/(π |Z_k|²),     S_A = η_A ħ ω_k/(2π R_k) [ (ω/ω_k)² + R_k²/|Z_k|² ],
    Z_k(f) = R(f) + r(f) + j(X(f) − X(f_k) f_k/f),   R_k = R(f_k) + r(f_k),
    K_k(f) = Q_a/(2π ω) (V/|Z_k|)⁴ / (S_th + S_A)²   (per second),   Θ(f) = Σ_k t_k K_k(f),

with `Q_a = 10⁶`, Θ₉₅ = χ²₁.ppf(0.9). Schedule: MQS tunings at `δ = 1/Q_a` from 5 MHz to `f_MQS = 149.896 MHz`,
`t = Θ₉₅/K_MQS`, `K_MQS = π Q_a V_MQS⁴/(2ω r² E²)`; cheapest first while `Σt ≤ 1 yr` (plus one) → `f_start`; strategy
`f_{n+1} = f_n (1 + max(1/Q_a, 1/Q_B))` from `f_start` to `f_MQS`, dwell = MQS dwell (interpolated) ×
`max(1, 10⁶/Q_B)`; tunings with `X < 0` dropped with their time. Sensitivity grid: the same recursion from 5 to
466.406 MHz. The figure plots M14 smoothed (A8).

**Calibration (inferred).** `JF_T_K = 20 mK`, `JF_B_T = 10 T` and `JF_RHO = 0.45 GeV/cm³` are inferred from the two
1-yr endpoints of [P]'s Fig. S3 single-pole curves (15.798 / 18.739 MHz for η_A = 1 / 20), which fix two unknowns, T
and the coupling scale. Only `B√ρ` is determined; 10 T × √0.45 is [R]'s 10.6049 A/m² per GeV⁻¹ T. With the 2022
tables present, `test_jf2022_port_on_the_2211_epoxy_table` checks that these constants put the η_A = 20 endpoint at
18.7386 MHz to within 2 kHz.

**Port and transcription.** The SI port (`jf2022_*`) equals the transcription to 2e-5 in g given the same strategy
(`test_jf2022_port_matches_notebook_transcription_synthetic`); the vectorised kernel equals the transcribed per-tuning
statistic to 1e-12 (`test_jf2022_fast_kernel_equals_verbatim`); the port's step grid, dwell factors, mask and
sensitivity grid equal the transcription's (`test_jf2022_port_schedule_grid_and_curve`). The port's own `f_start` can
differ from the transcription's by a few tuning steps (rounding of constants; the test allows 1e-5), which shifts the
comb of tunings, so the tests that compare g feed both the same strategy. With the epoxy table present, three points
of [P]'s Fig. S3 "Multipole" curve below the 1-yr endpoint, where only off-resonant tails contribute and g is smooth,
anchor the per-tuning normalisation to 5 % in g (`test_jf2022_port_multipole_vs_paper_below_endpoint`; a factor 4 in
Θ would be 41 % in g).

**Normalisation (inferred).** `K_MQS(M14)/K_MQS([S]) = 4`. [S] uses `⟨X²⟩ = π⁻¹∫S dω` with `S_V,th = 2kTR` and a
signal `S = (π/2)V²L` for a peak-V sine. [C]'s noise `(2/π)kTR/|Z|²` is [S]/π (an ordinary one-sided PSD per rad/s),
and its signal `|I|²/Δω` is (2/π)×[S] (the ½ of `⟨I²⟩ = |I|²/2` for a peak phasor is missing; [C]'s scale is the peak
`gB√(2ρ)`). The per-bin SNR is therefore 2× [S]'s, Θ 4×, and the absolute reach √2 optimistic in g. With [S]'s
defaults (Q_a = 6.8e5, ρ = 0.3) the total is ≈ 13× in Θ, 1.9× in g. M12 differs from [C] in two further ways: it
charges no time to tunings where X ≤ 0, while [C] charges them their MQS time and then discards the tunings, and it
uses [P]'s text normalisation (`c_s = π`). Ratios (`Θ/Θ₉₅` at an imposed endpoint) are unaffected.

**[C] against M12–M13, term by term.**

| item | [C] (M14) | M12–M13 |
|---|---|---|
| amplifier noise, frequency weights | imprecision `∝ (ω/ω_k)²` (current), backaction `∝ ω_k R_k` (voltage) | imprecision `∝ ω_k\|Z\|²/R_k`, backaction `∝ R_k ω²/ω_k` |
| normalisation, Q_a, ρ | π⁻¹ PSDs, Q_a = 10⁶, ρ = 0.45 | c_s = π, Q_a = 6.8e5, ρ = 0.3 |
| schedule | discrete comb, step `max(10⁻⁶, 1/Q_B)`, dwell × `max(1, 10⁶/Q_B)` | continuum, δ = 10⁻⁶, MQS dwell |
| plotted curve | Gaussian σ = 101 grid points, every 100th point | evaluated at the requested frequencies |
| thermal noise | `ħω(2n+1) r(f)` | `2kT r(f)` |
| MQS allocation | sort by time, cumulative cut (+1 tuning) | top-down to the live time |
| masking | X < 0 tunings charged their MQS time, then dropped with it | X ≤ 0 tunings cost no time |

The amplifier-noise frequency weights are the difference that matters off resonance: `jf2022_theta(amp_model="m13")`
swaps them to M13's form, and on a synthetic one-pole table that swap changes g above the tunings (200 and 400 MHz) by
more than a factor 2 (`test_jf2022_port_matches_notebook_transcription_synthetic`). [C]'s weighting is the mirror image
of [S]'s SQUID model (S3.5–S3.6: flat imprecision current, ω²-weighted backaction voltage, i.e. M13's form); M14
reproduces [C]'s choice, and `amp_model="m13"` gives [S]'s. The tests use only the 19 % epoxy toroid table behind
Fig. S3; the other multipole figures (Figs. 2/S2, Fig. S6) need tables that are not available.
