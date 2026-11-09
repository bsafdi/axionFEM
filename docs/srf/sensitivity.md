# SRF heterodyne sensitivity — `python/axionfem/heterodyne.py`

**Beta.** `axionfem.heterodyne` is a pure-Python model (no FEM) of a heterodyne axion search in a superconducting RF
cavity: a pump mode (TM020) is driven, and the axion converts pump photons into the signal mode (TE011) when the axion
mass matches the mode splitting. The module computes the scan rate, the optimal readout coupling, the cavity volume a
scan target requires and the RF load; its cost metric is a scaling metric for comparing designs, not a construction
estimate. The cavity enters only through geometry figures of merit (`SRFGeometry`), which the eigenmode side,
`axionfem.srf`, computes ([README.md](README.md)).

## Physics core

- Signal power in the mode, `mode_power`: P_sig = ¼ g² η² B0² ρ V Q1/ω1 (= Q1/4 × `haloscope.power_coefficient` with
  ω → ω1, C → η²).
- Receiver factor `gbar_srf`: Ḡ_SRF = τ²β²/[(1+β)³ √A (A+D)^{3/2}], with the flat noise A = ½ + n_add + τ n_e +
  (1−τ) n_l and the resonant term D = 4τβ(n_i − n_e)/(1+β)².
- Optimal coupling `beta_opt`, the exact maximiser: d ln Ḡ/dβ = 0 reduces to β² − (1 + k/2)β − 2 = 0 with
  k = 4τ(n_i − n_e)/A, so β* = [(1+k/2) + √((1+k/2)² + 8)]/2 → 2τ n_i/A (k ≫ 1) and → 2 (k → 0).
- Scan rate `scan_rate` (in Hz/s), checked against a direct quadrature of the per-tuning Fisher density
  ∫dν (S_a/S_N)² (`scan_rate_from_psd`) to 1e-6.
- Scan target `DEFAULT_TARGET` (= `Target()`), an illustrative default (set a `Target` for your own search): DFSZ with
  the lattice C_γ = 1.77 (`haloscope.g_axion(..., convention="qcd")`), 120.9 kHz to 1 MHz in 3 live years,
  ρ = 0.3 GeV/cm³, Q_axion = 6.8e5, Θ95 = 2.71. The same integration and target code
  reproduces the package's lumped requirement (C++ `requirement`) to 2e-3.
- The weak-signal statistic assumes λa/λB ≪ 1 in every bin. `asimov_density` and `fisher_integral` give the exact
  exponential-likelihood Asimov density 2[ln(1+r) − r/(1+r)] (dense scan, standard-halo line, β re-optimised at every
  mass); the derivation is in `fisher_integral`'s docstring.

## Scan model

The module implements one scan model, called model A in the code (`ScenarioA`, `design_A`):

- **Cavity:** one elliptical cell, scaled: ω0 = ξ c V^(-1/3), Q = G/R_s.
- **Code:** `ScenarioA`, `required_volume`, `required_volume_exact`, `design_A`, `cost_live`, `SCENARIOS_A`.
- **Field:** B_pk at the wall, B0 = B_pk / (B_pk/B0) of the geometry.
- **Scan:** the weak-signal rate with β optimal at every mass and no schedule constraint; an optional adiabatic
  schedule (`adiabatic_eps`); the exact statistic in `required_volume_exact`.
- **Cost metric:** `CostLawA`: niobium area, cryostat area and refrigeration of the RF load. It is a scaling metric
  for comparing designs, not a construction estimate, and its default coefficients are illustrative (no source is
  claimed for them).

The model-A benchmarks (`SCENARIOS_A`) are R_s = 1.5 / 20 nΩ (both modes), B_pk = 150 / 90 mT, A = 1 / 10, τ = 1,
n_e = n_l = 0 and a wall temperature of 2 K / 4.5 K (optimistic / conservative); `ScenarioA.T_K` sets both the wall
occupation n_i and the refrigerator factor. These presets are illustrative benchmarks: replace them with the
parameters of the cavity and readout under study.

## Geometry: `SRFGeometry` and its presets

The figures of merit enter only through `SRFGeometry(xi, eta, G_pump_ohm, G_signal_ohm, Bpk_over_B0, area_coeff,
Epk_over_cB0)`; their definitions are in [README.md](README.md). `Epk_over_cB0` is E_pk/(cB0), dimensionless: the
pump's peak surface E is `Epk_over_cB0 · c · B0` (`Epk_V_per_m`). The constructor
validates the values: finite, 0 < η ≤ 1, 2 ≤ ξ ≤ 100 (angular ξ), G in [10, 1e5] Ω, 1 ≤ B_pk/B0 ≤ 10,
A/V^(2/3) ≥ 4.836 (the sphere). A value of η10² passed for η10 cannot be caught by a range check; `from_modes` maps
`eta_10` itself. Presets (`heterodyne.GEOMETRIES`):

| preset | ξ | η10 | G pump / signal [Ω] | B_pk/B0 | A/V^(2/3) | E_pk/(cB0) | source |
|---|---|---|---|---|---|---|---|
| `PROJECTX_086` = `PROJECTX_086_WALL` (`"computed_wall"`, alias `"computed"`): **the computed default** | 7.6430 | 0.44921 | 524.47 / 604.48 | **1.4640** (largest wall \|B\|) | 6.1822 | 1.843 | `axionfem.srf`, 2D, cached record |
| `PROJECTX_086_EQUATOR_DIAGNOSTIC` (`"equator_diagnostic"`) | 7.6430 | 0.44921 | 524.47 / 604.48 | 1.4039 (\|B\| at the equator; **diagnostic, not a field limit**) | 6.1822 | 1.843 | same |

`SCENARIOS_A` use the computed cell `PROJECTX_086` by default; `dataclasses.replace(sc, geometry=...)` selects another
geometry. B_pk/B0 = 1.464 is the largest pump |B| anywhere on the wall, converged (2D at h = R_eq/40 and
R_eq/80 agree to 4e-5; 3D gives 1.464). The equator value 1.404 is kept only to separate the B_pk effect from the
integral figures: with it, the iris-side wall would run 4.3 % above the field limit.

The computed presets are built by `SRFGeometry.from_modes(srf.figures_of_merit(...), bpk="wall" | "equator")` from the
cached record `python/axionfem/data/srf_projectx086_geometry.json`, written by
`benchmarks/srf/srf_geometry_record.py --update-package-data` (Project X 650 MHz β = 0.9 cell with axial factor 0.86,
50 mm × 100 mm beam tubes; 2D order 3, h = R_eq/80, angle_tol 0.005; a second solve at h = R_eq/40 differs by ≤ 4e-5 in
every figure except E_pk, 2.4e-3). Nothing is typed in by hand. The record stores its inputs (cell, axial factor,
tubes, mode pair, mesh) and a fingerprint (sha256 of `srf.py`, `git describe --dirty`); `heterodyne.check_record`
refuses a record for another cell or a mesh coarser than R_eq/80 or order 3, and warns if `srf.py` changed or the tree
was dirty. The presets load lazily, so a missing record raises `FileNotFoundError` at first access. Without
`--update-package-data` the script needs an explicit `--out`; it never overwrites the package record by default.

## Model A with the computed geometry

`benchmarks/srf/model_a_geometry.py` (about 3 min on 8 cores) evaluates model A for the two illustrative presets on
both computed geometry presets and writes `benchmarks/srf/results/model_a_geometry.json`; the tests compare the
module's derived results with that record (`test_derived_results_match_the_record`) and use its values as regression
values. "weak" is the weak-signal rate with the closed-form β*, "exact" the Asimov statistic with β re-optimised at
every mass (`required_volume_exact`, n_f = 32). What the record and the tests check are properties of the model, not
the sizing of a particular experiment:

- **Wall against equator B_pk** on the same cell is the B0 effect alone: the volume ratio equals
  (1.4640/1.4039)^(3/2) = 1.06495 to 1e-3 for both presets (`test_model_A_wall_vs_equator_is_the_B0_law`). The scaling
  law is V ∝ ξ^(3/4) (A T/(τ Q_i))^(3/8) (η B0)^(−3/2) (`volume_scaling_exponents`; the A T and B0 exponents are
  checked numerically in `test_model_A_volume_scaling_law`); G_pump and A/V^(2/3) enter only P_RF and the cost metric.
- **The weak-signal volumes are lower bounds.** Where the weak-signal optimum reaches a per-bin λa/λB of order one
  (`peak_bin_snr`; for low-noise receivers near the top of a band), the exact statistic needs a larger V and favours
  overcoupling, which lowers the peak λa/λB; the record holds both sizings, and the tests check the band ends of
  `required_volume_exact`.
- **Adiabaticity.** The weak-signal optimum can scan faster than the sufficient adiabatic condition
  ν̇ ≤ 0.1πΔc min(Δc, ν_a/Q_axion), typically near the top of a band. Raising β where needed
  (`required_volume(..., adiabatic_eps=0.1)`) restores the condition at a small increase in V
  (`test_model_A_adiabatic_schedule`). The exact-statistic optimum can exceed the condition as well, so the schedule
  can bind there too (`test_exact_optimum_still_needs_the_schedule_at_1MHz`).
- **Peak surface E.** E_pk = (E_pk/(cB0)) · c · B0 on the iris noses. Compare it with the peak surface E at which CW
  cavities operate (about 32 MV/m in LCLS-II): field emission (dark current, Q-slope, x-rays, extra RF load) is not
  modelled.

## Tests

`tests/test_heterodyne.py`: the receiver factor against the lumped Ḡ under the substitution of the noise terms, β* as
the maximum and its limits, the closed-form scan rate against the PSD quadrature, the signal power in natural units,
the lumped requirement through the shared conventions, the volume scaling law, the geometry construction and
validation (from modes and from `srf` figures of merit; range checks), the computed presets and the
record's fingerprint check, the second-level caches against the record (`test_derived_results_match_the_record`),
model A with the computed geometry, the wall-against-equator B0 law, the adiabatic schedule, the Asimov density
limits, the Fisher integral's weak limit and the peak per-bin SNR, the hot-port bath quadratures, the band ends of
`required_volume_exact`, the endpoint guard and the bandwidth estimate. The BCS wrapper `rs_bcs_repo` has no unit
test.

## Limitations

- The exact-statistic dense-scan model assumes a quasistatic cavity response; a transient analysis near 1 MHz has not
  been done.
- The field dependence of R_s and pump-induced signal loss are not modelled.
- The model-A wall temperature and refrigerator factor are inputs: `refrigerator_factor` knows only 2 K (3.3) and
  4.5 K (1); pass `ScenarioA.refrigerator_factor` for other temperatures.
