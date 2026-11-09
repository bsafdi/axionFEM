# Surface resistance of RF walls: normal skin, anomalous skin, BCS, flux flow

!!! warning "Beta"
    The surface-resistance models are beta.

Implementation: `cpp/src/surface_resistance.cpp` (`axf::rs`), used through `spec::RFLoss::Rs(f)`; design format:
[design_format.md](../design_format.md) (`rf` models); tests: `cpp/tests/test_surface_resistance.cpp`,
`tests/test_surface_resistance.py`.

Conventions: SI, time dependence e^{iωt}, Z_s = R_s + iX_s = E_t/H_t at the wall, ω = 2πf. Q from a wall:
Q = G/R_s with G = ω∫μ|H|²dV / ∮|H_t|²dA (perturbative: R_s ≪ ωμ × size). Every model below is a function
R_s(f) per material; the solvers weight it by |H_t|² of the lossless mode (2D, cross-section and 3D cavity Q0,
2D driven sweep).

## 1. Classical skin effect (`skin`)

Local Ohm's law j = σE in a good conductor (σ ≫ ωε): E ∝ e^{-(1+i)z/δ}, δ = √(2ρ/μ0ω),

    Z_s = (1+i) ρ/δ,   R_s = √(π f μ0 ρ).

Valid for mean free path ℓ ≪ δ, ωτ ≪ 1, wall thickness and curvature radii ≫ δ (textbook; e.g. Jackson Sec. 8.1,
Pozar Sec. 1.7). Copper resistivity (CRC Handbook, 84th ed.): 1.543 (273 K), 1.678 (293 K), 1.712 (298 K), 1.725 (300 K)
×10⁻⁸ Ω m; 100 % IACS = 1.7241×10⁻⁸ Ω m (20 °C). Note 1.68×10⁻⁸ is the 293 K value, not the 300 K one (−2.6 % in ρ,
−1.3 % in R_s).

### 1.1 The builtin `copper` and the anomalous regime

**The builtin material `copper` is `anomalous`** (diffuse p = 0, quasi-static, ρℓ = 6.6×10⁻¹⁶ Ω m²) with
ρ = 1.7×10⁻¹⁰ Ω m (RRR ≈ 100 residual resistivity, mean free path ℓ = 3.9 μm) and T_K = 0.03 K. The classical `skin`
model with the same ρ is the builtin **`copper_classical`**, for like-for-like comparisons with classical impedance
boundary conditions such as COMSOL's. Ratio classical/RS for this ρ (exact diffuse RS integral; scipy quad and
`axf::rs` agree):

| f | 0.12 MHz | 1 MHz | 5 MHz | 10 MHz | 30 MHz | 100 MHz | 200 MHz | 1 GHz | 5 GHz |
|---|---|---|---|---|---|---|---|---|---|
| α | 0.063 | 0.52 | 2.6 | 5.2 | 16 | 52 | 105 | 525 | 2600 |
| R_skin / R_RS | 0.996 | 0.976 | 0.934 | 0.906 | 0.848 | 0.770 | 0.721 | 0.604 | 0.492 |

So the classical model is adequate only below ~1 MHz (≤ 2.4 % in R_s). For a scan that reaches tens to hundreds of
MHz (DMRadio-m³-like) the classical R_s is low by 9–23 % (the RS value is 10–30 % higher), so Q and T/Q come out
optimistic, and for cavities (GHz) it is low by 40–50 % (Q0 high by 1.65–2.2×). At α → 0 the two coincide (1 kHz:
4.4×10⁻⁵; 10 Hz: 4.5×10⁻⁷). `copper_classical` (and any `skin` wall) warns in the cavity loss evaluation at α > 1;
the MQS pickup-loss report does not warn for a `skin` wall in the anomalous regime.

The builtin-copper R_s enters every loss output only through `RFLoss::Rs`, and the geometry factors Γ and G do not
depend on it, so each loss output scales by R_RS/R_skin at its frequency between the two materials. The validation
benchmarks are like-for-like: the cavity validations (`benchmarks/cavity_2d_validation.py`, `tests/test_cavity.py`)
and the published cavity reproductions (classical impedance-BC references: CST, COMSOL, analytic) define explicit
`skin` walls with the reference σ; the arXiv:2211.00008 reproductions (`benchmarks/beyond_mqs_2211`) use
superconductor walls and COMSOL impedance tables; the published DMRadio benchmarks compare L and Ψ only (their
optional `copper_rf` is an explicit 293 K `skin` material). The cached field solutions in
`benchmarks/records/finite_band_fullwave.json` carry surface losses computed with the classical copper model; that
record's reach numbers do not use them, because both of its cases refuse the local reduction and R_rad dominates at
the top of the sweeps.

## 2. Anomalous skin effect (`anomalous`)

### 2.1 Model

When ℓ ≳ δ (pure metals at cryogenic temperature) the current at depth z depends on the field over a mean free path
around it (Pippard 1947; Reuter & Sondheimer 1948, Proc. R. Soc. A 195, 336, doi:10.1098/rspa.1948.0123). For an
isotropic Fermi surface in the relaxation-time approximation the bulk transverse conductivity at wave number q is

    σ(q) = σ0 F(qℓ),   F(t) = 3[(1+t²) arctan t − t] / (2t³),   F(0) = 1,   F(t) → 3π/(4t) (t → ∞).

The boundary condition for electrons hitting the surface is a specularity p (fraction reflected specularly; p = 0
diffuse). The half-space problem then has the closed forms (RS 1948; Dingle 1953; in SI as Cahill et al., IPAC2016
MOPMW038, Eqs. 3-8; the specular one follows from the even extension of the field):

    p = 0:  Z_s = iπωμ0 / ∫₀^∞ ln[1 + iωμ0σ(q)/q²] dq
    p = 1:  Z_s = (2iωμ0/π) ∫₀^∞ dq / [q² + iωμ0σ(q)]

In t = qℓ these depend only on β = μ0ωℓ²/ρ = 4α/3, where α = (3/2)(ℓ/δ)² = (3/4)μ0ωσ0ℓ² is the RS/Chambers
anomaly parameter. Both reduce to Sec. 1 when F → 1 (ℓ ≪ δ; ∫ln(1 + a/t²)dt = π√a).

**Extreme anomalous limit** (ℓ → ∞ at fixed ρℓ, using F ~ 3π/4t and ∫₀^∞ ln(1 + u⁻³)du = 2π/√3):

    R_∞ = [ √3 ρℓ (μ0ω)² / (16π) ]^{1/3}  (p = 0),    X_∞ = √3 R_∞,    R_∞(p = 1) = (8/9) R_∞(p = 0).

R_∞ depends only on ρℓ = m v_F/(n e²), not on purity: R_s ∝ f^{2/3}, so at fixed geometry Q0 ∝ f^{-2/3}, and raising
RRR stops helping once ℓ ≫ δ. Copper: R_∞ = 1.1234 mΩ (f/GHz)^{2/3} (diffuse, ρℓ = 6.6×10⁻¹⁶ Ω m²).
Sources for the limit: Métral, arXiv:1108.1643 Eqs. (19)-(21) (diffuse, with ρℓ = 6.6e-16 and 1.123 mΩ at 1 GHz);
Valentinis, arXiv:2003.06619 Eqs. (105)-(106) (specular, and "diffuse = 9/8 × specular"). The derivation here and a
numerical evaluation of the exact integral agree.

**Chambers' fit** across the crossover (Chambers 1952, Proc. R. Soc. A 215, 481, Eq. A.1; quoted by Métral and by
Ratzinger & Wang arXiv:2211.00135 Eq. 42): R_s/R_∞ = 1 + F α^{-G}, (F, G) = (1.157, 0.2757) diffuse and
(1.376, 0.3592) specular (relative to the specular R_∞), for α ≥ 3. It does not recover the classical limit and is
not used by the code; it agrees with the exact diffuse integral to < 2×10⁻³ for α = 3 … 10⁹ (unit test).

**General specularity 0 < p < 1.** There is no closed form. The code interpolates between the exact ends,
R(p) = R(1) + w(α, p) [R(0) − R(1)], with w tabulated (log10 α = −2 … 12 in steps of 0.5, p in steps of 0.1;
bilinear) from a numerical solution of the RS half-space integral equation,

    E''(z) = iωμ0 ∫₀^∞ [K(z − z') + p K(z + z')] E(z') dz',   K(s) = (3/(4ρℓ)) ∫₀¹ (1−u²)/u e^{−|s|/(ℓu)} du,

(`benchmarks/surface_resistance_specularity.py`; log `axf_wt/cavity_logs/rs/table_generation.log`): the solver
reproduces the p = 0 and p = 1 closed forms to 1×10⁻⁵ … 1×10⁻⁴ over the table, and w carries the remaining error to
R at ≲ 3×10⁻⁴ (off-grid test points: 2×10⁻⁴ at worst). w(α, 0.5) ≈ 0.40-0.55: a linear interpolation in p (w = 1 − p) overestimates R(0.5) by up to 1.5 %.
The specular/diffuse ratio is not monotone in α: R(1)/R(0) = 0.94 (α = 1), 0.844 (α ≈ 10³), → 8/9 (α → ∞).

**Relaxation (ωτ not small).** With τ = ℓ/v_F, σ(q, ω) = σ0/(1+iωτ) F(qℓ/(1+iωτ)) (Chambers; Cahill et al. Eqs. 5,
7; Xiao et al. arXiv:2004.11462 Eq. 2). The q-integrals above are evaluated with this σ when `v_F_m_s` > 0 (copper:
v_F = 1.57×10⁶ m/s). Size of the effect for copper (diffuse): −0.16 %, −0.53 %, −0.89 %, −1.5 % at 1, 5, 10, 20 GHz
for RRR 100 (ωτ = 0.016 … 0.31); −0.27 % … −1.7 % for RRR 1000 (ωτ = 0.16 … 3.1). Default: off (quasi-static).

### 2.2 Parameters

- **ρℓ (copper).** 6.6×10⁻¹⁶ Ω m² (Métral 1108.1643, with Chou & Ruggiero 1995); the free-electron value
  m v_F/(n e²) with n = 8.47×10²⁸ m⁻³, v_F = 1.57×10⁶ m/s is 6.58×10⁻¹⁶, so the number used is the free-electron one
  (the real Fermi surface of copper is not spherical — necks — and values fitted to anomalous-skin data scatter by a
  few %); Ciovati (arXiv:1501.07398, Eq. 18) uses 6.8×10⁻¹⁶ (his ρℓ units are misprinted "Ω·m⁻²", next to a
  missing π). The ±3 % spread changes R_∞ by ±1 %. No default: `rho_l_ohm_m2` is required.
- **ρ at temperature.** `rho_ohm_m` (resistivity at T_K) or `RRR` with `rho_300K_ohm_m` (ρ = ρ_300/RRR). RRR is the
  standard residual-resistivity ratio ρ(room T)/ρ(4.2 K), so ρ_300/RRR is the residual resistivity: it is accepted
  only for T_K ≤ 20 K (below ~10–20 K pure copper is at its residual value, so ρ(150 mK) = ρ(4 K)); above 20 K the
  phonon part (Matthiessen ρ = ρ_ph(T) + ρ_res; copper ρ(77 K) ≈ 2×10⁻⁹ Ω m) is not negligible and the parser asks for
  `rho_ohm_m` at T_K. RRR conventions differ (ρ(300 K), ρ(295 K) or ρ(273 K) over ρ(4.2 K): Calatroni arXiv:2006.02842
  Eq. 7 uses 300 K; Métral 273 K). Give the room-temperature resistivity that your RRR refers to.
- **Specularity.** p = 0 (diffuse) is the usual assumption for machined/etched copper; measured cavities lie below
  the p = 0 prediction anyway (Sec. 2.4), so p > 0 is rarely justified.

### 2.3 Regime of validity

Isotropic free-electron RS model; smooth surface (roughness comparable to the anomalous penetration depth X_∞/(ωμ0) ≈ 0.25 μm (f/GHz)^{-1/3} raises R_s); wall thickness
≫ ℓ and δ; no static magnetic field. In a static field B the cyclotron radius r_c = m v_F/(eB) ≈ 1.1 μm at 8 T for
copper is below ℓ = 3.9 μm (RRR 100): the RS model no longer applies and magnetoresistance (Kohler) and
cyclotron-orbit effects enter. The code therefore refuses `magnetoresistance` together with `anomalous`. **At
multi-tesla fields `anomalous` is an uncontrolled estimate**: feeding it a field-enhanced ρ is not a fix (with ρℓ
fixed, a larger ρ means a shorter ℓ, whereas orbit bending changes the trajectories, not the scattering time). The
primary route at field is a measured R_s(f) at the operating field (`table`); the zero-field `anomalous` value is then
at best a lower bound on the loss of a clean surface. The quasi-static default additionally needs ωτ ≪ 1 (above), and
relaxation combined with 0 < p < 1 uses the quasi-static weight w(α, p) (the ends include ωτ; the error is second
order in the ωτ-induced change of w and has not been quantified).

### 2.4 Comparison with measured copper cavities

- Cahill et al., "Measurements of copper RF surface resistance at cryogenic temperatures...", IPAC2016 MOPMW038,
  Table 1: S-band TM010 pillboxes (G = 252 Ω), Q0 = 17184 (293 K, 2.845 GHz) → 82931 (4 K, 2.855 GHz), ratio 4.83;
  UCLA twin 4.78. With ρ(293 K) = 1.678×10⁻⁸/0.95 (95 % IACS, as in the paper) the diffuse RS ratio is 4.35, 4.85,
  5.37, 5.93 for RRR 30, 50, 100, 400. The paper assumed RRR 400 and notes that theory comes out higher; the data
  correspond to an effective RRR ≈ 50 (test `test_copper_cavity_data_point`).
- X-band TE013 (11.424 GHz) in the same paper: R_s(300 K)/R_s(4 K) = 3.93, model 4.85 for RRR 400.
- Ratzinger & Wang, arXiv:2211.00135: 340 MHz coaxial cavity; the Chambers diffuse model overestimates Q by ~16 % for
  a 500 MHz cavity they cite.
- Rules of thumb: "Q of a 1 GHz cavity improves by about a factor of 4 from room temperature to 4 K" for OFHC copper
  (Brubaker thesis, arXiv:1801.00835, fn. 59); model (diffuse, ρ_300 = 1.725e-8): 5.4 (RRR 50), 6.1 (RRR 100),
  6.9 (RRR 400) at 1 GHz.

So measured cold Q0 sits below the diffuse-RS value: ~80-86 % in the two quantitative comparisons above, lower in
the rule of thumb and in cavities with joints and tuning rods. The model is an upper bound for a clean surface.

## 3. Superconductors: BCS + residual (`bcs`)

For T < T_c/2 and ħω ≪ k_BT ≪ Δ the Mattis-Bardeen surface resistance is (Ciovati arXiv:1501.07398 Eq. 34)
R_s ≃ (μ0²ω²λ³σ1Δ/k_BT) ln(2.246 k_BT/ħω) e^{−Δ/k_BT}: ∝ ω² (up to the log) and e^{−Δ/kT}/T. The engineering form used
here absorbs the slowly varying factors in a prefactor:

    R_s = A/T (f/f_ref)² exp(−(Δ/k_BT_c) T_c/T) + R_res.

Niobium: A = 2×10⁻⁴ Ω K at f_ref = 1.5 GHz, Δ/k_B = 17.67 K (Dhakal et al., arXiv:1202.0811 Eq. 4, "valid for
T < 0.5 T_c", T_c = 9.25 K, so Δ/k_BT_c = 1.910; also Podlech arXiv:1303.6552 Eq. 3, printed with a sign typo).
At 1.3 GHz this gives 10.93 nΩ (2 K) and 532.5 nΩ (4.2 K). Reference values from full BCS: TESLA (Aune et al.,
PRST-AB 3, 092001, Sec. 2.1.1) quote ≈ 15 nΩ (2 K) and ≈ 800 nΩ (4.2 K) at 1.3 GHz, i.e. the fit is ~30 % below
them; fitted Δ/k_BT_c for Nb ranges 1.83-1.92 and T_c 9.2-9.25 K across sources, and A depends on the mean free path
(λ_eff = λ_L√(1+ξ0/ℓ)). Residual resistance: a few nΩ for very pure, well-prepared Nb (TESLA: 3 nΩ ↔ Q0 = 10¹¹), up
to ~100 nΩ (Ciovati Sec. 3.2.3); it is preparation dependent and is required input (`R_res_ohm`).

**Frequency scaling.** The plain form scales A by f² and drops the factor ln(2.246 k_BT/ħω) of Ciovati Eq. 34. Away
from f_ref this biases R_BCS: from 1.5 to 10 GHz the log factor is 0.611× (4.2 K) and 0.541× (2 K) of its f_ref value,
so the f² law overestimates R_BCS by 1.64× and 1.85×. Option `log_factor: true` multiplies by
L(f, T)/L(f_ref, T), L = ln(2.246 k_BT/hf), i.e. R ∝ (f²/T) ln(2.246 kT/hf) e^{−Δ/kT} normalized at f_ref (requires
h f_ref < k_BT; L is clamped at 0 where hf > 2.246 kT). This restores the Mattis–Bardeen frequency dependence of the
low-T limit but not the mean-free-path dependence of A or the departure from the asymptotic form; it is still an
approximation. Without `log_factor` the cavity loss evaluation warns when L(f)/L(f_ref) differs from 1 by more than
10 % (for f_ref = 1.5 GHz: f outside 0.66–1.5 f_ref at 2 K, 0.61–1.6 f_ref at 4.2 K), and with either form it warns when
hf > k_BT. **Recommended above ~3 GHz (or far from f_ref): the tabulated R_BCS(f, T)** from a full BCS code (Halbritter's
SRIMP or equivalent) or measurement: interpolation log-log in f and ln R linear in 1/T (the Arrhenius dependence), no
extrapolation.

Validity: low RF field (no Q-slope), T < T_c/2 (enforced), ħω ≪ k_BT (at 20 mK and GHz, ħω > k_BT and the formula
is not valid, but R_BCS is then negligible against R_res; a warning is issued).

Nb3Sn: T_c ≈ 18 K (Trenikhina et al., arXiv:1707.05856); Δ/k_BT_c ≈ 2.2-2.5 (UNVERIFIED here, Posen & Hall SUST 30,
033004 (2017)); no A is recommended by this note.

## 4. Vortex (flux-flow) losses in a static field (`flux_flow`, additive; LARGE UNCERTAINTY)

A type-II superconductor in a static field H_c1 < H < H_c2 (the haloscope magnet) is in the mixed state; the RF current
moves the vortices. Overdamped, pinned vortex with thermal creep (Gittleman & Rosenblum, PRL 16, 734 (1966); Coffey &
Clem, PRL 67, 386 (1991); in the e^{iωt} convention of Pompeo & Silva, PRB 78, 094503 (2008), Eq. 2): η u̇ + k_p u =
Φ0 J (+ creep) gives the vortex resistivity

    ρ_v = ρ_ff (ε + i f/f0) / (1 + i f/f0),

- ε = 0 (no creep): ρ_v = ρ_ff/(1 − i f_p/f), Re ρ_v = ρ_ff f²/(f² + f_p²), f0 = f_p = k_p/(2πη) (depinning frequency);
- ε ∈ (0, 1] (creep): ρ_v → ε ρ_ff at f ≪ f0 (flux creep) and ρ_ff at f ≫ f0; in Coffey–Clem ε and f0 follow from
  the pinning barrier U/kT (Pompeo & Silva Eqs. 3–5); the code takes the measured/fitted pair (ε, f0) directly,
- f0 = 0: free flux flow ρ_v = ρ_ff,

with the Bardeen–Stephen flux-flow resistivity ρ_ff = ρ_n |B|/B_c2 (η = Φ0B_c2/ρ_n; Gurevich & Ciovati PRB 87, 054502
Sec. V). Inputs: `rho_n_ohm_m` (normal-state ρ just above T_c), `lambda_m` (London depth), `Bc2_T`, `f_depin_Hz` (f0),
`creep_epsilon` (ε, default 0), `B_T` (|B| in the wall, or the local static field), `orientation`.

**Surface impedance.** For a thick wall with local London + vortex response (no quasiparticle term — the
zero-field BCS/residual part is added separately) the Coffey–Clem complex penetration depth λ̃² = λ² − iρ_v/(μ0ω)
gives

    Z_s = iμ0ω λ̃ = √( iμ0ω (iμ0ωλ² + ρ_v) ),     ΔR_s = Re Z_s − Re Z_s(ρ_v = 0) = Re Z_s,

valid for any ratio |ρ_v|/(μ0ωλ²):

- |ρ_v| ≪ μ0ωλ² (strong pinning, low field): ΔR_s → Re ρ_v/(2λ) (the additive "thick London" term);
- |ρ_v| ≫ μ0ωλ² (vortex dominated): ΔR_s → Re √(iμ0ωρ_v), √(μ0ωρ_ff/2) = R_n,bulk √(B/B_c2) for free flux flow.

At haloscope parameters (5 GHz, 8 T) the ratio is ~100 (Nb3Sn: ρ_n 1.5e-7 Ω m, λ 110 nm, B_c2 25 T; NbTi: 6e-7, 300 nm,
11 T), and the full form gives 0.031 Ω and 0.092 Ω where the additive term gives 0.22 and 0.73 Ω (reproduced by the
unit test). Not included: quasiparticle
(normal-fluid) conduction in parallel (Coffey–Clem's 1 + 2iλ²/δ_nf² denominator; second order for T ≪ T_c), films
thinner than λ, the nonlinear B-dependence of ρ_ff (TDGL: ρ_ff/ρ_n = αB/[(α−1)B + μ0H_c2], α ≈ 0.4, Ghigo et al. Sci.
Rep. 13, 9315 Eq. 12), the anisotropy of B_c2 and of pinning, and trapped-flux hot spots (Gurevich & Ciovati find the
older (B/2B_c2)√(μ0ρ_nω/2) estimate low by ~4× for Nb and ~40× for Nb3Sn). B below B_c1 is accepted (field-cooled
walls trap flux; in a zero-field-cooled wall below B_c1 there are no vortices and the term should be dropped).

**Field orientation.** The RF current drives a vortex through the Lorentz force per length Φ0 J × v̂
(v̂ ∥ B0), so only the current component perpendicular to the vortices moves them: the displacement and the
dissipation scale with sin²θ, θ = angle(J_rf, B0). The code applies this as ρ_v → s ρ_v with

    s = 1 − (Ĵ·B̂0)²,   Ĵ ∥ n̂ × H  (2D: TM modes (−n_z, n_r) in the (r, z) plane; TE modes φ̂),

per wall quadrature point, with B0 the static field the form factor uses (`cavity.form_factor.B0`, default the applied
field with superconducting walls transparent) and |B| = |B0| there unless `B_T` is given; the wall's ΔR_s is the
|H_t|²-weighted mean of ΔR_s(B, s). `orientation` may instead be a fixed s ∈ [0, 1]; engines that do not know the
current direction (driven sweep, MQS losses) use s = 1, the upper bound; the 2D, cross-section and 3D cavity Q0 apply s.
In a solenoid haloscope this is the dominant effect: the TM010 barrel current is axial, parallel to B0, so s = 0 and
the barrel — which carries most of the TM010 wall loss — has no flux-flow loss in this model, while the end caps (B0
normal to the surface, current radial) get the full term; TE0 modes (azimuthal current) get it everywhere (test
`test_flux_flow_orientation_pillbox_axial_field`).

*Uncertainty of the orientation factor.* sin²θ is the standard angular law for a rigid, straight vortex lattice
(e.g. the Lorentz-force-free configuration J ∥ B of magnetoresistance experiments). Real walls deviate: a
misalignment δθ between magnet axis and barrel gives s ≈ δθ² (1 mrad: 1e-6, 1°: 3e-4); surface roughness tilts the
local normal and the vortex ends; vortices parallel to the surface sit in a surface layer with a Bean–Livingston barrier
and pin differently from the bulk; J ∥ B is subject to helical (force-free) vortex instabilities above a critical
current. Measured "parallel-field" losses are therefore small but not zero. Treat the barrel value as a lower bound and
bracket it with a fixed `orientation` (e.g. 1e-3 … 1e-2) for sensitivity studies. Measured in-field cavities (below)
lie well below the perpendicular-field estimate, consistent with orientation dominating.

**Use it for orders of magnitude and parameter studies; for predictions use a measured R_s(f) at the operating
field (`table` or `power`).** Parameters (ρ_n, λ, B_c2(T), f0, ε) of thin films vary by orders of magnitude with
preparation. The cavity output reports per wall `Rs_flux_flow_ohm` (with orientation), `Rs_flux_flow_perpendicular_ohm`
(s = 1) and `flux_flow_orientation_factor` (|H_t|²-weighted s), and a warning.

Measured superconducting haloscope cavities in field (for orientation): NbTi (QUAX), Q0 = 4.5×10⁵ at 9 GHz, 2 T, 4 K
(Alesini et al., PRD 99, 101101, arXiv:1903.06547); Nb3Sn film, Q0 = (5.3 ± 0.3)×10⁵ at 3.9 GHz, 6 T, 4.2 K (Posen et
al., arXiv:2201.10733); REBCO polygon cavity, no significant degradation 1-8 T at 6.93 GHz (Ahn et al.,
arXiv:1904.05111). Trapped-flux sensitivity of Nb at GHz: ~0.3-2 nΩ/mG (TESLA paper Eq. 4: R_mag = (B/2B_c2)R_n,
0.35 nΩ/mG at 1.3 GHz; Bafia et al. arXiv:2503.14616: ~2 nΩ/mG at 6 GHz, 10 mK).

## 5. Numerics

- `skin`, `power`, `bcs` (analytic), `flux_flow` (complex square root, per wall point): closed forms.
- `anomalous`, p = 0 and p = 1: trapezoid rule in x = ln(q/q0), q0 = 1/(field decay length) (β^{1/3}/ℓ or β^{1/2}/ℓ),
  x ∈ [−50, 50], h = min(0.05, atan(1/ωτ)/8). The integrands are analytic in a strip of half-width π/4 (or atan(1/ωτ))
  and decay exponentially at both ends, so the error is ~exp(−2π·(π/4)/h) — below 10⁻¹². ~2000 evaluations, ~0.1 ms.
  Checked against scipy adaptive quadrature to 1×10⁻⁸ (limited by quad) for copper.
- `anomalous`, 0 < p < 1: as above for the ends, plus the tabulated w(α, p) (Sec. 2.1), accuracy ≲ 3×10⁻⁴ in R
  (7×10⁻⁵ on the p grid, 2.5×10⁻⁴ off it; the error is the linear interpolation in p, which a finer p step would
  reduce).
- Crossover tests: the exact p = 0 and p = 1 integrals at α = 0.01, 0.1, 1, 3, 10 against an independent scipy `quad`
  (piecewise in ln t, epsrel 1e-13) to 1×10⁻¹⁰ (`cpp/tests/test_surface_resistance.cpp`).
- Tabulated `bcs`: reduced to an R_s(f) table at T_K when the design is parsed.

## 6. Shared code and the 3D path

All models live in `cpp/src/surface_resistance.cpp` (`axf::rs`), called only through `spec::RFLoss`:
`Rs(f)` (= `Rs_base(f)` + `Rs_flux_flow(f)` with the fixed orientation or s = 1, and `B_T`), `Rs_base(f)` (without the
vortex term), `Rs_flux_flow(f, B, s)` and `warnings(f)` — the model/validity warnings (keys `flux_flow`,
`skin_anomalous`, `bcs_log`, `bcs_hf_kT`) that every engine reporting wall losses should emit once per material and
key. The 2D cavity Q0 (`model.cpp`) uses `Rs_base` + the per-point orientation-weighted `Rs_flux_flow` and emits
`warnings`. The cross-section solver (`xsec.cpp`) does the same with s = 0 on the barrel for TM_z (axial current) and s = 1 on
the end caps and for TE_z. The 3D cavity Q0 (`model3d_fullwave.cpp`) uses the same models: (1) it pushes
`mt->rf.warnings(f)` once per (material, key) into the result warnings; (2) it writes `Rs_model` and, with a
flux_flow term, `Rs_flux_flow_ohm`, `Rs_flux_flow_perpendicular_ohm`, `flux_flow_orientation_factor`; (3) it uses
`Rs_base(f)` plus `Rs_flux_flow(f, |B0|, s)` weighted per wall quadrature point by |H_t|² with s = 1 − (Ĵ·B̂0)², Ĵ
along the tangential consistent-flux surface current, B0 = `fullwave.source` (the applied field; s = 1 where B0
vanishes, warned). Tests:
`tests/test_cavity3d.py::test_builtin_copper_q0_matches_2d` (3D and 2D Q0 with builtin copper) and
`::test_flux_flow_orientation_3d_pillbox` (TM010 in an axial field: s = a/(a + L) for barrel + caps).
