# SRF cell: mode mixing by wall deformations

!!! warning "Beta"
    The SRF modules (`axionfem.srf`, `axionfem.heterodyne`) are beta.

In a heterodyne SRF detector the pump mode (TM020) and the signal mode (TE011) of an elliptical cell are nearly
degenerate. A vibration of the wall mixes them, and the pump power that leaks into the signal mode is a noise
source. This page computes the mixing for the Project X 650 MHz β = 0.9 cell tuned to degeneracy (see
[README.md](README.md)). Script: `benchmarks/srf/srf_vibration_mixing.py` (stages `ceff`, `m2`, `report`), record
`benchmarks/srf/results/srf_vibration.json`; tests in `tests/test_srf_vibration.py`.

## The coupling

To first order (Slater perturbation theory) a normal wall displacement u_n couples two modes as

    g = (ω̄/4Ũ) ∮ dA u_n (B0·B1/μ0 − ε0 E0·E1) ≡ (ω̄/2)(u/L_c) η_mix,      L_c = V^(1/3).

For every body of revolution the integrand vanishes identically: the TM0 pump has B purely azimuthal and the TE0
signal has E purely azimuthal, so B0·B1 = E0·E1 = 0 pointwise. A vibration therefore mixes the pair only together
with a static out-of-round error of the cell, and η_mix is proportional to that error.

## 1. A random figure error: the coefficient c_eff

For a random, isotropic figure error of rms amplitude δ_s per octave of wavenumber, interacting with a diffuse
displacement field (local corrugation coupling −2(K_xK_y/K) B0 B1 h_K, averaged in quadrature), η_mix = c_eff δ_s/R_eq
with

    c_eff = (ln 2)^(−1/2) R_eq L_c sqrt(π I_4),   I_4 = ∮ (B0t B1t)² dA,   both modes normalized to ∫|B|² dV = 1.

This is a statistical model; only the surface fields are computed here. With the 2D wall fields of the cell (order 3,
angle_tol 0.4/n, |B_t| sampled 1e-6 R_eq inside the wall, 4 points per wall segment):

| h | c_eff (all walls) | cell only | G-average closed form |
|---|---|---|---|
| R_eq/20 | 3.7391 | 3.7390 | 3.4088 |
| R_eq/40 | 3.7384 | 3.7383 | 3.4075 |
| R_eq/80 | **3.7380** | 3.7379 | **3.4069** |

- c_eff is converged to 1e-4 in h. The beam tubes contribute 5.4e-5 of I_4 (2.7e-5 of c_eff).
- The "G-average closed form" replaces I_4 by its value for wall fields uniform over the wall, from (G, ξ,
  A/V^(2/3)) alone; it checks the normalization. The resolved value lies 10 % above it because I_4 is weighted toward
  the iris-side band where the pump's wall peak sits.
- c_eff depends on the cell's resolved wall peaks: an under-resolved B_pk (for example on a coarse structured mesh)
  gives a c_eff a few per cent low.

## 2. A coherent m = 2 (elliptical) deformation in 3D: a selection rule

**Setup.** The 3D mesh of the cell (h = R_eq/12, order 3, the same mesh for every ε) is mapped affinely, x → (1+ε)x,
y → (1−ε)y. This gives an elliptical cell with wall displacement ε r cos 2φ. Each deformed cell is re-solved with field
probes on a meridional grid (735 points × 8 azimuths). Every eigenmode's E is projected on the 2D TM020 (E_r, E_z) and
TE011 (E_φ) fields after averaging over φ (the m = 0 part).

**Extraction.** In the (TM020, TE011) subspace the coupling is |W| = Δω_pair sin θ cos θ, with θ the TE content of the
TM-like mode, which equals the TM content of the TE-like mode. A same-pattern vibration on top of a static ε would give
g = 2wεω̄ with W = wε²ω̄, so η_mix = 4wεL_c/R_eq and c_m2 = 4wL_c/R_eq.

| ε | f TM-like / TE-like [GHz] | θ (TM-like / TE-like) | \|W\|/ω |
|---|---|---|---|
| 0 | 1.416741 / 1.416788 | 6.9e-5 / 7.1e-5 | 2.3e-9 (mesh asymmetry floor) |
| 0.01 | 1.417550 / 1.417104 | 3.7e-5 / 3.1e-5 | 1.1e-8 |
| 0.02 | 1.419938 / 1.418044 | 1.6e-5 / 9.7e-6 | 2.1e-8 |
| 0.04 | 1.428971 / 1.421683 | 9.0e-6 / 6.4e-7 | 4.6e-8 |

There is no ε² coupling. |W|/ω stays at the 1e-8 level, below 5e-8 at ε = 0.04, while the diagonal shifts reach 9e-3.
The bound is w < 2.9e-5 at ε = 0.04 (c_m2 < 1.5e-4; the record's 7.96e-5 uses the mean of the two θ estimates), and
w < 1.1e-4 (c_m2 < 6e-4) at ε = 0.01. The two θ estimates disagree by ×14 at ε = 0.04 (W/ω = 4.6e-8 vs 3.3e-9), as
expected for the mesh-asymmetry noise floor (θ ≈ 1e-5–7e-5).

**Why this is exact for the static ellipse.** Under a reflection through any plane containing the axis (φ → −φ), the
polar E maps as (E_r, E_φ, E_z) → (E_r, −E_φ, E_z) and the axial B as (B_r, B_φ, B_z) → (−B_r, +B_φ, −B_z). So every
m = 0 TM mode (E_r, E_z, B_φ) has eigenvalue +1 and every TE0 mode (E_φ, B_r, B_z) −1, for *every* axial plane. The
affine x(1+ε), y(1−ε) cell keeps the two axial mirrors σ_xz, σ_yz (group C2v; the midplane mirror makes it D2h, but
the axial mirrors are what matters). TM-like modes are A1 and TE-like modes A2, so no perturbation that keeps an
axial mirror can mix them, at any order. **One axial mirror plane suffices**: any static error with a single mirror
plane through the axis (for example any cos mφ series about a common origin) leaves the pair unmixed. In the Slater
form ∮ δ(φ, z)(B0·B1 − E0·E1) dS the unperturbed integrand vanishes pointwise for every δ(φ, z); with the static-ellipse
fields B_a·B_b is odd under both mirrors (∝ sin 2kφ), so a cos 2φ vibration projects to exactly zero and a sin 2φ
vibration is allowed by symmetry.

**What the rule does and does not cover (vibration on top of a static error).** The coupling vanishes when the static
error and the vibration together keep a common axial mirror plane:

- a vibration of any A1 shape relative to the static ellipse (the same cos 2φ ellipse, a breathing mode, …): zero;
- a rigid rotation of the same ellipse (a sin 2φ vibration with the *same* (r, z) profile f(z) as the static ε f cos 2φ,
  which is again an ellipse, only rotated): its first-order term is ⟨b|J_z|a⟩α (J_z, the rotation generator of a
  vector field). TM020 is odd and TE011 even under the midplane mirror z → −z, and J_z commutes with that mirror, so
  this term is **exactly zero** for this pair, not merely suppressed by the pair splitting. Any z-symmetric
  deformation product leaves the pair unmixed for the same reason.
- **any other m = 2 pattern couples the pair at O(ε δ ω).** For a static ε f(z) cos 2φ plus a vibration δ h(z) sin 2φ,
  the sum has an axial mirror (at angle θ/4) only if ε f (cos θ − 1) + δ h sin θ = 0 for all z, which forces h ∝ f.
  Otherwise the shape is chiral, and once the product has a z-odd part it couples TM020/TE011 (opposite midplane
  parity). The twisted control below is exactly this case: R(a(z)) diag(1+ε, 1−ε) R(−a(z)) ≈ ε cos 2φ + 2ε a(z) sin 2φ
  with a ∝ z, and it mixes with w ≈ 0.7, c ≈ 3.3–3.6. Realistic flexural vibrations at 0.1–1 MHz have 5–15 mm
  wavelengths, so their (r, z) profile is generic: they are of this coupling type, not of the A1 type.

This is derived, and computed for the two 3D cases above and below.

**Positive control: the method sees allowed couplings.** The same mesh was mapped with the ellipse axes twisted to
±45° at the two ends (R(a) diag(1+ε, 1−ε) R(−a), with a = (π/4) z/z_end). This chiral cell has no mirror plane
through the axis. At h = R_eq/8, order 2:

| ε | θ (TM-like / TE-like) | \|W\|/ω | w = W/(ε²ω) | 4wL_c/R_eq |
|---|---|---|---|---|
| 0.02 | 0.291 / 0.286 | 2.85e-4 | 0.71 | 3.6 |
| 0.04 | 0.209 / 0.212 | 1.01e-3 | 0.64 | 3.3 |

The two estimates of θ agree, and W scales as ε² to within 10 %. The coupling is ~2e4 times the untwisted value at the
same ε. On the coarser mesh the untwisted floor is 2.6e-7 (`m2/n8_o2`).

**Reading.**

- The statistical c_eff of Sec. 1 is not contradicted by the selection rule: random errors and generic vibrations
  contain chiral (mirror-breaking) components.
- A *coherent* global m = 2 flattening couples the pair only for a vibration that does not share the static
  ellipse's mirror plane; it is **not** zero by symmetry for a generic vibration pattern. For a *diffuse* vibration
  field it matters little for another reason, the **wavenumber mismatch**: a global m = 2 error has K ≈ 2/R_eq,
  far below the elastic wavenumbers k_el, and the random-error model sees only the error spectrum at k_el.
- A coherent coefficient derived from Lasenby's Eq. 23 (40 kV/m per f = 1e-3 at 0.2 T) is 3.48× too low: a 3D solve
  and the first-order theory give 139 kV/m ([README.md](README.md), External benchmarks 4).
- The coherent number for the chiral (twisted-ellipse) pattern, 3.3–3.6, happens to be close to c_eff. That is a
  coincidence of this pattern, not a check of c_eff.

## Limitations

- c_eff rests on the random-figure-error model: an isotropic random figure error, a diffuse displacement field, and
  δ_s per octave. Only the surface fields are computed.
- A 3D solve with a *random* chiral figure error, which would test that model directly, is not part of this
  validation. It would need a mesh map with ~cm-scale ripple at k_el, far below the R_eq/12 mesh.
- The twisted control is h = R_eq/8, order 2 only. It shows sensitivity and is not a converged coefficient.
- These computations use the faceted wall (`wall="polygon"`) of the 2D and 3D cell.
