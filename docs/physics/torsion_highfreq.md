# High-frequency torsional mobility: multi-branch waveguide and wall-resolved FEM

!!! warning "Beta"
    `axionfem.torsion_waveguide` and the torsional analysis (`Results.torsion`, `Results.torsion_band`) are beta.

Validation: [validation/torsion_highfreq.md](../validation/torsion_highfreq.md) (`benchmarks/torsion_highfreq.py`,
records `benchmarks/records/torsion_highfreq_*.json`; tests `tests/test_torsion_highfreq.py`). The m = 0 torsional
module itself: [mechanics_thermal.md](mechanics_thermal.md#torsional-dynamics).

## Why

The Brownian spectrum S_q = 4 k_B T Re Y/ω² of the generalised coordinate q conjugate to a magnetic load is needed
at 0.1–1 MHz for vibration noise in lumped-element detectors. The loads are K × B on conductor surfaces, i.e.
azimuthal **surface tractions**. The one-branch T(0,1) infinite-waveguide asymptote holds only below the first
thickness-shear cutoff c_T/(2t) and for loads uniform through the wall; a surface traction above the cutoff drives
the T(0, n ≥ 2) branches and Re Y exceeds the T(0,1) value 4–17×. For a 1 cm steel wall the cutoff is 157 kHz, inside
the band. This page gives:

- **A**, the exact all-branch mobility of an infinite uniform tube for an arbitrary axisymmetric azimuthal load
  (volume and surface), damped, as `axionfem.torsion_waveguide`;
- **B**, the production FEM path: loads on named regions, surfaces and curves, band means with uncertainties, a
  mesh-resolution guard fixed by a convergence study, and a fast complex-symmetric direct solve;
- **F**, the exact modal sum of the finite free-free uniform tube, the third reference: it is the structure the FEM
  solves, and F − A is the finite-length scatter.

Time dependence e^{−iωt} throughout, as the FEM and the full-wave engine; structural damping μ → μ(1 − iη).

## A. The infinite tube, all branches

**Equations.** For m = 0 the azimuthal displacement u(r, z) decouples. In a tube a ≤ r ≤ b = a + t with
body force g(r, z) [N/m³] and surface tractions τ_a(z), τ_b(z) [Pa] on the inner and outer walls (positive
along +φ), the weak form per radian is, for every test function v,

    ∫∫ μ' [(∂_r u − u/r)(∂_r v − v/r) + ∂_z u ∂_z v] r dr dz − ρω² ∫∫ u v r dr dz
        = ∫∫ g v r dr dz + ∫ (a τ_a v(a, z) + b τ_b v(b, z)) dz,                   μ' = μ(1 − iη),

the traction-free conditions σ_rφ = μ(∂_r u − u/r) = 0 on the walls being natural.

**Radial eigenproblem.** The radial form a(ψ, φ) = ∫ (ψ' − ψ/r)(φ' − φ/r) r dr with weight r gives the self-adjoint
problem −(rψ')' + ψ/r = κ² r ψ with ψ' − ψ/r = 0 at r = a, b. Writing ψ = Z₁(κr) (Z a Bessel function) and using
Z₁'(x) − Z₁(x)/x = −Z₂(x), the wall condition is Z₂(κa) = Z₂(κb) = 0:

    ψ₀ = r,                                    κ₀ = 0  (T(0,1), the rigid-rotation profile)
    ψ_n = Y₂(κ_n a) J₁(κ_n r) − J₂(κ_n a) Y₁(κ_n r),   J₂(κa) Y₂(κb) − J₂(κb) Y₂(κa) = 0,   n ≥ 1

(a solid rod: ψ_n = J₁(κ_n r), J₂(κ_n b) = 0). κ = 0 is not a root of the Bessel determinant for a ≠ b
(it tends to (b² − a²)(b² + a²)/(2π a² b²) ≠ 0), so T(0,1) is the separate solution ψ₀ = r. The eigenvalues are
simple (Sturm–Liouville); for a thin wall κ_n ≈ nπ/t. With the wall condition Z₁' = Z₁/x, Lommel's integral gives the
norm in closed form, ∫ₐᵇ ψ_n² r dr = [r² ψ_n(r)²/2]ₐᵇ; the code normalises ψ̂_n = ψ_n/√N_n (orthonormal with weight r,
checked to 1e-6 by quadrature in the tests). The dispersion of every branch is exact:

    ω² = c_T² (k² + κ_n²),       f_n = c_T κ_n/(2π)  (cutoff),       v_g,n = c_T² k/ω.

**Projection.** Expand u = Σ_n ψ̂_n(r) U_n(z) and test with ψ̂_n(r) v(z). Orthogonality of the ψ̂_n in both the mass
and the stiffness form (a(ψ̂_m, ψ̂_n) = κ_n² δ_mn) gives, per branch,

    U_n'' + k_n² U_n = −G_n(z)/μ',     k_n² = ω²/(c_T²(1 − iη)) − κ_n²,
    G_n(z) = ∫ g(r, z) ψ̂_n(r) r dr + a τ_a(z) ψ̂_n(a) + b τ_b(z) ψ̂_n(b).

A surface traction projects through the wall values ψ̂_n(a), ψ̂_n(b), which do not decay with n, whereas a load
proportional to r (g = c r, the "torque-like" profile) projects on T(0,1) alone. A load uniform through the wall
projects weakly on n ≥ 1 (∫ψ̂_n r dr ≠ 0 but small); a surface traction projects with O(1) weight on every branch.

**Response.** The outgoing Green's function of U'' + k²U = −δ under e^{−iωt} is (i/2k) e^{ik|z|} (Im k > 0). With
q = 2π [∫∫ g u r dr dz + ∫ (a τ_a u(a) + b τ_b u(b)) dz] = 2π Σ_n ∫ G_n U_n dz,

    q = Σ_n (iπ/(k_n μ')) I_n(k_n),     I_n(k) = ∫∫ G_n(z) G_n(z') e^{ik|z−z'|} dz dz',     Y = −iω q.

For a separable load G_n(z) = R_n p(z) (R_n the radial projection, p the axial profile) I_n = R_n² I_p(k), with
closed forms (Im k ≥ 0, w the Faddeeva function):

    Gaussian p = exp(−(z−z₀)²/2σ²):   I_p = 2πσ² w(σk)
    top hat of width W:               I_p = 2(−W/α + (e^{αW} − 1)/α²),  α = ik
    ring δ(z − z₀):                   I_p = 1

Several terms, or a sampled profile, use the trapezoidal double integral on a uniform grid (with the kernel's kink
on the nodes), evaluated in O(N) per (branch, frequency) by the recursion A_j = e^{ikh} A_{j−1} + x_j, plus one
Richardson step. Its accuracy is limited by the grid step against the far evanescent branches' decay length 1/κ_n:
the tests hold it to 1e-4 of the closed forms; it was measured at 3.6e-7 for a 4 cm wall and 1.6e-4 for a 1 cm
wall at 0.6–1 MHz, where the Richardson correction reaches 17 % on the far branches. Prefer a closed-form profile.

**Lossless limit.** As η → 0, 1/(β² − k_n²) → P + iπ δ(β² − k_n²), so only the propagating branches contribute to
Re Y:

    Re Y = Σ_{n: f > f_n} π |Ĝ_n(k_n)|²/(ρ v_g,n),       Ĝ_n(k) = ∫ G_n(z) e^{−ikz} dz (symmetrised over ±k).

For n = 0 this is the T(0,1) asymptote of the m = 0 torsional module ([mechanics_thermal.md](mechanics_thermal.md#torsional-dynamics)), (2π∫w r² dr)²|P(k)|²/(2ρ c_T J), J = 2π∫r³dr. Each higher
branch adds a term with v_g → 0 at its cutoff: an integrable 1/√(f − f_n) singularity. **At finite η the damped
form is the right reference**: the band mean of a finite tube tends to the
damped infinite tube, which differs from the lossless one at first order in η with a sign set by the shape of
|P(k)|². A therefore computes every branch in the damped form (complex k_n); its T(0,1) branch equals the
single-branch integral T²/(πJμ) ∫₀^∞ |P|² ωηk²/((k² − k₀²)² + η²k⁴) dk to 1e-8 (test), and the
cutoff singularities are regularised by η.

**Branches kept, and the evanescent tail.** By default A keeps the largest of:
- every propagating branch at the highest frequency plus 100 evanescent ones (`n_evanescent`);
- every branch up to κ = 20 k_T(f_max): the tail below is an asymptotic form (κ ≫ k_T), and on a thick wall at high
  frequency (26 propagating branches on a 4 cm wall at 1 MHz) n_prop + 100 was 7.7e-4 short;
- for terms with a surface traction, branches up to κ_N s ≥ 5 for a Gaussian or sampled profile and ≥ 50 for a top
  hat, s the axial scale (σ, the width): the tail assumes the per-branch Re Y falls as 1/κ_n², which needs κ s ≫ 1,
  and a top hat's edges act as ring loads, a 1/(κW) correction per branch (2e-3 at κW = 5). Volume loads need no
  extra branches (their projections decay fast: 2e-12 between 126 and 250 branches).

At most 20 000 branches, with a note. Volume projections use a composite 16-point Gauss–Legendre rule, one panel
per half-wave of ψ_n (a single high-order rule would cost O(nq³)).

An evanescent branch (k_n ≈ iκ_n) contributes Re Y = O(η): for a Gaussian or top-hat profile I_p ≈ c/κ_n, so the
branch adds ≈ ωη R_n² c'/(μκ_n²). For a load uniform through the wall R_n decays and the sum converges fast; for a
**surface traction** R_n² = r_s² ψ̂_n(r_s)² tends to a constant (≈ 2r_s/t), so the remainder after N branches falls
only as 1/N: with 20 evanescent branches A would be 1.2 % low at η = 0.1. A therefore adds the analytic remainder

    Σ_{m>N} Y_m ≈ c̄/(D̄ (κ_N + D̄/2)),   c̄ = ½(Y_N κ_N² + Y_{N−1} κ_{N−1}²),   D̄ = (κ_N − κ_{N−4})/4,

from Y_m ≈ c/κ_m² and Σ_{m>N} 1/κ_m² ≈ 1/(D̄(κ_N + D̄/2)). The mean over the last **two** branches matters for
tractions on **both walls**: there R_n = a s_a ψ̂_n(a) + b s_b ψ̂_n(b) alternates with the parity of n
(ψ̂_n(b)/ψ̂_n(a) ≈ (−1)ⁿ√(a/b)), and a tail from the last branch alone would be doubled or dropped by parity, 0.5–1.5 % off at η = 0.1. Inner, inner + outer and inner − ½ outer tractions agree with a
brute-force sum (1000 and 2000 branches, extrapolated in 1/N) to ≤ 3.6e-5 at either parity of N (tests).
`truncation` reports the last branch's share, `tail` the remainder added (up to 6.8e-3 of Re Y in validation X). F
adds, for the branches beyond its modal sum (evanescent at every frequency and local, decay length 1/κ_n ≪ the
distance to the ends), the infinite tube's contribution of the next 100 branches and its tail. **A ring (δ) axial
profile on a surface** has I_p = 1, a 1/κ_n term and a logarithmically divergent Re Y: the strain energy of a line
load diverges, and hysteretic damping dissipates in proportion to it. That is physics, not a numerical defect: give
the load a finite axial width (A notes it and adds no tail).

**F, the finite free-free tube.** The ends z = ±L/2 are traction-free (∂_z u = 0), so the finite tube's modes are
exactly separable, Φ_nm = ψ̂_n(r) cos(mπ(z + L/2)/L)/√(2πρℓ_m) (ℓ_m = L, L/2), with
λ_nm = c_T²(κ_n² + (mπ/L)²) and unit modal mass, and q = Σ c_nm²/(λ_nm(1 − iη) − ω²),
c_nm = 2π R_n ∫ p φ_m dz/√(2πρℓ_m). Uniform η makes this exact for the damped problem. The FEM must equal F up to
discretisation (it does, to 3e-4 pointwise on a test tube with nodal loads); F's band mean tends to A
when the participating modal overlap Σ_n ηf L/v_g,n is ≳ 1.

**API.** `Tube(a, t, rho, mu=… | E=…, nu=…)`; `cutoffs_Hz(n)`, `kappa(n)`, `mode_shape(n, r)`;
`mobility(f, TubeLoad(axial, volume=w | inner=τ_a | outer=τ_b), eta)` → Y, Y per branch, cutoffs, projections,
truncation; `mobility_lossless(f, load)`; `finite_tube_mobility(f, load, eta, L)`. Axial profiles `Gaussian`,
`Rect`, `Ring`, `Sampled`.

## B. The wall-resolved FEM harmonic path

**Loads.** `Results.torsion(loads=[...])` takes, in addition to (and summed with) the nodal `load`:

- `{"region": R, "g_N_m3": v}`: a body force on the elements of region R;
- `{"surface": {"on": R, "from": S | "vacuum"}, "traction_Pa": v}`: an azimuthal traction on the faces R shares with
  S (`vacuum`: every non-structure element and the domain boundary);
- `{"curve": C, "traction_Pa": v}`: on a zero-thickness curve, applied once through the structure side.

v is a number, an expression in r and z [m], or one value per DOF (its P_p interpolant). Values are evaluated **at
quadrature points**, so a load restricted to a region or a boundary has exact edges (no smearing over an element,
unlike the nodal `load`), and the applied force and torque equal the closed forms to 1e-10 (tests). The response
lists, per entry and in total, ∫ applied dV or dA, ∫ (·) r and the axial support. The face integral uses the
element's own shape functions on the face (order 2p + 6 Gauss rule), as the mechanics module's pressure loads.

**Band driver.** `band = {"f_Hz": [f0, f1], "n": N}` (N odd) solves N equally spaced frequencies and returns in
`harmonic.band` the trapezoidal mean of Re Y and two uncertainties:
- `quadrature_uncertainty` = |mean on the full grid − mean on every second point|, an upper estimate of the grid's
  error (it is the coarse grid's error); resonance peaks are resolved when Δf ≲ ηf/4, and a warning fires when
  Δf > ηf_min/2;
- `ripple_sem` (n ≥ 17): the standard error of the means of eight disjoint sub-bands of Re Y minus its
  least-squares quadratic in f. **A diagnostic of the modal ripple, not a bound and not an estimate of the
  finite-length scatter.** In validation X (108 rows) it is below |F/A − 1| in 8 rows, all eight among the 14 with
  |F/A − 1| > 1 %, and about 350× above it at the median.
- `edge_estimate` (with `band.length_m = L`, the structure's axial length): E = rms(Re Y − quadratic trend) ·
  √2 · (c_T/L)/(π B), the band mean of a ripple with the T(0,1) even-mode spacing c_T/L at the worst edge phase,
  c_T the largest shear speed of the structure. It assumes the band spans many ripple
  periods, B ≫ c_T/L. It is **a conservative scale, not an estimate**, of how far the band mean of the finite
  structure may sit from a smooth (infinite-structure) value: on the uniform tubes of validation X (108 rows) it is
  58× the actual |F/A − 1| at the median, and below it in 6 rows, by up to 17 %. It is calibrated on uniform tubes
  only. For a definitive finite-length scatter, compare with F − A of a uniform analogue.

A failed factorization (`ok = 0`) makes the band request fail, naming the frequencies; a `freqs_Hz` request lists such rows in a warning. A `freqs_Hz` request also carries `harmonic.note`: the
mesh guard is calibrated on band means, so single-frequency values are not certified.

`Results.torsion_band(f_min, f_max, loads=…, eta=…, n=…)` is the shortcut. **Its default solver is serial STRUMPACK**
(serial Eigen SparseLU in a build without STRUMPACK): it
is robust in memory, and in the measured cases never more than about 1.8× slower than the fastest choice.

**Solver.** The operator A(ω) = K(1 − iη) − ω²M is complex symmetric (not Hermitian). K and M are put on one pattern,
so A at each frequency is a combination of their value arrays; `ComplexSymSolver` (`sym_solver.hpp`) factorizes it
with Eigen's SparseLU (COLAMD) or STRUMPACK's multifrontal LU in complex<double> (METIS nested dissection, OpenMP;
chosen by size as the real solver, `AXF_STRUMPACK_MIN_DOFS`, or by `solver.backend`), analyses the pattern once per
solver and reuses it for every frequency: STRUMPACK its METIS ordering and symbolic factorization, Eigen's SparseLU
only its column ordering and elimination tree (its supernodes are formed during the pivoted numeric factorization). The numerical factorization cannot
be reused: A changes with ω (a shift-invert Krylov reduction would need as many vectors as there are modes in the
band, thousands at 1 MHz). `solver.parallel_frequencies = p` runs p frequencies concurrently, each with its own solver and ⌊threads/p⌋ OpenMP
threads (nested level 2).

**`parallel_frequencies = "auto"` (opt-in).** It measures before it chooses:
1. It reads the job's memory first (`torsion_available_memory`): the smallest cgroup limit (v2 `memory.max`, v1
   `memory.limit_in_bytes`) over the process's cgroup and every ancestor, "/" included (in a container the cgroup is
   "/"), less that cgroup's usage without reclaimable file cache (`memory.current` − `inactive_file`), so other
   processes of the job count. Without a cgroup limit, the batch scheduler's allocation (`SLURM_MEM_PER_NODE`, or `SLURM_MEM_PER_CPU` × `SLURM_CPUS_ON_NODE`), less the process RSS. Everything is bounded
   by MemAvailable. A source that leaves 0 is kept. Without
   /proc (macOS, Windows) nothing is found, the available memory is 0 and auto runs serially.
2. With STRUMPACK, it solves the first frequency with STRUMPACK on all threads, timing the numerical factorization
   apart from the ordering and symbolic analysis (which took 2.8× the steady time), and keeps that solver for the
   rest.
3. It probes Eigen SparseLU (one solve, 2 threads) only when that can pay off and fits. It skips the probe when:
   - fewer than 8 frequencies are left;
   - STRUMPACK's fill is above 90 per DOF, which marks a compact section (thin walls 61–85, compact 96–107);
   - one Eigen solve, predicted as 1.5 × (20 B × 3 × STRUMPACK's fill + the matrix), would not fit in 80 % of the
     available memory.

   Eigen's fill was 1.7× STRUMPACK's on a thin wall and 2.9–3× on compact sections. The probe costs one Eigen
   solve: 3 s at 3.5e5 DOF on a thin wall, but 30–100 STRUMPACK frequencies on a compact one, hence the rules.
   Its memory is measured as the resident growth (at least 20 B per factor nonzero plus the matrix) × 1.5 for
   the factorization's freed workspace; that came out 1.3–1.5× above the measured peak.
4. It runs the rest as p = min(remaining, threads/2, 0.8 × available/memory per solve) concurrent Eigen solves, or
   serial STRUMPACK, whichever predicts the higher throughput. Concurrent Eigen solves are taken as 1.1× the probe's
   time when p > 1 (measured 0.99–1.09×).

The decision is a pure function (`torsion_auto_choice`), tested on measured sections. Near a tie,
repeated runs can choose differently; only round-off in Y changes. `harmonic.timing.parallel_choice` reports the
memory sources, the probes, the decision and its reasons.

Measured on one 32-thread node (P2 tube at 6 elements per wavelength at 1 MHz; validation, "Cost per
frequency"): seconds per frequency, effective (wall time of the sweep / number of frequencies),

| free DOFs | STRUMPACK, 1 at a time (32 threads) | STRUMPACK, 4 concurrent | Eigen SparseLU, 16 concurrent |
|---|---|---|---|
| 2.7e5 | 1.22 | 0.83 | 0.25 |
| 7.0e5 | 3.21 | 2.19 | 0.72 |
| 1.41e6 | 6.63 | 4.38 | 1.67 |

These numbers are for a **thin wall** (long, small separators, Eigen fill about 110 per DOF). The cost per
frequency is close to linear in the DOF count there. Concurrent STRUMPACK solves do not scale (8 at once is slower
than one: the dense kernels run through a single-threaded OpenBLAS build); concurrent Eigen solves do, at 3.2–3.5 kB
of memory per DOF each on this mesh (measured at 4.7e5 DOF; 72 GB peak at 1.41e6 DOF and 16 at
once). **On a compact cross-section** (a 0.1 × 0.1 m steel ring) Eigen's COLAMD fill is 300–400 per DOF,
its memory 7–9 kB per DOF per solve, and one Eigen solve at 1.16e6 DOF takes 273 s against STRUMPACK's 12.4 s: there
serial STRUMPACK wins. In a sweep of six sections at 3.5e5 DOF, concurrent Eigen was faster up to
a 2 cm × 1 m section, tied at 3 cm, and 1.8× slower from 5 cm × 0.4 m on. Hence serial STRUMPACK as the default, with
`"auto"` measuring instead of assuming (validation, section auto). A 201-point band
at 1.4e6 DOF on a thin wall takes about 6 min on one node. Backends and parallel settings agree to 1e-9 (tests).

**Mesh-resolution guard.** The relevant length is the shear wavelength λ_T = c_T/f: every branch has total wavenumber
ω/c_T (k² + κ_n² = ω²/c_T²), so both the radial and the axial resolution are set by λ_T at the highest frequency.
The guard measures, per structure region, N = λ_T(f_max)/h with h the longest vertex-to-vertex edge of each element
(about 1.35× the region's `mesh_size` on these meshes), and compares the minimum with thresholds per element order
from the convergence study B1 (band-mean error against F, 0.8–1 MHz, η = 0.01 and 0.003, inner-traction and volume
loads; the worst case):

| order | ~1 % (warn below) | refuse below | measured |
|---|---|---|---|
| P1 | 24 | 7 | 0.85 % at N = 24.1, 9.1 % at 7.35 |
| P2 | 3.5 | 1.5 (coarsest studied) | 1.00 % at 2.98, 0.21 % at 4.46, 4.2 % at 1.53 |
| P3 | 2 | 1.25 (coarsest studied) | 0.08 % at 2.15, 1.1 % at 1.53 |

`mesh_check = "error"` (default) refuses when the **1st percentile** of the elements of any **loaded region**, or of
the whole structure, is below the refuse threshold. One long-edged sliver in an unloaded part does not refuse a
request, but an under-resolved loaded region does (a pooled 1st percentile let a coarse loaded pad of 3 % of the
elements through). It warns when any region's **minimum** is below the warn threshold, with a
message worded for the mode in effect; `"warn"` only warns, `"off"` skips; the key is validated on every request. P1 is uneconomical (24 per
wavelength); P2 at 4–6 or P3 at 3 per wavelength is the production setting. Pointwise values converge much more slowly
than band means (the FE dispersion error shifts the resonances of a tube many wavelengths long): at η = 0.003 the
median single-frequency error is 22 % for P2 at N = 3 and 0.35 % at N = 9. Only band means are certified.

## Validation summary

The tables are in [validation/torsion_highfreq.md](../validation/torsion_highfreq.md). In brief:

- **A against the FEM** on a 0.6 m tube (a = 0.5 m, t = 0.04 m, η = 0.1, nodal loads): the FEM equals F pointwise to
  1.2e-4 (uniform) and 2.9e-4 (surface-concentrated); band means FEM/A = 0.999–1.007. Above the T(0,2) cutoff the
  surface-concentrated load's band mean is 5.3–10.1× its T(0,1) part, which A reproduces to 0.1 %.
- **Cross-validation** (108 band means: a = 0.1, 0.5, 1 m; t = 0.01, 0.02 m; η = 0.01, 0.03, 0.1; volume, inner,
  both-wall and outer loads; bands below the first cutoff and above 1, 2 and 5 cutoffs, up to 1 MHz for the 1 cm wall;
  P2 at 8 elements per wavelength): B = F to ≤ 2e-4 in every row; F/A within 1.4 % at participating modal overlap
  ≥ 1 and up to 5.7 % at overlap 0.08.
- **Convergence**: the guard table above; P3 reaches 1e-5 (surface) and 1e-6 (volume) band-mean error at 4.5 per
  wavelength.
- **A non-uniform structure** (a tube with an annular flange): the flange's effect is measured against the bare finite
  tube with the same load position (F exact, B equal to F to ≤ 1.5e-4): −5.6 % (47–140 kHz), +2.1 % (164–296 kHz)
  and −0.01 % (0.6–1 MHz); the nearby free end, not the flange, makes up most of the difference from the infinite
  tube.

## Validity

- A: infinite, uniform, isotropic tube (or rod), m = 0, one uniform η; any axisymmetric azimuthal load. Exact in
  the radial direction (no thin-wall approximation).
- A for a finite structure: the band mean only, within the finite-length scatter (F − A, quantified in the
  validation as a function of the participating modal overlap); individual frequencies are resonant.
- B: any axisymmetric structure of isotropic materials (bonded, free-free, one η), resolved per the guard; the
  same m = 0 limits as the module (no (u_r, u_z) dynamics, no m ≥ 1, no supports, no pre-stress).
- Not covered: frequency-dependent or region-dependent η, supports, and the general (u_r, u_z, m ≥ 1) problem.
