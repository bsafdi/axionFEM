# Adaptive frequency sweep of the 3D full-wave engine

Status: **opt-in (`fullwave.adaptive`), beta.**
- Code: `cpp/src/rational_fit.cpp` (`axionfem/rational_fit.hpp`: the fit and the sweep driver, no FEM), its call in
  `Model3D::fullwave_analysis` (`cpp/src/model3d_fullwave.cpp`).
- Spec: `fullwave.adaptive` in [design_format.md](../design_format.md) (3D full wave).
- Validation: `benchmarks/adaptive_sweep.py`, record `benchmarks/records/adaptive_sweep.json`; the driver's random
  trials `benchmarks/adaptive_sweep_trials.cpp`, record
  `benchmarks/records/adaptive_sweep_trials.json` (Validation, below).
- Tests: `cpp/tests/test_rational_fit.cpp` (ctest, cases 1-22 and 15b); `tests/test_adaptive_sweep.py` (default suite: the
  TEM line, Robin and PML terminations, an unconverged sweep, every solve refused, two ports on two ranks; `AXF_SLOW`:
  the lossy cavity); `tests/test_fullwave3d_dzdw_terminations.py` (dZ/dω with every termination).

A fixed sweep solves at the listed frequencies only. Between them nothing is known, and a self-resonance is reported
only as a sign change of L between two listed frequencies. The adaptive sweep instead fits a rational model to the
solves, places each new solve where the model is least certain, and stops when the model is accurate to a tolerance
over the whole band. It is the counterpart of COMSOL's adaptive frequency sweep. It is built on the fact that every 3D
solve already returns the exact dZ/dω (`dZ_domega_H`, [multiport_fullwave.md](multiport_fullwave.md)), with every
termination the engine has (A′(ω) includes the Robin groups' dβ/dω and the PML's stretch).

## Why a rational model

The discrete problem is A(ω) e = iω J with A(ω) = K − ω² M_ε(ω) + (exterior). For a closed structure with
frequency-independent ε, μ and tan δ, A is a matrix polynomial in ω, so every entry of Z(ω) = −Jᵀ e / I² and of
Ψ(ω) is a rational function of ω whose poles are the discrete eigenfrequencies with the ports open (complex with
loss). Conduction (σ/ω) keeps it rational. The radiating exterior's coefficients are rational in k too (spherical
Hankel functions are e^{ix} times polynomials in 1/x, so their log-derivatives γ_n are rational), so the truncated
discrete Z remains rational, of high degree. In a band, a few poles in or near the band plus a smooth remainder
describe it to high accuracy, which is what a low-degree rational fit exploits.

All outputs share the poles: they are bilinear forms of the same A(ω)⁻¹. One denominator is therefore fitted for all
entries of Z and of Ψ together (set-valued AAA).

## The fit

**Form.** AAA's barycentric form (Nakatsukasa, Sete and Trefethen, SIAM J. Sci. Comput. 40, A1494, 2018), in f [Hz]:

    r_k(f) = Σ_j w_j F_jk / (f − f_j)  /  Σ_j w_j / (f − f_j),      type (m−1, m−1).

It interpolates the solved values F_jk at the m support frequencies f_j for any weights w. The outputs k are every
entry of Z (row-major) and every entry of the Ψ vector.

**Units.** The fit itself runs in x = (f − c)/s, with c and s the solves' centre and half span, and the derivatives
multiplied by s. The barycentric weights do not change under this affine map, so the model maps back to Hz by
f_j = c + s x_j alone (each support point is set back to its solve's exact frequency). Fitted in Hz directly, a
support point's derivative row would carry an extra factor of the spacing in the fit's unit, so the rows' relative
weights would depend on the unit (in trials: up to 2.6× the solves, up to 14 % unconverged, false stops up to 9× the
tolerance). A sweep takes the same decisions in any unit (`test_rational_fit` case 8: the same frequencies in [1, 2]
and ×7e7 Hz; the trials below in both units).

**Weights.** w minimizes the linearized residual of all samples over ||w|| = 1: the right singular vector of the
smallest singular value of a stacked Loewner matrix (Eigen `JacobiSVD`). Rows, per output k and sample i:
- value at a sample outside the support: Σ_j w_j (F_ik − F_jk)/(x_i − x_j) = 0 (that is, F_i D(x_i) − N(x_i) = 0);
- derivative at a sample outside the support: Σ_j w_j [F'_ik/(x_i − x_j) − (F_ik − F_jk)/(x_i − x_j)²] = 0, the
  derivative of (F_ik + F'_ik (x − x_i)) D(x) − N(x) at x_i. With the value row it gives r'(x_i) = F'_ik;
- derivative at a support point: F'_ik w_i + Σ_{j≠i} w_j (F_ik − F_jk)/(x_i − x_j) = 0, which is r'(x_i) = F'_ik
  there.

**Hermite data.** dZ/dω is exact and free with every solve, so the derivative rows are used for every Z entry. The
engine takes them only from a row whose `dZ_domega_scope` begins with "exact"; any other row's Z is fitted by value.
Every termination of the 3D engine (PEC, PMC, the radiating exterior, impedance, scattering and sheet groups, PML) is
in that scope, and the scope names the ω-dependent terms the design has. `benchmarks/dzdw_check.py` checks it on
12 cases, 7 of them TEM-line terminations. Against Richardson-extrapolated centred differences they all agree to
2.4e-8, except the fixed-wavelength PML at 1.8e-7, whose solves stall at a true residual of about 1e-7. A term added
to A(ω) later must add its d/dω to A′ (`derivative_form` in the dZ/dω block) or leave the scope without "exact".
dΨ/dω is not free: it would need the source solve A⁻¹ b₀ per frequency. So Ψ contributes value rows only. Its poles
come from the shared denominator, which Z's derivatives help determine, and its numerator from interpolation at the
support points.

What the derivatives buy:
- Every solve gives two conditions for Z instead of one, so a given degree needs about half the solves.
- On `test_rational_fit` (two lossy poles in [1, 2] and an out-of-band background), 7 solves fit the band to
  2.9e-3 with derivatives and to 1.3 without.
- On the random resonant responses below (tolerance 1e-3, noise 1e-11, in Hz), the sweep converges in a mean of 8.8
  (lossless) and 8.4 (lossy) solves with derivatives, and 11.7 and 11.1 on values alone.

**Consistency.** Wrong derivative data cannot be interpolated together with the values. The fit then misses its own
solves, and Froissart doublets at the support points absorb the difference (with a dZ/dω missing the termination terms,
the matched line would converge with a misfit 100× the fit tolerance and the PML line with 4.7e-3). So:
- the sweep converges only if the fit reproduces its solves to 10 × the fit tolerance (`fit_consistent`);
- the derivative rows are dropped for the rest of the sweep, with a warning, as soon as either:
  - the two newest solves confirm the model (each within the tolerance of its prediction) while the fit cannot
    reproduce its own solves;
  - or the derivatives of two consecutive confirmed solves miss the model's prediction by more than 100 × the
    value's miss (in the derivative rows' measure: |F′ − r′| over the length within which F changes by itself, at
    most the distance to the nearest solve, relative to max(|F|, floor)).

  `hermite.dropped` says when and why, and the confirmations start again on values alone. Waiting for the estimate to
  pass as well would not work: with clearly wrong derivatives the doublets they cause keep it high.
- if the values alone are inconsistent too (solves noisier than the fit tolerance), the sweep ends unconverged at
  `max_solves`, with a warning that the fit misses its tolerance.

`test_rational_fit` case 11 adds a bias of 1e-2 of |Z| per band width to dZ/df, as a missing term of A′ did: the
rows are dropped after 15 solves, and the sweep converges at 17 with a true error of
8.2e-8. This detects gross inconsistency only. A bias that the fit can absorb within 10 × its tolerance passes: in the
trials, with a bias of 1e-4 of |Z| per band width, the rows stayed in 98 of 100 lossless sweeps, and 10 of them stopped
at up to 9.9× the tolerance (9 inside a lossless pole's neighbourhood; Trials, below).

**Weighting.** Each row is divided by max(|F_ik|, floor_k), the same pointwise measure the error estimate uses. The
derivative rows are multiplied by min(h_i, max(|F_ik|, floor_k)/|F'_ik|), with h_i the distance to the nearest other
sample: the length over which F changes by itself, at most the spacing. Without the pointwise weights, a solve near a
resonance (|F| of 1e5 times the median) dominates the least squares, and the rest of the band loses its accuracy (in early trials at tolerance 1e-6 this failed in about half the lossless
cases).

**Real and complex outputs.** Every output is fitted as complex. In a closed lossless model Z is imaginary and Ψ real
(up to the solves' residual), so one part is noise. The pointwise measure uses |F|, so that part only enters at its
own size, and nothing is imposed on the fit. The fit does not enforce Z(−ω*) = Z(ω)* either: the band lies at
positive frequencies, and the mirror poles at −Re p act as a smooth background there.

**Floor.** Per entry: floor = 0.01 × √(median |Z_jj| × median |Z_kk|) for `Z[j][k]`, the medians over the
solves, not below 1e-6 × the median largest |Z| entry; 0.01 × the median largest |Ψ| entry for every Ψ entry. A
group that is zero throughout (Ψ without an axion source) is not fitted. The floor keeps the measure finite at zeros
of an entry (a series resonance, Z_jk ≈ 0 by symmetry, a weak coupling):
- there the absolute error is held to tolerance × floor;
- elsewhere the measure is pointwise relative.

One floor shared by all Z entries (1 % of the largest) would hold a port whose |Z| is below 1 % of another's only to
an absolute error.

**Greedy (AAA).** Start from the mean, add as support point the solve with the largest weighted misfit (values or
derivatives), solve for w, and repeat until every solve is fitted to the fit tolerance, 1e-3 × the sweep's
tolerance and not below 1e-10. The solves themselves are accurate to about their true residual (gated at 1e-6,
normally 1e-10), and fitting below that fits noise.

**Froissart doublets.** Pole-zero pairs with negligible residues appear when the fit has more freedom than the data
support. Candidates are poles whose largest residue over the outputs, relative to the output's scale and the band
width, is below 100 × max(fit tolerance, achieved misfit), the misfit counting only up to 1e-6. That is the solves'
residual gate: a larger misfit means too few solves, not noise. Without the cap the cleanup can remove every pole of
an early fit. Noisy data fitted below their noise leave doublets with residues of a few
times the misfit (3.4× at 1e-9 noise, 11× at 1e-11, `test_rational_fit` case 3).

That threshold alone cannot tell a doublet from a weak genuine resonance: the two-port frame cavity's eigenmode poles
have strengths from 6.2e-6 to 0.35, and some lie below it. What tells them apart is the solves. A candidate's removal (its
nearest support point dropped, the weights solved again) is kept only if the solves stay fitted:
- first all candidates together (as Chebfun's cleanup), to max(fit tolerance, 100 × the capped misfit); overfitted
  noise leaves many doublets, and removing one leaves its neighbours' support points needed;
- once that is refused, one at a time, weakest first, to max(fit tolerance, 10 × the capped misfit); a refused
  candidate stays.

Both allowances stop at the consistency bound, 10 × the fit tolerance, unless the misfit already exceeds it (a
removal never makes the fit worse than that). The misfit that enters the cleanup is usually the greedy's stopping
value, just below the fit tolerance, not the data noise. A larger allowance (100 × that) would let a removal leave a
fit at 1e-5 to 4e-5, past the bound, and the consistency rule would then drop the exact derivative rows.

So a pole whose removal changes the fit at any solve by more than the allowance is kept. The FEM fits of the record
end with misfits of 2e-8 to 1e-6, so the allowance is up to 10 × the fit tolerance, 1e-5 of the output's scale at the
default tolerance. A pole that changes no solve by that much is not resolved
by the solves and is indistinguishable from a doublet. On 40 noisy samples overfitted on purpose (every sample a
support point), the cleanup removed 34 doublets and the error between samples fell from 6.3e-5 to 1.6e-9, the noise
level (case 3). Three in-band poles of strengths 1e-5 to 1e-4 of a strong one are kept
(case 7).

**Poles and residues.** The poles are the zeros of D(x) = Σ w_j/(x − x_j). They are the finite eigenvalues of the
arrowhead pencil (E, B), E = [0 wᵀ; 1 diag(x_j)], B = diag(0, 1, …, 1). They are computed as s + 1/μ from the
eigenvalues μ of (E − sB)⁻¹B, with the shift s in the upper half plane, away from the poles of a passive response
(e^{−iωt}: Im p ≤ 0). The residues are N_k(p)/D'(p). Q = Re p / (2 |Im p|).

**The model's derivative** (dense dZ/dω, L_eff). With j the nearest support point, δ = f − f_j, and n, d the sums over
the other support points, r = (w_j F_j + δ n)/(w_j + δ d), so
r' = [w_j (n − F_j d) + δ w_j (n' − F_j d') + δ² (n' d − n d')]/(w_j + δ d)². The quotient rule on N/D lost about
ε f/δ next to a support point, 3e-2 at δ = 1e-15 f; this form has no cancellation there and is the
support-point formula at δ = 0 (case 13: 1.9e-11 within 1e-12 of a support point).

## The adaptive loop

1. **Initial solves.** Solve at `initial_points` frequencies, the band ends included, log- or linearly spaced. These
   are `fullwave.frequencies` internally, so the mesh sizing (λ/8 at f_max) and the exterior map's multipole order
   (k R1 + 10 at f_max) are those of the whole band. The mesh, the assembly of everything frequency-independent, the
   port sources and the exterior projections are set up once, as for a fixed list. If none of them gives a usable
   solve, the sweep stops (`stop_reason` "no usable solve among the initial points", `converged: false`, a warning,
   `model` and `dense` null, no resonance) and results.json is written.
2. **Fit** all successful solves, with the cleanup.
3. **Estimate the error** on the candidates: a grid of max(2000, 4 × `dense_points`) points over the band (same
   spacing), plus points around each in-band pole of the model. With r a quarter grid step there:
   - a resolved pole (|Im p| ≥ r): its peak Re p and half-power points Re p ± |Im p|;
   - a pole on the real axis (|Im p| < r) that is confirmed: r⁻ (below) or the model fitted before the newest solve
     has a pole within tolerance × r. Then the points Re p ± r, and nothing closer: the pointwise relative error of
     a lossless resonance is not controlled closer to it. r⁻ alone locates poles far worse than the model, so with it
     alone a genuine lossless pole could stay unconfirmed for good, with the estimate peaking at its r/64 points,
     inside the zone where no solve is allowed;
   - an unconfirmed pole on the real axis: points from r/64 to r on either side. Spurious poles on the real axis
     (narrow spikes between solves) are the main failure without these points: the estimate would never look at
     them.

   The estimate is the larger of two terms:
   - **Degree.** |r − r⁻| in the fit's measure, where r⁻ is the same solves fitted with the last-chosen support point
     dropped (one degree fewer). Both interpolate the shared support points, so the difference peaks between solves,
     where the data constrain the model least. With one support point (a constant response, such as the matched
     line's Z) r⁻ is AAA's starting approximation, the mean of the solves.
   - **Change.** |r − r_prev|, where r_prev is the model fitted before the newest solve: how much that solve moved
     the model anywhere in the band.
4. **Stop** when three conditions hold:
   - the estimate is below `tolerance` on every candidate, except those within a quarter grid step of a solved
     frequency. The data pin the model there, and no solve may be placed there, so an estimate peaking there could
     never fall (a sweep could stall on f_min). Candidates near a refused frequency count;
   - each of the two newest solves agreed with the model fitted before it (|F_new − r_prev(f_new)| below `tolerance`
     in the same measure);
   - the fit is consistent (above).

   A sweep also stops, converged, on its confirmations alone when all of these hold:
   - four solves in a row came within tolerance / 100 of their predictions;
   - the newest solve moved the model by less than the tolerance;
   - the fit is consistent;
   - the gate: wherever the estimate exceeds the tolerance, the "spurious" poles q of r⁻ account for at least half
     of it. A spurious pole lies above the real axis by more than 1e-9 |q| and by at most a quarter of the band
     (`kSpuriousReach`), with no model pole within Im q. Its own term in r⁻, |res_q/(f − q)| with r⁻'s residue, in
     the fit's measure (relative to max(|r(f)|, floor), the largest over the outputs), is evaluated at every
     candidate (`rational::spurious_terms`); a candidate is excused where the terms reach half its estimate.

   The `stop_reason` then says so. It exists for sweeps like a lossy trial that a pole of r⁻ at 170.5 + 0.91i MHz held
   for 49 solves, while every solve agreed with its prediction to 1e-8. Without the gate, the closure stops 112-147 of
   300 noise-bound sweeps falsely.

   **What the gate does in practice** (82 800 trial sweeps). When r⁻ has a spurious pole,
   r − r⁻ is mostly that pole's term, so its term reaches half the estimate nearly everywhere: in 778 of 783 closures
   the terms excused at least 99 % of the candidates, and in 173 closures the excused estimate exceeded 10× the
   tolerance (up to 3.3e4×). The gate therefore mostly decides *whether* the closure may be taken (whether r⁻ has a
   qualifying pole at all), not where. What protects a closure is the four solves in a row within tolerance/100, the
   model change below the tolerance and the consistent fit. Measured:
   - closures were 0.57, 1.13 and 1.14 % of the converged sweeps at tolerance 1e-3, 1e-4 and 1e-5;
   - 4 closures were wrong, all at tolerance 1e-5 with noise 1e-9 (fit tolerance 10 × the noise, not flagged): at the
     flag boundary the closure is about as reliable as the normal stop, not more.

   A region-based gate (Re q ± Im q × max(3, 2 × peak/tolerance)) covers the whole band and can stall for a q below
   the band. Subtracting the spurious terms from r⁻ fails: such a term is cancelled by
   r⁻'s other poles. A local test (comparing r with a comparison model that lacks q, e.g. the fit with another
   support point dropped, and requiring that one below the tolerance) is not implemented; it is the open route.

   When the closure's other conditions hold and only the gate refuses, and the largest unexcused estimate among the
   candidates that may be solved reaches the tolerance, the next solve goes there instead of to the overall maximum.
   In trial 166 of the trials below the overall maximum sits at an excused term, and without the redirect 50 solves
   there confirm the model while the unexcused peak (a genuine lossy pole that r⁻ places 51 Hz off) is never solved.

   Otherwise, if `max_solves` is spent, stop with a warning (`converged: false`). Otherwise solve where the estimate is
   largest among the candidates that may be solved, and go to 2. Near-ties (within 1e-6, as with the flat estimate of
   a one-point fit) go to the candidate farthest from every attempted frequency, so that confirming solves probe the
   widest gaps. A candidate may not be solved:
   - within r of a pole on the real axis (a lossless resonance is singular there);
   - within a quarter grid step of a frequency already attempted;
   - near a refused one (below).

   If no candidate may be solved, the sweep stops unconverged ("no admissible candidate frequency"; `test_rational_fit`
   case 16).
5. **After convergence: Froissart doublets on the real axis are removed** (`rational::remove_invisible_poles`).
   Candidates are in-band poles of the converged model that
   - lie on the real axis (|Im p| below a quarter grid step r);
   - are invisible one candidate step away (strength / max(|Im p|, step) below the tolerance);
   - have a needle, |res/Im p| at the pole against the model next to it (at Re p ± r, in the fit's measure), that
     reaches the tolerance: a lower one cannot make the model wrong by the tolerance.

   For each candidate the support point nearest to it is dropped and the weights are solved again at that fixed
   degree. The candidate is removed (with that refit as the new model) only if the refit
   - has lost it (no pole within r of it): a doublet tied to that support point disappears with it, a genuine mode
     that the other solves determine does not;
   - reproduces the solves to 0.1 × the tolerance (`kInvisibleFraction`), in the fit's measure and against the
     output's scale;
   - changes on the candidate grid, outside r of every pole on the real axis, by at most
     max(0.1 × tol, min(0.25 × tol, tol − the sweep's estimate there)) in the fit's measure: 0.1 × tol, plus up to
     0.15 × tol of the estimate's own slack (`kInvisibleGridFraction`);
   - keeps every other pole on the real axis, matched one to one, moved by at most min(0.1 × tol × r,
     tol × 1e-6 |p|): next to a lossless pole a shift δp puts the model δp/|f − p| off, so the bound keeps that
     below the tolerance down to 1e-6 of the pole;
   - adds no pole on the real axis whose own needle reaches the tolerance.

   There is no retry on another support point. A refused candidate stays. `invisible_poles_removed` counts the
   removals, `sample_fit_error_before_removal` keeps the misfit before them, and `fit_consistent` is judged before
   them.

   Why: the cleanup's allowances stop at 10 × the fit tolerance, the consistency bound. In trial 88 of the trials
   below (lossless, tolerance 1e-5) that keeps a doublet at 164.915001 − 6.8e-10i MHz of strength 2.6e-14 of the
   output's scale. Its needle is far narrower than the candidates around it, so no estimate sees it, and the
   converged model is 24× (Hz) and 44× ([1, 2]) the tolerance off there without the removal. Its removal raises the
   misfit to 1.66e-7 and changes the model by up to 0.116× the tolerance in the low band (where the estimate is
   0.10-0.12×, and where it brings the true error from 0.080 to 0.020×); the model is then 0.16× off
   (`test_rational_fit` case 20).

   Why the checks are as they are:
   - checking the solves and the other poles only relative to |r(f)| would delete or move genuine weak modes next
     to a strong mode, where |r| is 1e2-1e3 × the output's scale (`test_rational_fit` case 22 keeps such a case as a regression);
   - without the needle condition, removals caused three wrong stops of 1.2-2.1× in about 9000 trial sweeps;
   - a full leave-one-out refit stops at a lower degree and drops a marginal pole however well it is located, and a
     retry on the next-nearest support point keeps the evidence-carrying solve, which blinds the solves check: that
     variant deleted 142 of about 17 700 well-located genuine weak modes (Q 1e5-1e7) in a close-pair scan;
   - letting other real-axis poles move by 0.1 × tol × r (8.7e-9 relative at the default grid) is up to 8.7× the
     tolerance at 1e-6 from a lossless pole, so the shift bound includes tol × 1e-6 |p|;
   - a grid allowance of 0.1 × tol refuses trial 88's removal; against the output's scale the grid check would refuse
     any removal next to a strong mode (a shift of its pole well inside the allowance changes the model by 0.4 × the
     tolerance in that measure 100 kHz away). The weak modes there are protected by the fixed-degree test and the
     pole-by-pole check instead.

**Refused frequencies.** A frequency may fail:
- a validation gate refuses it (multiport true residual, source or loaded checks: `retain_rejected`, which stops a
  fixed list);
- a single-port solve is not accepted (true residual ≥ 1e-6);
- an output is not finite.

The sweep then records it in `fullwave.adaptive.refused`, with the rejected row's diagnostics, warns, and moves on.
The rejected row's file that `retain_rejected` writes (the whole result with status `provisional_rejected`) is renamed
`fullwave_adaptive_refused_f_{index}.json` (`diagnostics_file` in the entry), so that a sweep that goes on does not
leave a file that reads as its own failure. The frequency counts against `max_solves` and is left out of
the fit. No later solve comes within an eighth of the gap between the successful solves around it (at least a quarter
grid step): a solve that fails tends to fail near it too, and each attempt costs a full solve. In `test_rational_fit`
case 5 (a refused 6 % of the band) this cuts the refused attempts from 10 to 2. A failed solve gives no prediction, so
it breaks the run of confirming solves. The estimate still covers the region around a refused frequency, so a sweep
that cannot solve where the model is uncertain ends at `max_solves` with a warning rather than claiming convergence.

**MPI.** Every rank fits the same allreduced numbers. The decisions (stop, next frequency) are broadcast from
rank 0, so that a rounding difference between ranks could never desynchronize the collective solves. A broadcast
frequency that was already attempted is recorded as failed rather than solved twice (case 12 checks that a changed
decision is obeyed).

**Fixed lists are unchanged.** The per-frequency body is a lambda that both modes call. A fixed list calls it for each
listed frequency exactly as the loop does. With a fixed list every output is bit-identical to a build without the
adaptive sweep, except `dZ_domega_H` (and its scope text) of designs with Robin groups or PML, which gained the
termination terms (Validation). The self-resonance rule for fixed lists (sign changes of L) is unchanged.

## Outputs

- `fullwave.sweep`: the solved rows, as for a fixed list, sorted by frequency. Rows the fit left out are kept, with
  their solver diagnostics. Field-export files keep the solve-order index.
- `fullwave.adaptive`:
  - `settings` (with `fit_tolerance`, `floor_fraction`, `candidates`, and in words the error measure, the estimate,
    the stopping rule, the Hermite data and the pole classes);
  - `converged`, `stop_reason`, `solves`, `fitted_solves`, `solved_frequencies_Hz` (solve order), `refused`;
  - `hermite`: `used_in_final_fit`, and `dropped` (null, or when and why the derivative rows were dropped);
  - `fit_consistent`: the final fit reproduces its solves to 10 × the fit tolerance (before step 5's removals);
  - `invisible_poles_removed` and `sample_fit_error_before_removal` (step 5);
  - `noise_bound`: the fit tolerance is below `settings.noise_bound_factor` (10) × `settings.pole_noise`, the
    solves' largest relative port residual. Then the fit follows the solves' errors and the stop, converged or not,
    is unreliable; the run warns and names the tolerance that would not be noise-bound (Limits);
  - `estimated_error` and `last_prediction_error` at the stop;
  - `history`: per fit, the solves, support points, fit misfit, doublets removed, the estimate and where it peaks,
    its model-change part, the next frequency, and the newest solve's prediction error;
  - `model`: `support_Hz`, `weights`, `values` (support points × outputs, with the output names), all `poles_Hz`
    (null when nothing was solved);
  - `poles`: every in-band pole of the model, with its class (below). Each entry has:
    - `f_Hz` (Re p), `pole_im_Hz`, and `pole_uncertainty_Hz` (standard errors of Re p and Im p, below);
    - `reproduced_by_refits` (the leave-one-out refits that have the pole, of all);
    - `Q` with `Q_uncertainty` when the loss is resolved (and the pole not `unresolved`); otherwise, unless
      `unresolved`, `lossless_within_fit` with the lower bound `Q_min`;
    - the residue of `Z[0][0]`, and `relative_strength` (the largest residue over the outputs, relative to the output's
      scale and the band width);
    - `contribution_one_step_away`, and `between_Hz` (the solved frequencies around it);
  - `dense`: the model on the band grid (`dense_points`, same spacing) plus each narrow in-band lossy pole's peak and
    half-width points (null when nothing was solved). It holds:
    - port 0's `Z_re_ohm`, `Z_im_ohm`, `L_H`, `Psi_re`, `Psi_im`;
    - `Z_matrix_*`, `Psi_vector`, and `dZ_domega_H` (the model's derivative, for L_eff);
    - `estimated_relative_error` at each point: the degree term and the change from the model before the newest
      solve. After a removal (step 5) it compares the final model with r⁻ and the model before the newest solve, both
      of the pre-removal fit; so it is not exactly the estimate the sweep stopped on, and at a removed needle r⁻ may
      still carry it.
- `fullwave.self_resonances`: the `poles` entries of class `resonance`, and only if the sweep converged. An
  unconverged sweep lists none: its poles stay in `fullwave.adaptive.poles`, the warning says so, and
  `fullwave.self_resonances_determined` is false (true for a converged sweep). Q is null for a pole whose
  loss is not resolved (below), flagged `lossless_within_fit`, with the bound `Q_min`.
- Python: `axionfem.circuit.fullwave_dense(results)` (also `Results.fullwave_dense()`) returns the dense curve as
  arrays. `axionfem.circuit.fullwave_model(results)` evaluates the model at any frequency. Both raise a ValueError
  when the sweep has no model.
- Touchstone: `Results.write_touchstone(path, dense=True)` (also `axf export-touchstone --dense`) writes the dense
  curve's port matrix as S-, Z- or Y-parameters in exp(+j omega t) ([touchstone.md](touchstone.md)). Its passivity is
  checked on the dense points only, not between them.

**Pole classes** (`rational::classify_poles`). Calling every in-band pole with |Im p| ≤ Re p a self-resonance would
include upper-half-plane poles, background poles of Q 0.5 and residues of 1e-16; the classes below sort them.

**The uncertainty of a pole.** Whether a pole is lossy, lossless or growing is decided on its Im p against its own
uncertainty, not against a fixed line. The fit's resolution of Im p is 1e-12 to 1e-9 of |p|, depending on its
conditioning. With a fixed line Im p > 1e-12 |p|, the four-port gap case's strongest resonance (a lossless model,
every solve a support point) would be:
- `non_passive`, and so dropped from `self_resonances`, in 58-62 of 200 noise realizations;
- given a spurious Q ≈ 1e11 in another 50-63.

Superconducting cavities are lossless (PEC) here, so this is the case that matters. σ_Re and σ_Im
(`pole_uncertainty_Hz`) combine in quadrature three parts:
- **Leave-one-out:** the jackknife over the refits of the solves with one solve left out, each with the final fit's
  options. It measures how much the pole rests on the solves.
  - One high-leverage refit can dominate it: the refit without the only solve near a narrow pole, which the sweep
    places there by design. On the lossy cavity's solves under phase noise σ_Im was then up to 4000× the actual
    scatter of Im p, and Q were withheld. That is conservative, and it stays.
  - Dropping that refit (a jackknife without the refit farthest from the median) is not done: that refit is also the
    only one that shows a *bad* near-pole solve, and the FEM's errors are largest exactly there (the system is
    nearly singular next to a pole). Under that error structure a trimmed σ leaves 8-15 % of resolved Q beyond 3σ (|z| up
    to 34) and flips lossless poles in mixed models; the plain jackknife gives ≤ 0.7 % (`test_rational_fit` case 21).
  - A median absolute deviation fails the same way more strongly.
  - Keeping the trimming would need the dropped solve's own error measured (re-solving the pole-carrying solves at a
    tighter `AXF_FW_RTOL` where the trimmed and plain classes differ); a proxy (noise refits scaled by the resonant
    amplitude) fixed the synthetic cases but withheld the TEM line's Q in every realization, and was not adopted.
- **Offset:** the distance of the model's pole from the mean of the refits' poles. The spread is about the refits'
  own mean, so without this term a model that is itself the outlier looks certain. In one case every refit put a Q 1e11
  pole at Im p ≈ −9.1e-4 Hz, the 6-point model put it at +1.85e-3 Hz, and the spread was 1.05e-5 Hz: it would be
  `non_passive` at 176 σ (`test_rational_fit` case 17).
- **Noise:** the RMS deviation over 8 refits of the solves perturbed by relative Gaussian noise of the solves' own
  size, a fixed seed. It measures how much the pole rests on their last digits. That size, `settings.pole_noise`, is
  the largest relative residual of the fitted port solves, within [1e-12, 1e-6].
  - The perturbation has a random phase, v (1 + ν (g₁ + i g₂)/√2), as the error of a complex-arithmetic solve does
    (a design with loss or a radiation boundary). A real factor changes magnitudes only, and misses the scatter of a
    Q 1e11 line's Im p by 900×.
  - An output whose every solve is real, or every solve imaginary, keeps a real factor. That is a lossless model
    solved in real arithmetic (its Re Z is exactly −0.0), whose errors have no phase; a phase there would invent a
    loss.
  - The draws are a Box-Muller transform of `std::mt19937`'s raw output, which the C++ standard specifies, instead of
    `std::normal_distribution`, which is implementation-defined. A marginal pole's class can
    still differ in the last bit of `std::log`, `std::sin` and `std::cos` between math libraries. Within one binary
    it is deterministic and the same on every rank.

Either part alone missed cases. On the cavity's solves under 1e-10 noise, a mode's leave-one-out spread was 1.3e-5 Hz,
while the noise moved its Im p by 3e-4 Hz. A refit's pole is matched to the model's within half the distance to the
model's nearest other pole.

The comparison model r⁻ is not used for this. It locates poles about 1e6 times worse than the model, and it hides
genuine Q of 1e9-1e11.

**Loss and growth.** Let line = max(1e-12 |p|, 3 σ_Im).
- `Q` = Re p/(2|Im p|), with `Q_uncertainty` = Q σ_Im/|Im p|, is given only if −Im p > line.
- A passive pole that does not meet that, and is not `unresolved`, is lossless within the fit, with
  `Q_min` = Re p/(2 max(1e-12 |p|, |Im p| + 3σ_Im)), a lower bound at 3σ: the true |Im p| can be up to
  |Im p| + 3σ_Im. (Re p/(2 line) would be up to 2× higher, and with a lossy pole next to the target 1-2 of 160-270
  such bounds per trial row would exceed the true Q, by up to a factor 1.21.)
- A pole with Im p > line grows in time under e^{−iωt}. A passive structure has none.

**Classes**, in order:
- `unresolved`: fewer than half the leave-one-out refits reproduce the pole, or its position is uncertain by more
  than max(|Im p|, one candidate step). Its position, width and passivity are not determined by the solves.
- `non_passive`: Im p > line.
- `negligible`: the pole's largest residue over the outputs, relative to the output's scale, divided by
  max(|Im p|, one candidate step), is below the tolerance. That is the pole's contribution at its half-width, or one
  candidate step away for a narrower pole: it changes no output by the tolerance anywhere the estimate looks.
- `background`: Q < 2 (from Im p as fitted). Its half-power width exceeds a quarter of its frequency, so it shapes a
  trend of the band rather than a peak (the DMRadio-Core models have one at 82-85 MHz with Q 0.46-0.50).
- `resonance`: otherwise.

A weak genuine mode can be `negligible` at a loose tolerance. Every pole stays in `model.poles_Hz`.

The cost is M + 8 refits of the final solves (M solves), once per sweep, after the loop, on every rank. Measured at
M = 30: 0.4 s at 1 port, 0.8 s at 4, 6 s at 16 and 26 s at 32 ports (19 s at 4 ports with
M = 60). That is small against FEM solves but grows with the ports. The removal after convergence (step 5) adds one
weight solve per needle candidate, when there is one.

## Defaults and why

| setting | default | why |
|---|---|---|
| `tolerance` | 1e-3 | Z and Ψ to 0.1 %. F = Ψ⁴/L² then carries about 0.4-0.6 %, below a typical mesh error. The scan-time quadrature with 4 fixed points was 6.5 % off on the DMRadio-Core band. The spec accepts 1e-6 to 0.1: below 1e-6 the fit tolerance reaches the solves' own accuracy. |
| `initial_points` | 5 | With derivatives, 5 solves already fix a type (4, 4) fit of Z. Fewer gives the error estimate too little to compare, more costs solves that a smooth band does not need. |
| `max_solves` | 30 | In the trials at the default tolerance the sweep converges in 8.4-9.2 solves on average, none took more than 15, and none ran out. The resonant FEM cavity with 8 poles in the band took 20. 30 bounds a pathological band's cost. |
| `spacing` | `log` | As `f_min`/`f_max`/`points`. Over a narrow band it is nearly linear. |
| `dense_points` | 400 | Enough for plots and for a trapezoid scan-time quadrature over a smooth band. The model can be evaluated anywhere else. |

## Validation

### Trials on random resonant responses (the driver alone)

`benchmarks/adaptive_sweep_trials.cpp` runs `rational::adaptive_sweep` and `rational::classify_poles` on synthetic
single-port responses; the record is `benchmarks/records/adaptive_sweep_trials.json` (110 rows; the source hashes are
recorded). No row has a removal that made the grid wrong (`removal_made_grid_wrong` 0 throughout). The protocol:
- Z = −0.3if + Σ i a_n f / (p_n² − f²) with dZ/df, and Ψ = 1 + Σ b_n p_n² / (p_n² − f²), in x = f/unit;
- 0-3 poles in the band [unit, 2 unit], two above it and one below;
- strengths a_n = 10^U(−2, 0), b_n normal (σ 0.3);
- lossless (real p_n) or lossy with Q = 10^U(1, 5); `mixed`: 0-6 in-band poles, each lossless or lossy with
  Q = 10^U(1, 7) (seed 99, a hard set);
- weak-mode families: one extra mode of strength 1e-7 or 3e-7 and Q 1e7 or lossless, at a
  random in-band frequency or 3e-5, 3e-4 or 3e-3 (relative) from the first strong in-band mode (`AXF_TRIAL_WEAK`,
  `AXF_TRIAL_WEAK_NEAR`; trials without an in-band mode are skipped);
- relative Gaussian noise on every value and derivative: magnitude only (a real factor), or with a phase
  (`AXF_TRIAL_PHASE=1`: 1 + ν (g₁ + i g₂)/√2, "phase noise");
- 5 log-spaced initial points, `max_solves` 60, fit tolerance as the engine's;
- 300 trials per row, the same responses in each unit (seed 7).

"True" is the final model against the exact response on 20001 points, in the fit's measure, outside 1e-6 of a real
pole. "Wrong" counts sweeps that converged with a true error above the tolerance. `needle_wrong` counts converged
models whose own near-real poles that match no true pole carry an error above the tolerance at Re q (1, 1 ± 1e-9,
1e-8, 1e-7): needles that narrow are invisible to the 20001-point grid.

**Random streams.** By default the responses and the noise come from one random stream. A driver change that alters
one sweep's number of solves therefore changes the *responses and the noise* of every later trial of the row, so two
drivers' rows are not the same trials after the first that differs. Two tools make comparisons exact:
- `AXF_TRIAL_STREAMS=1`: a noise stream per trial (trials and highq modes); the responses then stay the same too;
- `AXF_TRIAL_STATE_OUT=t:FILE` / `AXF_TRIAL_STATE_IN=FILE`: the random state before trial t, written by one driver
  and read by another, replays exactly that trial. The states of three hard trials (236, 166 and 88) are in
  `benchmarks/data/adaptive_sweep_replays/`, and the record replays them.

Each cell: mean (max) solves; not converged; wrong (worst true error / tolerance). The last column counts the wrong
ones whose worst point lies outside |f − p| < |δp|/tolerance of a lossless pole (Limits: inside it no fit controls
the pointwise error), [1, 2] + Hz. In lossy rows every wrong sweep counts there.

| poles | tolerance | noise | band [1, 2] | band [1e8, 2e8] Hz | wrong outside the pole neighbourhood |
|---|---|---|---|---|---|
| lossless | 1e-3 | 1e-11 | 8.80 (15); 0; 1 (1.4) | 8.80 (15); 0; 1 (1.4) | 0 |
| lossy | 1e-3 | 1e-11 | 8.44 (13); 0; 0 | 8.44 (13); 0; 0 | 0 |
| lossless | 1e-3 | 1e-8 | 9.22 (15); 0; 3 (1.7) | 9.22 (15); 0; 3 (1.7) | 0 |
| lossy | 1e-3 | 1e-8 | 8.72 (14); 0; 0 | 8.72 (14); 0; 0 | 0 |
| lossless | 1e-5 | 1e-11 | 10.40 (18); 0; 2 (1.1) | 10.32 (17); 0; 2 (1.5) | 0 |
| lossy | 1e-5 | 1e-11 | 9.64 (14); 0; 0 | 9.64 (14); 0; 0 | 0 |
| lossless | 1e-5 | 1e-8 | 39.07 (60); 138; 10 (19) | 40.75 (60); 152; 8 (18) | 2 + 2 |
| lossy | 1e-5 | 1e-8 | 15.33 (60); 4; 7 (5.2) | 16.81 (60); 7; 3 (11) | 7 + 3 |
| lossless, values only | 1e-3 | 1e-11 | | 11.73 (60); 1; 3 (3.1) | 1 |
| lossy, values only | 1e-3 | 1e-11 | | 10.98 (17); 0; 0 | 0 |

- **Default tolerance.** 2400 sweeps over both units, lossless and lossy, and two noise levels. None ran out of
  solves, none needed more than 15, and every wrong one (up to 1.7×) lies inside the neighbourhood of a lossless pole.
  Needles above the tolerance: 1 and 12 of 300 in the lossless rows, none in the lossy ones.
- **Tolerance 1e-5, noise 1e-11.** Every wrong stop lies inside a lossless pole's neighbourhood (up to 1.5×). The
  replayed hard cases converge: trial 236 (lossless, phase noise 1e-8) closes after 9 solves (true error 0.0025× the
  tolerance), trial 166 (lossy, phase noise 1e-9) converges after 16 (0.0002×), and trial 88 (lossless, tolerance
  1e-5) stops after 9 solves at 0.157×.
- **Noise-bound rows** (tolerance 1e-5, noise 1e-8): every sweep is flagged `noise_bound`; 46-51 % of the lossless
  ones and 1-2 % of the lossy ones end at `max_solves`; of those that converge, 5-6 % are off by up to 19×
  (lossless) and 1-2 % by up to 11× (lossy).
- **Units.** At tolerance 1e-3 the two units agree in every count and mean; at 1e-5, rounding-level differences
  change a few decisions.
- **Derivatives.** Values alone need 11.7 and 11.0 solves instead of 8.8 and 8.4.

**Pole classes in the trials.**
- Every true in-band pole of the default rows is found, as `resonance` in all but 3 of about 7900.
- No lossless pole is classed `non_passive` or given a Q in any row of the record (about 4000 in the all-lossless
  rows, 1500 in the mixed 1000-sweep row, 1300 under phase noise at the default tolerance).
- No resolved Q lies beyond 3σ in any row of the record, and no `Q_min` exceeds the true Q. With one near-pole solve
  corrupted by 24× the noise (`test_rational_fit` case 21), a few do (Limits).

**Inconsistent derivatives** (`derivative_bias`): b × |Z| per band width added to dZ/df, Hz, tolerance 1e-3, 100
trials per cell. Each cell: mean solves; not converged; Hermite dropped; wrong (worst); spurious resonances.

| bias b | lossless | lossy |
|---|---|---|
| 1e-6 | 9.1; 0; 0; 1 (1.8); 0 | 8.9; 0; 0; 1 (1.2); 0 |
| 1e-4 | 13.2; 0; 2; 10 (9.9); 4 | 12.1; 0; 1; 0; 2 |
| 1e-2 | 41.0; 38; 46; 6 (17); 2 | 41.3; 48; 40; 3 (12); 9 |

- At b = 1e-2, 38 and 48 % of the sweeps run out of solves despite the early drop of the derivative rows.
- At b = 1e-4 the rows stay in 98-99 of 100 sweeps. The fit absorbs the bias within 10 × its tolerance, and 10 of 100
  lossless sweeps stop at up to 9.9× the tolerance; 9 of them lie inside a lossless pole's neighbourhood. This
  inconsistency is not detected (Limits). With a noise stream per trial (record rows): 5 (14×) and 1 (1.1×) of 100.

**High Q** (`highq`): one in-band pole of quality Q among out-of-band ones, Hz, tolerance 1e-3, 300 sweeps per cell
(mean 7.9-8.1 solves, all converged).

| true Q | noise | Q resolved | median Q error | loss not resolved: lossless within the fit, with Q_min | Q_min above the true Q | resolved Q beyond 3σ |
|---|---|---|---|---|---|---|
| lossless | 1e-12, 1e-10, 1e-8; phase 1e-10 | 0 | — | 300 each (none `non_passive`) | — | — |
| 1e11 | 1e-12, 1e-10, 1e-8 | 176, 162, 94 | 1.9e-4, 1.3e-4, 3.0e-4 | 124, 138, 206 | 0 | 0 |
| 1e11 | phase 1e-10 | 97 | 8.2e-4 | 203 | 0 | 0 |
| 1e10 | 1e-12, 1e-10, 1e-8 | 238, 218, 186 | 2.7e-5, 2.3e-5, 7.0e-5 | 62, 82, 114 | 0 | 0 |
| 1e10 | phase 1e-10 | 165 | 1.2e-4 | 135 | 0 | 0 |
| 1e9 | 1e-12, 1e-10, 1e-8 | 284, 267, 261 | 3.1e-6, 3.4e-6, 1.5e-5 | 16, 33, 39 | 0 | 0 |
| 1e9 | phase 1e-10 | 242 | 2.3e-5 | 58 | 0 | 0 |

- Every pole in this table is classed `resonance`.
- `Q_min` is a 3σ bound: none exceeds the true Q.
- The resolved counts are those of the plain jackknife; a trimmed jackknife would resolve 34-95 % more at 1e11 and
  18-51 % more at 1e10, but can be blind to one bad near-pole solve (Limits).

**The FEM runs' own solves under noise** (`refitnoise`): the fitted solves of the four-port, tree-cotree, cavity,
lossy cavity and TEM runs (`benchmarks/adaptive_sweep.py samples`), refitted with the engine's options under relative
noise of 1e-12, 1e-10 and 1e-9 (magnitude, 100 realizations), and at each run's own `pole_noise` with and without a
phase (200 realizations). Per realization z = (fit − noise-free fit)/σ, for Im p and for a resolved Q.
- Every lossless pole stays a resonance with no Q in every realization, magnitude or phase noise: 2 + 2 + 8 + 1 poles.
- No resolved Q lies beyond 3σ in any row (largest |z| 0.81).
- |z| of Im p exceeds 3 only for the lossless four-port case under magnitude noise at its own level: 53 and 3 of 200
  (up to 8.7). Those shifts stay inside the 1e-12 |p| line, so no class changes.
- The lossy cavity's Q under phase noise at its own level (1.5e-9): withheld in 14 of 200 realizations at 178 MHz and
  73 at 240 MHz (σ inflated by the high-leverage refit, conservative); at 1e-8, 132 and 181. Under magnitude noise
  every Q but 1-7 per mode is resolved.
- The TEM line's Q of 1e9, 1e10 and 1e11 are resolved in every realization under magnitude noise; under phase noise at
  the TEM run's level (3.9e-9) the Q 1e11 line's in 44 of 200.
- A fixed line instead of σ would make the four-port case's 140.37 MHz pole `non_passive` or give it a Q in about
  60 % of realizations; ctest 15b checks that it does not.

### FEM validation (`benchmarks/adaptive_sweep.py`, record `benchmarks/records/adaptive_sweep.json`)

The record's `builds` names the build of every run; the gap, cavity and high-Q runs used 4 MPI ranks, the
DMRadio-Core runs 16.

**The FEM solves' algebraic errors** are not part of the record. Next to a pole the system is nearly singular and a
solve's error relative to F is largest, which is where the jackknife's high-leverage refit sits (above). For a
tolerance close to the solves' accuracy, re-solve the adaptive run's frequencies as a fixed list at a tighter
`AXF_FW_RTOL` on the same mesh and compare.

**Poles against eigenmodes** (`cavity`). The two-port frame (two gap ports, uniform source) in its closed PEC sphere
(R1 0.7 m), adaptive over 100-270 MHz, tolerance 1e-3, linear spacing. The same run computes the 16 lowest eigenmodes
of the same mesh (AME).
- The discrete Z and Ψ are rational in ω with poles at the discrete eigenfrequencies (ports open). So the model's
  poles must sit on them, up to the fit and the eigensolver.
- The sweep converged after 20 solves (5 initial) with 18 support points (estimate 4.7e-4, newest solve against its
  prediction 5.0e-4), with the derivative rows and a consistent fit. It found all 8 eigenmodes of the band as poles,
  each classed `resonance`, and no other in-band pole. That includes the close triple at 253.59, 255.50 and 256.09
  MHz (0.75 % and 0.23 % apart).
- Pole against eigenfrequency: at most 1.0e-9 relative, the weakest mode (253.59 MHz, strength 6.2e-6, σ_Re 36 Hz,
  1.4e-7 relative); the other seven within 2.2e-11.
- Every pole is lossless within the fit: no Q, `Q_min` 4.1e11 to 5e11 (σ_Im 2.3e-6 to 5.7e-5 Hz). All 20 leave-one-out
  refits reproduce each pole.
- Strengths (the largest residue relative to the output's scale and the band width) range from 6.2e-6 to 0.35. Three
  of the eight lie below the cleanup's candidate threshold (1e-4 here) and are kept because the solves see them. One candidate step away they contribute 0.012 to 690 of the output's scale, above the tolerance, so
  none is `negligible`.
- **Against solves it never saw** (`cavity_check`): 27 fixed-list solves at the midpoints between the adaptive solves
  and at the band points farthest from them. The model's largest error there is 8.5e-8 (Z 3.8e-8, Ψ 8.5e-8). The
  tolerance is 1e-3.
- The eigensolver's lowest returned mode is spurious (7.3 Hz, relative residual 1.0), so `eigenmodes.converged` is
  false and the run warns ("3D eigenmodes incomplete or nonfinite…"). Every other mode converged, up to 308 MHz, so
  the comparison covers the band.

**Q of lossy poles** (`cavity_lossy`): the same model with a dielectric slab (ε_r 2, tan δ 1e-3, 0.2 m cube at the
centre).
- The sweep converged after 23 solves with 19 support points (estimate 1.6e-5). All 8 poles are classed `resonance`,
  each with a resolved Q.
- Re p is within 2.1e-9 of the lossless eigenfrequencies of the same mesh (the eigensolver uses Re ε). The loss shifts
  Re p only at second order in tan δ.
- Reference Q = 1/(tan δ p_e), with p_e the slab's share of the eigenmode's electric energy (first order in tan δ).
  Seven modes list the slab among their three largest regions.
- Their fitted Q, from 6.7e4 to 6.9e6, agrees to at most 1.7e-5 (the other six within 6e-7). At Q 6.9e6 the pole is
  17 Hz below the real axis at 240 MHz, with σ_Im 0.44 Hz.
- Against 30 direct solves it never saw (`cavity_lossy_check`), the model's largest error is 8.2e-7 (Z 8.2e-7, Ψ
  6.2e-7).

**High Q in the FEM** (`tem_tand_*`). The shorted TEM line (nx 12, order 2) filled with a
dielectric of loss tangent tan δ has one pole near 249.83 MHz with Q = 1/tan δ. Each line is adaptive over 100-400 MHz
at tolerance 1e-3.

| tan δ | solves | class | Q / (1/tan δ) | Q_uncertainty / Q | σ_Im |
|---|---|---|---|---|---|
| 0 | 7 | resonance, lossless within the fit, Q_min 5e11 | — | — | 2.8e-7 Hz |
| 1e-11 | 7 | resonance | 1.00024 | 1.5e-3 | 1.9e-6 Hz |
| 1e-10 | 7 | resonance | 0.99995 | 1.5e-4 | 1.9e-6 Hz |
| 1e-9 | 7 | resonance | 0.999998 | 2.9e-5 | 3.7e-6 Hz |

**Adaptive against a dense sweep, DMRadio-Core** (`core_adaptive`; `core_dense_0..3`; 16 ranks each). The reduced Core design (`benchmarks/core_readout_ports.py` c360, the
full-circumference uniform feed) runs on a coarse mesh, so that 61 frequencies are affordable:
- order 2, h 0.32 m, surface 0.05 m: 1.2M unknowns, 6-14 min per frequency;
- radiation boundary, 70-140 MHz, tolerance 1e-3, log spacing.

The dense sweep is 61 log-spaced frequencies (the 4-point sweep among them) in four parallel parts. Each part also
solved 140 MHz, and the four rows agree exactly, so the parts share mesh and operator. They are fixed lists, so they do
not depend on the adaptive code (invariance, below).
- **Solves:** the adaptive sweep converged after 7 solves (5 initial), 5 support points, estimate 2.8e-5, against 61
  (76 min on 16 ranks). The band has no resonance: its one in-band pole, at 82.4 MHz with Im p = −90.2 MHz
  (Q 0.46), is classed `background`, and `self_resonances` is empty (`self_resonances_determined` true). The model's
  nearest pole outside is 152.6 MHz with Q ≈ 330, the first self-resonance just above the band (an extrapolation,
  unchecked).
- **Model against the 61 direct solves:** Z to 1.2e-7, Ψ to 5.1e-7 (tolerance 1e-3). L_eff = (ω dX/dω + X)/(2ω),
  from the model's derivative, matches the solves' exact dZ/dω to 3.2e-7.
- **Scan time** (`core_readout_ports.scan_time_yr`: the published Q, T, η(ω), DFSZ, trapezoid in ln f; the reduced
  design's years, so only ratios matter): the 61 solves give 26.084 yr. Against that:

  | quadrature | scan time | relative to 61 solves |
  |---|---|---|
  | the 4-point subset (70, 88.2, 111.1, 140 MHz) | 27.787 yr | +6.5 % |
  | the 7 adaptive solves alone | 26.991 yr | +3.5 % |
  | the adaptive model at the 61 frequencies | 26.084 yr | +5.6e-7 |
  | the adaptive model's dense curve (400 points) | 26.079 yr | −1.6e-4 |

  The 4-point error matches the 6.2 % found on an axisymmetric 61-point sweep of the same design. The model
  reproduces the 61-solve quadrature to 6e-7. Its 400-point curve differs from the 61-point trapezoid by 1.6e-4, the
  quadrature error of 61 points. Seven solves thus replace the 61 for this purpose.

**Gap validation** (`gap`). On one small mesh per case, an adaptive sweep (tolerance 1e-3, linear spacing,
`max_solves` 30) against a dense fixed list of 41 linearly spaced frequencies on the same mesh. The model is compared
with every dense solve in the fit's measure:

| case | design (band) | ports | solves (support points) | estimate | in-band poles | model against the 41 solves (Z, Ψ) |
|---|---|---|---|---|---|---|
| `tree_cotree` | the two-port frame in its PEC sphere, STRUMPACK with the tree-cotree gauge (115-150 MHz) | 2 | 9 (7) | 7.9e-5 | 2 resonances, 121.21 and 139.20 MHz, lossless | 2.0e-8, 2.1e-7 |
| `pml` | the TEM line (nx 12, order 2) with a PML end (100-300 MHz) | 1 | 7 (4) | 8.7e-5 | none | 2.0e-7, — |
| `robin_absorbing` | the TEM line with an absorbing (scattering) end | 1 | 7 (1) | 4.0e-7 | none | 6.1e-7, — |
| `robin_impedance` | the TEM line with copper impedance plates and an absorbing end | 1 | 7 (4) | 1.6e-7 | 1 unresolved | 2.3e-9, — |
| `four_ports` | the two-port frame with its slot split into 4 gap ports (115-150 MHz) | 4 | 8 (8) | 9.6e-5 | 2 resonances, 121.17 and 140.37 MHz, lossless | 1.8e-6, 1.8e-6 |
| `refusal` | the shorted TEM line with a 50 Ω validation shunt, GMRES capped at 32 iterations, one rank | 1 | 7 (5), 250 MHz refused | 6.4e-4 | 1 resonance, 249.827 MHz, lossless | 1.2e-7, — |

Every case converged with its derivative rows (`hermite.used_in_final_fit`, `fit_consistent`), and the model matched
every dense solve to 1.8e-6 or better against the tolerance 1e-3. Notes:
- The TEM line carries no axion source, so Ψ is zero and not fitted (—).
- `robin_absorbing`: the matched line's Z is constant to 6e-7 over the band, so one support point fits it. The
  comparison model of a one-point fit is the solves' mean (above), and the sweep converges in 7.
- `robin_impedance`: the one in-band pole, a weak pole of the impedance plates' √ω dependence, is `unresolved`: the
  leave-one-out refits do not pin it.
- `four_ports`: the pole at 140.37 MHz lies 4e-4 Hz below the real axis; with its uncertainty (σ_Im 3.2e-3 Hz) it is
  lossless within the fit, not a resonance of Q 1.8e11.
- `refusal`: a deliberately starved configuration. The initial point 250 MHz, 0.07 % from the line's
  quarter-wave pole, failed the multiport residual gate. The sweep recorded it in `refused` (with its diagnostics
  file), went on, and converged in 7 solves. The pole is where the unrefused run has it (249.827 MHz), and the model
  matches the uncapped dense solve at 250 MHz to 1.2e-7. With four ranks the same cap refused all five initial points:
  the sweep then stopped with "no usable solve among the initial points", unconverged, and
  wrote its results, as the default test `test_adaptive_with_every_solve_refused` checks.
- No configuration needed to be refused.

**Fixed lists are bit-identical** (`invariance`: a build without the adaptive sweep against one with it). Four
fixed-list designs were solved by both builds, and every output value was compared except timings:
- `stage1` (the two-port frame): two ports, source check, shunt loads: 981 values, 704 floats;
- `stage1_lossy`: tan δ slab: 1053, 754;
- `stage1_radiating`: radiation boundary, exterior dZ/dω: 833, 617;
- `stage1_gauged`: tree-cotree STRUMPACK: 1222, 865.

There were 0 differences. None of the four has a Robin group or PML, whose `dZ_domega_H` gained the termination
terms with the adaptive sweep.

The default test `test_adaptive_rows_equal_a_fixed_list` also checks that the adaptive sweep's rows equal a fixed
list's at the same frequencies, bit for bit.

## Limits

- **What the tolerance covers.** It bounds the model against the solves of this mesh (the discrete response), not
  the discretization error. Mesh convergence (h/p, `refine_and_compare`, the error indicator per solve) is a
  separate question, as for a fixed list; `refine_and_compare` does not take adaptive designs and says so.
- **Many resonances.** The model's degree grows with the poles in or near the band, and each needs about a solve.
  The band of 100-270 MHz of the two-port frame's cavity, with 8 modes, takes 20 solves (Validation); a band with
  about 40 eigenmodes needs about 40 solves or more. Split a rich band, or raise `max_solves`.
- **Closely spaced poles.** Two poles closer than the spacing of the solves near them are resolved only if the solves'
  accuracy carries them: the fit sees their sum through the tails. The estimate then rises near them, and the sweep
  places solves there. Poles closer than about tolerance × their half-width merge into one at that tolerance.
- **Weak, very high-Q modes.** A pole narrower than the candidate grid step is found from its tails, if the data are
  accurate enough, and its peak and half-power points are added as candidates. Its Q is then as accurate as the fit.
  A pole that changes no solve by more than the cleanup's allowance (up to 10 × the fit tolerance, 1e-5 of the
  output's scale at the default tolerance) is indistinguishable from a doublet and can be cleaned away; its peak
  height is its residue over |Im p|. After convergence an in-band pole on the real axis that is invisible one
  candidate step away, has a needle reaching the tolerance, disappears from the fixed-degree refit without its
  nearest support point and passes step 5's checks is removed too. A weak genuine mode that only that one solve
  carries can meet that: in a close-pair scan 36 of about 19 600 found weak modes were deleted at
  tolerance 1e-3, and the near, far, fine and band measures were made wrong in 1, 4, 1 and 2 of 62 216 sweeps.
  A high-Q, very weakly coupled mode can therefore be missed while it is invisible at the solved frequencies, and a
  weak one that is kept can be classed `negligible`. And a weak mode makes the stop less reliable: with one mode of
  strength 1e-7 to 3e-7 added to the trials' lossless responses, 6-15 % of converged sweeps exceed the tolerance on
  the 20001-point grid, against 0.3 % without it; close pairs 3e-4
  apart, 3.5-9.2 % (13 % at 3e-3).
- **High-Q and superconducting cavities.** The driven operator A(ω) has no RF surface loss: conductors and
  superconductors are PEC, and the `rf` surface-resistance models enter only the eigenmode wall-loss post-processing.
  A superconducting cavity is therefore a lossless model here, and its resonances come out `lossless_within_fit` with
  a bound `Q_min`, not a Q. A loss that A(ω) does contain (dielectric tan δ, conduction) is resolved when
  −Im p > 3 σ_Im. The FEM TEM line resolves Q of 1e9-1e11 to 2.4e-4 or better. In 8-solve synthetic sweeps (Trials)
  a Q of 1e11 is resolved in 31-59 % of them, 1e10 in 62-79 % and 1e9 in 87-95 % (with phase noise 1e-10: 32, 55
  and 81 %; record rows 27-35 and 81-83). The rest are reported lossless within the fit with a 3σ bound Q_min,
  which is the honest answer when the fit cannot tell; they stay `resonance`.
- **Near a lossless pole.** The pointwise measure near a pole on the real axis is relative error ≈ |δp|/|f − p|, δp
  the fit's pole error (1e-12 relative in the tests, about the noise in the trials). It exceeds the tolerance inside
  |f − p| < |δp|/tolerance whatever the fit, and the estimate does not look closer than a quarter grid step.
  In the trials every false stop at the default tolerance lay inside it, within 3e-6 (relative) of a lossless pole.
  At tolerance 1e-5 with noise 1e-11, one response (trial 88) would stop 24-44× off at a near-real doublet that the
  capped cleanup keeps; with the removal after convergence (The adaptive loop, step 5) it is 0.16× off.
- **Needles remain in about 1 % of converged models**. A near-real pole of the model that
  matches no true pole can carry a needle far narrower than any candidate (up to 1e10× the tolerance at its centre),
  which the trials' 20001-point grid cannot see either. In a scan of 82 800 trial sweeps, evaluated at the model's own
  near-real poles, 922 converged models (1.1 %) had such a needle above the tolerance (0.16, 0.49 and 2.7 % at
  tolerance 1e-3, 1e-4 and 1e-5), against 1229 without the removal. The removal takes only needles that the fixed-degree refit without their nearest
  support point loses and whose removal passes its checks; the others stay. The record counts them
  (`needle_wrong`), and the removals that turned the grid wrong (`removal_made_grid_wrong`).
- **The removal trades a needle for a small change elsewhere.** It may change the model by up to 0.25 × the tolerance
  on the grid (where the estimate leaves the slack) and move other real-axis poles by min(0.1 × tol × r,
  tol × 1e-6 |p|). In the 82 800-sweep scan no removal made the trials' grid wrong (0 of 303), but in 25 the grid
  error grew without crossing the tolerance, and in a close-pair scan a few models were made wrong next to a weak
  mode the removal deleted.
- **Noise.** The fit tolerance is 1e-3 × the tolerance, not below 1e-10. A tolerance within about 1000× of the solves'
  accuracy fits noise. In the trials with noise 1e-8 at tolerance 1e-5 (fit tolerance = noise), 49-51 % of the
  lossless sweeps ran out of solves, and 5 % of the converged ones were off, by up to 18.5×. The spec refuses a
  tolerance below 1e-6. With the default solver settings (true residual 1e-10) keep it at 1e-5 or above, and raise it
  where the solves are less accurate (tree-cotree at low frequency, AMS near high-Q poles, PML: about 1e-7).
- **Radiating models.** Z(ω) has a branch point at ω = 0 in the continuum, from the outgoing Hankel functions. The
  truncated discrete map is rational, but a band that reaches down toward kR1 ≪ 1 sees the static limit's structure,
  which needs more degree. Bands far above the static limit are smooth. The model is a band model: outside
  [f_min, f_max] it extrapolates without any check.
- **Dispersive boundaries.** An impedance wall's Z_s ∝ √ω (a branch point at ω = 0), a sheet's tan(k d/2) and the
  PML's stretch are not rational in ω. Their dZ/dω is exact (above), and the fit approximates them like any other
  smooth dependence in the band, but the pole interpretation (eigenfrequencies) then holds only approximately. The
  gap validation's Robin and PML lines converged in 7 solves or fewer.
- **The estimate is not a bound.** Agreeing models and two confirming solves make a false stop rare, not impossible
  (Trials, above). A resonance invisible at every solved frequency, with negligible tails, is not found by any
  sampling method. The stop needs at least two solves beyond the initial points (four for the closure on tight
  confirmations).
- **σ rests on a noise proxy.** `pole_noise` (the largest relative port residual) is the noise level of the refits.
  The FEM's actual errors are not uniform: they are largest for a solve next to a pole (a nearly singular system),
  and they carry a phase in a lossy design but none in a lossless one (real arithmetic). The noise refits perturb
  every solve alike, so they can overstate the typical solve's error and understate the near-pole one's. σ is an
  estimate, not a bound; where a Q matters, re-solve the pole-carrying frequencies at a tighter `AXF_FW_RTOL`.
- **σ is conservative with one high-leverage solve, and can still undercover with one bad one.** The plain jackknife lets the refit without the only solve near a narrow pole
  inflate σ (up to 4000× the actual scatter): some well-resolved Q are withheld (the lossy cavity's Q in 5 % of
  realizations at its own phase noise, 20 % at 1e-8, ×24 near-pole errors included). With one near-pole solve
  corrupted by 24× the noise at the worst phases, 4 of 80 Q 1e9 poles still lie beyond 3σ, up to 9σ
  (`test_rational_fit` case 21); with that error structure applied to the sparse TEM sweeps, 2-4 of 64-66
  resolved Q at tan δ 1e-10 and the 2-3 resolved at 1e-11 do, and one lossless-like Q 1e11 pole is `non_passive`.
  - **A measured cover, not adopted.** Adding in quadrature the pole's RMS deviation over 8 refits per solve within
    1e-3 of the pole, with that solve alone perturbed by 24ν and a random phase (8 fits per near-pole solve), brings
    case 21 from 4 to 1 beyond 3σ and keeps the lossy cavity's resolved Q, but withholds the TEM line's Q 1e11 in
    every realization (100 → 0 of 100 under magnitude noise, 23 → 0 under phase noise). Like an amplitude-scaled
    proxy, it cannot tell designs with noisy near-pole solves from
    designs without them; it would fit as an opt-in "pessimistic σ" for the former.
- **Noise-bound sweeps are flagged, not prevented**. The flag fires when the fit tolerance
  is below 10 × `pole_noise`; the sweep itself is unchanged. In the trials (tolerance 1e-4, phase noise; record rows):

  | fit tolerance / noise | flagged | lossless: converged but wrong (worst) | mixed: converged but wrong (worst) |
  |---|---|---|---|
  | 1 | yes | 35 of 300 (39×) | 68 of 300 (40×) |
  | 10 | no | 10 (18.5×) | 18 (35×) |
  | 33 | no | 5 (6.8×) | 9 (7.4×) |
  | 100 | no | 2 (12×) | 4 (2.1×) |

  The stop's reliability degrades gradually with the ratio, and below about 100 the trials' mixed models still stop
  wrong in 1-6 % of the sweeps, so the line at 10 is a guard against the worst regime, not a guarantee above it.
  Most of these wrong stops lie inside a lossless pole's neighbourhood (below), not all. The flag is also
  conservative for FEM runs whose residual overstates their errors: the PML gap case at the default tolerance
  (`pole_noise` 1.76e-7, fit tolerance 1e-6) is flagged, yet its model matched 41 dense solves to 2.0e-7.
  It can also miss: next to poles a solve's error can be well above `pole_noise` (the trials use 24×), so for such a
  solve a sweep whose fit tolerance is 10-100× `pole_noise` is effectively at 0.4-4× the noise, the regime of the
  margin rows above. Such a run gets a second warning ("may be noise-bound there"), not the flag.
- **The closure's gate is not local.** In practice it excuses at least 99 % of the candidates whenever r⁻ has a
  spurious pole, so the closure rests on its tight confirmations, the model-change condition and a consistent fit
  (The adaptive loop). Closures are 0.6-1.1 % of converged sweeps, and at the noise-bound flag's boundary (fit
  tolerance 10 × the noise) 4 of 315 closures at tolerance 1e-5 were wrong (up to 6.1×); there the closure is about as
  reliable as the normal stop. A local comparison-model test is not implemented.
- **Moderately inconsistent derivatives.** The consistency rule and the early drop catch derivative data that the fit
  cannot absorb (a bias of 1e-2 of |Z| per band width). A smaller inconsistency that the fit absorbs within 10 × its
  tolerance passes, and can mislead the model between solves. With a bias of about 1e-4 of |Z| per band width, 10 of
  100 lossless trial sweeps stopped at up to 9.9× the tolerance (9 inside a pole's neighbourhood), and 2-4 per 100
  sweeps listed a spurious resonance. No source of such data is known in the engine; a new term of A(ω) without its
  d/dω in A′ would be one.
- **Cost.** The greedy recomputes a full `JacobiSVD` of the stacked Loewner matrix for every support size, on every
  rank: about M·K·m³ per fit, with M solves, K = n² + n outputs and m support points. That is negligible against a FEM
  solve at the sizes so far (4 ports, 8 support points), but grows quickly with many ports; a QR of the stacked
  matrix before the SVD would cut it (not implemented).
- **Scope.** 3D designs only. The axisymmetric engine refuses `fullwave.adaptive`; the fit itself
  (`axionfem/rational_fit.hpp`) has no FEM dependence and could serve it.
- **The scene** stays at the initial point nearest `scene_frequency`, the first by default.
- **Surface losses** (`R_loss_ohm`, `Q_loss_tuned`, ...; [fullwave_surface_losses.md](fullwave_surface_losses.md)) are not part of the
  model. They are evaluated at the solved rows only, and the dense curve does not carry them. Their Γ has a double pole
  wherever Z has a simple one, and the perturbative loss fails near those poles anyway. Away from the poles, R_loss
  between the solves is smooth enough to interpolate from the rows. They are post-processing of the solved fields and
  put nothing into A(ω), so the exact dZ/dω (and its "exact" scope, which the fit relies on) is unchanged by them.
