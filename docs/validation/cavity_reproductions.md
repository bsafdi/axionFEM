# Cavity haloscopes: reproductions of published simulations

**Beta.** Two studies: 2D axisymmetric reproductions of published cavity simulations ("Per benchmark (2D)" below), and
3D reproductions (RADES filters, CAPP sector cells) with further checks of the 2D study (BI-RME probe, QUAX 2201
absolute f, Stern B/E) and a blind check (35 independently derived numbers agree to ≤ 3e-8; "3D benchmarks and
further checks" below). Scripts in `benchmarks/cavity/repro/`, records `benchmarks/records/cavity_repro_<name>.json`.

## Conventions

- **C** is the eps-weighted form factor with V = the cavity volume, metal excluded ("C_vac"). For HAYSTAC,
  `C_full` rescales to the volume with the rod included (V_full/V_vac = 1.332). The QUAX cavities have no metal
  inside, and the compared quantity is C with V = the full cavity volume **including the cone recesses** (2004,
  2201). The `C_full` column of the QUAX records uses V = pi R^2 L (cylinder only) and is not compared.
- **Converged** means order 3 on the middle mesh of a study with at least three order-2 meshes. Every mesh error
  quoted below is at least 10x smaller than the difference being judged, except where a row says otherwise.
- **Signs.** Relative diffs are (published − axionFEM)/axionFEM. Absolute diffs (MHz) are published − axionFEM.
- **Verdict categories.**
  - *agrees (predicted)*: no parameter was tuned on this quantity; the agreement is within a tolerance fixed
    before the run (refs README) or within the printed rounding.
  - *agrees (after fitting X)*: X was chosen with this quantity (or a close relative) in view; only the residual
    is evidence.
  - *consistent (circular)*: the input was inferred from this same number; the row checks bookkeeping only.
  - *not decidable*: the reference value or the 2D model cannot settle it (reason given).
  - *disagrees (cause: …)*: outside tolerance; the cause is given if known, "unexplained" otherwise.

## Independent cross-check

An independent axisymmetric solver that shares no code with axionFEM (`axq1.py`: scalar H_phi
formulation of the m = 0 TM problem, bilinear Q1 elements on a graded tensor grid, PEC as the natural BC, Q_wall by
PEC perturbation) and an exact Bessel transfer-matrix model of z-invariant layered cylinders (`layered.py`) are in
`benchmarks/cavity/repro/independent/`, with the driver scripts (`hay.py`, `hayq.py`, `hayaxle.py`,
`clam.py`, `pill.py`, `alt2201.py`) and the axionFEM side of the clamshell sensitivity (`clam_sens_axionfem.py`). The
scripts print their results; their outputs are not stored as records, so this cross-check is rerun by hand, not by a
test. Run by hand, they agree with axionFEM to ≤ 1e-5 wherever the comparison is well posed (for example,
`hay.py` at the nominal 0.254 mm gap converges under grid refinement to the recorded f = 5.86096 GHz and
C_vac = 0.77255 to better than 1e-5); rerunning the scripts reproduces these comparisons:
- HAYSTAC: f, C_vac, Q, the gap sweep and the smeared-axle estimate;
- QUAX 2201 and 2004 flat-end layered models;
- the clamshell without a gap.

So, to the extent of these by-hand checks, none of the remaining discrepancies (HAYSTAC f, QUAX 2201 absolute f,
clamshell C) is a solver error.

## Summary table

| benchmark | quantity | published | axionFEM (converged) | diff | verdict | notes |
|---|---|---|---|---|---|---|
| ADMX thesis Tab 6.1 (4 cylinders) | f TM010, TM011, TM020 (12 entries) | 0.547 … 5.73 GHz | closed form to ≤ 1e-7 | −0.34 % … +0.22 % | **agrees (predicted)**, all 12 | tolerance = half a unit of f plus the propagated half units of R and H (TM011 includes H) |
| ADMX thesis | C010, C020 | 0.69166, 0.13127 (closed form) | 0.6916603, 0.1312713 | ≤ 6.5e-7 | agrees (predicted) | C011 ≤ 1e-17 (exactly 0) |
| KLASH Tab 2 (R = 2 m) | f TM010 | 57.37 MHz | 57.3713 MHz | −2.2e-5 | **agrees (predicted)** | L = 4 and 3.5 m |
| KLASH Tab 2 | Q0 ratio L = 4 m / 3.5 m (sigma-independent) | 1.0479 | 1.0476 | +0.03 % | **agrees (predicted)** (rounding ±0.67 %) | |
| KLASH Tab 2 | absolute Q0 TM010, sigma = 5.8e7 | 153 000 / 146 000 | 152 820 / 145 874 | +0.12 % / +0.09 % | consistent (circular) | refs inferred sigma = 5.8e7 from this table |
| BI-RME 2107.03137 Tab I | f TM01p, p = 0–5 | 2.55920 GHz (TM010) … | R = 45 mm: 2.549830 GHz | +0.33 … +0.37 % | disagrees at the stated R (cause: radius offset) | |
| BI-RME | f TM01p after a one-parameter fit | — | R fitted: 44.839 mm (refs: 44.835) | residual ≤ 7.7e-5 over 6 modes | **agrees (after fitting R)** | an L-only fit leaves 3.7e-3: the offset is a radius, not a length |
| BI-RME | Q0 TM010 (sigma = 6e7) | 33 069 | 33 466 (R = 45), 33 410 (R = 44.835); **3D study, with the pin: 33 231** | −1.0 % empty; **−0.5 % with the pin** | disagrees (cause: half is the probe pin; rest unexplained) | see the BI-RME probe section |
| BI-RME | C TM010 | 0.68505 | 0.69166 empty; **with the pin at the depth fitted to the p ≥ 1 C: 0.68591** | −0.96 % empty; **−0.13 % with the pin** | agrees (after fitting the pin depth) | probe-induced TM01p mixing reproduces the whole Table I C column |
| QUAX hollow sapphire 2004.02754 | f pseudo-TM030 | 10.915 GHz | 10.9028 GHz (baseline cones); 10.9141 (as-built tubes) | +0.11 % | **agrees (predicted)** (refs ~0.3 %) | cone variants change f by < 1 MHz |
| QUAX 2004 | Q0 (R_s 5.5 mOhm, tan d 2e-6) | 1.91e6 | 1.987e6 | −3.9 % | disagrees (cause: probably cone/bore geometry, refs' main systematic; unproven) | cone variants give 1.95–2.00e6; flat ends give 0.88e6 |
| QUAX 2004 | p_e (derived from Tab 1) | 0.167 | 0.1671 | 0 % | **agrees (predicted)** | |
| QUAX 2004 | wall-only Q (derived from Tab 1) | 5.4e6 | 5.91e6 | −9 % | disagrees (cause: as for Q0) | |
| QUAX 2004 Tab 1 | Δf for R1, R2, t1, t2, eps 11.5 | −145, −13, −145, −15, −25 MHz | −143.4, −13.3, −145.2, −17.1, −23.9 MHz | −1.6, +0.3, +0.2, +2.1, −1.1 MHz | **agrees (predicted)**, all 5 within the printed rounding | rounding: ±5.5 MHz (10.77, 10.77, 10.90, 10.89 printed), ±1 MHz (10.902); R1 and t1 give the same Δf, as in the paper (f is set by the tube-1 outer face) |
| QUAX 2004 Tab 1 | Δf, eps = 9.39 | +235 MHz | +172 MHz | +63 MHz | disagrees (cause: probable typo in the paper) | independent layered model +170 MHz; refs flag the row |
| QUAX 2004 Tab 1 | Q, tan d = 1e-5 / 1e-7 | 0.54e6 / 4.98e6 | 0.5435e6 / 5.38e6 | −0.6 % / −7.5 % | agrees (predicted) / disagrees (cause: as for Q0, wall-dominated) | |
| QUAX 2004 Tab 1 | Q, t2 row | 0.86e6 | 0.65e6 | | not decidable (avoided crossing in our model: C split over 3 modes) | |
| QUAX 2004 | flat-end cross-check f, C030 | 10.883 GHz, 0.041 (independent layered model) | 10.8826 GHz, 0.0410 | < 4e-5, < 1 % | agrees (predicted) | |
| QUAX two-shell 2201.04223 | Δf(eps 11.44 → 11.2), 4.2 K | +18.9 MHz | +18.58 MHz; **18.77 MHz rescaled by f** | +0.7 % (rescaled); +1.7 % raw | agrees at 0.7 % after f rescaling (tolerance post hoc) | no tolerance was set in refs before the run; printed ±0.1 MHz, residual 0.13 MHz; same within 0.02 MHz for every cone/gap/contraction variant; 11.44 → 11.1: 26.9 vs 26.49 MHz |
| QUAX 2201 | Δf(T), contraction only (eps 11.44) | +8.8 MHz | +8.79 (set A) / +9.12 (set B) MHz | 0 % / −4 % | **agrees (predicted)** (set A, within ±0.1 MHz) | total 293 → 4.2 K: 27.7 vs 27.37 MHz |
| QUAX 2201 | p_e | 0.170 | 0.1698 | +0.1 % | **agrees (predicted)** (refs ~1 %) | 0.1671–0.1724 over all variants |
| QUAX 2201 | C030 trend vs eps (11.44, 11.3, 11.1 rel. to 11.2) | +0.93, +0.31, −0.62 % | +0.73, +0.31, −0.31 % | — | **agrees (predicted)** within printed rounding (±0.32 % per difference) | |
| QUAX 2201 (soft) | f, 4.2 K, eps 11.2 | 10.4745 GHz | 10.3685 GHz | +1.02 % | disagrees (cause: not determinable from the paper; no single stated input fits; a common ~1 % geometry scale is not excluded) | not a solver issue (layered model agrees); no single stated input within tolerance |
| QUAX 2201 (soft) | C030 | 0.0321 | 0.03252 | −1.3 % | agrees (after choosing the 2004 cones) | the cones were adopted because they fit C, Q_Cu and the volume (flat ends: 0.0378) |
| QUAX 2201 (soft) | Q_Cu (R_s 4.83 mOhm) | 1.10e7 | 1.133e7 with the 2004 cones | −2.9 % | agrees (after choosing the 2004 cones) | flat ends 2.74e6 |
| QUAX 2201 | cavity volume at 293 K | 1.0776 L | 1.07766 L (D 57.1 mm); 1.0739 L (D 57.0 mm) | | agrees with D = 57.1 mm (Fig. 1a) only | with the text's 57 mm the cones do not close the gap |
| HAYSTAC one-rod, θ = 0 (2006.01248) | f TM010-like | 5.835 GHz (Fig. 3 lower line) ± 0.015 | no axles 5.86096 GHz; smeared axles 5.8538 GHz | −26.0 / −18.8 MHz | **not decidable from Fig. 3** | f(θ) is even in θ but the Fig. 3 lines are straight down to θ ≈ 0.5°: f(0) is an extrapolation with > 15 MHz uncertainty (see HAYSTAC below) |
| HAYSTAC | C_vac vs Fig. 6c (θ ≈ 0.9° point, 5.827 GHz) | 0.751 (C_full 0.564) | no axles 0.7726 (C_full 0.5798); smeared axles 0.698 | −2.8 % / +7.7 % | not decidable (the two 2D models bracket it; 3D with axles) | the angle effect is second order (dC/dθ = 0 at θ = 0) |
| HAYSTAC | no-gap limit (coax, exact) | 5.866558 GHz, C_vac 0.806157 | 5.866560 GHz, 0.806158 | 3e-7, 1e-6 | agrees (predicted) | |
| HAYSTAC | empty barrel TM010 f, C, Q (5.8e7) | 2.25871 GHz, 0.69166, Q 30 444 (closed form); paper Q 3.07e4 | 2.258711 GHz, 0.6916603, Q 30 444 | Q vs paper +0.8 % | agrees (predicted) | |
| HAYSTAC θ = 0 | Q0 (sigma 5.8e7) | 4637 (Fig. 6b; not a target) | 9198 | — | n/a | the gap faces cut Q from 13 854 (no gap) to 9198 |
| QUAX clamshell 2410.07774, 0° | C030 prototype (0.73 mm top gap) | 0.235 | largest-C mode 0.2524 (range 0.15–0.28 under ±0.2 mm / ±0.02 eps); cluster C sum 0.2854 (0.2846–0.2890) | — | **not decidable (avoided crossing)** | the paper value lies inside the single-mode range; see clamshell below |
| QUAX 2410 | Q ratio prototype / pill-box | 4.32 | 3.78 nominal; 3.15–4.09 over the sensitivity scan; 4.09 least mixed | +5 % (least mixed) | **not decidable (avoided crossing)** | mixing lowers Q; paper/axionFEM absolute Q: 2.29 (prototype), 2.01 (pill-box) |
| QUAX 2410 | C TM010 pill-box | 0.667 | 0.69166 (exact) | −3.6 % | disagrees (cause: HFSS systematic; closed form exact) | |
| QUAX 2410 | C030 prototype without gap | ~0.34 (refs, independent layered model) | 0.3366 | | agrees (predicted) | the clean 2D benchmark; layered-exact |
| QUAX 2410 | ideal design C max | 0.476 | 0.476 at r1 = 11.27 mm (t = 3.01 mm) | 0 | agrees (after fitting r1) — not a test | t = 3.97 mm (text): max C 0.4695; shell inner radius not given |
| QUAX 2410 | ideal design f, Q ratio ideal / pill-box (at the fitted r1) | "11 GHz"; 2.39 | 11.021 GHz; 2.36 | +1.7 % (Q ratio) | **agrees (predicted)** | f excludes the second C = 0.476 crossing beyond r1 = 12 mm (f > 11.18 GHz); the paper's node rule (0.278 r_c = 7.8 mm) is not what was built |
| QUAX 2410 | f prototype vs Fig. 7a at 0° | 11.021 GHz | 11.0444 GHz | −0.21 % | not decidable (3D) | Fig. 7b Q at 0° (1.95e5) is ~45 % below its neighbours, the signature of a crossing in the HFSS model too |

## 3D benchmarks and further checks

Scripts `benchmarks/cavity/repro/{rades_5cav_1803,rades_6cav_alt_2002,capp_multicell_3d,birme_probe_port,
quax_2201_offset,stern_B_variants,blind_check}.py`; records `benchmarks/records/cavity_repro_<name>.json`;
figures next to the scripts. 3D solves: `fullwave` rf_only eigenmodes (STRUMPACK shift-invert) + `cavity` (3D) on
shared 3D helpers in `_common.py` (`design3d`, `solve_many3d`, `rounded_box_xz`). Same conventions and verdict
categories as the 2D study. A rerun with later solver and mesher defaults changed the records by ≤ 2e-9 relative;
two rejected RADES 2020 iris readings (U_floor, U_floor_half) were not rerun and are flagged in the record.

| benchmark | quantity | published | axionFEM (converged) | diff | verdict |
|---|---|---|---|---|---|
| RADES 2018 5-cavity (3D, + exact 2D) | f, 5 modes | 8.428 … 8.710 GHz | 2D exact 8.419693 … 8.705556 GHz (3D 0.14-0.34 MHz higher) | +4.4 … +8.3 MHz (0.05-0.10 %) | disagrees (small; cause unexplained; corners +0.3 MHz max) |
| RADES 2018 | eigenvector sign patterns | Table 2 | identical for all 5 | — | agrees (predicted) |
| RADES 2018 | C mode 1 | 0.65 | 0.69034 (2D = 3D to 3e-5; 0.7054 pocket volume) | −6 % | disagrees (cause unexplained) |
| RADES 2018 | C modes 3, 5 | 8.1e-5, 6.4e-6 | 2D 1.07e-4, 7.5e-6 | −24 %, −15 % | disagrees (cause unexplained; cancellation residuals) |
| RADES 2018 | Q, 5 modes (sigma 2e9) | 40 386 … 48 048 | 2D, edge-extrapolated: 38 298 … 46 883 (excluded h^(2/3) fit 0.5-1.1 % higher; 3D 0.8-1.6 % lower, unresolved) | +5.5 … +2.5 % | disagrees (cause unexplained; iris-edge loss) |
| RADES 2020 6-cavity (3D) | k2(w), inductive iris | Fig. 5 left | 1.0 % rms | — | agrees (predicted), ~1 % |
| RADES 2020 | k1(h), capacitive irises | Figs 5 right, 6 right | 0.4 % / 1.2 % rms for the centred-slot readings | — | agrees (after choosing the reading among 6) |
| RADES 2020 | f, 6 modes | 8.298 … 8.698 GHz | 8.29713 … 8.69995 GHz (order 3; + h_slot/6 shift up to +0.9 MHz) | −2.8 … +0.9 MHz | modes 1-3: agrees (after choosing the iris reading); modes 4-6: disagrees (small, 1.7-2.8 MHz) |
| RADES 2020 | C mode 4 | 0.534 | 0.5448 | −2.0 % | disagrees (2 %; cause unexplained) |
| RADES 2020 | C modes 2, 6 | 1.257e-3, 3.42e-5 | 1.13-1.36e-3, 3.2-4.1e-5 (mesh range) | — | not decidable (not converged) |
| RADES 2020 | Q0 2 K / 300 K | 40 657 … 44 454 / 6 957 … 7 597 | flux 37 930 … 42 849 (pointwise 40 096 … 44 101) / 6 420 … 7 273 | +2.3 … +7.2 % (pointwise +0.3 … +1.4 %) / +2.4 … +8.4 % | disagrees (est. +1 … +6 %; not resolved) |
| CAPP 1710 Tab. 2 (3D sector cells) | Q, N = 2, 4, 6, 8 | 17 794 / 16 151 / 14 320 / 12 907 | sigma 5.998e7: 17 794.0 / 16 151.1 / 14 320.1 / 12 906.9 (5.96e7: 0.32 % lower) | ≤ 9e-6 | absolute: consistent (circular; sigma inferred from this column); N-ratios: agree (predicted) |
| CAPP 2205 Tab. 2 | C, N = 2, 4, 6, 8 | 0.64 / 0.65 / 0.63 / 0.61 | 0.6440 / 0.6493 / 0.6341 / 0.6132 | within print | agrees (predicted) |
| BI-RME Tab. I (2D, pin) | C TM010 … TM015 | 0.68505, 5.2e-3, 2.7e-4, 5.3e-5, 1.7e-5, 7.2e-6 | at d_fit = 5.63 mm: 0.68591, 5.22e-3, 2.71e-4, 5.30e-5, 1.71e-5, 7.2e-6 (at d_c = 5.45 mm: 0.68680, 4.4e-3, …) | TM010 −0.13 %; p ≥ 1 −0.4 … +0.2 % (at d_c: −0.26 %, +14 … +19 %) | agrees (after fitting the pin depth to the p ≥ 1 C); residual −0.13 % unexplained |
| BI-RME Tab. I | Q TM010 | 33 069 | 33 231 (d_fit) | −0.5 % | disagrees (cause unexplained; half of the empty-cylinder deficit explained by the pin) |
| QUAX 2201 | absolute f | 10.4745 GHz | 10.3685 GHz | +1.02 % | disagrees (not determinable; no single stated input fits; a common ~1 % geometry scale is not excluded) |
| Stern B / E | C convention / geometry | Fig. 6 | B: not a geometry typo of the kinds tested; most likely vacuum-volume normalization (4 of 5 points); E: the Table I geometry (posts touch) misses the Λ range | — | B: not decidable; E: disagrees (cause: probably a Table I typo; not tested here) |
| blind check | 35 numbers, 27 independent (KLASH, QUAX 2004 flat, 2410 no gap, coax) | independent exact solutions | ≤ 2.6e-8 | — | agrees (predicted) |

### RADES 5-subcavity WR-90 filter (1803.01243; `rades_5cav_1803`)

- **Reference:** Table 1 (p. 8, 2 K column: a 22.86, b 10.16, L1 = L5 26.68, L2-4 25.00, W 8.00, t 2.00 mm) and
  Table 2 (p. 9: CST eigenmodes, f to 1 MHz, G^2, Q_u at sigma = 2e9 S/m). 2107.03137 Table II is the same table
  (refs). Not given: the corner radius ("rounded to facilitate machining", p. 7) and the iris centring (assumed).
- **Two models.**
  - **3D** (the target of this study). The closed domain is the union of the five pockets and four centred full-height
    apertures. 1/8 model with mirror planes x = 0 (PMC), y = 0 (PEC) and z = 0 (PMC for modes 1, 3, 5; PEC for
    modes 2, 4). Walls: classical skin, sigma = 2e9. B0 along y. Iris surroundings refined (h/3 to h/6). Seven
    meshes, 3.4e4 to 2.8e6 DOFs per parity, orders 2 and 3. The full model at h = 2 mm agrees with the 1/8 models to
    5e-6 (f), 1e-6 (C, mode 1) and 0.2 % (Q). Probes at the pocket centres reproduce Table 2's five sign patterns.
  - **Exact 2D reference.** The filter is uniform along
    y (full-height irises), so its TE10-family modes E_y(x, z) are the TM_z modes of the cross-section solver
    (`"symmetry": "xsec"`) on the (x, z) plane, with end caps at y = +-b/2 (`cavity.length = b`, which includes the
    top and bottom walls in Q). The 16 re-entrant iris edges are refined from 0.2 to 0.025 mm (order 2), plus
    order 3. f and C come from order 3 at 0.05 mm (order 2 at 0.025 mm agrees to 2e-7 in f).
  - **Wall loss at the re-entrant edges.** H ~ r^(-1/3) there, and the consistent-flux and pointwise G converge
    slowly from opposite sides. A joint fit of both to one limit with the edge exponent h^(1/3) fits the four meshes
    to 0.3-0.4 ohm. The h^(2/3) alternative fits 6-9x worse and is excluded; the record keeps its limit as the upper
    end of `Q0_bracket`, 0.5-1.1 % above the h^(1/3) value (mode 1 largest).
  - **The 3D values are low.** The "converged" 3D runs still carry the solver's resolution warning (flux/pointwise
    disagreement 0.03-0.07 on the wall). Their Q is 0.8-1.6 % low and their f 0.14-0.34 MHz high relative to the
    2D reference. The 2D model is therefore quoted as the axionFEM value. The 3D C agrees with 2D to 3e-5 for mode 1.
- **Corner radius 0.5 / 1 / 1.5 mm** (3D; the y-parallel edges of every pocket): f +0.06 / +0.11 / +0.27 MHz at
  most, Q within 0.15 %, C(mode 1) +0.3 % at most (record `corner_radius_sweep_vs_sharp_o2_h1.4`). Corner rounding
  is therefore far too small to explain the f and Q differences below.

| mode | pattern | f CST [GHz] | f axionFEM (2D exact) | CST - axionFEM | C CST | C axionFEM | Q CST | Q axionFEM (2D, edge-extrapolated) | CST/axionFEM - 1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | + + + + + | 8.428 | 8.419693 | +8.3 MHz | 0.65 | 0.69034 | 40 386 | 38 298 | +5.5 % |
| 2 | + + 0 - - | 8.454 | 8.447737 | +6.3 MHz | 3.2e-7 | 0 (symmetry) | 42 033 | 40 438 | +3.9 % |
| 3 | - + + + - | 8.528 | 8.522211 | +5.8 MHz | 8.1e-5 | 1.07e-4 | 43 654 | 42 116 | +3.7 % |
| 4 | - + 0 - + | 8.625 | 8.620316 | +4.7 MHz | 1.6e-12 | 0 (symmetry) | 45 882 | 44 502 | +3.1 % |
| 5 | - + - + - | 8.710 | 8.705556 | +4.4 MHz | 6.4e-6 | 7.5e-6 | 48 048 | 46 883 | +2.5 % |

(3D, order 3, 2.6e6 DOFs per parity: f 8.42004 / 8.44799 / 8.52243 / 8.62050 / 8.70569 GHz, Q 37 698 / 39 943 /
41 626 / 44 045 / 46 505; record `comparison`.)

- **f: disagrees (cause: unexplained, small).** CST is 4.4-8.3 MHz (0.05-0.10 %) above the exact model. The printed
  rounding is 0.5 MHz and the corner effect <= 0.3 MHz. The mode spacing agrees to 1-7 %: axionFEM 285.9 MHz vs
  CST 282 MHz between modes 1 and 5, so our iris coupling is about 1 % stronger. Neither a common offset nor the
  spacing follows from any stated dimension. A 1e-3 discretization error of CST's FIT is plausible but unproven.
  Measured 2 K peaks (Table 3, sanity only) are 18-22 MHz below axionFEM.
- **C(mode 1): disagrees (cause: unexplained).** The exact 2D value 0.69034 (3D 0.6903; 0.7054 with V = the pocket
  volumes, Eq. A.21) against the printed 0.65.
  - The box TM110 / pillbox C are validated to 1e-5 (cavity_3d.md), and the 2D and 3D models agree.
  - Physically, a filter mode is not a TE101 mode: its guide wavelength is longer than 2L (it sits 0.2-0.5 GHz below
    the isolated TE101), so its z-profile is flatter than a half sine and C > (8/pi^2)^2 = 0.657 is expected.
  - The paper's two-digit 0.65, "very close to 0.657", is therefore low by ~6 % for the stated geometry. Its
    normalization volume is not stated beyond Eq. A.21.
- **Small C (modes 3, 5): disagrees (cause unexplained; tiny cancellation residuals).** CST is 24 % / 15 % below
  the exact values: 1.07e-4 and 7.5e-6.
  - The exact values are close to converged. Under iris-edge refinement 0.2 / 0.1 / 0.05 / 0.025 mm (order 2,
    record `xsec_reference.runs`) C3 is 8.04 / 9.64 / 10.36 / 10.65e-5 and C5 6.44 / 7.12 / 7.39 / 7.51e-6; the
    differences shrink by a factor of about 0.4 per halving, so the limits lie 1-2 % above the finest values, far
    less than the differences from CST.
  - The 3D values drift strongly with the mesh (mode 3: 7.6e-6 to 9.3e-5) and are not used.
  - Modes 2 and 4 are exactly 0 by the z-mirror symmetry (CST 3e-7, 2e-12: noise).
- **Q: disagrees (cause unexplained).** CST is +2.5 … +5.5 % above the converged sharp-edge Q (+2.0 … +4.3 % above the excluded h^(2/3) limit). The excess grows with the iris current and is largest for mode 1. The skin model is
  CST's (classical, the same sigma).
  - The excess follows the pattern of our own flux/pointwise disagreement.
  - Our under-resolved 3D pointwise-current Q happens to match CST to 0.08-0.38 % (record `Q_rel_vs_pointwise`).
  - CST's iris coupling is also about 1 % weaker than ours in both filters: the mode 1-5 spread here and the RADES 2020
    mode 1-6 spread. That is the direction a coarse edge representation would give.
  - Together these suggest, but do not prove, that CST's edge discretization is the cause.

### RADES 6-subcavity alternating filter (2002.07639; `rades_6cav_alt_2002`)

- **Reference:** Table 1 (p. 17, 2 K design) and Table 2 (p. 18). Table 2 gives CST f, G^2 and Q0, at 2 K with
  sigma = 2.008e9 and at 300 K with the thermally expanded dimensions and 5.8e7. Figs 4-6 (pp. 14-15) give the iris
  layouts and the CST coupling curves: k2(w), k1(h) for the plain iris, and k1(h) with g = 1.1 mm "protrusions".
  These curves are digitized in refs.
- **Iris geometry chosen from the paper's own coupling curves, never from Table 2.** The capacitive slot position
  and the protrusion layout are not dimensioned (refs gap).
  - The drawings: Figs 4 bottom and 6 left show metal below the slot. Fig. 6 left shows a U-shaped opening whose side
    channels rise to the ceiling. The Fig. 7 inset shows metal above and below the opening.
  - The paper's method was used: two cavities joined by one iris, with a PMC/PEC mirror plane at the iris centre, and
    k = (f_e^2 - f_o^2)/(f_e^2 + f_o^2) (Eq. 3.2). The paper does not give the resonator length; l_int = 27 mm is
    our assumption.
  - Six discrete readings were tested (3D, slot element size h_slot/3; rms over the digitized range):

  | iris reading | compared with | rms deviation from the CST curve |
  |---|---|---|
  | inductive, centred aperture w | Fig. 5 left (w = 5-13 mm) | 1.0 % |
  | plain slot at the floor | Fig. 5 right | 18 % |
  | plain slot centred in y | Fig. 5 right | **0.4 %** (+0.8 % at h = 1 mm with the slot refined to h_slot/6) |
  | floor slot + side slots g, full height | Fig. 6 right | 50 % (not rerun with the current defaults) |
  | floor slot + side slots g/2 | Fig. 6 right | 56 % (not rerun with the current defaults) |
  | centred slot + side slots g, full height | Fig. 6 right | 12 % |
  | **centred slot + side slots g from the slot to the ceiling** | Fig. 6 right | **1.2 %** (+1.6 % at h = 1 mm, h_slot/6) |

  - The as-built "quasi-capacitive" iris is therefore a centred slot of height h whose two side slots (width
    g = 1.1 mm) run from the slot up to the ceiling. This is a discrete identification with a decisive margin
    (1.2 % against >= 12 %), consistent with the drawings: **agrees (after choosing the reading)**. The two
    floor-slot U readings were not rerun with the later defaults and are flagged in the record. Every reading that was
    rerun reproduced to 3e-13.
  - Inductive curve: **agrees (predicted)** at the ~1 % level (1.0 % rms over w = 5-13 mm, record `couplings`). The
    paper's text value |k| = 0.0248 at w = 8.90 is itself 4.2 % above its own curve (refs inconsistency).
- **Filter model.** 1/4 model: x = 0 PMC, z = 0 PMC/PEC, and no y mirror because the irises are not y-symmetric.
  - Meshes: order 2 at h = 2, 1.5 and 1 mm (slots at h_slot/3); order 2 at 1.5 mm with the slots at h_slot/6
    (5.0e6 DOFs per parity); order 3 at 1.5 mm (2.6e6 DOFs per parity). h_slot/9 did not fit one node.
  - Limitation: the inductive-iris edges stay at hmesh/3 in every mesh. On the RADES 2018 irises the
    corresponding 3D resolution (o3_h1.4) gives Q 0.9-1.9 % below the exact 2D value. The 3D Q below are therefore probably ~1 % low,
    the excess quoted against CST correspondingly high.

| mode | f CST [GHz] | f axionFEM (order 3) | CST - axionFEM | with the h_slot/6 shift | G^2 CST | C axionFEM | Q0 2 K CST / axionFEM [flux, pointwise] | Q0 300 K CST / axionFEM |
|---|---|---|---|---|---|---|---|---|
| 1 | 8.298 | 8.29713 | +0.9 MHz | +0.8 MHz | 1.8e-9 | 0 (sym.) | 42 878 / [41 036, 42 576] (+4.5, +0.7 %) | 7 300 / 6 962 (+4.9 %) |
| 2 | 8.319 | 8.31936 | -0.4 MHz | -0.4 MHz | 1.257e-3 | 1.13e-3 … 1.36e-3 | 43 758 / [42 784, 43 636] (+2.3, +0.3 %) | 7 448 / 7 273 (+2.4 %) |
| 3 | 8.395 | 8.39558 | -0.6 MHz | -0.8 MHz | 1.0e-8 | 0 (sym.) | 43 626 / [42 290, 43 409] (+3.2, +0.5 %) | 7 426 / 7 224 (+2.8 %) |
| 4 | 8.508 | 8.50917 | -1.2 MHz | -2.1 MHz | **0.534** | **0.5448** | 40 657 / [37 930, 40 096] (+7.2, +1.4 %) | 6 957 / 6 420 (+8.4 %) |
| 5 | 8.613 | 8.61400 | -1.0 MHz | -1.7 MHz | 1.5e-8 | 0 (sym.) | 44 168 / [42 643, 43 864] (+3.6, +0.7 %) | 7 563 / 7 210 (+4.9 %) |
| 6 | 8.698 | 8.69995 | -1.9 MHz | -2.8 MHz | 3.42e-5 | 3.2e-5 … 4.1e-5 | 44 454 / [42 849, 44 101] (+3.7, +0.8 %) | 7 597 / 7 250 (+4.8 %) |

(The h_slot/6 shift is the order-2 change from h_slot/3 to h_slot/6 at h = 1.5 mm, added to order 3. 300 K is order 3
at h = 1.5 mm, as 2 K. The C ranges are over all meshes. The record has every mesh.)

- **f, modes 1-3: agrees (after choosing the iris reading)**, within 0.9 MHz. **Modes 4-6: disagrees (small,
  1.7-2.8 MHz; CST below axionFEM by 2-3e-4; cause unexplained).** Our remaining mesh uncertainty is ~1 MHz
  (h_slot/9 not run, inductive edges unrefined) and the printing is to 1 MHz (+-0.5 MHz). The sign matches the ~1 %
  weaker CST iris coupling noted below. f was not looked at when the reading was chosen.
- **C mode 4: disagrees slightly (2.0 %; cause unexplained).** 0.5448, stable to 1e-3 over all meshes (0.555 with the
  pocket volume). For scale, the paper's Fig. 11 Monte Carlo gives a nominal 0.5340 but a mean of 0.5383 for +-30 um
  perturbations, about 1 % of CST-side variation. Modes 2 and 6 are **not decidable**: they vary over the meshes by
  20 % and 27 %. Modes 1, 3 and 5 are z-odd and exactly 0 (CST 1e-9 to 1e-8: noise).
- **Q: disagrees (cause unexplained; not resolved at our mesh).** CST is +2.3 … +7.2 % above our consistent-flux Q
  and +0.3 … +1.4 % above our pointwise-current Q (2 K); at 300 K it is +2.4 … +8.4 % above the flux Q.
  - The two estimators bracket the edge-converged value. With the RADES 2018 calibration (the flux Q ~1 % low), the
    CST excess is roughly +1 … +6 %. It is largest for mode 4, whose field reverses in the three capacitive slots.
  - The pattern matches RADES 2018: CST close to our under-resolved pointwise Q, and a CST iris coupling ~1 % weaker
    than ours (the mode 1-6 spread is 400 MHz in CST and 402.8 MHz here).

### BI-RME probe in the BI-RME convention (`birme_probe_port`)

A matched (lossy/PML) coax termination seems the natural model. Reading the method (2107.03137 Sec. 2, Eqs. 1-2,
pp. 3-5) changes the question. BI-RME expands the field in the resonant modes of the cavity **with the port surfaces
covered by PEC**; the port enters only through magnetic currents on that surface. Its Table I f_m, Q_m (wall loss
over the cavity surface, Sec. 2.2) and C_m are therefore properties of the cavity with the coax **shorted at the port
plane**, not of the cavity loaded by a matched line. A matched termination would only add the loading (Q_L) and is
not needed for Table I. axionFEM has no eigenmode solver with absorbing boundaries (PML/scattering are for driven
sweeps on imported meshes only), so a driven matched-port check of the coupling formula below was not run (it would
need an axisymmetric port or PML on a native mesh; [cavity/coupling.md](../cavity/coupling.md) has such a port for a
pillbox).

- **Model:** R = 44.835 mm (the 2D-study fit), L = 1 m, sigma = 6e7 (skin). A copper pin of radius 0.635 mm (part of the
  lossy wall) protrudes d into the cavity from the centre of the top cap. The coax annulus (0.635-2.11 mm, PTFE eps
  2.08, Z_c = 49.9 ohm) is closed at the port plane by the lossy copper wall (its loss is ~5e-5 of P_wall at critical
  coupling; effectively a short): either at the wall plane (s = 0) or 1 mm up the line (s = 1 mm). The pin side and
  the region under the pin tip are refined. Grid d = 0-20 mm with two pin-zone meshes (0.3 and
  0.15 mm): near d_c (d <= 8 mm) f changes by <= 1e-5 and C of the mixed modes by <= 2 %. At d = 20 mm the change is
  3e-3 in f and 5 % in C and Q; those depths are not used.
- **Critical-coupling depth** (the paper's only statement: "designed under critical coupling operation regime for the
  TM010 mode", p. 17). In the shorted-port basis the port sees a series resonance. Terminating the line in Z_c
  extracts P_ext = |I|^2 Z_c / 2, with I the pin current at the port plane, so Q_ext = 2 omega W / (|I|^2 Z_c).
  - The correction is O(1/Q_ext). The pin-current normalization was checked against the analytic TM010 field
    (I at d = 0).
  - The grid brackets Q_ext(d) = Q0(d), and secant steps on direct solves give **d_c = 5.453 mm (s = 1 mm) /
    5.501 mm (s = 0)**, with beta = Q0/Q_ext = 1 to 3e-4. The port-plane choice (0.05 mm) dominates the uncertainty
    of d_c; current extraction and mesh contribute ~0.01 mm.
  - C varies like d^5-6, so a linear interpolation in d would be wrong; all values below are direct solves.
- **Results** (s = 1 mm; s = 0 agrees to 2e-4 in C010 and 3 % in the p >= 1 C). Direct solves at d_c and at the depth
  d_fit, where the Table I C of the five p >= 1 modes are matched. The record has the per-p crossing depths from a
  log-quadratic interpolation through direct solves at d_c and d_c +- 0.25 mm. They are 5.628 / 5.622 / 5.629 /
  5.627 / 5.627 mm, so d_fit = 5.627 mm with a spread of 0.007 mm, and beta = 1.13.

  | mode | C Table I | C axionFEM at d_c | C axionFEM at d_fit | Q Table I | Q axionFEM at d_fit | Q empty cylinder |
  |---|---|---|---|---|---|---|
  | TM010 | 0.68505 | 0.68680 | 0.68591 | 33 069 | 33 231 | 33 410 |
  | TM011 | 0.00523 | 0.00440 | 0.00522 | 33 261 | 31 936 | 32 063 |
  | TM012 | 2.7e-4 | 2.3e-4 | 2.71e-4 | 33 120 | 32 111 | 32 145 |
  | TM013 | 5.31e-5 | 4.5e-5 | 5.30e-5 | 33 292 | 32 262 | 32 280 |
  | TM014 | 1.71e-5 | 1.5e-5 | 1.71e-5 | 33 494 | 32 453 | 32 466 |
  | TM015 | 7.25e-6 | 6.2e-6 | 7.2e-6 | 33 782 | 32 691 | 32 701 |

- **Mechanism.** The TM01p modes of the 1 m cavity are 4-40 MHz apart. The pin at the top cap breaks the z
  symmetry and mixes them, so TM010 hands part of its C to TM011, TM012, ....
  - The sum over the six modes stays 0.6915, independent of d to 6e-5 (Table I: 0.6906; ideal cylinder 0.69166).
  - This is why Table I has non-zero C for p >= 1, which an ideal cylinder cannot have.
- **Verdicts** (signs: (published - axionFEM)/axionFEM).
  - C: **agrees (after fitting the pin depth to the five p >= 1 C)**.
    - One depth, d_fit = 5.627 mm, reproduces all five to -0.4 … +0.2 %. That depth is 0.17 mm deeper than the
      first-order critical-coupling depth (beta = 1.13).
    - The residual, TM010 -0.13 %, equals Table I's own sum-rule deficit (0.69063 vs 0.69150). Its cause is
      unexplained.
    - Without the fit, at d_c: TM010 -0.26 %, p >= 1 +14 … +19 %, i.e. **disagrees (cause: probe and port geometry
      not given)**.
  - TM010 Q: the pin brings the deficit from -1.0 % to **-0.5 %**. **Disagrees (cause unexplained).**
  - TM01p (p >= 1) Q: Table I is 3.1-4.1 % above our model (TM011 +4.1 %, p = 2-5 +3.1-3.3 %). The pin does not
    explain it; left out as in the 2D study.
    - A hypothesis, not a demonstrated cause: Table I's Q_p/Q_0 for p = 2-5 is 1.041-1.044 of
      the closed-form ratio. (L + 2R)/(L + R) = 1.043 is the factor by which a TM01p Q comes out too high if its end
      caps are weighted like TM010's. With that factor the whole Table I Q column would sit a uniform ~1 % below the
      closed form.
  - Not refitted: R was fitted on the empty cylinder. The pin shifts TM010 by -0.29 MHz (-1.1e-4), more than the
    fit residual (7.7e-5); this has no effect on C or Q at this level.
- The 2D study's "C and Q 1 % low, unexplained" becomes:
  - the C pattern is reproduced by probe-induced mode mixing at one fitted pin depth, with a 0.13 % residual;
  - Q is half explained.

### QUAX 2201 absolute f (`quax_2201_offset`)

The 2D study swept the obvious inputs (cones, end gap, contraction set, D_cav, stated tube tolerances) and found none that
supplies the 106 MHz (+1.02 %). Here every input is solved for, one at a time, so that f lands on 10.4745 GHz, and the
resulting change is tested against the other published targets (exact layered Bessel model with the 2D study's cone
offset +4.5 MHz and cone C factor 0.861; the layered flat-end f equals axionFEM's to 1e-6, C to 4e-6, p_e to 2e-5):

| changed input | nominal -> required | C (Table II 0.0321) | p_e (0.170) | Delta f(eps) (18.9 MHz) |
|---|---|---|---|---|
| R_cav | 28.55 -> 24.13 mm (-4.4) | 0.0557 (+74 %) | 0.220 (+29 %) | 24.12 (+28 %) |
| tube-1 inner radius | 10.875 -> 10.716 mm (-0.16) | 0.0298 (-7.1 %) | 0.168 | 18.55 |
| tube-1 thickness | 2.35 -> 2.19 mm | 0.0304 (-5.2 %) | 0.164 (-3.5 %) | 18.11 (-4.2 %) |
| tube-2 inner radius | 20.90 -> 19.21 mm | 0.0462 (+44 %) | 0.184 | 20.32 |
| tube-2 thickness | 1.94 -> 0.85 mm | 0.0565 (+76 %) | 0.193 | 21.32 |
| eps both tubes | 11.2 -> 9.96 | 0.0311 (-3.1 %) | 0.178 (+4.5 %) | 19.60 (+3.7 %) |
| eps tube 1 only | 11.2 -> 9.82 | 0.0282 (-12 %) | 0.175 | 19.30 |
| eps tube 2 only | none down to 5 | | | |
| all dimensions x s | s = 0.98988 | 0.0325 (+1.3 %) | 0.1698 (-0.1 %) | 18.77 (-0.7 %) |
| tube radii x s (cavity fixed) | s = 0.98968 | 0.0322 (+0.3 %) | 0.1690 (-0.6 %) | 18.68 (-1.1 %) |

Conclusion:
- No single stated input reproduces f within its stated tolerance without breaking C or p_e.
  - The tube-1 radius would need 2.1x its stated tolerance (-0.16 vs +-0.075 mm), and C then drops 7 %.
  - The tube-1 thickness would need 6.3x its tolerance, with C -5 % and p_e -3.5 %.
  - The tube-2 changes (no tolerance stated) spoil C by 44-76 %.
  - eps would have to be 9.96, not the measured 11.2.
- A common ~1 % scale of the tube geometry (or of the whole cavity) is **not excluded** by any target. With the tube
  radii x 0.9897 the model gives f, C (+0.3 %), p_e (-0.6 %) and Delta f(eps) (-1.1 %) together.
  - This is not positive evidence: a uniform scale leaves C and p_e unchanged by construction.
  - Only single inputs and two scales were tested, not combinations within tolerance.
  - A uniform 1 % scale is far above any thermal contraction (sapphire 0.08 %, copper 0.33 %).
- The same group's 2410.07774 prototype uses the same inner tube and agrees with axionFEM to -0.21 % in f (2D study),
  so a wrong tube-1 drawing alone is unlikely.

Verdict unchanged: **disagrees (cause: not determinable from the paper; no single stated input fits; a common ~1 %
geometry scale is not excluded)**. Open question for the authors: which dimensions the HFSS model used.

### Stern design B and E (`stern_B_variants`)

- **E**: with the Table I post diameter 0.375 R the moving and stationary posts touch radially, and the Lambda range
  is 0.263-0.338 GHz m against the paper's 0.240-0.294 ([cavity_xsec.md](cavity_xsec.md), record
  `cavity_xsec_stern.json`): **disagrees (cause: probably a typo in Table I).** A smaller post diameter is the
  likely intended value; no candidate geometry for E is tested with a record in this repository.
- **B**: 9 discrete candidates were compared (record `cavity_repro_stern_B_variants.json`): the Table I row; post diameter 0.25 / 0.275 /
  0.325 / 0.35 R; post-location radius (R_MOV = R_STAT) 0.413 / 0.454 R; and the A and C rows (i.e. a legend swap).
  - The Table I geometry reproduces the Lambda range (0.2174-0.2754 vs 0.2145-0.2756 GHz m) and Q0 (1.0 % rms).
    On the four points both cover, the Table I geometry is within 0.20 % of C_vac, r = 0.413 R 1.4-4.6 % off
    (its 2.6 % rms excludes the Lambda = 0.219 point; it also has a narrower Lambda range and Q0 2.0 % rms).
  - No candidate brings C_full closer than 7 % rms (Table I: 8.0 %).
  - With the Table I geometry, 4 of the 5 comparable points match C_vac to <= 0.2 %. The first point
    (Lambda = 0.219) matches C_full instead (0.4842 vs 0.4835) and is 10 % off C_vac. The paper's Lambda range also
    starts slightly below ours.
  - Conclusion: **B is not a geometry typo of the kinds tested. The most likely reading is that most of the
    published B curve was normalized with the vacuum (post-excluded) volume, unlike A, C, D, F, G.** This is a
    reading, not a proof: **not decidable**. Compare B in the C_vac convention and flag its first point.

### Blind check (`blind_check`)

An independent derivation, given only the specifications of four problems and no numbers or code from axionFEM
(KLASH; QUAX 2004 with flat ends and the Table 1 one-parameter changes; the QUAX 2410 tube without gaps; the
HAYSTAC-like coax without gaps), solved them exactly (layered Bessel matching, closed forms). Its values are in
`benchmarks/cavity/repro/blind_check.py`. axionFEM (2D, order 3) agrees on all 35 compared numbers
(27 independent; 8 are derived from others) to <= 2.6e-8.

### CAPP multi-cell, ideal sector cells in 3D (`capp_multicell_3d`)

- **Reference:** 1710.06969 Table 2 (p. 3): Q of the N = 2, 4, 6, 8 cell cavity with zero-thickness partitions,
  Rc = 25 mm, sigma = 5.96e7 S/m stated, "no end caps": analytical 17 794 / 16 151 / 14 320 / 12 907, COMSOL
  17 819 / 16 151 / 14 310 / 12 851 and a second COMSOL column 17 836 / 16 335 / 14 325 / 12 930; Table 3 ratios;
  2205.01319 Table 2 (p. 7): C = 0.64 / 0.65 / 0.63 / 0.61.
- **Model:** zero-thickness partitions that meet on the axis decouple the cells, so the in-phase mode of the N-cell
  cavity has the f, Q and C of one cell. One cell = a `revolve` sector (0 < phi < 360/N) whose two flat sides are the
  partitions (lossy, same copper as the barrel). The z-independent TM_(N/2,1,0) mode is exact for any length; the end
  plates (L = Rc) are separate conductor bodies, so Q without end caps = G_outer/R_s. Mesh study h = Rc/6, /9, /12
  (order 2) and order 3 at Rc/9; order 3 is within 2e-6 (f), 1e-6 (C), 1.2e-5 (Q) of the independent closed form
  (`closed_form()` in the script, quadrature), with flux/pointwise G disagreement <= 2.2e-3.
- **Result:**

  | N | f [GHz] | C (2205 Tab. 2) | C axionFEM | Q, sigma 5.96e7 | Q, sigma 5.998e7 | Tab. 2 analytical | COMSOL |
  |---|---|---|---|---|---|---|---|
  | 2 | 7.31296 | 0.64 | 0.64402 | 17 737.5 | 17 794.0 | 17 794 | 17 819 / 17 836 |
  | 4 | 9.80153 | 0.65 | 0.64928 | 16 099.9 | 16 151.1 | 16 151 | 16 151 / 16 335 |
  | 6 | 12.17679 | 0.63 | 0.63411 | 14 274.7 | 14 320.1 | 14 320 | 14 310 / 14 325 |
  | 8 | 14.48266 | 0.61 | 0.61317 | 12 866.0 | 12 906.9 | 12 907 | 12 851 / 12 930 |

- **Verdicts.**
  - C: **agrees (predicted)** within the printed two digits.
  - Table 3 omega/omega0 (1.59, 2.14, 2.65, 3.16) and Q/Q0 (0.68, 0.62, 0.55, 0.50): **agree (predicted)**.
  - The three N-ratios of the Table 2 Q are predicted too (they do not depend on sigma) and agree to 1e-5.
  - Absolute Q: with the stated sigma = 5.96e7 every printed "analytical" value is 0.32 % above axionFEM. With
    sigma = 5.998e7 (COMSOL's built-in copper) all four agree to <= 9e-6. The refs inferred 5.998e7 from this same
    column, so this match is **consistent (circular)**, as for KLASH in the 2D study. The 3D FEM adds only that axionFEM's
    3D Q equals the closed form to 1e-5.
  - The paper's own COMSOL columns scatter by -0.4 to +1.1 % around these values (their 3D meshes).
- Order-3 3D runs carry the solver's "3D order >= 3 is not validated" warning. These CAPP cells are a partial
  validation of order 3: f, C and Q are within 1e-5 of the closed form.

## Per benchmark (2D)

### Analytic empty cylinders (ADMX, KLASH, BI-RME)

- **ADMX** (`admx_tab6_1`): order 2 at h = R/4, R/8, R/16 converges as h^4 (TM010 4e-6 → 2e-8). Order 3 at R/8 is
  within 1e-7 of the closed form for all three modes in all four cavities. Every table entry agrees with the closed
  form to its printed precision. The tolerance is half a unit of the printed f plus the half units of R and H,
  propagated as df/f = (k_r/k)^2 dR/R + (k_z/k)^2 dH/H (H enters TM011 only).
- **KLASH** (`klash_tab2`): f and G = Q0 R_s agree with the closed forms to 2e-9 (order 3).
  - The sigma-independent test is the ratio Q(L = 4 m)/Q(L = 3.5 m): 1.0476 vs the table's 1.0479.
  - The absolute Q (0.1 % above axionFEM) only confirms the sigma = 5.8e7 S/m that refs inferred from this table.
- **BI-RME** (`birme_tab1`): the empty cylinder matches the closed forms (f 2e-8, Q 1e-7 at R/16).
  - At R = 45 mm every TM01p frequency is 0.33–0.37 % below the table.
  - R is a fit, not a prediction. A one-parameter least-squares fit of R to the six TM01p frequencies gives
    R = 44.839 mm with residuals ≤ 7.7e-5. Fitting L instead leaves 3.7e-3. So the evidence is that a pure radius
    offset explains the six frequencies; f agreement at the fitted R is not a test.
  - **TM010 Q and C are 1.0 % low; unexplained.** An axisymmetric model of the centred top-cap probe was tried at
    R = 44.835 mm, pin depth d = 0–15 mm. The pin is r = 0.635 mm in a PTFE coax 0.635/2.11 mm, with a 10 mm stub
    shorted at its end.
    - C hardly moves up to d = 4 mm (0.69165 → 0.69039), then falls steeply (0.68769 at 5 mm, 0.68070 at 6 mm).
    - This strongly nonlinear onset is the shoulder of the shorted stub's own resonance, which dominates from
      d ≈ 7 mm. The stub is an artificial closed termination (a matched line cannot be closed in an eigenmode model).
    - So the "match" at d = 5–6 mm says nothing about the real probe. A closed-stub probe model can produce the
      deficits only near its own resonance. A matched (lossy or PML) coax termination is needed to say more.
  - The table's TM01p (p ≥ 1) Q values rise with p, 3 % above the exact closed form. Its p ≥ 1 C values
    (5e-3 … 7e-6) are exactly 0 for the ideal cylinder. Both columns are left out of the comparison, as refs advise.

### QUAX hollow sapphire (2004.02754, `quax_hollow_sapphire_2004`)

- **Geometry:** Fig. 4 table plus Fig. 10. Cone base diameter 20.7 mm, height 6 mm, frustum to a 4 mm × 20 mm bore
  closed at its end. R_cyl is the inner radius. The paper itself supports this reading: the R1 and t1 rows give the
  same Δf (−145 MHz), because f is set by the tube-1 outer face.
- **Mesh study:** h = 1.0, 0.7, 0.5 mm, order 2, plus order 3 at 0.7 mm, with the cone zone at h/2. Relative changes
  are at most 3.7e-5 (f), 1.9e-5 (C) and 1.8e-4 (Q), at the coarsest mesh. The production mesh is within 5e-6 of
  converged.
- **f and p_e:** f is within 0.11 % of 10.915 GHz (predicted; refs tolerance ~0.3 %). With the measured (as-built)
  tube dimensions it gives 10.9141 GHz. p_e = 0.1671 equals the value derived from Table 1, 0.167.
- **Q:** Q0 is 4 % above 1.91e6, outside the cone-variant range. The cone variants (height 5–7 mm, base 21.4 mm,
  sharp apex, no bore) move Q by −1.7 … +0.9 %; flat ends halve Q. Refs name the cone geometry as the main
  systematic, but this is not demonstrated.
- **Table 1 Δf:** the tolerance is the printed rounding: half a unit of the row's f plus half a unit of the 10.915 GHz
  reference. That gives ±5.5 MHz for 10.77, 10.77, 10.90 and 10.89, and ±1 MHz for 10.902 and the loss-tangent rows.
  - All five geometry/eps rows agree, t2 (+2.1 MHz) included.
  - eps = 9.39 is off by 63 MHz. Refs and an independent layered model (+170 MHz) both point to a typo in the paper.
  - The Table 1 Q column shows the same +4–5 % offset in every row except t2. That row and eps = 9.39 sit at avoided
    crossings in our model: the C030 strength is shared among two or three modes.
- **Not done:** the Fig. 3 radial field profiles (not compared; `Results.cavity_mode_field` exports the mode
  fields). Table 2 needs 3D.

### QUAX two-shell (2201.04223, `quax_two_shell_2201`)

The cones are not dimensioned. Baseline: the 2004 cones (6 mm, base radius 10.35 mm, 4 mm × 20 mm bore). Swept:
- cone height 0–10 mm and base radius 10.35 / 10.8 mm;
- tube end gap 0–1.5 mm per end;
- contraction set A (Cu −3.26e-3, sapphire −8e-4) and set B (2410.07774 Table I);
- tube-1 ID ±0.075 mm and OD ±0.025 mm (the stated diameter tolerances), D_cav 57.0 mm, and tube-2 inner radius
  ±0.1 mm.

Mesh study at the baseline (h = 1.0, 0.7, 0.5 mm, plus order 3): the production mesh is within 2.6e-5 of converged.

- **Hard targets.** Refs README lists them without numerical tolerances (written before this run).
  - Δf(eps 11.44 → 11.2): **18.77 MHz after rescaling by f** (raw 18.58 MHz) vs 18.9 MHz, i.e. 0.7 %. The rescaling
    is fair because Δf ∝ f p_e Δeps/(2 eps) and the absolute f is 1 % off. Table II prints f to 0.1 MHz, so the
    0.13 MHz residual is just outside the printed rounding. Any % tolerance on this row is post hoc.
  - Δf(T) from contraction: 8.79 vs 8.8 MHz (set A; within the printed 0.1 MHz). Set B gives 9.12 MHz.
  - p_e: 0.1698 vs 0.170.
  - The C trend vs eps agrees within the printed rounding.
  - None of these moves by more than 0.04 MHz (or 0.1 % in p_e) across the cone, gap and contraction variants.
- **The 2004 cones are a modelling choice made with the soft targets in view.** With them, the refs' flat-end
  problems largely disappear:
  - C030 0.0325 vs 0.0321 (+1.3 %; flat ends 0.0378);
  - Q_Cu 1.13e7 vs 1.10e7 (flat ends 2.7e6);
  - cavity volume at 293 K 1.07766 L vs the stated 1.0776 L. This uses D_cav = 57.1 mm (Fig. 1a). With the text's
    57 mm the cylinder alone is 1.0717 L and the cones give 1.0739 L, which does not close the gap.

  So the sister design's cones are probably what was simulated, but C and Q_Cu are "agrees (after choosing the
  cones)", not predictions.
- **Remaining discrepancy: absolute f, −1.02 %** (10.3685 vs 10.4745 GHz). It is not a solver issue: the
  independent layered model gives the same flat-end f to 1e-6.
  - Cones (< 5 MHz), end gap (< 1 MHz), contraction set (< 1 MHz), D_cav (< 1 MHz) and the stated tube tolerances
    (≤ 18 MHz together) cannot supply 106 MHz.
  - f is set by the outer face of tube 1 (17 MHz per 0.025 mm). A tube-1 outer radius about 0.16 mm smaller
    (OD −0.32 mm, i.e. t1 ≈ 2.19 mm) would close the gap. That is outside the stated OD tolerance.
  - Shifting the whole tube 1 inward by 0.159 mm (the refs' "fit R1") reproduces f but drops C to 0.0297 (−8 %).
  - Alternative readings of the dimensions (layered model at 4.2 K, `independent/alt2201.py`): tube-2 "41.8" read as
    an OD gives 10.4866 GHz; tube-2 t halved, 10.4518; tube-1 OD 26.13 mm, 10.4724. None is a natural typo that gives
    10.4745 GHz.
  - Left as an unexplained 1 % offset of the paper's absolute f.

### HAYSTAC one-rod, θ = 0 (2006.01248, `haystac_rod_centred`)

- **Model:** axisymmetric approximation — centred floating rod, 0.254 mm gaps, alumina axles omitted.
- **Mesh study:** the gap zone is graded from gap/2 to gap/8 at bulk h 3 → 1.5 mm, order 2, plus order 3. It
  converges to 4e-7 (f) and 3e-6 (C). The independent Q1 solver gives 5.86095 GHz (Richardson), C_vac ≈ 0.7725 and
  Q 9197, against axionFEM's 5.860955 GHz, 0.77255 and 9198.
- **Limits are exact:** the no-gap (coax) limit and the empty barrel reproduce the closed forms (3e-7).
- **The reference f(θ = 0) is not decidable from Fig. 3** (`fig3_reference_analysis` in the record):
  - by the mirror symmetry of the pivot geometry, f(θ) is even, so df/dθ = 0 at the concentric position;
  - the digitized Fig. 3 lines are straight from their first points (lower line −18.5 MHz/deg over 0.46–2.89°;
    about −28 MHz/deg on average to 10.9°), with no flattening;
  - quadratic fits f0 − a(θ + θ0)² need an origin offset θ0 = 1.7 / 2.1 / 5.9° (lower line, θ < 4 / 6 / 8°;
    f0 = 5.847 / 5.851 / 5.892 GHz) and 0.5 / 1.6 / 3.7° (upper line; f0 = 5.851 / 5.861 / 5.883 GHz). The fit is not
    unique. That the θ < 6° fit of the upper line lands on 5.861 GHz, the axionFEM value, is therefore not evidence
    either way;
  - so either the plotted θ = 0 is not the concentric configuration, or the branch near 0 is dominated by an
    interaction. In both cases f(0) = 5.835 GHz is an extrapolation with an uncertainty larger than 15 MHz.
  - CST's resolution of a 0.254 mm gap in a 254 mm cavity is also unknown. Here f moves 5 MHz between 0 and 0.25 mm of
    gap, and 30 MHz at 0.74 mm.
- **Results, one verdict per model** (published − axionFEM):

  | model | f [GHz] | Δf vs 5.835 | C_vac | C vs Fig. 6c (0.751) |
  |---|---|---|---|---|
  | no axles | 5.86096 | −26.0 MHz | 0.7726 | −2.8 % |
  | smeared axles | 5.8538 | −18.8 MHz | 0.698 | +7.7 % |

  - The smeared annulus is first order only. It is exact for E normal to the plates in the gap, but it ignores the
    axle inside the endcap holes and inside the rod, and the steel spindle.
  - The Fig. 6c point (5.827 GHz) lies at θ ≈ 0.9° on the digitized lower line, not at θ = 0, and 8 MHz below the
    "θ = 0" value read from Fig. 3. The effect on C is second order (dC/dθ = 0 at θ = 0), probably < 1 %.
  - The two 2D models bracket the published C. Which applies is a 3D question.
- **Ambiguities investigated:**
  - **Gap sweep, 0–1 mm:** f falls slowly (5.8666 → 5.861 → 5.851 → 5.818 GHz at 0, 0.254, 0.5, 1 mm). f = 5.835
    needs about 0.74 mm per end, but C_vac there is about 0.54 (plus a 5.8865 GHz partner with C 0.229), far below
    Fig. 6c. No single gap fits both.
  - **Gap asymmetry (0.127/0.381 mm):** −1 MHz.
  - **Axle estimate:** the 0.25 in alumina axle footprint, smeared over an annulus in each gap (eps_eff = 1.58, area
    weighted): −7.2 MHz and C_vac −10 %. The refs' "+2–3 MHz" axle systematic is probably an underestimate.
  - **Mode spectrum:** the next m = 0 TM modes are 26 MHz (z-odd, C = 0) and 34.5 MHz (even, C = 0.025) above the
    TM010-like mode. Both are confirmed independently, the even one by the Q1 solver with the z-symmetry plane. The
    Fig. 3 "upper line" may be an m ≠ 0 mode that only 3D can show.
- **Next (3D):** compare the slope df/dθ and C at θ ≈ 5–20° (isolated stretches) with the axles, not f(0).

### QUAX clamshell at 0° (2410.07774, `quax_clamshell_2410`)

- **Model:** prototype at 4 K, with the tube standing on the bottom plate and a 0.73 mm top gap. Flat ends; antennas
  and pins omitted.
- **Mesh study:** h = 1.0, 0.7, 0.5 mm, plus order 3, with the gap zone at gap/4. Changes are 1e-6 (f) and 7e-4 (C).
  This is convergence at one fixed detuning from the spectator modes (below). It is not a physical uncertainty on C.
- **Avoided crossing at 0.73 mm.** The TM030-like mode has two partners within 6 MHz. They are high-order TM0np
  modes: tube energy fraction 0.21–0.23 vs 0.093, end-zone fraction ~10× higher. In the z-invariant no-gap cavity
  they have C = 0; the top gap breaks the z symmetry and mixes them with TM030. Converged (order 3) mode cluster:

  | f [GHz] | C | Q0 | p_e | tube energy fraction |
  |---|---|---|---|---|
  | 11.04309 | 0.0287 | 1.129e5 | 0.098 | 0.231 |
  | 11.04443 | 0.2524 | 1.430e5 | 0.176 | 0.093 |
  | 11.04682 | 0.0043 | 1.487e5 | 0.109 | 0.208 |
  | **sum** | **0.2854** | | | |

- **Sensitivity scan** (production mesh; cavity and tube length shifted together by dL, gap fixed; or eps shifted).
  dL = 0, ±0.05, ±0.1, ±0.15, ±0.2 mm and deps = ±0.01, ±0.02. Record: `avoided_crossing_sensitivity`; figure: 4th panel.

  | change | largest-C mode: C | its share of the sum | its Q0 | cluster C sum |
  |---|---|---|---|---|
  | dL −0.20 mm | 0.2774 | 0.96 | 1.548e5 | 0.2890 |
  | dL −0.15 mm | 0.2460 | 0.86 | 1.469e5 | 0.2875 |
  | dL −0.10 mm | 0.1517 | 0.53 | 1.193e5 | 0.2865 |
  | dL −0.05 mm | 0.2322 | 0.81 | 1.375e5 | 0.2858 |
  | nominal | 0.2526 | 0.89 | 1.431e5 | 0.2854 |
  | dL +0.05 mm | 0.2503 | 0.88 | 1.426e5 | 0.2851 |
  | dL +0.10 mm | 0.2169 | 0.76 | 1.355e5 | 0.2849 |
  | dL +0.15 mm | 0.1524 | 0.54 | 1.459e5 | 0.2847 |
  | dL +0.20 mm | 0.2311 | 0.81 | 1.538e5 | 0.2846 |
  | eps −0.02 | 0.2580 | 0.91 | 1.478e5 | 0.2849 |
  | eps −0.01 | 0.2600 | 0.91 | 1.462e5 | 0.2852 |
  | eps +0.01 | 0.2283 | 0.80 | 1.369e5 | 0.2856 |
  | eps +0.02 | 0.1683 | 0.59 | 1.261e5 | 0.2859 |

  A 0.1 mm length change (2.4e-4, below the Cu/sapphire contraction uncertainty) or a 0.01–0.02 change in eps moves
  "the" TM030 C between 0.15 and 0.28 and Q0 between 1.19e5 and 1.55e5. The cluster C sum stays 0.2846–0.2890.
  The independent Q1 solver shows the same swing with its z mesh (C 0.20–0.29) and the same sum (0.285–0.286).
- **C030 verdict: not decidable.** The paper's 0.235 lies inside the single-mode range. HFSS's own mesh error is
  certainly larger than the 1e-4 detunings involved, so the 0.235 is one sample of an ill-conditioned quantity. The
  Fig. 7b Q at 0° (1.95e5, ~45 % below its neighbours, zig-zagging between 0.1° steps) is the same signature in the
  HFSS model.
- **Gap sweep:** largest-C mode / cluster C sum = 0.3366 / 0.3366 (no gap), 0.2886 / 0.3231 (0.05 mm),
  0.2715 / 0.3114 (0.1), 0.2597 / 0.2965 (0.25), 0.2554 / 0.2885 (0.5), 0.2526 / 0.2854 (0.73), 0.2494 / 0.2833 (1.0).
  The largest-C curve mixes the genuine gap effect with the detuning trend; the cluster sum is the gap effect. The
  no-gap C = 0.3366 (layered-exact, independent) is the clean 2D benchmark.
- **Q ratio prototype / pill-box: not decidable.** Paper 4.32. axionFEM: 3.78 nominal, 3.15–4.09 over the scan,
  4.09 at the least-mixed point (dL −0.2 mm, 96 % of the cluster C). The independent Q1 solver gives ≈ 1.53–1.57e5 for
  the least-mixed Q (ratio ≈ 4.1). The prototype/ideal ratio moves the same way: 1.61 nominal, 1.74 least mixed,
  paper 1.80. The paper/axionFEM absolute Q factor (2.29 prototype, 2.01 pill-box) is not a common factor either.
- **Ideal design** (shell inner radius not given). With t = 3.01 mm (Table II caption), the inner radius r1 is
  **fitted** so that C = 0.476: r1 = 11.27 mm. C is therefore not a test. The predictions at that r1 are:
  - f = 11.021 GHz, against "11 GHz" in Table II. f also excludes the second C = 0.476 crossing beyond r1 = 12 mm,
    where f > 11.18 GHz;
  - the Q ratio to the pill-box, 2.36 vs 2.39 (+1.7 %).

  With t = 3.97 mm (text), C never exceeds 0.4695. The paper's design rule (inner face at the empty-cavity node,
  0.278 r_c = 7.8 mm) is clearly not what was built at 11 GHz.

## Not done here (needs 3D or a solver feature)

- TE111 (m = 1) entries of the ADMX and KLASH tables.
- QUAX 2004 Fig. 3 field profiles (not compared).
- QUAX 2004 Table 2 (split shells), clamshell Fig. 7 (opening angle), HAYSTAC θ > 0 and the θ = 0 axles.
- The end-cap share of the copper loss (2201: 16 %): `losses.walls` reports per connected conductor body, and the
  end plates and the barrel form one body. Splitting it would need a solver change (not made here).
- BI-RME probe with a matched (lossy/PML) coax termination: not needed for Table I (BI-RME's modes are
  shorted-port modes); a driven matched-port check would need an axisymmetric port/PML on native meshes.
