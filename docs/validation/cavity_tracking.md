# Cavity tuning sweeps and mode tracking: validation

**Beta.** Code: `python/axionfem/cavity_track.py` (exported from `axionfem.cavity`); user docs:
[design_format.md](../design_format.md), "Cavity tuning sweeps and mode tracking". Verdict categories as in
[cavity_reproductions.md](cavity_reproductions.md): *agrees (predicted)*,
*agrees (after fitting X)*, *consistent*, *not decidable*, *disagrees (cause)*. Differences are published − axionFEM
unless stated. Nothing here was tuned to match a published number; the one fitted parameter (the HAYSTAC θ origin)
is named in its row.

## 1. Tracker (tests/test_cavity_track.py)

| case | reference | axionFEM | verdict |
|---|---|---|---|
| two-level model, κ = 2 MHz, slope 1e8 Hz | gap 2κ = 4 MHz at x0 = 0; f± = f0 ± √((sx)² + κ²) | gap to 1e-3, \|x0\| < 1e-3, f to 1e-3 Hz; C exchanges on the adiabatic branches, stays on the diabatic one | agrees |
| two-level + uncoupled mode | two true crossings, overlaps 1 | found, overlaps > 0.999 | agrees |
| x0 midway between two samples (n = 6…54) | one event | one event; diabatic() recovers the character | agrees |
| 41 / 201 / 801 samples over the same crossing | the same event | one event each, gap to 1 % | agrees |
| spectator mode inside the gap | one avoided crossing of the coupled pair | found, gap to 2 % | agrees |
| three-level crossing | character of the lowest branch ends on the highest | multi-branch event; diabatic() recovers it | agrees |
| κ = 0.1 MHz, step 1000× the width | bisection resolves a 0.2 MHz gap | gap to 2 % | agrees |
| κ = 0 with refinement (default tolerance) | stays a true crossing | x0 to 1e-4, overlaps > 0.999 | agrees |
| degenerate pair rotated at random each step (equal and 1:10 probe norms) | members followed, no ambiguity | yes | agrees |
| pair degenerate at the first step, split after | no crossing | none in 20 noisy trials | agrees |
| rejected best match (O = 0.45) | interval ambiguous, refined | yes | agrees |
| sweep range narrower than the crossing / minimum at the edge | warning | warning | agrees |
| steep level crossing two flat levels (Δ/κ = 4–14), 241 points | diabatic high-C line carries the steep level's character at every point (exact eigenvectors) | max deviation ≤ 0.2 in C at all points | agrees |
| narrow crossing (2κ = 0.2 MHz) next to an unrelated mesh-split doublet | event kept | kept | agrees |
| probe vectors of distinct modes overlapping by > 0.5 | no false ambiguity | none (Löwdin orthonormalization) | agrees |
| axisymmetric pillbox, a = 0.1 m, L swept | TM020/TM012 true crossing at L* = 2πa/√(χ02² − χ01²) = 0.126455 m (closed form: 0.126455 m, 2633.8198 MHz) | x0 to 2e-4 relative, f_mid to 1e-3, overlaps > 0.99 | agrees (predicted) |
| same + dielectric disk on an end cap | avoided crossing (perturbation theory: non-zero matrix element) | avoided, gap > 1 MHz | consistent (qualitative: the gap value is not predicted) |
| cross-section rectangle 0.1 × b, b swept | TM21/TM12 and TM31/TM13 true crossings at b = a, 3.35178 / 4.74013 GHz | x0 to 1e-4, f to 1e-3 | agrees (predicted) |
| same + ε = 2 post, r = 4 mm at (31, 27) mm | first-order degenerate perturbation theory (finite-post integrals, derived independently of the code): gaps 37.6983 / 6.60803 MHz at b = 99.6951 / 99.7738 mm; neglected orders 5–10 % / ~10 % | 39.97 / 6.495 MHz at 99.755 / 99.778 mm (+6.0 % / −1.7 %) | agrees (predicted, within the blind error estimate) |
| HAYSTAC cross-section (no gaps), θ = 0–180° | method of particular solutions (independent) | TM010-like branch: f to ≤ 1.4e-6, C to ≤ 6e-6 at 10–150° | agrees (predicted) |
| 3D dielectric rod moved off axis (AXF_SLOW) | one continuous TM010-like branch, f rising | see the suite run | — |

Figure: docs/images/cavity_track_xsec_rod.png (HAYSTAC cross-section mode map, tracked branches, crossings).

## 2. HAYSTAC one-rod cavity in 3D with the alumina axles (2006.01248, CST)

Model (benchmarks/cavity/tracking/haystac_rod_3d.py; all dimensions from the refs JSON with their sources): barrel
R = 2 in, H = 10 in; rod r = 1 in, 9.98 in long, 0.010 in gaps; rod centre at d(θ) = 2 (0.475 in) sin(θ/2);
alumina axles (ε_r 9.8, not given in the paper) of 0.25 in OD at the pivot across both gaps; copper σ = 5.8e7 S/m
at 293 K (not given); z-mirror symmetry (PEC plane: the TM010-like family; PMC plane for the odd family). Mesh
h = 6 mm, rod surface 3 mm, gap regions 0.7 mm, order 2 (≈ 0.72 M DOFs); mesh study at θ = 10° (0.33 / 0.72 /
1.48 M DOFs): f 5.587250 / 5.586518 / 5.586199 GHz, C 0.45800 / 0.45738 / 0.45708, Q0 9071 / 8964 / 8890 (production
error ≈ +0.6 MHz, +0.15 % in C, +1.5 % in Q0). θ = 0–32° in 1° steps plus bisection (record:
benchmarks/records/cavity_track_haystac_rod_3d.json; figure docs/images/cavity_track_haystac_rod_3d.png).

Differences in this section are axionFEM − published (the opposite of the general convention above).

| quantity | published (digitized) | axionFEM | verdict |
|---|---|---|---|
| df/dθ of the TM010-like line, 5–14° (Fig. 3 lower line, linear fit) | −32.26 ± 0.42 MHz/deg | −32.39 MHz/deg | **agrees (predicted)**, 0.4 % |
| f(θ), 0–14.5° (20 digitized points of the lower line) | lower line | +18.7 MHz (0.32 %) above it; rms about the constant offset 1.9 MHz (digitization ±4 MHz) | **disagrees by 0.32 % (cause open)**: not the mesh (production +0.6 MHz), not the axle end-cap holes (≤ 1.2 MHz at 5–10°); a θ-origin shift alone fits worse (0.65°, rms 4.5 MHz; a joint fit returns θ0 = 0.04°), so the offset is in frequency, not in the angle |
| upper parallel line of Fig. 3 (2–10°) | identity unknown (refs: "plausibly TM011-like") | odd-in-z family (PMC run): the meridional (E_azimuthal_fraction < 0.01) mode just above our TM010-like one lies +16.0 / +17.0 / +16.9 MHz above the upper line at 2 / 6 / 10° | **consistent**: the upper line is the odd-in-z partner, with the same ≈ +17 MHz offset as the even line. At 14° the odd modes are in a crossing cluster (pick −15 MHz) |
| f(θ = 0) | 5.835 GHz | 5.851 GHz | not compared (refs README: not decidable from Fig. 3) |
| Fig. 6 points (1 × 2 in rod, red triangles) | given vs f only; their CST angles θ_CST are read off the Fig. 3 lower line; compared at θ_CST (equal θ) and at equal f, with the spread over ±1° | | |
| … 4.917 GHz (θ_CST = 31.3°) | C 0.388, Q0 7.31e3 | equal θ: C 0.389, Q0 7.16e3; ±1°: C 0.381–0.392, Q0 6.2–8.3e3 | C **agrees (predicted)** (+0.2 %, ±1.5 % over ±1°); Q0 **consistent** (−2 %; Q0 changes ~18 %/deg here and σ is not stated) |
| … 5.391 GHz (θ_CST = 15.3°, crossing cluster) | C 0.306, Q0 3.46e3 (a dip) | equal f: C 0.218, Q0 2.53e3; equal θ: 0.379, 4.75e3; ±1°: C 0.22–0.37, Q0 2.5e3–1.1e4 | **consistent** (a C and Q dip at this frequency in both); not a precision target (refs) |
| … 5.827 GHz (θ_CST = 0.9°) | C 0.751, Q0 4.64e3 (3× below the copper estimate) | C 0.639–0.649 (equal f / equal θ), at most 0.649 anywhere in the sweep; Q0 3.5e3–4.7e3 (4.0–5.8e3 within ±1°) | C **disagrees (−15 %; cause open: axle and gap modelling; the 3D mesh study is at 10° only and the gap mesh, 0.7 mm in a 0.254 mm gap, is coarser than the converged 2D grading)**. Q0 **not decidable** (it hinges on mixing with lossy rod-end modes near θ ≈ 2° in our model, unverified by a mesh study there). For reference, 3D at θ = 0: C 0.632, Q0 3191; 2D without axles ([cavity_reproductions.md](cavity_reproductions.md)) 0.773 / 9198, smeared axles C 0.698 |
| axle sensitivity (5/10/15°) | — | no axles: Δf = +1.1 / +6.9 / −8.6 MHz; axles 10 mm into end-cap holes: −0.4 / −1.2 / −10.2 MHz | the offsets near crossings (10°, 15°) are crossing shifts, not a uniform loading |

TM010-like identification: the largest-C mode of each step (the standard identification) and the tracked diabatic
branch (`Tracking.branch("max_C", diabatic=True)`) coincide; both follow the Fig. 3 lower line through 0–32° with
the ≈ +18 MHz offset, through the C dips at 5.18 and 5.39 GHz (crossing clusters at 15–25°). The sweep: 33 angles +
59 bisection steps. The probe vectors are Löwdin-orthonormalized: without it, distinct modes overlap by up to 0.6–0.8
on the probe lattice and 64 of 91 intervals are flagged ambiguous; with it, 8 of 91.
Geometry check: all 92 tracked steps have the same cavity volume (7.72737e-4 m³ in the kept half; spread 3e-8).
Reruns with later mesher and full-wave defaults (HAYSTAC θ = 2, 6, 10, 14, 31°; TASEH θ = 90° on two meshes) left f, C
and Q0 unchanged to the printed digits.

## 3. TASEH single rod in 3D (2205.01477, HFSS)

Model (benchmarks/cavity/tracking/taseh_rod_3d.py; dimensions quoted from 2205.01477 p. 4–5, as in the refs JSON):
D = 50 mm, L = 120 mm; copper rod 4 mm × 114 mm whose rotation axis is 19 mm from the centre with the rod centre
3.5 mm from it; transverse end arms (baseline A: arm centrelines at ±57 mm, arm surface 1 mm from the end walls) and
a knuckle on the rotation axis; Teflon axles 3 mm, ε_r 2.1 (not given); θ = 0 = rod farthest from the axis (refs
inference); square corners (the stated 3 mm fillets collide with the baseline rod end near θ = 0 — variant below);
copper σ = 5.8e7 S/m at 293 K (HFSS "bulk copper", not printed); z-mirror symmetry (PEC: even family; PMC: odd).
Mesh h = 5 mm, rod surface 0.8 mm, rod-end regions 0.5 mm, order 2 (≈ 0.37 M DOFs). Mesh studies (0.17/0.37/0.82 M
DOFs): θ = 90°: f 4.761516 / 4.761166 / 4.761091 GHz, C 0.65604 / 0.65591 / 0.65588, Q0 18327 / 18345 / 18351;
θ = 0° (rod 0.5 mm from the barrel): f 4.657523 / 4.657280 / 4.657215, C 0.67776 / 0.67766 / 0.67764, Q0 18071 /
18098 / 18107 (rod surface 0.3 mm: 18100). Production error ≤ 0.1 MHz, 5e-5 in C, 0.05 % in Q0. θ = 0–180° in 5°
steps (record benchmarks/records/cavity_track_taseh_rod_3d.json, figure docs/images/cavity_track_taseh_rod_3d.png;
all 37 steps have the same cavity volume). The rod is not built from cylinders and spheres with tangent contacts:
such a union loses a sub-volume at some angles (θ = 105–145° here), which the volume check catches.

Differences: published − axionFEM.

| quantity | published (digitized Fig. 5) | axionFEM | verdict |
|---|---|---|---|
| f(θ), 73 points 0–180° | 4.6616 … 4.9511 GHz | 4.6573 … 4.9447 GHz; mean diff +3.8 MHz (0.08 %), rms 4.1, max 6.9 MHz | **agrees (predicted)** (refs tolerance: a few MHz) |
| f(θ) shape | — | θ-origin shift + constant offset fitted: 0.46°, rms 1.4 MHz, max 3.9 MHz | agrees (after fitting the θ origin and an offset; shape check only) |
| tuning range 0→180°; df/dθ at 50–100° | 289.5 MHz (text: 290); 2.003 MHz/deg (text: ~2) | 287.5 MHz (−0.7 %); 1.976 MHz/deg (−1.3 %) | **agrees (predicted)** |
| TM010 mode crossings 0–180° | "no mode crossing" (p. 5) | no crossing or avoided crossing on the tracked branch (0 events, 0 ambiguous intervals) | **agrees** |
| other modes (Fig. 5(a)) | rising mode 4.84→5.1 GHz, flat pair 5.12/5.16 GHz, 5.29 GHz | odd-in-z run: the rising mode 4.849→5.112 GHz and the pair 5.12/5.158 GHz (TM011- and TE113-like); odd mode at 4.55 GHz at θ = 0 (falling below 4.5 by 30°) not in Fig. 5(a); even: a mode 5.28→5.37 GHz rising where Fig. 5(a) has a flat 5.29 GHz line | consistent for the rising mode and the pair; the 4.55 and 5.29 GHz differences unexplained (not targets) |
| C(θ) shape | C(180)/C(0) = 0.926 | 0.924 | **agrees (predicted)** (0.2 %) |
| C(θ) absolute | 0.640 … 0.592 (mean 0.618) | 0.678 … 0.626 (mean 0.654); published 5.5 % lower at every angle (rms 5.5 %, max 5.6 %) | **disagrees (−5.5 %; probable cause: the HFSS C evaluation)**: the same V convention (rod excluded, 0.234 L); a 4 mm rod 0.5 mm from the wall cannot lower C from the empty-cylinder 0.6917 by the 7.5 % HFSS shows at θ = 0 (ours −2 %); the paper itself quotes 0.60–0.61 in the PRL and 0.67 elsewhere (refs inconsistencies), and HFSS gave 0.667 for an exact-0.6917 pillbox in QUAX 2410 ([cavity_reproductions.md](cavity_reproductions.md)) |
| Q0(θ) | mean 1.85e4, falling 1.93 → 1.69e4 | mean 1.79e4 (published +3.3 %), 1.81 → 1.66e4; ours dips by 4 % towards θ = 0 (rod 0.5 mm from the barrel; mesh-converged) where HFSS is flat | **consistent** (σ not stated; +3.3 % ↔ σ 7 % higher); the θ < 40° shape differs (the rod–wall clearance at θ = 0 is an inference, refs) |
| modelling variants (θ = 0/60/120/180°) | — | rod ends variant B (114 mm total, 3 mm gaps): Δf −1.4/−0.8/−1.8/−3.6 MHz, C −0.2…−0.7 %, Q0 −1…−3 %; no Teflon axles: Δf ≤ 0.19 MHz, C, Q0 ≤ 0.4 %; 3 mm corner fillets (θ ≥ 60° only, see the model note): Δf +5.4/+5.7/+6.1 MHz, C +0.1…+0.3 %, Q0 +2.5 % | none of them explains the 5.5 % C offset; with fillets the mean f difference would change sign (≈ −2 MHz) and the Q0 gap shrink to ≈ 1 % |
