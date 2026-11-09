# Cavity-haloscope reference data

Reference values for reproducing published cavity simulations with axionFEM. One JSON per benchmark. All JSONs use
the same keys: citations, software, geometry (each parameter with its source), materials, conditions, conventions,
reference_values (tables copied exactly; digitized curves), measured_values (sanity only), gaps, inconsistencies,
solver_class, priority, readme_row.

Digitization: every digitized figure was checked against an overlay of the extracted points on a crop of the
published figure (the crops and overlays are not redistributed here). Scripts are in `digitize/`. Run them as `python3 -I digitize/<script> <pdf>`,
importing numpy before fitz. Each curve records its axis calibration and its uncertainty in data units.
- Vector extraction (exact PDF path vertices): RADES 2002.07639 Figs 5, 6, 10, 13.
- Raster extraction (embedded bitmaps; native resolution or 600 dpi colour masks): all other figures.

## Index

| Benchmark | JSON | Quantities | Geometry complete? | 2D-axisym possible? | Needs 3D? |
|---|---|---|---|---|---|
| Analytic: ADMX thesis 2408.03444 Tab 6.1 | `analytic_admx_thesis_tab6_1.json` | f TM010/TE111/TM011/TM020, 4 empty cylinders (+ closed-form check) | yes | yes (TE111 needs m=1) | no |
| Analytic: KLASH 1707.06010 Tab 2 | `analytic_klash_tab2.json` | f, Q0(300 K) TM010/TE111 | yes (sigma inferred 5.8e7) | yes (TE111 needs m=1) | no |
| Analytic: BI-RME 2107.03137 Tab I | `analytic_birme_tab1.json` | f, Q, C TM01p, D = 90 mm, L = 1 m | nearly (probe depth) | yes (centred pin) | no |
| 1. RADES 5-subcavity WR-90 | `rades_5cav_1803.json` | 5 modes f, G², Q0 (CST, 2 K); measured f | nearly (corner radius) | no | yes |
| 2. RADES 6-subcavity alternating | `rades_6cav_alt_2002.json` | 6 modes f, G², Q0 (2 K/300 K); k(w), k(h), S21 (vector-digitized) | nearly (capacitive-iris slot position/outline) | no | yes |
| 3. QUAX two-shell sapphire | `quax_two_shell_2201.json` | hard: Δf(ε), Δf(T), p_e; soft: absolute f (~1%), C030 (~15%); Q_Cu cone-dominated (unreliable) | nearly (cones, tube end gap: sweep) | **yes** | no |
| 3b. QUAX hollow sapphire | `quax_hollow_sapphire_2004.json` | TM030 10.915 GHz, Q = 1.91e6 (cone-sensitive), Tab 1 sensitivities (except ε = 9.39 row), Fig 3 radial field shapes; Tab 2 split shell. R_CV / C030 = 0.065 withdrawn | yes | **yes** (Tab 2: no) | Tab 2 only |
| 3c. QUAX single-shell clamshell | `quax_clamshell_2410.json` | Tab II C (~4%); Fig 7 f, C030 vs opening angle; all simulated Q ≈ 2× analytic (ratios/trends only) | nearly (hinge) | 0° only | yes |
| 4. TASEH single rod | `taseh_rod_2205.json` | f(θ), Q0(θ), C(θ) (HFSS Fig 5); coupling vs probe depth; sim vs meas | nearly (θ origin, rod end arms, σ) | no (empty cavity only) | yes |
| 5. HAYSTAC one-rod | `haystac_rod_2006.json` | f(θ): two separated near-θ=0 lines (lower = TM010-like) + mode cloud (Fig 3); Q, C, QC²V² vs f (Fig 6); intruders; C010 = 0.4654 (convention unknown, consistency check only) | nearly (axles, vernier, σ) | **approximately** at θ = 0 (centred rod, 0.254 mm gaps; alumina axles omitted, +2–3 MHz systematic); empty barrel | yes (θ > 0) |
| 6. Stern multi-post/vane | `stern_multipost_1603.json` | Q0, C, FoM vs Λ = fR: 14 series = 13 Table I designs + unmatched Fig 6b "H" (256 points; 241 without H) | no (rotation angles, vane shape, design labels; Fig 7a 8-post inset ≠ Table I K) | no (z-invariant: 2D cross-section) | yes, for axionFEM today |
| 7. CAPP multi-cell (concept) | `capp_multicell_1710.json` | Tab 2 Q for 2–8 sector cells; Tab 3 ratios; Tab 1 f, Q, C | yes (Tabs 2–3); no (Tab 1) | no (closed form) | Tab 1 only |
| 7. CAPP multi-cell (gap study) | `capp_multicell_2205.json` | Tab 2 closed form; Tab 3 optimal gap; Figs 5–6 C, C²V²Q vs gap | yes (Tab 2); no (Figs 5–6) | no | Monte Carlo points |
| 8. ADMX Run 1A/1B two-rod | `admx_two_rod.json` | Run 1B Fig 2 rod positions, symmetric vs anti-symmetric (equal f); f = 773 MHz at (180°, 270°), 656.4 MHz at (120°, 0°), range 580–890 MHz; Run 1A S12 mode map vs angle; C(f) Run 1A (COMSOL, 6% band) and Run 1B (CST); crossings table | nearly: pivot radius 122 mm and arm 58 mm are **INFERRED** (±2 mm); armatures and ports missing | no. A **2D cross-section is feasible** (z-uniform, no end gaps) | yes: end gaps, armatures, mode crossings, C with the field map |

## Suggested order for the 2D axisymmetric solver

1. The three analytic empty-cylinder tables.
2. QUAX 3b (2004.02754): fully specified, and the authors simulated it themselves on a PMC sector. Q depends on the cone geometry.
3. QUAX 3 (2201.04223): benchmark differences (Δf(ε), Δf(T), p_e). Sweep the cone dimensions and the tube end gap (0–1.5 mm); fit R1 before comparing absolute f and C.
4. HAYSTAC centred-rod subcase (θ = 0, coax with end gaps). It is only approximately axisymmetric: the CST model has 0.25 in alumina axles crossing both gaps, so carry a +2–3 MHz systematic. The no-gap limit is analytic (5.8666 GHz; C = 0.806 with the vacuum volume, 0.605 with the full barrel volume). With gaps, the target is the lower (TM010-like) Fig 3 line, f(0) = 5.835 GHz. Quote it as 5.83–5.85 GHz with ±0.015 GHz tolerance, because a second line sits 14 MHz above it. C ≈ 0.75 in the paper's convention. Sweeping the gap tests the "G < 250 µm" statement in 1611.07123.
5. QUAX 3c at 0° opening.

CAPP sector cells, Stern and the ADMX two-rod cavity are not axisymmetric. Their ideal versions are z-invariant (the posts and partitions run the full height), so a 2D cross-section solver handles them (ADMX: rods touching both end caps, no armatures; `digitize/admx_two_rod_2d.py` is a validated stand-alone finite-difference reference for that limit); axionFEM has one (`"symmetry": "xsec"`, [docs/validation/cavity_xsec.md](../../../docs/validation/cavity_xsec.md)), and the end gaps, armatures and ports still need 3D. The CAPP tables are closed-form checks that need no FEM (`digitize/capp_sector_analytic.py`).

## Validation targets (what to trust, at what tolerance)

| Benchmark | Good hard targets (tolerance) | Soft / do not use |
|---|---|---|
| Analytic tables (ADMX, KLASH, BI-RME) | closed forms, exact (<1e-4); table entries within rounding (ADMX ±0.24–0.31%) | BI-RME TM01p (p ≥ 1) Q and C; BI-RME f is offset +0.35% uniformly (R = 44.835 mm fits) |
| QUAX 2004 hollow sapphire | Tab 1 sensitivities Δf (≤ 2 MHz agreement in exact model), f 10.915 GHz (~0.3%, cone-sensitive), Q 1.91e6 (cone geometry is the main systematic), Fig 3 field shapes | ε = 9.39 row; R_CV / C030 = 0.065 (withdrawn); Fig 3 absolute f (third geometry) |
| QUAX 2201 two-shell | Δf(ε) (19 MHz), Δf(T) (9 MHz), p_e = 0.170 (~1%), C trends vs ε/T | absolute f (~1%), absolute C (~15%) unless R1 and end gap are fitted; Q_Cu/G (cone-dominated); Q_Sapph(4.2 K) |
| QUAX 2410 clamshell | no-gap C030 (0.3366, layered-exact; ~0.34), pill-box C, ideal-design Q ratio, Fig 7a f(θ) shape/slope (3D) | all simulated Q in absolute value (≈2× analytic): use ratios/trends only; Tab II prototype C (0.73 mm gap) and the prototype Q ratios: not precision targets (avoided crossing) |
| HAYSTAC one-rod | empty barrel and no-gap coax (analytic); Fig 6 Q at 4.25–4.55 GHz (~5%); df/dθ and C at θ ≈ 5–20° (3D) | f(θ = 0) = 5.835 GHz from Fig 3 (lines linear to θ → 0: not decidable); C010 = 0.4654 (unknown convention); θ ≈ 0 Q = 4.6e3; Fig 3 14.5–19.5° (crossing cluster, ±0.03 GHz) |
| TASEH | Fig 5 f(θ) (few MHz after fitting the θ origin), C(θ) ~2% | quoted 4.56 GHz empty f; PRL C 0.60–0.61; C = 0.67 |
| RADES 1803 / 2002 | CST eigenfrequencies (MHz level), G² | Q (σ assumptions); S21 peaks (probe-loaded); k(w) at w = 8.90 (~4% figure-vs-text mismatch) |
| CAPP 1710 / 2205 | Tab 2–3 closed forms (exact; quadrature); Fig 6 ideal curves only in kept ranges (2-cell C ≤ 0.0565) | Monte Carlo points (asymmetric "statistical deviation", unknown model); C convention ~7% |
| Stern | Λ-trends of Q, C, FoM for designs with trustworthy geometry (both C conventions) | K design (inset ≠ Table I), H/I labels, σ (5.8–6.0e7 unconstrained) |
| ADMX two-rod (Run 1A/1B) | 2D cross-section with INFERRED pivot (r_p 122, a 58 mm): f at the Fig 2 rod positions, symmetric = anti-symmetric (caption; 2D model agrees within 0.3%); f(180°, 270°) = 773 MHz, f(120°, 0°) = 656.4 MHz, 580–890 MHz range (all ±1%: 2D ignores the 0.5 mm end gaps and the armatures); C trends (C_anti < C_sym) | absolute C (field map, unknown V convention; expect C_2D ≈ 1.3× published); Fig 8c absolute angles (good to ~3°); Q (only measured Q_L); 2010.00169 "0.09 m" trajectory radius |

## Main geometry gaps (proposals in each JSON)

- **RADES:** corner radius (sweep 0–1.5 mm). In 2002.07639, the capacitive-iris slot position (calibrate on the k1(h) curve).
- **QUAX 2201:** cone dimensions (take them from the 2004 drawing, then sweep). They control Q_Cu. The tube end gap is also a sweep parameter.
- **TASEH:**
  - the θ origin;
  - the rod end-arm geometry, which leaves the rod-end gaps ambiguous;
  - wall σ is not stated (use 5.8e7);
  - the probe mirror image (run both).
- **HAYSTAC:** σ and axle details; the angles of the Fig 6 points (inferred from Fig 3); the seven-rod pivot radius.
- **Stern:** moving-post rotation angles (sweep and compare against Λ). Figure labels do not match Table I (H/I), and the Fig 7a 8-post inset (rings ~0.31R/0.65R, posts ~0.2R) contradicts Table I K (0.55/0.55). Both must be resolved before these designs are used as benchmarks.
- **Form-factor volume convention (all post/rod/partition geometries):** papers do not say whether V in C is the full cavity volume or the vacuum volume (posts/rods excluded). The factor is 1/(1 − area fraction): Stern 5–21% depending on design, CAPP partitions ~7%, HAYSTAC rod 1.33. axionFEM must report C under both conventions, and comparisons quote both.
- **CAPP 2205 Figs 5–6:** Rc, height, partition thickness and the Monte Carlo model are not given.
- **ADMX two-rod:** the pivot radius and arm length are not published. Both are INFERRED (r_p = 122 ± 2 mm, a = 58 ± 2 mm) from two independent methods: pivot-ring and rod-dot fits on Run 1B Fig 2 / 2010.06183 Fig 16, and a 2D fit to the published simulated frequencies (`digitize/admx_two_rod_infer.py`). The alumina armatures, the ports and the end-gap details ("~0.020 in") are also not published.

## Notable inconsistencies (details in JSONs)

- **ADMX two-rod:** 2010.00169 says the rods sweep "0.09 m radius circular trajectories", but Fig 2 shows an arm of 58–60 mm, and with a = 90 mm the two rods would collide at θ = 180°. 2312.16668 says COMSOL, but its Fig 8a is a CST render. Its Fig 8a label (656.4 MHz at 120°) is 3–4 MHz below the Fig 8c branch at 120°. The rod material is reported as copper, Cu-plated stainless steel, or a hollow OFHC tube, depending on the source.
- **2107.03137 Table II** is identical to 1803.01243 Table 2, so it is not an independent cross-check.
- **2002.07639** gives h = 0.93 mm in the text but 1.000 mm in Table 1. The w marker in Fig 5 sits at 9.06 mm, not 8.90.
- **2201.04223 Table I:** the column labelled "σ_eff" is really 1/R_s. 2410.07774 then propagates this as a conductivity.
- **TASEH:** C is 0.62–0.63 in 2205.01477 and 0.60–0.61 in the PRL (the digitized curve gives 0.616–0.631). The quoted empty TM010 of 4.56 GHz differs from the analytic 4.590 GHz.
- **HAYSTAC:**
  - 2006.01248 Table I swaps two volumes.
  - Only the θ ≈ 0 Fig 6b Q point (4.6e3) is about 3× low. The 4.25–4.55 GHz points agree with the analytic copper estimate (1.385e4), so 4.6e3 is not a 2D target.
  - Volume convention explains most of the gap between C ≈ 0.5 and C ≈ 0.75, but not all of it: 0.751 × 1.5455/2.0593 = 0.564, leaving a 12% residual.
  - Fig 3 has two parallel lines near θ = 0, 13–27 MHz apart. The original trace jumped between them; they are now digitized as separate series.
  - f(θ) must be even in θ, yet both Fig 3 lines are straight down to θ ≈ 0.5° (quadratic fits need a 1.6–2.1° origin offset, not unique): f(0) = 5.835 is an extrapolation with > 15 MHz uncertainty.
  - C010 = 0.4654 has an unknown volume convention and is in tension with Fig 6c.
- **CAPP 1710 Table 2:** the analytic Q corresponds to σ = 5.998e7 S/m, not the stated 5.96e7. 5.998e7 is COMSOL's built-in copper. Eq. 16 has a typo; the corrected form n³/(π(n²−1)) reproduces Tab 3.
- **CAPP 2205 Eq. 7** as printed gives C ≡ 0. The corrected Bessel order works for m = 1–4 and fails at m = 0, so `capp_sector_analytic.py` uses quadrature.
- **QUAX 2410:** every simulated Q (Tab II, Fig 7b) is about 2× the analytic value for the stated σ. The measured Q agrees with the analytic value.
- **QUAX 2410 prototype at 0°:** the 0.73 mm top gap puts TM030 at an avoided crossing with high-order TM0np modes; its single-mode C (0.15–0.28) and Q depend on 1e-4 detunings, the cluster C sum (0.285) does not.
- **QUAX 2201:** an exact model at the stated dimensions gives f 0.9% low and C 15% high, but reproduces the Δf(ε), Δf(T) and p_e. Q_Cu is set by the unknown cones (flat ends give 2.7e6 vs 1.1e7).
- **QUAX 2004:** R_CV / C030 = 0.065 cannot be reproduced (flat-end C030 = 0.041). The Tab 1 ε = 9.39 row is a possible typo. The Fig 3 field plot uses a third geometry (10.71 GHz).
- **Errors in the earlier literature survey:**
  - It is missing the Sidecar row of Tab 6.1.
  - It wrongly treats 2107 Tab II as independent.
  - It labels the HAYSTAC "4.3999 vs 4.4006" pair as simulated vs measured; both values are simulated.
