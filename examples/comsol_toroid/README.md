# Running the COMSOL cross-check: the fiducial toroid of arXiv:2211.00008

This directory holds a cross-check of axionFEM's 3D full-wave solver against COMSOL Multiphysics (RF Module), for
users who have COMSOL. Running it produces one validation plot, `validation_toroid.pdf`, and its JSON summary, which
compare COMSOL's and axionFEM's port impedance Z(f) and axion open-circuit voltage V_oc(f) on the same detector, from
the magnetoquasistatic (MQS) limit, through the first self-resonance, up to 60 MHz. The two codes need not agree to
the last digit; the criteria of section 5 say how close they should be, and the differences should shrink as COMSOL's
mesh is refined. The axionFEM side is already computed (`axionfem_reference.json`); the COMSOL side has not been run
yet.

## 1. Why this is a non-trivial check

The two codes share only the geometry description and the source formula. Everything else is independent:

| | axionFEM (reference, already computed) | COMSOL |
|---|---|---|
| mesh | its own gmsh mesh, order-2 Nédélec, two resolutions | COMSOL's own mesh from the exported Java geometry (refine it, section 4) |
| port | current fed through the port *volume* | lumped port on a *face* |
| impedance | Z from the port-driven solve | `emw.Zport_1` from the 'imp' run |
| axion voltage | **reciprocity**: V = −iω ∫ E_port·B0 dV / I, without solving with the axion source | **direct**: the 'volt' run solves with the axion current Je = B0 and reads the port voltage (converted to open circuit) |
| time convention | e^{−iωt} | e^{+jωt}, so COMSOL's numbers are complex conjugates of axionFEM's |

Agreement therefore tests geometry translation, meshing, the lumped-port model, reciprocity, the sign and time
conventions, and the material-loss implementation (lossy case). The physics changes over the band:
- the MQS plateau, L ≈ const below ~5 MHz;
- the self-resonance, where Im Z changes sign, near 20 MHz;
- capacitive and multipole behaviour above it, with cavity modes of the PEC sphere.

## 2. The detector

The two designs are `comsol_toroid_lossless.json` and `comsol_toroid_lossy.json`, both axionFEM design files:
- **Toroid:** rectangular cross-section, inner radius 0.5 m, outer radius 1 m, height 0.5 m, 2.5 cm thick PEC
  (superconducting) walls.
- **Slit:** a planar 2 cm slit, at y ∈ [−1, 1] cm for x > 0, cuts the walls.
- **Port:** the readout port is the box x ∈ [0.975, 1.0], y ∈ [−0.01, 0.01], z ∈ [−0.25, −0.20] m. It bridges the
  slit at the bottom of the outer wall, like the "readout bridge" of the paper's Fig. 1.
- **Enclosure:** a closed PEC sphere of radius 1.5 m, centred on the toroid.
- **Axion source:** Je = Je_scale·B0, with B0 = B_φ = 1 T in the cavity (ρ ∈ [0.525, 0.975] m, |z| ≤ 0.225 m) and
  0 elsewhere. `Je_scale` = 1 A m⁻² T⁻¹ is axionFEM's κ.
- **Lossy variant:** a dielectric support ring with ε_r = 3 and tanδ = 2·10⁻³, at r ∈ [0.6, 0.9] m on the cavity
  floor, z ∈ [−0.225, −0.175] m.

Note: `L_MQS_H` in `axionfem_reference.json` is axionFEM's static (MQS) pickup inductance with an **open**
boundary. It is not the low-frequency limit of this closed-sphere problem, so do not compare COMSOL with it. Use
the full-wave rows at 1 and 5 MHz (lossy: 5 MHz only).

**Reference frequencies.** Lossless: 1, 5, 10, 15, 20, 25, 30, 40, 50, 60 MHz. Lossy: the same without 1 MHz
(axionFEM's iterative AMS-GMRES solver did not reach its residual gate for the lossy case at 1 MHz in a full sweep;
this is an axionFEM limitation, not a physics change. Each reference frequency was then solved in its own job,
and all of them converged). The exported Java models use exactly these lists; `compare` matches COMSOL rows to axionFEM's by
frequency.

## 2a. Expected values (axionFEM, COMSOL convention)

From `axionfem_reference.json`, level B (refined mesh: lossless 4.51 M DOFs, lossy 4.74 M; order-2 Nédélec,
AMS-GMRES, one batch job per frequency, every solve accepted with true relative residual ≤ 2·10⁻⁷ against a
gate of 10⁻⁶). COMSOL convention: Zport_1 = R + jωL, so Im Zport_1 > 0 is inductive. \|V_oc\| is the
open-circuit voltage for B0 = 1 T, Je_scale = 1. The last column is axionFEM's own mesh spread (level A, 2.3 M
DOFs, relative to level B).

**Lossless**

| f [MHz] | Im Zport_1 [Ω] | L = Im Zport_1/ω [µH] | \|V_oc\| [V] | A/B − 1: Im Z, \|V_oc\| |
|---|---|---|---|---|
| 1 | +5.0415 | +0.8024 | 0.94543 | −0.2 %, −0.1 % |
| 5 | +26.623 | +0.8474 | 5.0110 | −0.0 %, +0.0 % |
| 10 | +64.862 | +1.0323 | 12.351 | +0.4 %, +0.5 % |
| 15 | +156.70 | +1.6626 | 30.437 | +1.9 %, +2.1 % |
| 20 | +2434 | (pole) | 486.8 | at the pole (level A: +4409) |
| 25 | −215.42 | −1.3714 | 44.858 | −5.2 %, −5.0 % |
| 30 | −102.86 | −0.5457 | 22.624 | −3.4 %, −3.1 % |
| 40 | −40.811 | −0.1624 | 10.860 | −2.5 %, −2.1 % |
| 50 | +3.9474 | +0.0126 | 2.4554 | −0.4 %, −1.2 % |
| 60 | −50.380 | −0.1336 | 14.749 | −5.6 %, −4.2 % |

**Lossy**

| f [MHz] | Re Zport_1 [Ω] | Im Zport_1 [Ω] | L [µH] | \|V_oc\| [V] | A/B − 1: Im Z, \|V_oc\|, Re Z |
|---|---|---|---|---|---|
| 5 | 2.906e−4 | +26.722 | +0.8506 | 5.0302 | −0.0 %, +0.1 %, +9 % |
| 10 | 3.612e−3 | +66.055 | +1.0513 | 12.586 | +0.6 %, +0.7 %, +10 % |
| 15 | 3.607e−2 | +167.98 | +1.7823 | 32.679 | +2.8 %, +3.0 %, +15 % |
| 20 | 52.45 | −5429 | (pole) | 1089 | at the pole (level A: R = 10.5, Im = −2333) |
| 25 | 8.006e−2 | −184.19 | −1.1726 | 38.533 | −5.8 %, −5.6 %, −3 % |
| 30 | 2.662e−2 | −93.085 | −0.4938 | 20.618 | −4.0 %, −3.7 %, +1 % |
| 40 | 7.908e−3 | −38.016 | −0.1513 | 10.264 | −3.0 %, −2.5 %, +3 % |
| 50 | 2.377e−4 | +4.0280 | +0.0128 | 2.4367 | −0.1 %, −1.4 %, +6 % |
| 60 | 1.890e−2 | −43.372 | −0.1150 | 13.185 | −6.5 %, −4.7 %, −2 % |

**First self-resonance** (first sign change of Im Z, interpolated linearly in 1/Im Z between the 15/20 and 20/25 MHz
points, as `compare` does): lossless **20.41 MHz** (level A 20.22, +0.9 % A→B); lossy **19.85 MHz** (level A 19.66,
+1.0 %). The ε_r = 3 support ring lowers it by ≈ 0.55 MHz. Im Z changes sign again near 49 MHz and between 50 and
60 MHz (a narrow feature, probably a sphere/cavity mode, that the reference frequencies do not resolve; the
interpolated crossing near 59 MHz is not a precise location). `compare` therefore leaves the 20, 50 and 60 MHz points
out of the pointwise criteria; the dense COMSOL sweep shows the feature.

**Convergence status.**
- *Converged:* the MQS plateau (f ≤ 10 MHz). L(1 MHz) = 0.802 µH and \|V_oc\| change by ≤ 0.7 % from A to B.
- *Not converged (level A to B spread):* everything at and above 15 MHz. The SRF moves by +1 % from A to B, and the points above it move by
  3–6 %, mostly through that SRF shift (Im Z ∝ 1/(f − f_SRF) nearby). This is comparable to the 5 % Im Z/\|V_oc\|
  criterion and the 2 % SRF criterion of section 5, so a COMSOL result that misses those criteria by a few % in the
  direction A → B (higher SRF, larger \|Im Z\| above it) may be axionFEM's discretization error, not a disagreement.
  Also not converged: the lossy Re Z at 5–15 MHz, which moves by 9–15 % from A to B (the 15 % criterion).
- No level C was run. The `rows[*].residual` in the JSON are the true relative residuals of the solves.

## 3. Files

| file | what |
|---|---|
| `ComsolToroidLossless.java`, `ComsolToroidLossy.java` | the COMSOL models at the reference frequencies, exported by axionFEM (`--physics fullwave`) |
| `ComsolToroidLosslessDense.java`, `ComsolToroidLossyDense.java` | the same with 238 frequencies, 0.5–59.75 MHz in 0.25 MHz steps, for the curves in the plot |
| `axionfem_reference.json` | axionFEM's results at mesh levels A (the designs) and B (refined), with COMSOL's expected values |
| `check_toroid.py` | `compare` makes the plot and the pass/fail summary; `export` regenerates the Java (Python only); `solve` reruns axionFEM (needs the C++ build; not needed for the comparison) |

## 4. What to run

All commands run from this directory unless stated otherwise. `$REPO` is the repository root.

1. **Environment.**
   - Find the COMSOL launcher (`comsol`, `comsolbatch`, `comsolcompile`, e.g. under `/usr/local/comsol*/bin` or
     `C:\Program Files\COMSOL\...\bin`).
   - Check its version and that the RF Module is licensed (`comsol batch -help`, or the license manager).
   - For `compare`, make a Python environment with numpy and matplotlib (`pip install numpy matplotlib`). It
     does not need the axionFEM C++ build: if `import axionfem` fails (no `_core`), `check_toroid.py` loads the
     pure-numpy `python/axionfem/comsol_physics.py` of this repository directly. No `PYTHONPATH` is needed.
2. **First solve, reference frequencies, lossless.**
   - Run `comsolcompile ComsolToroidLossless.java`, then `comsol batch -inputfile ComsolToroidLossless.class`.
   - This writes `ComsolToroidLossless_impedance.txt` and `ComsolToroidLossless_voltage.txt` (rows
     `freq_MHz re im`) and `ComsolToroidLossless.mph`.
   - Calls marked `UNVERIFIED` in the Java have not been run in COMSOL (docs/comsol_export.md section 4). **If
     compiling or running fails, fix the Java minimally** using the COMSOL Programming Reference Manual, or by
     recording the same step in the GUI and saving as Java. Note every change (the API name, the old call, the new
     call and why): reported as an issue, such notes are what the exporter needs to be fixed.
   - Then sanity-check the result before going on:
     - Im(Zport_1) > 0 (inductive) at 1 MHz, with L = Im Z/ω close to the axionFEM value in
       `axionfem_reference.json` (COMSOL_expected).
     - The voltage file is nonzero.
     - The domain/boundary selections hit the intended entities. Open the .mph in the GUI if possible, or print
       the selections. Check four things:
       - the port is the face x = 1.0 of the port box;
       - the PEC walls are all faces of the sheath;
       - `ecd1` covers all air/dielectric domains;
       - the outer sphere is PEC.
3. **Mesh convergence in COMSOL.**
   - The export uses COMSOL's physics-controlled default mesh. Solve the lossless reference model at least three
     times, with the mesh made progressively finer.
   - Use the size presets (Normal, Fine, Finer), or set the maximum element size, for example 0.15, 0.10 and
     0.07 m. Refine near the slit and the port. COMSOL should resolve the 2 cm slit and the 2 × 2.5 × 5 cm port
     with several elements across.
   - Keep the quadratic (default) discretization.
   - Record the number of DOFs and the solve time for each mesh. Keep each level's tables in a subdirectory,
     `comsol_out/mesh_<level>/`.
   - The finest level that runs in reasonable time is the one compared.
   - **The convergence trend is part of the evidence.** COMSOL's L at 1 MHz and its SRF should move toward
     axionFEM's level-B values as the mesh is refined, or should bracket them.
4. **Lossy reference model** (`ComsolToroidLossy.java`). Run it at the finest mesh setting of step 3, applied the
   same way.
5. **Dense sweeps** (`*Dense.java`): 238 frequencies each, at the finest affordable mesh. These give the curves.
   - Measure the time of one frequency first. If the whole sweep would take more than about 6 h, thin it to every
     second frequency, and say so in the notes.
   - Do the lossless case first.
6. **Compare.** Put all the final `*_impedance.txt` and `*_voltage.txt` in one directory, e.g. `comsol_out/final/`.
   Then run:

   ```bash
   python check_toroid.py compare comsol_out/final -o validation_toroid.pdf
   ```

   This writes `validation_toroid.pdf` and `validation_toroid.json`, with per-point relative differences,
   pass flags, the SRF locations and an overall `pass`. The dense tables are optional (curves and the dense SRF
   only); the reference-frequency tables are required. `--ref FILE` selects another axionFEM reference JSON.
7. **Sharing a result.** The comparison JSON holds every number the plot shows. A useful report (for example in an
   issue) gives the COMSOL version, the mesh levels with their DOFs and times, the convergence trend, the pass/fail
   of each criterion with its numbers, and every change made to the exported Java.

## 5. Acceptance criteria (COMSOL finest mesh vs axionFEM level B)

| quantity | where | criterion |
|---|---|---|
| L = Im Z/ω | f ≤ 5 MHz (MQS plateau) | within 2 % |
| \|V_oc\| | f ≤ 5 MHz | within 3 % |
| Im Z | f > 5 MHz, more than 10 % in frequency from a pole/zero crossing | within 5 % |
| \|V_oc\| | the same points | within 5 % |
| Re Z (lossy) | the same points (R/\|X\| is only 10⁻⁵–10⁻³ here, so R must come from the loss, not from solver noise) | within 15 % |
| first self-resonance (first − → + sign change of Im Z, axionFEM convention) | both codes' reference-frequency points, interpolated the same way (linear in 1/Im Z across a pole) | within 2 % |

axionFEM's own A→B mesh spread is 3–6 % above the SRF and 1 % on the SRF (section 2a): weigh a marginal failure
there against it, and against the direction of axionFEM's refinement trend.

The plot overlays the following, with the SRFs as vertical lines:
- COMSOL's dense curves (grey);
- axionFEM's points (black circles);
- COMSOL's reference-frequency points (maroon crosses);
- relative differences per frequency, where the axionFEM coarse-vs-fine difference (grey dashed) shows
  axionFEM's own discretization error, and the green dotted lines are the acceptance bands. Points within 10 % of
  an SRF are left out of these panels and of the pointwise criteria.

The SRF from the dense COMSOL sweep is reported as `COMSOL_dense` in the JSON (more precise than the
reference-point interpolation, but not used for pass/fail: axionFEM has only the reference points).

**How to read a failure.**
- A difference that is **systematic and shrinks with COMSOL refinement** is resolution. Report the trend.
- A **constant few-% offset in L at low frequency** that does not shrink is most likely the port model (volume
  feed vs face port; docs/comsol_export.md "Port face"). Report its size; do not tune it away.
- A **factor of 2, a sign, or a complex conjugate** in V is a convention error. Check the open-circuit conversion
  and the time convention (section 1) before anything else.
- **Wrong Re Z in the lossy case only** points to the material mapping. COMSOL gets σ = 2π f ε0 ε_r tanδ through
  `electricconductivity`, with the frequency input.
- Near poles, pointwise ratios are meaningless; compare the pole positions.
