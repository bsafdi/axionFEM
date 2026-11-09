# S-, Y- and Z-parameters and Touchstone files

The implementation is in `python/axionfem/touchstone.py`. The validation is in
`benchmarks/touchstone_check.py` (record `benchmarks/records/touchstone_check.json`) and
`tests/test_touchstone.py`.

A full-wave sweep (`fullwave` section, both engines) reports the impedance of every port. A 3D multiport sweep gives
the full matrix in `Z_matrix_re_ohm`/`Z_matrix_im_ohm`, and an adaptive sweep (`fullwave.adaptive`) adds a rational
model and its dense curve. This page shows how those results become S-, Y- and Z-parameters for circuit simulators,
network analysers and `scikit-rf`.

## Time convention

The engines use the time dependence $e^{-i\omega t}$, so a passive inductor has $Z=-i\omega L$
(`docs/physics/multiport_fullwave.md`, `axionfem.circuit`). The Hermitian part $(Z+Z^\dagger)/2$ is the dissipation,
and $\operatorname{Re} Z \ge 0$ for a passive port. Touchstone files and circuit tools use $e^{+j\omega t}$, where the
same inductor has $Z=+j\omega L$. A physical response is the same in both conventions:

$$
\operatorname{Re}\!\left[Z e^{-i\omega t}\right]=\operatorname{Re}\!\left[\overline{Z}\,e^{+j\omega t}\right],
$$

so the engineering-convention matrix is the element-wise complex conjugate of the engine's. There is no transpose, so
the port order and reciprocity ($Z=Z^T$, which the engine's bilinear port functional gives) are unchanged. The
conversion relations below are algebraic and hold in either convention, but $S$ computed from the engine's $Z$ is
$\overline{S}$ of the file's. Every function in `axionfem.touchstone` returns engineering-convention values. The
conversion functions take a required `convention` argument: `"axionfem"` conjugates the input, and `"engineering"`
uses it as given. A wrong sign would show up as an inductor whose $S_{11}$ turns counter-clockwise with frequency;
the tests check $\operatorname{Im} S_{11} > 0$ for an inductor below $|Z|=z_0$.

A sign check: a solve of the two-port frame design (`benchmarks/designs/stage1_twoport.json`) gives
$\operatorname{Im} Z<0$ and $L=-\operatorname{Im}Z/\omega>0$ for its inductive frame (for example $-2.8248\,\Omega$ at
1 MHz, $L=0.4496\,\mu$H; these numbers are not stored in a record). The exported file gives
$\operatorname{Im}Z=+2.8248\,\Omega$, and $L=\operatorname{Im}Z/\omega$ reproduces the solve's `L_H` to $10^{-12}$.

## Conversions

The reference impedances $z_{0,k}$ are real and positive, one value or one per port. With $D=\operatorname{diag}(\sqrt{z_0})$
and $z=D^{-1}ZD^{-1}$:

$$
S=(z-1)(z+1)^{-1},\qquad Z=D(1+S)(1-S)^{-1}D,\qquad Y=Z^{-1}.
$$

For real $z_0$ these are the power-wave and pseudo-wave scattering parameters, which coincide there. They agree with
`scikit-rf` (`s_def="power"`) to $10^{-12}$, also for per-port $z_0$. Complex reference impedances are refused,
because the two definitions differ for them. $S$ is symmetric exactly when $Z$ is. The network is passive exactly when
$\sigma_{\max}(S)\le 1$, which is equivalent to $(Z+Z^\dagger)/2\succeq 0$.

API (`axionfem.touchstone`; `circuit` re-exports `port_parameters`, `write_touchstone` and `read_touchstone`):

- `port_parameters(results, dense=False, surface_losses=False)` returns:
  - `f_Hz` (sorted) and `Z` ($n_f\times n\times n$, engineering convention);
  - `ports` (names in matrix order; port 1 is the pickup) and `port_details` (`toward`, region or boundary,
    terminals);
  - `source` and `notes`;
  - `Re_Z_unresolved_Hz` and `Re_Z_unresolved` (per output point);
  - `Z_error_ohm` (an absolute error estimate of Z);
  - `surface_loss_beyond_perturbation_Hz`.
- `write_touchstone(results, path, kind="S"|"Z"|"Y", z0=50, dense=False, surface_losses=False, passivity_tol=1e-9)`.
  It is also available as `Results.write_touchstone(path=None, ...)` and `circuit.write_touchstone`. The default
  path is `<workdir>/<design>[_dense]_<kind>.s<N>p`; the design name is sanitised to letters, digits, `_`, `-` and
  `.`.
- `write_touchstone_data(path, f_Hz, Z, *, convention, kind, z0, ports, port_details, comments)` takes a $Z$ from
  anywhere. `convention` is required, as for `z_to_s`.
- `read_touchstone(path, nports=None)` returns `f_Hz`, `kind`, `z0_ohm`, `data`, `S`, `Z`, `ports`, `comments`,
  `noise_rows_ignored`, `Z_undefined_Hz` and `S_undefined_Hz`.
- Conversions and checks: `z_to_s`, `s_to_z`, `z_to_y`, `y_to_z`, `equilibrated_condition`, `to_engineering`,
  `reciprocity_error`, `passivity`, `safe_stem`.
- Command line: `axf export-touchstone results.json [-o OUT.sNp] [--kind S|Z|Y] [--z0 50] [--dense] [--surface-losses]`.
  - The default output is `<results dir>/<design>[_dense]_<kind>.s<N>p`.
  - Exit status: 0 written and passive; 3 written but not passive; 2 nothing written, with the error on stderr.
    Every exception gives 2, never Python's 1.
- MCP: `export_touchstone(name, kind, z0, dense, surface_losses)` writes
  `runs/<name>/touchstone/<name>[_dense]_<kind>.sNp`, so exporting S and then Y keeps both files. Every entry point
  overwrites an existing file and says so (`overwrote` in the report).

## What is exported

- **Solved rows** (`dense=False`) come from `fullwave.sweep`: `Z_matrix_*`, or `Z_re_ohm`/`Z_im_ohm` for a 2D one-port.
  Each frequency is a full solve, sorted, and repeated frequencies are refused.
- **The adaptive model's dense curve** (`dense=True`) comes from `fullwave.adaptive.dense` (`circuit.fullwave_dense`).
  - It holds to the sweep's tolerance where the sweep converged. A sweep that did not converge is exported with a
    warning.
  - The rows' `Re_Z_unresolved` flags are carried over. `Re_Z_unresolved_Hz` lists the flagged solves, a dense point
    counts as flagged when a neighbouring solve is, and the file note says that the model was fitted to such rows.
- **Losses.** $Z$ contains what the solve contains: radiation, bulk dielectric and conduction loss, and Robin, PML and
  sheet terminations. It does not contain the perturbative conductor surface loss (`R_loss_ohm`), which the rows carry
  separately (`docs/physics/fullwave_surface_losses.md`). `surface_losses=True` adds its **resistive part** only:
  - For a one-port, it adds `R_loss_ohm` to $\operatorname{Re}Z$. This is the first-order series resistance that
    `Q_total` uses.
  - For a multiport, it adds `R_loss_matrix_ohm`. That matrix equals the impedance perturbation only for in-phase
    port fields, so it is refused when the matrix is complex or $Z$ has a resistive part.
  - **The surface reactance is not added.** That is the internal inductance of a normal metal, equal to $R_{\rm loss}$
    there, or the kinetic inductance of a superconductor, $X_s/R_s\approx 10^2$–$10^5$ times larger. Omitting it is
    consistent with the engine's own L and Q, but the file is not a full lossy-conductor model. A capacitor-tuned
    resonance would move by about half its linewidth for a normal metal, and more for a superconductor. The file
    note says so.
  - Rows with $R_{\rm loss}/|X| > 0.1$ (`R_loss_over_abs_X`, the engine's own limit) are warned about, listed in
    `surface_loss_beyond_perturbation_Hz` and noted in the file. There the perturbation is off by more than about 1 %.
    Near a self-resonance it grows like $1/\Delta^2$ while the true resistance stays bounded.
  - The adaptive model does not carry the losses, so `dense=True` with `surface_losses=True` is refused.

## The Touchstone file

- **Character set.** The file is written as US-ASCII, as the Touchstone 2.1 general syntax rules require.
  - In names, control characters (newlines included) become spaces.
  - Every other character outside 0x20–0x7E becomes `?`: non-ASCII letters such as µ or subscripts, and the
    Unicode line separators U+0085, U+2028 and U+2029.
  - So a name can neither inject a data line nor make the file unreadable.
  - The reader decodes bytes as Latin-1 (never fails) and splits lines only at CR/LF.
- **Comment block** (`!`). It holds:
  - the axionFEM version, the time convention and that $Z$ was conjugated;
  - the parameter type and the reference resistance;
  - the design, the source (rows or dense, with tolerance and convergence), and the loss and `Re_Z_unresolved` notes;
  - $\sigma_{\max}(S)$ over the band and the reciprocity error;
  - the port order with each port's polarity and terminals (`!   port k: name | toward [x, y, z]; region ...`).
    A reversed port flips the sign of its off-diagonal entries. `read_touchstone` reads the names back.
- **Option line.**
  - Touchstone v1.0: `# Hz S RI R 50` (or `Z`/`Y`).
  - A per-port reference resistance, for S-parameters only, uses the Version 1.1 form `# Hz S RI R r1 ... rn` with
    per-port power waves (Touchstone 2.1 specification, Version 1.1 option line; scikit-rf reads it).
  - Z and Y with per-port resistances are refused by the writer and the reader, because readers do not agree on
    their normalisation. A single $z_0$, or `z_to_s` in Python, covers those cases.
- **Data.**
  - Frequencies are strictly increasing, with 17 significant digits: a round trip is exact to the last bit for S.
  - A two-port block is `N11 N21 N12 N22` on one line, the v1 exception. Three ports and more are written row by
    row, at most four complex values per line, each row starting a new line.
  - The tests check the raw tokens on disk with non-reciprocal data, so that $S_{21}\neq S_{12}$ and a transposition
    shows.
- **Normalisation.** Z and Y are stored normalised, $Z/R$ and $YR$: in a Y file, 25 Ω at R = 50 is `2 0`. The
  Touchstone 2.1 specification says that the option line's reference resistance "defines … the normalization used on
  G-, H-, Y-, and Z-parameter data in Version 1.0 and Version 1.1 files", and the normalised admittance is
  $y = Y/Y_{\rm ref} = YR$. `read_touchstone` de-normalises it.
  - scikit-rf 2.2 (`skrf/io/touchstone.py:664-666` in 2.2.0, which multiplies g, h, y and z alike by z0) reads Z
    files correctly.
  - It multiplies a v1 Y file by $R$ instead of dividing, so its admittances come out $R^2$ too large. Exchange S or
    Z files with it.
- **Extension.** It must be `.s<N>p`, since readers take the port count from it.
- **`read_touchstone`** handles:
  - Hz/kHz/MHz/GHz, RI/MA/DB and the v1 default option line;
  - values spread over any number of lines;
  - the v1.1 per-port `R` for S.
- **What it refuses:**
  - data lines before the option line (v1.0/1.1 require the option line first);
  - tokens that are not decimal or scientific numbers (`nan`, `inf`, `1_0`, hex), and values that overflow;
  - Touchstone v2 keywords;
  - H and G parameters.
- **A missing option line** is warned about, because the specification requires one; the v1 defaults
  `# GHz S MA R 50` are then used.
- **The 2-port noise block** starts where a frequency is at or below the previous one. The remaining numbers must
  form rows of 5 with increasing frequencies, otherwise the file is refused (a repeated network frequency is not
  taken for noise data). The ignored rows are counted (`noise_rows_ignored`) and warned about.
- **Files whose Z or Y does not exist** are read: an ideal thru, an open, or a series element given as Y. `S` and the
  stored `data` are always returned. `Z` (and S of a Y file) is NaN where it does not exist, with a warning naming
  the file and the frequencies (`Z_undefined_Hz`, `S_undefined_Hz`). `z_to_s` and `s_to_z` name the singular
  matrices (e.g. `Z = -z0`).

## Checks reported, never repaired

- **Passivity.** $\sigma_{\max}(S)$ per frequency.
  - Frequencies where it exceeds $1+10^{-9}$ (`passivity_tol`) are listed in the report and raise a `RuntimeWarning`.
  - The data are written as computed: nothing is clipped, symmetrised or projected. The engine does not symmetrise
    $Z$ either (`multiport_fullwave.md`), so in a lossless structure $\sigma_{\max}$ exceeds 1 at the level of the
    solver's reciprocity error. In a closed lossless 3D solve $\operatorname{Re}Z$ is exactly 0 and the excess stays
    near $10^{-11}$.
  - In a radiating design an unresolved $\operatorname{Re}Z\approx-\rho|X|$ gives an excess of about
    $\rho\,2|x|/(1+x^2)$ with $x=X/z_0$, i.e. about $\rho$ at $|X|=z_0$. Violations at rows flagged
    `Re_Z_unresolved` are therefore listed apart (`passivity_violations_at_unresolved_Re_Z_Hz`), and the warning
    names a tighter `AXF_FW_RTOL` as the remedy. Violations at resolved rows point to the mesh.
- **Reciprocity.** The largest $\|Z-Z^T\|_F/\|Z\|_F$ over the band.
- **$Y=Z^{-1}$ is checked in two ways.**
  - `z_to_y` warns where the equilibrated condition number (of $D^{-1}ZD^{-1}$, $D=\operatorname{diag}\sqrt{|Z_{kk}|}$)
    exceeds $10^{12}$, i.e. where the ports are nearly dependent. A graded but well-determined matrix, such as a
    floating port at low frequency, is not flagged.
  - **Every export, whatever the kind, reports Z's own estimated relative error** (`Z_relative_error_max`, in the
    report and the file note), and the frequencies where it is 0.1 or more (`Z_poorly_determined_Hz`).
    - For the rows it is the normwise $\|\delta Z\|/\|Z\|_2$ from the error indicator, plus the solve's true
      relative residual.
    - The indicator's `Z_signed_error_estimate_ohm` is preferred where given. The unsigned `Z_error_estimate_ohm`
      counts the electric and magnetic parts with the same sign and is called an upper estimate: near a
      self-resonance it lies above the error (`fullwave_error_indicator.md`).
    - For the dense curve it is the model's estimated relative error.
    - Where nothing is known (a direct 2D solve) the report and the file say "no error estimate", and a Y export says
      "Y not checked".
  - **A Y export warns only where the inversion amplifies an error that is otherwise small** (`Y_ill_determined_Hz`):
    relative error × equilibrated condition number > 0.1 while the relative error itself is < 0.1.
    - Where Z's own error is 0.1 or more, Z is the problem; that is reported for every kind, and Y is not blamed.
    - The dense curve's barycentric model shares one denominator over all entries, so its error near a pole moves
      the entries together (δZ ≈ εZ). That error is therefore taken as proportional, which inversion does not amplify
      (δY = −εY).
    - A one-port is never amplified ($\kappa = 1$). Its Y has Z's relative error, which is large near a series
      resonance ($Z\to 0$) and harmless at a one-port antiresonance ($Y\to 0$).
  - **Real exports are quiet.** The S, Z and Y exports of the TEM line and of the two-port frame's adaptive rows and
    dense curve raise no warning (benchmark gate). The frame's rows report Z poorly determined at all 9 rows from the
    unsigned indicator (relative estimate up to 2.2e3), which is an upper estimate on that coarse mesh.
- **The report's `warnings`** list every warning raised on the way (non-convergence, surface-loss limits, passivity,
  Y), and each is also emitted as a Python warning.

## Validation

`tests/test_touchstone.py` (seconds; in the quick subset `tests/premerge.txt`) covers:

- **Closed forms:**
  - a series RLC one-port: S and Y to $10^{-13}$, and $S_{11}=(R-z_0)/(R+z_0)$ at resonance;
  - matched lossless and lossy uniform lines: $S_{11}=0$ and $S_{21}=e^{-\gamma l}$, with
    $\sigma_{\max}=e^{-\alpha l}$;
  - a mismatched line against the cascade formula;
  - per-port $z_0$.
- **Records:** the two-port frame's multiport record.
- **Unclipped passivity warning.**
- **Round trips:** 1, 2, 3 and 5 ports, for S, Z and Y.
- **Layout and normalisation, with non-reciprocal data:**
  - the raw on-disk tokens of a 2-port (S21 ≠ S12) and of a 3-port with distinct entries;
  - the raw normalised values (`0.5 0` in a Z file, `2 0` in a Y file for 25 Ω at R = 50);
  - a hand-written isolator-like 2-port;
  - S of a non-reciprocal Z against Pozar's Table 4.2.
- **The reader:**
  - units, formats and the noise block, with its validation;
  - files without Z (thru, open, series Y);
  - per-port v1.1 `R`;
  - refusal of data before the option line.
- **Interfaces and errors:** the dense curve, the command line, the MCP tool and the refusals.
- **scikit-rf cross-check**, when it is importable: reading our files, our reader on its writer, and per-port power
  waves.

`benchmarks/touchstone_check.py` adds engine runs (the numbers are in the record):

- `tem_short`: the shorted parallel-plate TEM line in 3D with an adaptive sweep over 100–400 MHz, through its
  quarter-wave pole, against $S=(jZ_c\tan kl-z_0)/(jZ_c\tan kl+z_0)$. Both the rows and the dense curve are checked.
- `coax2d_lossy`: the axisymmetric coax with a lossy dielectric against its transmission-line closed form, with
  $|S|<1$.
- `stage1_adaptive`: the two-port frame with an adaptive sweep: reciprocity and passivity of the rows and
  of the dense curve.
- `records`: the committed two-port frame cases (lossless, reversed port, lossy, floating port).
- `skrf`: every file written by the other cases *in this run*, read by scikit-rf.
- Gates (`evaluate` in the script; `tests/test_touchstone.py` checks them on the committed record):
  - `tem_short`: |S − S_exact| < 5e-5 and ||S| − 1| < 1e-9;
  - `coax2d_lossy`: |S| < 1 and |S − S_exact| < 5e-3;
  - lossless cases: σ_max − 1 < 1e-9;
  - `stage1_adaptive`: S asymmetry < 1e-8;
  - lossy record: 1 − SᴴS positive definite;
  - S, Z and Y exports of `tem_short` and `stage1_adaptive`, rows and dense: no warning;
  - scikit-rf: S identical (difference exactly 0) for every file this run wrote. A run without scikit-rf records
    the skip in `gates_skipped` instead of passing silently.
- A full run writes a fresh record with `git_head`. A partial run replaces only its cases, each stamped with its
  own date, `git_head` and version.

Results (the committed record: 4 MPI ranks, z0 = 50 Ω; every gate holds):

| case | points | σ_max(S) | check |
|---|---|---|---|
| `tem_short` rows | 7 solves | 1 to 2e-16 | \|S − S_exact\| ≤ 1.8e-5 (the solves' Z error); Im S11 > 0 below c/8l |
| `tem_short` dense | 400 | 1 + 3.5e-14 | \|S − S_exact\| ≤ 1.8e-5 through the quarter-wave pole (7 solves, estimate 4.8e-7) |
| `coax2d_lossy` | 6 | 0.99999999936 | \|S − S_exact\| ≤ 2.4e-3, Z to 4.1e-3 (the mesh's, as in tests/test_fullwave.py); \|S\| < 1 at every point |
| `stage1_adaptive` rows | 9 solves | 1 + 2.1e-11 | \|\|Z − Zᵀ\|\|/\|\|Z\|\| ≤ 8.9e-11, S asymmetry 4.1e-11 |
| `stage1_adaptive` dense | 400 | 1 + 1.3e-10 | Z asymmetry 4.5e-10, S 2.6e-10; poles 121.2 and 139.2 MHz |
| record, lossless / reversed | 2 | 1 + 3.1e-12 | asymmetry 6.2e-12; reversed port 2 flips S12 only |
| record, lossy | 1 | 1 − 3.3e-8 | min eig(1 − SᴴS) = 6.6e-8 > 0; set by the resistance matrix's smallest eigenvalue, 0.82 µΩ (`resistance_min_eigenvalue_ohm`: the two ports are almost perfectly coupled), not by Re Z11 = 3.7 mΩ |
| record, floating port | 1 | 1 + 4e-13 | asymmetry 8e-13 |

No passivity violation at the default tolerance 1e-9. In the lossless cases σ_max − 1 sits at the level of the
solver's reciprocity error, as expected for an unsymmetrised Z. scikit-rf reads every file to the same S (0
difference) and Z (≤ 5.7e-13 relative), and its own `is_reciprocal` agrees. Its passivity is not a separate check:
it would be our SVD of an identical S.

## Limits

- Touchstone v2 (mixed-mode data, explicit matrix format, keyword sections) is not written or read. v1.1 per-port
  reference resistances are handled for S only.
- Only S, Z and Y are handled (no H or G).
- The dense curve is a fitted model. Its passivity is checked on the dense points only, not between them.
- Exporting does not certify that the solves are converged in mesh and order: S inherits the error of Z.
