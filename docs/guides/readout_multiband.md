# Readout and multi-band scans

In lumped-element experiments, a readout circuit turns axion-induced pickup flux into a current measured by a
SQUID. A capacitor tunes the pickup and input coil to resonance. The circuit
combines thermal noise from resistive elements with the amplifier's measurement
noise and backaction on the coil. In this guide we compute readout noise and
plan multi-band scans.

## From pickup flux to readout noise

The signal transfer and noise referred to the pickup's electromotive force are

$$
H_s(\omega)\equiv\frac{\Phi_{\rm SQ}}{\mathcal E_{\rm pickup}},
\qquad
N_V(\omega)\equiv\frac{S_\Phi^{\rm tot}(\omega)}{\lvert H_s(\omega)\rvert^2}.
$$

Here $\Phi_{\rm SQ}$ is the SQUID flux and $S_\Phi^{\rm tot}$ its total noise
power spectral density (PSD). $H_s$ has units of $\mathrm{Wb/V}$ and $N_V$ of
$\mathrm{V^2\,s}$. These PSDs are half the usual one-sided per-Hz PSDs.

With NumPy and SciPy installed, we tune to 1 MHz. `Q` sets the pickup loss; the default amplifier has
uncorrelated noise and is matched for scan rate. We compute the effective circuit
parameters and input-referred noise to characterize the readout.

```python
from math import pi
from axionfem.circuit import series_readout

f0 = 1e6
readout = series_readout(
    L_p_H=1e-6, L_in_H=1e-7, f_Hz=f0,
    Q=1e6, T_K=0.03, M_H=1e-9,
)
eq = readout.equivalent(f_Hz=f0)
print(eq["L_eff_H"], eq["Q_loaded"], eq["T_eff_K"])
print(readout.noise_psd(2 * pi * f0)["N_V_V2_s"])
```

The first line gives whole-loop inductance in henries, loaded quality factor,
and effective noise temperature in kelvin; the second gives input-referred
noise. We use whole-loop quantities for [scan reach](../../examples/lumped_element_demo.ipynb):
input-coil and lead inductances contribute even though they collect no signal.
The model neglects SQUID input impedance; the resonance reduction applies
locally around an isolated pole.

## Tuning across several bands

The multi-band and multipole scans (`axionfem.multipole`) are beta.

When a detector's size approaches an electromagnetic wavelength, its impedance
can contain several resonances. A multi-band scan distributes time among tunable
frequency intervals. In the engineering convention $Z=R+iX$, positive $X$ is
inductive; a series capacitor or inductor cancels the source reactance:

$$
C=\frac{1}{\omega X}\quad(X>0),\qquad
L_t=-\frac{X}{\omega}\quad(X<0).
$$

The effective inductance is half the slope of the total tuned reactance with
respect to angular frequency, including the tuning element.

We start with an ideal 100 nH pickup and an illustrative coupling of
$10^{-6}\,\mathrm{H\,T\,m^2}$. The same axionFEM calls accept sampled full-wave responses. We tabulate the
pickup response and plan a multi-band scan to find the time required for the
target reach.

```python
import numpy as np
from axionfem import multipole

f = np.linspace(10e6, 150e6, 501)
omega = 2 * np.pi * f
table = multipole.source_table(
    f, 1j * omega * 100e-9, omega * 1e-6,
    convention="exp(+jwt)", source_unit="kappa_SI",
)
plan = multipole.multiband_scan(
    table, Q=1e5, live_time_yr=1.0,
    g_target=lambda f: np.full_like(f, 1e-15), rho_GeV_cm3=0.3, Q_a=6.8e5,
)
print(plan["bands"])
print(plan["t_req_s"])
```

The voltage column is the open-circuit amplitude per unit SI axion source
coefficient, identified by `source_unit="kappa_SI"`. For [COMSOL data](comsol.md),
we specify source normalization and any port load. axionFEM impedance data use
`convention="exp(-iwt)"`. Here `Q` specifies total resonator quality factor;
alternatively, `Q_mqs` adds intrinsic loss to the table's resistance.

`bands` lists intervals in Hz and their tuning-element type. Defaults allow
30–300 pF capacitors and 10–500 nH inductors, removing 2.5 MHz at each band edge.
`t_req_s` is the time to reach $10^{-15}\,\mathrm{GeV^{-1}}$ across accepted bands;
`g_target_allocation_GeV` gives reach after rescaling to the requested live time.

## Exporting port parameters to circuit tools

A full-wave sweep's port impedances can be written as a Touchstone file and simulated with an external circuit tool
(SPICE, ADS, Microwave Office, `scikit-rf`). The engine's $e^{-i\omega t}$ impedances are complex-conjugated into the
$e^{+j\omega t}$ convention these tools assume, so an inductor reads $Z=+j\omega L$:

```python
res = design.solve()                                   # a design with a fullwave section
report = res.write_touchstone("pickup.s2p", kind="S", z0=50)   # the solved rows
res.write_touchstone("pickup_dense.s2p", dense=True)   # an adaptive sweep's dense model curve (beta)
print(report["sigma_max"], report["passivity_violations_Hz"], report["reciprocity_error"])
```

From the command line, run `axf export-touchstone runs/NAME/results.json --kind S --z0 50 [--dense]`. It exits with
0 when the file was written and is passive, 3 when it was written but is not passive, and 2 on an error. The file is
Touchstone v1 with option line `# Hz S RI R 50`, increasing frequencies and a comment giving the port order and
polarity (port 1 is the pickup).

The report gives $\sigma_{\max}(S)$, which must not exceed 1 for a passive structure. A violation is reported as a
warning and the data are written as computed, never clipped. The report also gives the asymmetry of $Z$, which for a
reciprocal design is zero to solver precision (about $10^{-11}$).

The perturbative conductor surface loss is not part of $Z$. `surface_losses=True` adds its resistive part to the solved
rows, but not the surface reactance. `axionfem.touchstone.read_touchstone` reads files back. Per-port reference
impedances are written for S-parameters only, as a Touchstone v1.1 option line. Z and Y do not depend on z0: export
them with one z0, or compute per-port S with `z_to_s`. Details are in
[S-, Y- and Z-parameters and Touchstone files](../physics/touchstone.md).

`multiband_scan` uses local poles. axionFEM provides `scan.finite_band` to combine responses across resonances at
each fixed tuning with a frequency-dependent readout within a capacitor-tunable
interval. Further reading: [dc SQUID readout chains](../../examples/squid_chain_demo.ipynb) (beta)
or the [API reference](../api.md).
