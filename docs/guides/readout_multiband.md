# Readout circuits, SQUID chains and multi-band scans

In a lumped-element experiment a readout circuit turns the axion-induced flux in the pickup into a current measured
by a SQUID. A capacitor tunes the pickup and the SQUID input coil to resonance. The noise combines the thermal noise
of the resistive elements with the amplifier's imprecision and its backaction on the coil. This guide computes
readout noise, attaches a dc SQUID amplifier chain, and plans scans across several tunable bands.

## From pickup flux to readout noise

The signal transfer and the noise referred to the pickup's electromotive force are

$$
H_s(\omega)\equiv\frac{\Phi_{\rm SQ}}{\mathcal E_{\rm pickup}},
\qquad
N_V(\omega)\equiv\frac{S_\Phi^{\rm tot}(\omega)}{\lvert H_s(\omega)\rvert^2}.
$$

Here $\Phi_{\rm SQ}$ is the SQUID flux and $S_\Phi^{\rm tot}$ its total noise power spectral density (PSD). $H_s$
has units of $\mathrm{Wb/V}$ and $N_V$ of $\mathrm{V^2\,s}$. These PSDs are half the usual one-sided per-Hz PSDs
([readout circuit](../physics/readout_circuit.md)).

With NumPy and SciPy installed, we tune to 1 MHz. `Q` sets the pickup loss; the default amplifier has uncorrelated
noise and is matched for scan rate. We compute the effective circuit parameters and the input-referred noise:

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

The first line gives the whole-loop inductance in henries, the loaded quality factor and the effective noise
temperature in kelvin; the second the input-referred noise. Whole-loop quantities enter the scan reach: input-coil
and lead inductances contribute even though they collect no signal. The model neglects the SQUID input impedance; the
resonance reduction applies locally around an isolated pole. `series_readout` also takes the pickup's Ψ
(`Psi_SI=res.Psi`) and its inductance from a solve, and `readout.draw()` draws the circuit (the `draw`
extra).

## dc SQUID amplifier chains (beta)

`axionfem.squid` models published dc SQUID amplifier chains, a first-stage dc SQUID followed by a SQUID array and a
preamplifier, with the noise of every stage referred to the first-stage flux
([dc SQUID chains](../physics/squid_chain.md), after arXiv:2504.20398). It is beta. A chain replaces the default
amplifier of a readout:

```python
from axionfem import squid

print(squid.presets())                    # 'C1', 'C1+low+RT', ..., 'C1+high+cryo'
chain = squid.chain("C1+low+RT")
print(chain.eta(f0))                      # minimum noise temperature in units of hbar*omega/(2 k_B)
readout.squid = chain.squid_for_resonator(f0, 1e6, 1e-6 + 1e-7, 0.03)   # f0, Q, loop inductance, T
n = readout.noise_psd(2 * pi * f0)
print(n["N_V_thermal_V2_s"], n["N_V_backaction_V2_s"], n["N_V_imprecision_V2_s"], n["N_V_V2_s"])
```

`squid_for_resonator` couples the chain at its scan-optimal input; the noise PSD then splits into the thermal,
backaction, imprecision and correlation parts. `chain.draw()` draws the chain.

## Tuning across several bands

!!! warning "Beta"
    `axionfem.multipole` (`multipole.source_table`, `multipole.multiband_scan`, `scan.multipole_scan`,
    `scan.reach_2211`) is beta ([specification](../physics/multipole_scan.md)).

When a detector's size approaches an electromagnetic wavelength, its impedance can contain several resonances. A
multi-band scan distributes time among tunable frequency intervals. In the engineering convention $Z=R+iX$, positive
$X$ is inductive; a series capacitor or inductor cancels the source reactance:

$$
C=\frac{1}{\omega X}\quad(X>0),\qquad
L_t=-\frac{X}{\omega}\quad(X<0).
$$

The effective inductance is half the slope of the total tuned reactance with respect to angular frequency, including
the tuning element.

We start with an ideal 100 nH pickup and an illustrative coupling of $10^{-6}\,\mathrm{H\,T\,m^2}$. The same calls
accept sampled full-wave responses. We tabulate the pickup response and plan a multi-band scan to find the time
required for the target reach:

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

The voltage column is the open-circuit amplitude per unit SI axion source coefficient, identified by
`source_unit="kappa_SI"`. For [COMSOL data](comsol.md), give the source normalization and any port load. axionFEM's
own impedance data use `convention="exp(-iwt)"`, and `multipole.table_from_results` builds a table from a full-wave
solve. Here `Q` is the total resonator quality factor; alternatively, `Q_mqs` adds intrinsic loss to the table's
resistance.

`bands` lists intervals in Hz and their tuning-element type. Defaults allow 30–300 pF capacitors and 10–500 nH
inductors, removing 2.5 MHz at each band edge. `t_req_s` is the time to reach $10^{-15}\,\mathrm{GeV^{-1}}$ across
the accepted bands; `g_target_allocation_GeV` gives the reach after rescaling to the requested live time. The methods
are those of arXiv:2211.00008 and arXiv:2302.14084 ([multi-band scans](../physics/multipole_scan.md)).

## Finite-band scans

`multiband_scan` treats each band through its local pole. `axionfem.scan.finite_band` instead combines the response
across resonances at each fixed tuning with a frequency-dependent readout, within a capacitor-tunable interval, and
evaluates the exact circuit noise ([finite-band validation](../validation/finite_band.md)). Port impedances of a
full-wave solve can also be exported to external circuit tools as Touchstone files
([Full wave](fullwave.md#touchstone-export)).

## Next steps

- [Walkthrough notebook](../../examples/readout_squid.ipynb): readout of a solved detector, SQUID chains and the
  noise budget.
- [API reference](../api.md): every argument of the circuit, SQUID and scan functions.
