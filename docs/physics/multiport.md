# Multiport circuit closure

`axionfem.multiport.MultiportCircuit` extends the existing `Circuit` MNA
implementation with a complex reciprocal impedance block. `MultiportReadout`
uses the existing `Squid` transfer and noise conventions while preserving a
coherent vector of axion EMFs. The FEM extraction is documented separately;
this layer neither fits frequency data nor assigns a modal quality factor.

For exp(-i omega t), Z_L=-i omega L, Z_C=i/(omega C). A branch obeys

\[
 V_p-V_m=\sum_k Z_{jk}I_k-\mathcal E_j.
\]

The FEM emits Z_jk=-integral E_k dot j_j for unit drives, and its passive
device voltage is v_j=-integral E dot j_j. Circuit **p→m follows the gap
current profile**, opposite current entering that device port. Thus circuit
branch voltage is -v_j and branch current is minus the incoming FEM current.
With E_a driven by J_a=kappa_a B0, put E_j=V_a,j=-i omega kappa_a Psi_j in
the circuit's EMF mapping. The open branch has V_p-V_m=-V_a,j; the load sees
+v_j. Independent terminations carry I_c=(Z+Z_load)^-1 V_a. This orientation
is tested against actual direct-source and directly loaded FEM voltages by
the research workspace's `code/check_fem_circuit_bridge.py`.

```python
import numpy as np
from axionfem.circuit import Squid
from axionfem.multiport import MultiportCircuit, MultiportReadout

c = MultiportCircuit()
# Z_of_w returns the full matrix in this exact order and orientation.
c.multiport("field", [("cap", "gnd", "a"), ("read", "gnd", "b")],
            Z_of_w, T_K=0.030)
c.capacitor("tune", "a", "gnd", C_increment_F)
c.inductor("L_in", "b", "gnd", 100e-9)
sq = Squid(M_H=1e-9, input_coil="L_in", S_Phi=S_Phi, S_J=S_J,
           eta_A=10)
r = MultiportReadout(c, sq,
                    lambda w: dict(zip(("cap", "read"), -1j*w*Psi_of_w(w))))
signal_flux = r.signal(2*np.pi*120900)
noise = r.noise_psd(2*np.pi*120900)
```

The example assumes `Z_of_w`, complex `Psi_of_w`, `C_increment_F` and SQUID
noise properties were supplied/validated. It is not a physical capacitor
or readout prescription. A meshed bus or minimum capacitance is already in
Z; adding it again in the circuit double counts it. Capacitor settings and
loss elements remain fixed while evaluating a noise spectrum at one tuning.

The matrix is added to the branch block of MNA, so all existing resistors,
capacitors, ideal transformers and real mutual inductors remain usable. The
solver reuses one factorization for a batch of EMF patterns. Reciprocity is
Z=Z^T, not Z=Z^dagger. No circuit solve silently symmetrizes the supplied Z.

## Dissipation and noise

At a uniform bath temperature use the full Hermitian dissipative matrix

\[
 R_H=(Z+Z^\dagger)/2,\qquad
 S_V=\hbar\omega[2n(\omega,T)+1]R_H.
\]

Off-diagonal correlations matter. For several temperatures supply
`baths=[LossBath(name, R_b, T_b), ...]`, where each R_b is positive
semidefinite, in ohms, and their sum is R_H. Matrices or functions of omega
are allowed. The accounting test is relative to the **loss**, including weak
loss eigendirections; a large reactance cannot hide an omitted resistor.
No participation is inferred from a DC field or a stated total Q.

Thermal noise at a linear output with row transfer h is h S_V h^dagger.
The implementation evaluates it as a sum of nonnegative eigenfactor
contributions. An eigenvalue below -8 epsilon_machine times the loss norm
is rejected. Eigenvalue roundoff within that bound is projected to zero in
the covariance only; `thermal_noise(...)["loss_diagnostics"]` and
`loss_diagnostics(omega)` report the raw minimum eigenvalue and projection
norm. No finite-size FEM passivity repair is performed. Inputs with numerical
reciprocity defects can fail this strict test even if a lossless reference
should be passive. Tighten their solves or make an explicit, bounded,
provenance-recorded reciprocal/passive reduction upstream. Passivity at sample
points does not prove passivity of a fitted continuous-frequency model.

The source covariance is coherent, proportional to V_a V_a^dagger; the
readout solves the full complex vector before squaring its amplitude. SQUID
imprecision is added at the flux output; backaction enters as the series EMF
+i omega M J in the designated input-coil branch (the mutual stamp Z=-i omega M
seen from the SQUID loop; `readout_circuit.md` Sec. 4.2 and `squid_chain.md`
Sec. 4). PSDs retain the existing Foster
normalization, half the usual one-sided-in-Hz PSD.

`reference_noise(omega, Psi_ref)` supplies an exact scalar noise referral
for the existing finite-band kernel **only where the coherent signal is
nonzero**. A signal zero raises an explicit error: the current C++
`NoiseCurve` rejects infinite noise. `inverse_reference_noise` returns zero
at that point. `output_snr_rate` directly integrates the output signal/noise
PSD using supplied angular-frequency quadrature and handles exact coherent
zeros. It does not interpolate or certify the frequency grid, and the C++
finite-noise table still cannot represent a zero by an infinite referred PSD.

## Validation and limitations

`tests/test_multiport.py` compares with independent two-loop Kirchhoff
equations, power balance, physical resistor-mixture noise, two temperatures,
the existing scalar SQUID circuit, port reversal and coherent cancellation.
It rejects nonreciprocal/nonpassive matrices, missing weak noise channels,
and an invalid scalar referral at a zero. These are circuit tests, not FEM
mesh convergence or a detector design qualification. Small-model FEM and
source checks are in [validation/multiport_stage1.md](../validation/multiport_stage1.md).

Remaining: frequency-continuous passive fitting, loaded eigenmode completeness,
physical capacitor motion/ESL qualification, loss participation, prescribed
mode-Q calibration, zero-aware finite-band scan adapter, SQUID input loading
qualification, and hardware-limited matching. The single-source
`Readout.equivalent` reduction is not silently reused for an arbitrary source
vector. Shared `Circuit` behavior is unchanged. `axionfem.multiport_components`
adds exact line, physical symmetry, modal-energy loss calibration and component-source helpers
in [multiport_components.md](multiport_components.md); their analytic tests
do not substitute for detector/component FEM qualification.
