# Multiport component and energy helpers

`multiport_components.py` uses `exp(-i omega t)`, SI quantities and peak phasors.
These helpers implement circuit identities; their tests do not substitute for
FEM geometry, component-state, source-quadrature or mesh-convergence checks.

For the odd slit/even midplane sector, let full currents be upper modules,
lower modules, centered readout. The physical reduced coordinates are the
pair-total module currents and actual readout current. The lift is
`H = [[I/2,0],[I/2,0],[0,1]]`. Hence `Z_eff=H.T Z_full H`,
`Psi_eff=H.T Psi_full`, and `G_eff=H.T G_full H`. Quarter currents are half
these physical currents, giving `Z_eff=Z_q`, `Psi_eff=2 Psi_q`, `G_eff=G_q`.
Full energy at fixed mirrored quarter fields is four times quarter energy.
A module pair has `C_eff=2 C_module`, `ESL_eff=ESL_module/2`; the readout
impedance is unchanged. Thermal covariance follows the physical impedance's
Hermitian part, not the coherent-source image factor.

A uniform lossless line has entering currents at both ends. With
`theta=omega sqrt(L_total C_total)` and `Z0=sqrt(L_total/C_total)`,
`Z_line=i Z0 [[cot(theta),csc(theta)],[csc(theta),cot(theta)]]`.
`LosslessTransmissionLine` supplies this reciprocal matrix, exact terminated
input impedance and distributed fields. Its energy matrix is
`G_line=-d Im(Z_line)/d omega /4`; direct distributed energy quadrature,
open/short/matched loads and generic circuit stamps are tested. The open-port
impedance representation is singular at half-wave poles and explicitly refuses
them; a different terminal representation is needed there.

`calibrate_modal_loss` takes actual modal currents and a positive energy
matrix from every FEM field column. After including omitted component, line
and input-coil energy, it scales a specified positive resistance participation
so `omega U/(I^dagger R I/2)=Q`. The resistor realization and matching bath
temperature stay fixed through that tuning's bandwidth. It must not be used
to recalibrate loss at every off-center frequency. A fixed-capacitor circuit
with a distributed line and input coil verifies the resulting linewidth.
Existing physical loss must be accounted before assigning an extra reservoir.

`fixed_squid` constructs positive imprecision/backaction PSDs rather than
relying on `eta_A` alone. The default is one-sided flux amplitude
`1 microPhi0/sqrt(Hz)`, `eta_A=10`, `M=1 nH`, and zero imprecision/backaction
correlation. PSDs use the package's Foster convention:
`S_Phi=(1e-6 Phi0)^2/2`, `S_J=(10 hbar/2)^2/S_Phi`. These are a specified
noise scenario, not a measured amplifier. Hold PSDs and physical coupling
fixed throughout each tuning's bandwidth.

`MultiportReadout.output_snr_rate` directly integrates output signal/noise,
including exact coherent zeros. Given a normalized angular-frequency line
shape, `S_signal=(pi/2) kappa^2 (omega/omega_a)^2 |Phi/kappa|^2 L(omega)`;
the weak-signal information rate is `integral(S_signal/S_noise)^2 d omega/(2pi)`.
It performs no interpolation and does not establish grid adequacy. Source
zeros contribute zero information rather than infinite referred noise.

Component elimination must include `q_eff=q_g-A_gc A_cc^-1 q_c` as well as
the admittance Schur complement. `eliminate_component` checks the algebra;
`omitted_overlap_bound` evaluates a Cauchy bound only when supplied the actual
electric energy and local integral of `B0^2`. Removing PEC plates does not
remove overlap in component vacuum. `fit_series_capacitor` extracts candidate
C/ESL from at least three de-embedded samples and reports residuals. A low
residual is not an internal-mode search or three-state hardware qualification.
