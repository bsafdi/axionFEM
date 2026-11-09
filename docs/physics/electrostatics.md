# Electrostatics: capacitance matrices and the readout-gap capacitance

The design key `electrostatics` (`{"boundary": "open" | "grounded", "notes"}`, [design_format.md](../design_format.md))
enables a capacitance analysis in both engines. It solves div(ε grad φ) = 0 with every conductor body
(`superconductor` and `conductor` materials) an equipotential and reports, under `results["electrostatics"]`:

- `bodies` and `capacitance_F`, the Maxwell capacitance matrix in farads;
- `capacitance_to_ground_F` and `pairs`, the equivalent lumped network: the capacitor between i and j is −C_ij with
  the others grounded, and −Ĉ_ij with them floating (Ĉ = C_PP − C_PF C_FF⁻¹ C_FP, the exact reduction);
- `checks`, labelled by what each one tests (`checks.what_each_check_tests`), and `solver` diagnostics.

A `port` (poloidal) pickup also gets `pickup.port_capacitance`: the readout-gap capacitance and a lumped
self-resonance estimate (below). A design with only an `electrostatics` section needs no magnet; its Ψ, L_p and 𝓕
are zero. Python: `Results.electrostatics`, `Results.C_matrix`, `Results.port_capacitance`. Validation:
[axisymmetric](../validation/electrostatics.md), [3D](../validation/electrostatics3d.md); tests
`tests/test_electrostatics.py`, `tests/test_electrostatics3d.py`.

## Axisymmetric

Code: `cpp/include/axionfem/electro.hpp`, `cpp/src/axi_electro.cpp` (`ElectroProblem`,
`AxiModel::electrostatics_analysis`). Weak form on the meridian half-plane, per radian, on the H1 space of the
magnetoquasistatic engine:

    a(φ, v) = ∫ ε r (∂_r φ ∂_r v + ∂_z φ ∂_z v) dr dz + b_DtN(φ, v),
    W = 2π · ½ φᵀKφ  (J),      Q_k = 2π Σ_{i tied to k} (Kφ)_i  (C).

1. **No axis condition.** φ is a scalar: the weight r makes r = 0 a natural boundary, and a conductor touching the
   axis is an ordinary equipotential (unlike the A_φ problem, where ψ = rA = 0 is imposed on the axis).
2. **Conductors by elimination.** Each connected conductor body (zero-thickness `curves` included; regions touching
   at a corner share a whole circle of contact and form one body) is one master degree of freedom, either
   prescribed (Dirichlet) or floating with a prescribed net charge. A floating master's row is the natural condition
   Σ(Kφ) = Q/2π, so "zero net charge" needs no constraint equation.
3. **Exact exterior map** (`boundary: "open"`). Outside the domain sphere R_b,
   φ = Σ_{n≥0} a_n (R_b/ρ)^{n+1} P_n(cos θ), which gives the exterior energy W_ext = 2π·½ Σ_n c_n p_n² with
   p_n = Σ_b φ_b ∫N_b P_n r dl/R_b and c_n = (ε_out/R_b)(n+1)(2n+1)/2. The n = 0 term carries the net charge to
   infinity (it is absent from the magnetostatic map, whose multipoles start at n = 1), so an isolated sphere gives
   C = 4πε₀a exactly in the continuum. 80 multipoles. `boundary: "grounded"` fixes φ = 0 on the sphere instead. The
   map assumes a uniform permittivity outside; a dielectric reaching the boundary falls back to ε₀ with a warning.
4. **Charges from the reactions.** The Schur complement stays implicit, so the matrix is not symmetric by
   construction and its symmetry measures the accuracy of the linear solves. One factorization serves all bodies.
5. **Islands.** A dielectric region enclosed only by insulating walls has no reference potential; such components
   are pinned to zero and reported as `solver.pinned_dofs`.

**Checks.** Two probe the discretization: `charge_gauss_*` (the reaction charges against the strong ∮εE·dA over
the conductor surfaces) and `exterior_flux_*` (the strong flux −∮ε∂φ/∂n through the outer semicircle against the
net charge; the DtN map imposes the exterior Neumann data only weakly, so this difference converges with h and p).
Four are algebraic identities of the discrete system that test the solves: `symmetry_*`, `energy_*` (per column and
at a mixed potential vector), `floating_solve` (a real floating-conductor solve against the matrix reduction) and
`exterior_monopole_*` (2π·1ᵀKφ equals the exterior monopole charge identically).

## The readout-gap capacitance

The port of a poloidal pickup is a gap between two faces of **one** conductor body (a CORE-type cavity is a "C" in
the meridian plane, a coaxial solenoid is shorted at the top): the body is continuous around the port, so the
geometry alone defines no capacitance across the readout. Driving the two terminal faces while the rest of the same
body is an equipotential makes the energy diverge logarithmically at the port corners. Two lumped estimates are
reported:

- `C_gap_port_region_F`: the port region alone, the two terminal faces at V = ±½, natural (insulating) conditions on
  the faces open to the cavity;
- `C_gap_F`: the same two electrodes on the **whole** domain, with the rest of their own body insulating (n·D = 0)
  and every other conductor floating. This adds the fringing field at the two mouths of the gap.

**The one rigorous relation.** With the terminals driven at ±½ the full-domain field energy is
W_full = ½C_gap + ⅛(C_A,ground + C_B,ground). Restricted to the port subdomain, the full-domain solution is
admissible for the region-only problem and the energy density is positive, so

    C_gap_port_region ≤ C_gap + ¼(C_A,ground + C_B,ground),

exact for the discrete problems and emitted as `checks.dirichlet_bound`. It is loose; C_gap ≥ C_gap_port_region is
an empirical observation, not a theorem.

**Self-resonance estimates.** `self_resonance_Hz` = 1/(2π√(L_p C_gap)) and `self_resonance_from_port_region_Hz` use
the two capacitances; a warning fires when the estimate falls inside the scan band. Neither is a bound: C_gap is only
the gap's share of the structure's capacitance, which raises the estimate, while a uniform line resonates at
(π/2)/(2π√(L C_total)), which raises the true pole. The estimate exceeds the first pole only while
C_gap/C_total < (2/π)² = 0.405. For the solenoid and core templates C_gap/C_eff is 0.09 and 0.16, and the lumped
value is 3.4× and 2.5× the first pole of L(ω). With a `fullwave` sweep that brackets the pole, `vs_fullwave` reports
that pole, the ratio and C_eff = 1/((2πf)²L_p) (axisymmetric engine only).

## 3D

Code: `cpp/src/model3d_electro.cpp` (`Model3D::electrostatics_analysis`). The design key and the result keys are those
of the axisymmetric solver, so a design revolved with `axionfem.convert.to_3d` can be solved either way. H1 of order
`mesh.order` on the MPI mesh of the magnetoquasistatic solve:

    a(φ, v) = ∫ ε_eff grad φ · grad v dV,      W = ½ φᵀKφ,      Q_i = u_i · (Kφ),

with u_i the indicator vector of conductor body i.

1. **Unit-potential Dirichlet solves.** The Maxwell matrix is defined by its unit-potential columns: 1 V on body j,
   0 V on the others. The elimination is done once, and the columns reuse one BoomerAMG setup (PCG). A floating
   conductor is recovered exactly from the matrix, so there is no separate floating solver; a prescribed nonzero
   charge on a floating body is not implemented in 3D.
2. **Charges as one global reduction** (Q_i = u_i·(Kφ) with the un-eliminated K), so the matrix does not depend on
   the rank count beyond the solver tolerance.
3. **The open boundary is the infinity shell** that the 3D mesher always builds, [R₁, R₂ = 1.25 R₁] mapped onto
   [R₁, ∞) by f(ρ) = R₁(R₂ − R₁)/(R₂ − ρ). For the scalar problem the map gives
   ε_eff = ε det J J⁻¹ J⁻ᵀ = ε_out [(f²/(f′ρ²)) r̂r̂ᵀ + f′(I − r̂r̂ᵀ)] with J = diag(f′, f/ρ, f/ρ), and φ = 0 on the
   shell's outer sphere is φ → 0 at infinity. A net charge gives φ = Q/(4πε f) outside, linear in ρ inside the shell:
   the monopole is exact in the continuum, and its discrete error is a finite-element error that h/p refinement and
   radius comparisons quantify. `boundary: "grounded"` leaves the shell out and grounds the sphere at
   `mesh.domain_radius`. A dielectric must not reach that sphere: ε_out = ε₀ is assumed there without a check.
4. **Topology on the serial mesh.** Conductor bodies, the port terminal patches, island detection and both surface
   integrals run on the serial mesh every rank holds; per-rank markers are synchronized over shared DOFs.
5. **A permittivity floor of 1e-12 ε₀** in elements left out of the field problem keeps every matrix row non-empty;
   no free DOF belongs to a conductor element, so the floor does not change the solution beyond 1e-12.
6. **Checks.** Symmetry and the energy identities test the solves; `charge_gauss_*` and `exterior_monopole_*`
   (the net charge against the flux through the sphere at `mesh.domain_radius`) are discretization checks, at the
   per-cent level on coarse 3D meshes; `mixed_potential_solve` is a real solve at a non-unit potential vector against
   C·V; `floating_solve` checks the matrix reduction against the surface-flux charges.

**Readout gap in 3D.** The same two estimates; `C_gap_F` is the mutual capacitance −C_AB after the floating-body
reduction, and the ±½-drive capacitance C_differential = (C_AA + C_BB − 2C_AB)/4 is reported separately
(`C_differential_F`). Only C_differential carries the Dirichlet lower bound C_gap_port_region
(`checks.dirichlet_bound`). A 3D `port` pickup always pays for the readout-gap analysis (four solves), even without an
`electrostatics` section.

## Limitations

- The readout-gap capacitance is a lumped-model quantity; the full-wave pole of L(ω) is the reference for a
  distributed structure. `C_gap_F` has a mesh dependence of a few per cent (the electrode edges are
  Dirichlet–Neumann junctions); quote two digits.
- The potential and the surface charge density are not exported to scenes or VTU files.
- Dielectric loss (tan δ) does not enter the capacitance analysis.
- Zero-thickness conductor sheets exist in axisymmetric designs only; in 3D, model sheets as thin regions.
