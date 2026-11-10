# axionFEM

axionFEM is a finite-element package for electromagnetics, structural mechanics, heat conduction, fluid flow and electrochemistry, with a Python interface, a command line and an interactive field viewer. We built it to design axion dark-matter experiments, which include lumped-element detectors, cavity haloscopes and SRF heterodyne cavities, and it has dedicated tools for them. The solvers themselves are general. Axisymmetric geometries are drawn from polygons, arcs and boolean combinations; 3D geometries from primitives, revolved or extruded profiles and imported CAD (STEP) solids. The same code applies to magnets, superconducting devices, RF structures and resonators in other settings.

**What it solves**

- Magnetostatics and magnetoquasistatics: superconducting, resistive and iron-core magnets and shields; fields, inductances and forces, with exact open boundaries.
- Full-wave electromagnetics: impedances and S-parameters (Touchstone export), radiation losses, eigenmodes and self-resonances.
- Superconductors: ideal superconductors with flux or current constraints, critical surfaces of NbTi, Nb₃Sn and REBCO, and the HTS critical state.
- Electrostatics: capacitance matrices.
- Circuits: networks with the noise of every element, and multiport networks from field solutions.
- Mechanics and heat: stresses from Lorentz forces and cooldown, and steady heat conduction with a temperature-dependent conductivity.
- Shape sensitivities and optimization: adjoint gradients, with local, global and derivative-free optimizers.
- Fluids (beta): Stokes and Navier–Stokes flow, steady and transient, inductionless MHD, and heat transport with natural convection.
- Electrochemistry (beta): stiff chemical kinetics, reactors, and current distributions in electrolytes.

Axisymmetric problems run on one machine; 3D problems use MPI.

**Walkthroughs**

- [Lumped-element detectors](../examples/lumped_element_demo.ipynb): magnets, pickup circuits, and scan sensitivity.
- [dc SQUID readout chains](../examples/squid_chain_demo.ipynb) (beta): amplifier noise and detector coupling.
- [Cavity haloscopes](../examples/cavity_demo.ipynb) (beta): resonant modes, form factors, and sensitivity.
- [SRF heterodyne cavities](../examples/srf_demo.ipynb) (beta): pump and signal modes in superconducting radio-frequency cavities.
- [Mechanics and thermal](../examples/mechanics_thermal_demo.ipynb): forces and stresses in a winding, cooldown, and heat loads.
- [Fluids](guides/fluids.md) and [chemistry](guides/chemistry.md) (beta): incompressible flow and current distributions in electrolytes.
- [COMSOL export and import](guides/comsol.md): exchange of designs, meshes, and model settings.

[Installation](guides/install.md) ·
[Quick start](../examples/notebook_quickstart.ipynb) ·
[API reference](api.md)

Citing axionFEM, contact and issues: [About](about.md)

![Toroidal detector in the axionFEM viewer](images/viewer_toroid.png)

*Pickup field in a toroidal detector.*

![Overlap density of the pump and signal modes in an SRF cavity](images/home/card_srf_light.png#only-light){ width="49%" }
![Overlap density of the pump and signal modes in an SRF cavity](images/home/card_srf_dark.png#only-dark){ width="49%" }
![Von Mises stress in a solenoid winding under its own Lorentz force](images/home/card_mech_light.png#only-light){ width="49%" }
![Von Mises stress in a solenoid winding under its own Lorentz force](images/home/card_mech_dark.png#only-dark){ width="49%" }

*Left: overlap of the pump and signal fields in an SRF heterodyne cavity (beta). Right: von Mises stress in a solenoid winding under its own Lorentz force.*

![Natural convection in a differentially heated square cavity](images/home/card_fluids_light.png#only-light){ width="49%" }
![Natural convection in a differentially heated square cavity](images/home/card_fluids_dark.png#only-dark){ width="49%" }
![Primary current distribution on a disk electrode](images/home/card_chem_light.png#only-light){ width="49%" }
![Primary current distribution on a disk electrode](images/home/card_chem_dark.png#only-dark){ width="49%" }

*Left: natural convection in a heated square cavity at Ra = 10⁵, temperature and streamlines (beta). Right: primary current distribution on a disk electrode (beta).*

Contributors: Joshua N. Benabou, Joshua W. Foster, Benjamin R. Safdi.
