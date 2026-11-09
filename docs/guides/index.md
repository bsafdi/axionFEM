# User guide

The user guide is organized by task. Each detector area has a guide page (what to set up, which calls to make, what
to check) and a walkthrough notebook that runs the same steps end to end. The [physics pages](../physics/index.md) give
the equations and conventions behind each result, and the [validation pages](../validation/index.md) what each
capability was checked against.

Start with [Concepts](concepts.md) (designs, solves, results and units) and the
[quick start](../../examples/quickstart.ipynb). [Scope and limitations](limitations.md) lists what is validated,
what is beta and what is not implemented.

<div class="grid cards" markdown>

-   **Lumped-element detectors**

    Toroidal, solenoidal and CORE magnets with a pickup: coupling Ψ, inductance L_p and the scan figure of merit.
    Sequential or MPI source build.

    [:octicons-arrow-right-24: Magnetoquasistatic designs](lumped_element.md) ·
    [Walkthrough](../../examples/lumped_element.ipynb)

-   **Full wave**

    Coupling, inductance and impedance beyond the quasistatic limit; radiation, resonances, multiport circuits and
    Touchstone files. Sequential or MPI source build.

    [:octicons-arrow-right-24: Full-wave guide](fullwave.md) · [Walkthrough](../../examples/fullwave.ipynb)

-   **3D designs and clusters**

    Slits, ports and other non-axisymmetric geometry; MPI runs and batch jobs. MPI source build.

    [:octicons-arrow-right-24: 3D guide](3d.md)

-   **Readout and SQUID chains**

    Readout circuits, noise referred to the pickup, dc SQUID chains (beta), multi-band scans (beta) and finite-band scans.

    [:octicons-arrow-right-24: Readout guide](readout_multiband.md) ·
    [Walkthrough](../../examples/readout_squid.ipynb)

-   **Cavity haloscopes** <span class="axf-badge">beta</span>

    Eigenmodes, form factors, Q₀, tuning with mode tracking, haloscope sensitivity and receiver coupling.

    [:octicons-arrow-right-24: Cavity guide](cavity.md) · [Walkthrough](../../examples/cavity_haloscope.ipynb)

-   **SRF heterodyne cavities** <span class="axf-badge">beta</span>

    Pump and signal modes, their overlap, wall peaks, degeneracy tuning and heterodyne sensitivity.

    [:octicons-arrow-right-24: SRF guide](srf.md) · [Walkthrough](../../examples/srf_heterodyne.ipynb)

-   **Mechanics and thermal**

    Magnetic forces, elastic stresses, cooldown, heat conduction with k(T); 3D with an MPI build.

    [:octicons-arrow-right-24: Mechanics guide](mechanics_thermal.md) ·
    [Walkthrough](../../examples/mechanics_thermal.ipynb)

-   **Reach, sensitivities and optimization**

    Scan reach, required field and size, adjoint shape sensitivities, local and global optimization.

    [:octicons-arrow-right-24: Optimization guide](optimization.md) ·
    [Walkthrough](../../examples/reach_optimization.ipynb)

-   **Fluids** <span class="axf-badge">beta</span>

    Stokes, Navier-Stokes, convection and inductionless MHD on meshes given as arrays. MPI source build.

    [:octicons-arrow-right-24: Fluids guide](fluids.md) · [Walkthrough](../../examples/fluids_beta.ipynb)

-   **Chemistry** <span class="axf-badge">beta</span>

    Kinetics, reactors and equilibrium; current distributions; porous electrodes. 3D parts need an MPI build.

    [:octicons-arrow-right-24: Chemistry guide](chemistry.md) · [Walkthrough](../../examples/chemistry_beta.ipynb)

</div>

Tools that apply to every area:

- [Solvers and numerical controls](solvers.md): which solver runs for which design, element order, mesh size,
  threads and the solver diagnostics.
- [Visualization and viewer](visualization.md): notebook widget, HTML export, PNG snapshots, VTU files.
- [COMSOL interoperability](comsol.md): export of geometry and meshes, import of `.mph` models (beta).
- [AI assistants (MCP)](mcp.md): the MCP server for AI assistants.
