---
hide:
  - navigation
  - toc
---

# axionFEM { .axf-visually-hidden }

<div class="axf-hero" markdown>
<div class="axf-hero__text" markdown>

<p class="axf-hero__title">axionFEM</p>

Finite-element simulation for axion dark-matter detectors: lumped-element detectors, cavity haloscopes, SRF
heterodyne cavities and their readout.

[Install](guides/install.md){ .md-button .md-button--primary }
[Quick start](../examples/quickstart.ipynb){ .md-button }
[Validation](validation/index.md){ .md-button }

</div>
<div class="axf-hero__media" markdown>

![The toroidal detector of the quick start: static field on the cut planes and pickup-mode field lines](images/home/hero_toroid_light.png#only-light)
![The toroidal detector of the quick start: static field on the cut planes and pickup-mode field lines](images/home/hero_toroid_dark.png#only-dark)

</div>
</div>

<div class="axf-caption" markdown>
--8<-- "docs/images/home/captions.md:hero"
</div>

axionFEM computes what a detector design needs: for lumped-element detectors, the signal coupling Ψ, the pickup
inductance and the scan figure of merit from magnetoquasistatic and full-wave solutions; for cavity haloscopes,
eigenmodes, form factors and quality factors; for SRF heterodyne cavities, pump and signal modes and their overlap;
and for each, the noise, scan rate and reach of the readout chain. A C++17 core built on MFEM performs the solves; a
Python API, the `axf` command line, an interactive viewer and an MCP server sit on top.

<div class="grid cards axf-cards" markdown>

-   ![Cavity modes with an off-centre tuning rod](images/home/card_cavity_light.png#only-light){ loading=lazy }
    ![Cavity modes with an off-centre tuning rod](images/home/card_cavity_dark.png#only-dark){ loading=lazy }

    **Cavity haloscopes** <span class="axf-badge">beta</span>

    Eigenmodes, form factors, Q₀ with wall models, tuning sweeps and scan rate.

    ??? quote "About this figure"
        --8<-- "docs/images/home/captions.md:cavity"

    [:octicons-arrow-right-24: Cavity guide](guides/cavity.md)

-   ![Overlap density of the pump and signal modes of an SRF cavity](images/home/card_srf_light.png#only-light){ loading=lazy }
    ![Overlap density of the pump and signal modes of an SRF cavity](images/home/card_srf_dark.png#only-dark){ loading=lazy }

    **SRF heterodyne cavities** <span class="axf-badge">beta</span>

    Pump and signal modes, their overlap, wall fields and degeneracy tuning.

    ??? quote "About this figure"
        --8<-- "docs/images/home/captions.md:srf"

    [:octicons-arrow-right-24: SRF guide](guides/srf.md)

-   ![Noise budget of a resonant readout with a dc SQUID chain](images/home/card_readout_light.png#only-light){ loading=lazy }
    ![Noise budget of a resonant readout with a dc SQUID chain](images/home/card_readout_dark.png#only-dark){ loading=lazy }

    **Readout and SQUID chains** <span class="axf-badge">SQUID chains, multi-band scans: beta</span>

    Readout circuits, amplifier noise referred to the pickup, multi-band scans.

    ??? quote "About this figure"
        --8<-- "docs/images/home/captions.md:readout"

    [:octicons-arrow-right-24: Readout guide](guides/readout_multiband.md)

-   ![Von Mises stress in a solenoid winding under its own Lorentz force](images/home/card_mech_light.png#only-light){ loading=lazy }
    ![Von Mises stress in a solenoid winding under its own Lorentz force](images/home/card_mech_dark.png#only-dark){ loading=lazy }

    **Mechanics and thermal**

    Magnetic forces, elastic stress, cooldown contraction and heat loads with k(T).

    ??? quote "About this figure"
        --8<-- "docs/images/home/captions.md:mech"

    [:octicons-arrow-right-24: Mechanics guide](guides/mechanics_thermal.md)

-   ![Published haloscope limits next to the limits recomputed by axionFEM](images/home/card_reach_light.png#only-light){ loading=lazy }
    ![Published haloscope limits next to the limits recomputed by axionFEM](images/home/card_reach_dark.png#only-dark){ loading=lazy }

    **Reach across mass** <span class="axf-badge">beta</span>

    Haloscope sensitivity from cavity, magnet and receiver parameters.

    ??? quote "About this figure"
        --8<-- "docs/images/home/captions.md:reach"

    [:octicons-arrow-right-24: Cavity guide](guides/cavity.md#sensitivity-and-reach)

-   ![Natural convection in a differentially heated square cavity](images/home/card_fluids_light.png#only-light){ loading=lazy }
    ![Natural convection in a differentially heated square cavity](images/home/card_fluids_dark.png#only-dark){ loading=lazy }

    **Fluids** <span class="axf-badge">beta</span>

    Stokes and Navier-Stokes flow, convection and inductionless MHD (MPI build).

    ??? quote "About this figure"
        --8<-- "docs/images/home/captions.md:fluids"

    [:octicons-arrow-right-24: Fluids guide](guides/fluids.md)

</div>

<div class="grid cards" markdown>

-   :material-sine-wave: **Full-wave response**

    Coupling and inductance beyond the quasistatic limit, radiation, multiport circuits, Touchstone export.

    [:octicons-arrow-right-24: Full-wave guide](guides/fullwave.md)

-   :material-cube-scan: **Viewer**

    Notebook widget, single-file HTML pages and PNG snapshots of every solve.

    [:octicons-arrow-right-24: Visualization](guides/visualization.md)

-   :material-swap-horizontal: **COMSOL**

    Export geometry, meshes and physics; import `.mph` models (beta).

    [:octicons-arrow-right-24: COMSOL guide](guides/comsol.md)

-   :material-robot-outline: **AI assistants**

    An MCP server lets assistants build designs, run solves and read results.

    [:octicons-arrow-right-24: MCP guide](guides/mcp.md)

-   :material-flask-outline: **Chemistry** <span class="axf-badge">beta</span>

    Kinetics and reactors, current distributions, porous electrodes.

    [:octicons-arrow-right-24: Chemistry guide](guides/chemistry.md)

-   :material-server-network: **3D and clusters**

    The MPI build: 3D designs, slits and ports, batch jobs.

    [:octicons-arrow-right-24: 3D guide](guides/3d.md)

</div>

## Checked against published results

ABRACADABRA-10 cm · DMRadio-50L · DMRadio-m³ · DMRadio-GUT · DMRadio-Core · HAYSTAC · RADES · CAPP · QUAX ·
the beyond-magnetoquasistatic results of arXiv:2211.00008 · closed-form and manufactured solutions.
The [validation pages](validation/index.md) give the references, tolerances and tests.

## Citing

J. N. Benabou, J. W. Foster, B. R. Safdi, to appear (2026). [How to cite](about.md#citing-axionfem)
