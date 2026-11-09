# API reference

Lumped-element detectors, cavity haloscopes and SRF heterodyne cavities search for axion dark matter. Here we list the public objects of the `axionfem` Python package, grouped by topic.

## Designs and solves

axionFEM provides functions and classes for creating designs from templates or files, solving them and accessing the results.

::: axionfem.design.template
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.design.templates
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.design.load
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.design.Design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.results.Results
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.results.Results3D
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.results.requirement
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.global_opt.optimize
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.global_opt.global_optimize
    options:
      heading_level: 3
      show_root_full_path: false

## Lumped-element scan and reach

`scan.multipole_scan` (with `scan.reach_multiband` and `scan.reach_2211`, module `axionfem.multipole`) is beta.

::: axionfem.scan.reach
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.scan.g_qcd
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.scan.reach_fullwave
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.scan.finite_band
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.scan.multipole_scan
    options:
      heading_level: 3
      show_root_full_path: false

## Readout circuit and SQUID chains

The dc SQUID chains (`axionfem.squid`) are beta.

::: axionfem.circuit.series_readout
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.circuit.Readout
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.squid.chain
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.squid.Chain
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.squid.benchmark
    options:
      heading_level: 3
      show_root_full_path: false

## Cavity haloscopes

Everything in this section is beta, as are the related modules `axionfem.cavity_xsec`, `axionfem.cavity_coupling`
and `axionfem.haloscope_presets`.

::: axionfem.cavity.pillbox_design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.cavity.coax_design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.cavity.pillbox_modes
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.cavity.surface_resistance
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.cavity_track.sweep
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.cavity_track.track
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.cavity_track.Tracking
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.haloscope.Cavity
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.haloscope.Receiver
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.haloscope.scan_rate
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.haloscope.beta_opt
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.haloscope.g_limit_scan
    options:
      heading_level: 3
      show_root_full_path: false

## SRF heterodyne

Beta: `axionfem.srf` and `axionfem.heterodyne`.

::: axionfem.srf.elliptical_cell_design
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.srf.tune_degeneracy
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.srf.find_mode
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.srf.figures_of_merit
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.heterodyne.SRFGeometry
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.heterodyne.scan_rate
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.heterodyne.required_volume
    options:
      heading_level: 3
      show_root_full_path: false

## Touchstone (S-, Y- and Z-parameters)

::: axionfem.touchstone.write_touchstone
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.touchstone.read_touchstone
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.touchstone.port_parameters
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.touchstone.z_to_s
    options:
      heading_level: 3
      show_root_full_path: false

## COMSOL

The import (`axionfem.comsol_import`) is beta; the export is not.

::: axionfem.comsol.export_comsol
    options:
      heading_level: 3
      show_root_full_path: false

::: axionfem.comsol_import.import_comsol
    options:
      heading_level: 3
      show_root_full_path: false
