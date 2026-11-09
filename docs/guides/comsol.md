# COMSOL export and import

Lumped-element detectors and cavity haloscopes can be modeled in axionFEM and [COMSOL Multiphysics](https://www.comsol.com). In this guide we use axionFEM to export a detector's geometry to COMSOL or read a saved COMSOL model into an axionFEM design. Exporting geometry lets us set up a calculation in COMSOL; importing a model transfers its mesh and supported electromagnetic settings. Neither conversion needs COMSOL installed on the conversion machine.

## Design export to COMSOL

We export the toroid template [design](concepts.md) to Java so that its geometry can be rebuilt in COMSOL.

```python
import axionfem as af
from axionfem.comsol import export_comsol

info = export_comsol(af.template("toroid"), "Toroid.java", format="java")
print(info["file"])
```

This writes `Toroid.java`, which rebuilds the geometry and named region selections. The returned dictionary lists the output file, selections and geometry approximations. On a machine with COMSOL, we run `comsolcompile Toroid.java`, then `comsol batch -inputfile Toroid.class` to create `toroid.mph`.

The default export contains geometry and selections; we assign materials and physics in COMSOL. To export the mesh instead, we use `format="mphtxt"` and an `.mphtxt` output path; a design is meshed first. A 3D design can also use `format="step"`; opening that file in COMSOL requires CAD import support.

For a 3D design with a gap-port pickup, `physics="fullwave"` adds supported materials, electromagnetic settings and a frequency study to the Java or mesh export. We check the returned summary for source and boundary approximations.

## COMSOL .mph model import

!!! warning "Beta"
    The `.mph` import (`axionfem.import_comsol`, `axf import-comsol`) and the 2D full-wave solver it uses for 2D
    models are beta. Read the import report before relying on an imported model.

We save a 3D **Electromagnetic Waves, Frequency Domain** model with its tetrahedral mesh, using **Mesh → Build All** before saving. We then import the saved model and print its report summary to inspect the transferred settings.

```python
import axionfem as af

d = af.import_comsol("detector.mph", "imported/design.json")
print(d.import_report.summary)
```

The returned `Design` uses the stored mesh. The output directory contains `design.json`, `design.msh`, `design_mesh_info.json` and an import report in Markdown and JSON. The printed summary describes the imported mesh and electromagnetic settings. We read the report before solving: it identifies transferred settings and approximations, including frequency-dependent materials represented by constant values. Unsupported physics that would change the result stops the import with an explanation.

2D and 2D-axisymmetric models import as well: they run on a 2D full-wave solver that takes the stored triangle mesh. Impedance, transition (thin layer) and scattering boundary conditions, and perfectly matched layers, are imported in 2D and 3D. The details and their validation are in the [COMSOL import reference](../comsol_import.md); the export options are in the [COMSOL export reference](../comsol_export.md).

The [API reference](../api.md) lists conversion options.
