# Visualization

Lumped-element axion experiments use static magnetic fields and pickup circuits.
In this guide we use the axionFEM viewer to display detector materials and magnetic
fields in a browser or Jupyter notebook. Axisymmetric designs have a revolved cutaway and a meridional
section; full 3D designs show surfaces and field slices. We rotate, pan and zoom,
then select a field quantity or probe a point.

## Viewing and saving a result

We display the result `res` from the
[quick start](../../examples/notebook_quickstart.ipynb) and save its HTML viewer and
scene for later display.

```python
res.show(height=520)
res.export_html("toroid.html")
scene = res.scene()
res.save_scene("scene.json")
```

`show` displays a notebook widget. The HTML file includes both the scene and
viewer, so it can be opened or shared without a network connection. The scene
is a Python dictionary; its JSON file stores geometry, sampled fields and
results for later display without another solve.

We select `Bmag` for the static magnetic-field magnitude in tesla, or
`pickup_Bmag` for the pickup field per ampere of readout current. The colour bar
gives the units. Poloidal field lines in an axisymmetric scene follow contours
of the flux function

$$
\psi = r A_\phi.
$$

Here $r$ is cylindrical radius and $A_\phi$ is the azimuthal vector potential;
`psi` has units $\mathrm{Wb/rad}$. A purely toroidal field has no poloidal field lines.
Line spacing depends on the chosen contour levels.
Displayed fields are interpolated samples, so solver peak values can exceed
the colour-bar maximum.

## Scenes and PNG snapshots

The viewer and renderer read the same `axionfem.scene/1` format. Its `regions`
describe materials, `field` or `slices` hold sampled quantities, `summary`
contains solver results, and `design` stores the input document. Coordinates
are in metres: radius and height for axisymmetric scenes, Cartesian coordinates
for full 3D scenes. A scene contains display samples, not the full finite-element
solution.

We render the saved scene as a PNG to view the pickup-field magnitude.

```python
from axionfem.render import save_png

save_png("scene.json", "pickup.png", view="section",
         quantity="pickup_Bmag", log_scale=True)
```

This writes a pickup-field image using a software renderer, without a browser
or GPU. We use `view="3d"` for a cutaway and `quantity="none"` for materials only.
Logarithmic colours show magnitudes; we use a linear scale when a component's sign
matters. We set matching `vmin` and `vmax` when comparing images, `width` and
`height` for pixel dimensions, and `annotate=False` to hide the results panel.

The [API reference](../api.md) lists display options.
