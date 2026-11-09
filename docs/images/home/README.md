# Homepage graphics

Everything in this folder and in `../logo/` is written by
[`docs/images/make_homepage_figures.py`](../make_homepage_figures.py). Do not edit the files by hand:
change the script and run it again.

| file | what it shows |
|---|---|
| `hero_toroid_{light,dark}.png` (2400 x 1000) | the quick-start toroid as a 3D cutaway: static \|B\| on the cut planes, pickup-mode field lines in the bore |
| `card_cavity_*` | E_z of the three lowest TM modes of a copper cavity with an off-centre tuning rod |
| `card_srf_*` | overlap density of the degenerate TE012 pump and TM013 signal of the published 60 L closed cylinder of Lasenby |
| `card_readout_*` | noise of a resonant readout of the quick-start toroid with a published dc SQUID chain |
| `card_mech_*` | von Mises stress of an illustrative solenoid winding under its own Lorentz force |
| `card_reach_*` | published haloscope limits next to the same limits recomputed from the published parameters |
| `card_fluids_*` | natural convection in a differentially heated square cavity |
| `card_chem_*` | primary current distribution on a disk electrode (shown in the chemistry guide) |
| `social_preview.png` (1280 x 640) | repository social preview (uploaded in the repository settings at release) |
| `manifest.json` | axionFEM version and build features, SHA-256 of the script, design parameters, solver settings, convergence checks, quoted numbers, status, wall times |
| `captions.md` | captions generated from the manifest, one snippet section per figure |
| `../logo/` | icon, wordmarks (text converted to paths), favicons |

Cards are 1200 x 800 px with transparent backgrounds, in a light and a dark variant (ink `#1f2430` and `#e6e8ec`).
The hero uses the renderer's own backgrounds, `#ffffff` and `#13151a`; a container behind it should use the same
colour. Colour maps: viridis for magnitudes and coolwarm, symmetric about zero, for signed fields.

## Regenerating

The script needs a source build with Python bindings and the mesher (`axf-mesher`, or `AXF_MESHER` pointing to it),
plus NumPy, SciPy, matplotlib and Pillow. The fluids card needs an MPI build, because the fluids module exists only there.
From the repository root:

```bash
python docs/images/make_homepage_figures.py                     # everything, a few minutes on one node
python docs/images/make_homepage_figures.py --only cavity,srf   # some figures; the others stay in the manifest
python docs/images/make_homepage_figures.py --skip-unavailable  # skip the fluids card on a build without fluids
python docs/images/make_homepage_figures.py --check             # files present, manifest version = installed version
```

`--work DIR` sets the folder for the solver runs (default `runs/homepage`, which git ignores). The readout card
and the social preview reuse the hero solve, so asking for either also redraws the hero.

For a given build the output is deterministic: rcParams are fixed, the PNGs carry no metadata, and the C++
renderer is byte-deterministic, so images and captions are byte-identical between runs. In the manifest the
`seconds` fields change, and a different OpenMP thread count can change the last digit of relative differences that
sit at the round-off level (about 1e-9).

## Using the captions

Pages include a caption with pymdownx.snippets, so caption numbers are never typed by hand:

```markdown
--8<-- "docs/images/home/captions.md:hero"
```

Each figure's `status` in the manifest is set by the `STATUS` table at the end of the script, and a beta caption
starts with **Beta.** Fluids and chemistry are beta features, and the modules behind the cavity, SRF, readout and
reach figures are not declared validated, so those figures are beta as well.
