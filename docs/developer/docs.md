# Documentation and figures

The website is built with MkDocs and the Material theme from `docs/`, the walkthrough notebooks in `examples/` and an
API reference generated from the docstrings. It builds without the compiled core.

## Building the site

```bash
pip install -r docs/requirements.txt -c docs/constraints.txt
mkdocs serve                        # live preview on http://127.0.0.1:8000
mkdocs build --strict               # what CI checks: any warning fails the build
bash scripts/docs/build_offline.sh  # a site/ that opens from disk (file://); formulas still need a network
```

`docs/constraints.txt` pins the versions the site was last built and checked with; `docs/requirements.txt` gives the
ranges. The documentation workflow (`.github/workflows/docs.yml`) runs `mkdocs build --strict` on every pull request
and push to `main`; deployment to GitHub Pages runs only when started by hand (`workflow_dispatch`).

## How the pieces fit

- **Pages** are Markdown files in `docs/`. They keep their paths, because code, tests and other pages link to them;
  the navigation titles are set in the `nav` of `mkdocs.yml`. New user pages go to `docs/guides/`, developer pages to
  `docs/developer/`. `exclude_docs` in `mkdocs.yml` keeps the reference material (validation, physics notes,
  developer notes, the CLI and C++ references) out of the site; links to it go to GitHub.
- **Links** are written for GitHub: relative paths to other pages, to source files, benchmarks and records.
  `scripts/docs/hooks.py` rewrites them at build time, to site pages where the target is part of the site and to the
  GitHub repository otherwise; a link to a file that does not exist stays broken, so `--strict` reports it.
- **The API reference** (`docs/api.md`) uses mkdocstrings with static analysis (griffe): it never imports
  `axionfem`, so the C++ core need not be built. Docstrings are NumPy style; `scripts/docs/griffe_sphinx_roles.py`
  renders Sphinx roles such as `:func:`.
- **Notebooks.** `scripts/docs/gen_pages.py` copies the notebooks listed in `SITE_NOTEBOOKS` into the site, and
  mkdocs-jupyter renders them with their stored outputs; the site build does not execute them. A listed notebook that
  does not exist fails a strict build. After changing a notebook, execute it again with
  `python scripts/docs/execute_notebooks.py examples/NAME.ipynb` (no argument: all site notebooks; it needs
  `nbclient`, `nbformat` and `ipykernel`): it runs each notebook from its own directory with the inline matplotlib
  backend, removes widget views and execution timestamps, and lists output lines with absolute paths, user or host
  names, tracebacks or timings for review. The released notebooks are executed with an MPI build that has STRUMPACK,
  so that their 3D cells run.
- **Formulas** are typeset by MathJax (`pymdownx.arithmatex`); `docs/javascripts/mathjax.js` configures it.

## The homepage figures

The hero, the cards, the social preview and the logo files in `docs/images/home/` and `docs/images/logo/` are written by
`docs/images/make_homepage_figures.py`; never edit them by hand. Every picture is axionFEM output: the script runs
the solves, writes the PNGs in a light and a dark variant, a `manifest.json` with the design parameters, solver
settings, convergence checks and the numbers each caption may quote, and `captions.md`. The README and the home page
use some of these figures; the README's static-field caption is the `hero` section of `captions.md`
(`scripts/docs/check_readme.py` checks the copy), and a page can include a caption with pymdownx.snippets:

```markdown
--8<-- "docs/images/home/captions.md:hero"
```

To regenerate, use a source build with Python bindings and the mesher (an MPI build for the fluids card), plus NumPy,
SciPy, matplotlib and Pillow:

```bash
python docs/images/make_homepage_figures.py                     # everything; a few minutes on one node
python docs/images/make_homepage_figures.py --only cavity,srf   # some figures; the others keep their manifest entries
python docs/images/make_homepage_figures.py --check             # files present, manifest version = installed version
```

Commit the script first and regenerate from a clean checkout: the manifest records the script's SHA-256 and whether
the checkout had uncommitted changes. For a given build the output is deterministic (fixed rcParams, no PNG metadata,
the byte-deterministic C++ renderer), so an unchanged figure regenerates byte-identical. The `STATUS` table at the end
of the script sets which captions are labelled beta. `docs/images/home/README.md` lists every file.

Other figure scripts in `docs/images/`: `make_circuit_figure.py` draws the readout and SQUID-chain schematics of the
physics pages with the package's own `draw()` methods, and `make_cavity_figure.py` the cavity mode gallery of the
validation pages. Validation figures are written by their benchmark scripts in `benchmarks/`.
