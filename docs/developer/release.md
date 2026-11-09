# Releasing

A release is a snapshot of the source tree at a tagged commit, with wheels for PyPI and a matching documentation
site. This page lists the steps.

## Before the tag

1. **Version.** Set the new version in three places, which `tests/test_packaging.py` checks agree:
   `pyproject.toml` (`project.version`), `CMakeLists.txt` (`project(axionFEM VERSION ...)`) and
   `cpp/include/axionfem/core.hpp` (`axf::kVersion`).
2. **Changelog and citation.** Add the release section to `CHANGELOG.md`, and update `version` and `date-released` in
   `CITATION.cff`.
3. **Tests.** Run the full suite on an MPI build (`AXF_SLOW=1 AXF_MPI=1`, [Running the tests](testing.md)) and check
   that CI is green on the commit to be tagged.
4. **Generated material.** Regenerate what depends on the code and has changed: validation records (their scripts in
   `benchmarks/`), the walkthrough notebooks' outputs (`python scripts/docs/execute_notebooks.py` on an MPI build with
   STRUMPACK), and the homepage figures
   (`python docs/images/make_homepage_figures.py`, then `--check`; [Documentation and figures](docs.md)). Never edit
   generated numbers by hand. Commit a regenerated figure together with the generator change that caused it, and look
   at the images themselves (titles and labels): a text search does not see them.
5. **Documentation.** `mkdocs build --strict` passes, the user guide and `llms.txt` describe the released API, and
   beta labels match the code.
6. **Clean tree.** The snapshot contains source, documentation, tests, examples and the data they read; no build
   products, run directories, logs or machine-specific files.

## Tag and publish

1. **Tag** the release commit `vX.Y.Z` and push the tag. The wheels workflow (`.github/workflows/wheels.yml`) builds
   the wheels for every platform and Python version and the sdist, and installs and tests the wheels on clean runners.
2. **Publish to PyPI** by starting the wheels workflow by hand on the tag with `publish` set. It uploads through PyPI's
   trusted publishing from the `pypi` environment, so no API token is stored in the repository; configure the
   trusted publisher on PyPI and protect the environment (required reviewers) before the first upload. Before the
   first upload, also make `README.md` render on PyPI (its images and links are relative to the repository: use
   absolute `https://raw.githubusercontent.com/...` and documentation-site URLs, or a separate PyPI readme).
3. **Deploy the documentation** by starting the documentation workflow by hand on `main` (`workflow_dispatch`).
   GitHub Pages must be enabled with "GitHub Actions" as the source.
4. **Announce** the release on GitHub with the changelog section, and upload `docs/images/home/social_preview.png` as
   the repository's social preview (Settings, General).

## After the release

Open the next version's section in `CHANGELOG.md`. A fix to a release goes into a patch version (`vX.Y.Z+1`) with the
same steps.
