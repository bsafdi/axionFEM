# Vendored three.js

The viewer uses [three.js](https://threejs.org) (MIT license, see
`LICENSE-three.txt`). The files here are unmodified copies from one pinned
release so that exported pages work offline and builds are reproducible.

- Release: **three 0.185.1** (r185), published 2026-07-01.
- Source: the npm tarball `https://registry.npmjs.org/three/-/three-0.185.1.tgz`
  (registry integrity
  `sha512-5aojFCXKwnjBRZvUnt3WFfEcvUJgkN5LlijRFN95hMy8WVkG4I0QNcJE+OuWvuJ0bOdStrbfXn0pkd6/QyiAlg==`,
  verified on download). The same files are served at
  `https://cdn.jsdelivr.net/npm/three@0.185.1/<path>` (hashes checked to match).
- Why 0.185.1: it is the newest release that ships the official minified ES
  module builds (`build/three.module.min.js`, `build/three.core.min.js`); the
  0.186 packages no longer include them.

| file here | path in the package | sha256 |
|---|---|---|
| `three.module.min.js` | `build/three.module.min.js` | `86bcee248b64f44bcfc23c331ae74619061957d59cab040171dcb6fb5900beb6` |
| `three.core.min.js` | `build/three.core.min.js` | `05b2609338c76cd65daf74f3ac515bc9a5045e1b3b33edc07d8c9bd55250fa90` |
| `addons/controls/OrbitControls.js` | `examples/jsm/controls/OrbitControls.js` | `faabb4e8dfd9235ee4a9fd7c9a3d75f90f1689dbd4944bd6fd32117dacec5f93` |
| `addons/environments/RoomEnvironment.js` | `examples/jsm/environments/RoomEnvironment.js` | `55f466192cc84298755a424c5e040345006b2ee1455589b3b54126c2ea4123f4` |
| `LICENSE-three.txt` | `LICENSE` | `8b378ebe60e2fe500158cb0ac71cb5e8b7d92953c2abcc63a0eb90499653b5bc` |

`three.module.min.js` imports `./three.core.min.js`. For single-file HTML
export, `python/axionfem/viewer.py` rewrites that one specifier to the bare
name `axf-three-core` (resolved by the page's import map) before inlining; the
files on disk are never modified. `tests/test_viewer.py` checks the hashes
above.

To update: download the new release's files to the same paths, update this
table and `THREE_VERSION` in `python/axionfem/viewer.py`, then run
`pytest tests/test_viewer.py` and check the rendering (docs/viewer.md).
