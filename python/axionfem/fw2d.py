"""2D full-wave results on an imported mesh (``"symmetry": "planar"`` or axisymmetric with ``mesh.file``).  Beta.

Designs of this kind come from :func:`axionfem.import_comsol` on COMSOL 2D and 2D-axisymmetric models
(docs/comsol_import.md); formulation: docs/physics/fw2d.md.  The C++ engine (``cpp/src/fw2d.cpp``, formulation in ``cpp/include/axionfem/fw2d.hpp``)
solves the in-plane family driven by a boundary lumped port and the eigenmodes of both
families; every reported quantity is the 3D one (axisymmetric: full revolution; planar: thickness D).

Family names follow the existing engines: axisymmetric TE = E_phi (out of plane), TM = (E_r, E_z); planar
TM = E_z (out of plane, as the cross-section solver), TE = (E_x, E_y).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .results import Results

__all__ = ["Fw2DResults"]


class Fw2DResults(Results):
    """Results of a 2D imported-mesh full-wave design.  ``data["fullwave"]`` holds ``sweep`` (per frequency:
    ``Z_re_ohm``, ``Z_im_ohm``, ``L_H``, ``Psi_*``, ``V_oc_*``, ``power_W_per_A2``), ``eigenmodes`` (per family:
    ``modes`` with ``f_Hz``, ``Q_wall``, ``Q_dielectric``), ``ports``, ``boundary_groups``, ``pml``."""

    def __init__(self, model, design, workdir: Path):
        super().__init__(model, design, workdir)

    def sweep(self) -> dict:
        """Sweep columns as arrays: f_Hz, Z (complex, axionFEM convention e^{-i omega t}), L_H, Psi (complex),
        V_oc (complex)."""
        rows = self.fullwave.get("sweep", [])
        f = np.array([r["f_Hz"] for r in rows])
        Z = np.array([r["Z_re_ohm"] + 1j * r["Z_im_ohm"] for r in rows])
        Psi = np.array([r["Psi_re"] + 1j * r["Psi_im"] for r in rows])
        return {"f_Hz": f, "Z": Z, "L_H": np.array([r["L_H"] for r in rows]), "Psi": Psi,
                "V_oc": -1j * 2 * np.pi * f * Psi}

    def modes(self, family: str | None = None) -> list[dict]:
        out = []
        for fam, block in self.fullwave.get("eigenmodes", {}).items():
            if family is not None and fam != family:
                continue
            for i, m in enumerate(block.get("modes", [])):
                out.append({"family": fam, "index": i, **m})
        out.sort(key=lambda r: r["f_Hz"])
        return out

    def sample(self, x1, x2, *, frequency_index: int | None = None, mode: tuple | None = None) -> dict:
        """Fields at points: ``frequency_index`` (driven field per port ampere) or ``mode=(family, index)``.
        Returns complex arrays E (n, 3), H (n, 3) with components (r, z, phi) or (x, y, z), and ``valid``."""
        x1 = np.atleast_1d(np.asarray(x1, float)).ravel().tolist()
        x2 = np.atleast_1d(np.asarray(x2, float)).ravel().tolist()
        if (frequency_index is None) == (mode is None):
            raise ValueError("give exactly one of frequency_index and mode")
        if mode is None:
            d = json.loads(self._model.sample("sweep", "", int(frequency_index), x1, x2))
        else:
            d = json.loads(self._model.sample("mode", str(mode[0]), int(mode[1]), x1, x2))
        E = np.array(d["E"], float)
        H = np.array(d["H"], float)
        return {"E": E[..., 0] + 1j * E[..., 1], "H": H[..., 0] + 1j * H[..., 1],
                "valid": np.array(d["valid"], bool), "omega": d["omega"], "components": d["components"]}


# ------------------------------------------------------------------------------------------------ test geometry
def mapped_mesh(mapping, nx: int, ny: int, region=None, region_uv=None):
    """Structured second-order triangle mesh of the image of the unit square under ``mapping(u, v) -> (x1, x2)``
    (vectorized).  Mid-edge nodes are mapped too, so curved images (arcs) are represented to second order.

    Returns ``(points (n, 2), tri6 (m, 6) gmsh order, tri_region (m,), edges (k, 3) gmsh line3 order, edge_side (k,))``
    with sides 0: v = 0, 1: u = 1, 2: v = 1, 3: u = 0.  ``region(xc, yc) -> int`` labels triangles by their mapped
    centroid (default 0); ``region_uv(uc, vc) -> int`` by the centroid in the unit square instead (robust on
    coarse curved meshes, where the straight-sided centroid can fall on the wrong side of an arc)."""
    N = 2 * nx + 1, 2 * ny + 1
    uu, vv = np.meshgrid(np.linspace(0, 1, N[0]), np.linspace(0, 1, N[1]), indexing="ij")
    X, Y = mapping(uu, vv)
    pts = np.column_stack([np.ravel(X), np.ravel(Y)])
    idx = lambda i, j: i * N[1] + j  # noqa: E731
    tris, reg = [], []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = idx(2 * i, 2 * j), idx(2 * i + 2, 2 * j), idx(2 * i + 2, 2 * j + 2), idx(2 * i, 2 * j + 2)
            # two triangles a-b-c and a-c-d with mid nodes (gmsh tri6: v0 v1 v2 m01 m12 m20)
            t1 = [a, b, c, idx(2 * i + 1, 2 * j), idx(2 * i + 2, 2 * j + 1), idx(2 * i + 1, 2 * j + 1)]
            t2 = [a, c, d, idx(2 * i + 1, 2 * j + 1), idx(2 * i + 1, 2 * j + 2), idx(2 * i, 2 * j + 1)]
            for t in (t1, t2):
                tris.append(t)
                xc, yc = pts[t[:3]].mean(axis=0)
                if region_uv is not None:
                    ij = np.array([divmod(int(q), N[1]) for q in t[:3]], float)
                    reg.append(region_uv(ij[:, 0].mean() / (N[0] - 1), ij[:, 1].mean() / (N[1] - 1)))
                else:
                    reg.append(region(xc, yc) if region else 0)
    edges, side = [], []
    for i in range(nx):
        edges.append([idx(2 * i, 0), idx(2 * i + 2, 0), idx(2 * i + 1, 0)]); side.append(0)
        edges.append([idx(2 * i, N[1] - 1), idx(2 * i + 2, N[1] - 1), idx(2 * i + 1, N[1] - 1)]); side.append(2)
    for j in range(ny):
        edges.append([idx(N[0] - 1, 2 * j), idx(N[0] - 1, 2 * j + 2), idx(N[0] - 1, 2 * j + 1)]); side.append(1)
        edges.append([idx(0, 2 * j), idx(0, 2 * j + 2), idx(0, 2 * j + 1)]); side.append(3)
    tris = np.array(tris)
    # orientation: counter-clockwise (positive area)
    p = pts[tris[:, :3]]
    area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    neg = area < 0
    tris[neg] = tris[neg][:, [0, 2, 1, 5, 4, 3]]
    return pts, tris, np.array(reg), np.array(edges), np.array(side)


def write_mesh(path, pts, tri6, tri_attr, edges, edge_attr):
    """gmsh MSH 2.2 (second-order triangles type 9, lines type 8), as the COMSOL importer writes 2D meshes."""
    path = Path(path)
    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{len(pts)}\n")
        for k, p in enumerate(pts):
            f.write(f"{k + 1} {p[0]:.17g} {p[1]:.17g} 0\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(edges) + len(tri6)}\n")
        k = 1
        for e, a in zip(edges, edge_attr):
            f.write(f"{k} 8 2 {a} {a} " + " ".join(str(int(v) + 1) for v in e) + "\n")
            k += 1
        for t, a in zip(tri6, tri_attr):
            f.write(f"{k} 9 2 {a} {a} " + " ".join(str(int(v) + 1) for v in t) + "\n")
            k += 1
        f.write("$EndElements\n")
