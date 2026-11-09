"""Cross-section cavity haloscopes (``"symmetry": "xsec"``; beta).

A cavity whose cross-section does not change along z (posts, vanes, partitions, dielectric rods spanning the full
length, no end gaps) has z-independent modes that solve a scalar Helmholtz problem on the (x, y) cross-section:

- TM_z (E = E_z z^): -div((1/mu_r) grad E_z) = k0^2 eps_r E_z, E_z = 0 on every conductor. With flat end caps
  these are exact 3D modes (TM_mn0) for any length; they are the ones that couple to an axial B0.
- TE_z (H = H_z z^): cutoff modes of the guide, C = 0; z-independent TE modes do not satisfy end caps (they need
  p >= 1). Optional, for mode maps within the 2D model.

Modes with z dependence (TM_mnp / TE_mnp, p >= 1; hybrid modes of dielectric-loaded guides) are not captured: that
is the 3D solver's job. :func:`z_dependent_spectrum` lists them for vacuum cross-sections (separable).

Design section (docs/design_format.md, "Cross-section cavities"): ``"symmetry": "xsec"``, regions with shapes in
(x, y), ``fullwave`` (modes, families, target_f_Hz) and ``cavity`` (interior_point, wall_material, length,
end_cap_material, losses). ``Design.solve()`` returns :class:`XsecResults` (``cavity_modes()``,
``cavity_mode_field()``, ``plot_mode()``...).

Builders: :func:`xsec_design`, :func:`cylinder_rods_design` (a cylinder with N rods placed by (x, y) or by pivot,
arm and angle), :func:`sector_cell_design` (CAPP ideal multi-cell). Sweeps: :func:`sweep`, :func:`rod_angle_sweep`,
:func:`plot_mode_map`. Closed forms: :func:`disk_tm_modes`, :func:`disk_te_modes`, :func:`coax_tm_G2d`,
:func:`sector_cell_mode`.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from .results import Results

C_LIGHT = 299792458.0
MU0 = 1.25663706212e-6  # H/m (CODATA 2018, as cpp/include/axionfem/units.hpp)
Z0 = MU0 * C_LIGHT

__all__ = ["XsecResults", "xsec_design", "cylinder_rods_design", "sector_cell_design", "sweep", "rod_angle_sweep",
           "plot_mode_map", "disk_tm_modes", "disk_te_modes", "coax_tm_G2d", "sector_cell_mode",
           "z_dependent_spectrum", "skin_rf", "eccentric_rod_tm_mode", "eccentric_dielectric_rod_tm_mode"]


# ---------------------------------------------------------------------------------------------------- results
class XsecResults(Results):
    """Results of a cross-section (``"symmetry": "xsec"``) cavity design. ``data`` is results.json:
    ``fullwave.eigenmodes.{TM,TE}.modes[]`` with ``f_Hz``, ``form_factor`` (C, C_full, A_m2, A_full_m2), ``losses``
    (walls per body, end_caps, G_ohm, G_ohm_2d, Q_wall, Q_wall_2d, Q_dielectric, Q0, Q0_2d), ``energy_fraction``;
    ``fullwave.cavity`` (area, walls, regions); ``warnings``."""

    def __init__(self, model, design, workdir: Path):
        super().__init__(model, design, workdir)

    def cavity_modes(self, family: str | None = None) -> list[dict]:
        """Modes as flat rows sorted by frequency: ``family`` ("TM"/"TE"), ``index`` (within the family), ``f_Hz``,
        ``C`` (V = cavity area x length, conductors excluded), ``C_full`` (area enclosed by the outer wall), ``A_m2``,
        ``A_full_m2``, ``V_m3`` (with cavity.length), ``G_ohm`` (Q R_s, all walls and the end caps if a length is
        given), ``G_ohm_2d`` (barrel walls only), ``Q_wall``, ``Q_wall_2d``, ``Q_dielectric``, ``Q0``, ``Q0_2d``,
        ``T_over_Q_K``, ``residual_true``, ``energy_fraction``, ``walls``, ``end_caps``. Missing quantities: None."""
        rows = []
        for fam, block in self.fullwave.get("eigenmodes", {}).items():
            if family is not None and fam != family:
                continue
            for i, m in enumerate(block.get("modes", [])):
                ff, lo = m.get("form_factor", {}), m.get("losses", {})
                rows.append({"family": fam, "index": i, "f_Hz": m["f_Hz"], "C": ff.get("C"), "C_full": ff.get("C_full"),
                             "C_rms": ff.get("C"), "A_m2": ff.get("A_m2"), "A_full_m2": ff.get("A_full_m2"),
                             "V_m3": ff.get("V_m3"), "G_ohm": lo.get("G_ohm"), "G_ohm_2d": lo.get("G_ohm_2d"),
                             "Q_wall": lo.get("Q_wall"), "Q_wall_2d": lo.get("Q_wall_2d"),
                             "Q_dielectric": lo.get("Q_dielectric"), "Q0": lo.get("Q0"), "Q0_2d": lo.get("Q0_2d"),
                             "T_over_Q_K": lo.get("T_over_Q_K"), "residual_true": m.get("residual_true"),
                             "energy_fraction": m.get("energy_fraction", {}), "walls": lo.get("walls", {}),
                             "end_caps": lo.get("end_caps")})
        rows.sort(key=lambda r: r["f_Hz"])
        return rows

    @staticmethod
    def _arrays(d: dict, row: dict) -> dict:
        import numpy as np

        out = {k: (np.asarray(v) if not isinstance(v, (str, int, float)) else v) for k, v in d.items()}
        out["elem"] = np.asarray(out["elem"]).astype(np.int64)
        if "tris" in out:
            out["tris"] = np.asarray(out["tris"]).astype(np.int64).reshape(-1, 3)
        out["Emag"] = np.sqrt(out["E_x"] ** 2 + out["E_y"] ** 2 + out["E_z"] ** 2)
        out["Hmag"] = np.sqrt(out["H_x"] ** 2 + out["H_y"] ** 2 + out["H_z"] ** 2)
        out["mode"] = {k: row.get(k) for k in ("family", "index", "f_Hz", "C", "C_full", "A_m2", "G_ohm", "Q_wall",
                                                   "Q_dielectric", "Q0")}
        return out

    def cavity_mode_field(self, i=0, family: str | None = None, *, subdiv: int = 2) -> dict:
        """Fields of a mode on the cavity mesh (``subdiv`` intervals per element edge): ``x``, ``y`` [m], ``tris``,
        ``E_x``, ``E_y``, ``E_z``, ``H_x``, ``H_y``, ``H_z``, ``Emag``, ``Hmag``, ``eps_r``, ``mu_r``, ``elem``,
        ``mode``. ``i`` indexes :meth:`cavity_modes` (or with ``family`` the index within the family; a row works too).
        Normalization: int eps_r |E|^2 dA = 1 over the cross-section (E in m^-1, per unit length); H is the matching
        physical field (int mu_r |H|^2 dA = 1/Z0^2). Real amplitudes of E_c = i E, H_c = H (e^{-i omega t}). Sign: TM
        modes with int E_z dA > 0 (when it is not round-off), else the largest E component at max |E| positive.
        TM_z: (E_z, H_x, H_y); TE_z: (E_x, E_y, H_z)."""
        row = self._cavity_mode_row(i, family)
        return self._arrays(self._model.mode_mesh(row["family"], int(row["index"]), int(subdiv)), row)

    def cavity_mode_sample(self, i, x, y, family: str | None = None) -> dict:  # type: ignore[override]
        """Fields of a mode at points (x, y) [m]; NaN outside the cavity (same keys as :meth:`cavity_mode_field`)."""
        import numpy as np

        row = self._cavity_mode_row(i, family)
        xx, yy = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        d = self._model.mode_sample(row["family"], int(row["index"]), [float(v) for v in xx.ravel()],
                                    [float(v) for v in yy.ravel()])
        out = self._arrays(d, row)
        for k, v in list(out.items()):
            if isinstance(v, np.ndarray) and v.size == xx.size:
                out[k] = v.reshape(xx.shape)
        return out

    def cavity_mode_probe(self, i, x: tuple, y: tuple, family: str | None = None) -> dict:  # type: ignore[override]
        """Fields on a regular grid ``x = (x0, x1, nx)``, ``y = (y0, y1, ny)``; arrays (ny, nx); ``X``, ``Y``."""
        import numpy as np

        X, Y = np.meshgrid(np.linspace(x[0], x[1], int(x[2])), np.linspace(y[0], y[1], int(y[2])))
        out = self.cavity_mode_sample(i, X, Y, family)
        out["X"], out["Y"] = X, Y
        return out

    def save_cavity_modes(self, directory: str | Path | None = None, modes=None, *, subdiv: int = 2) -> list[Path]:
        """One ``<FAM>_<index>.npz`` per mode (arrays of :meth:`cavity_mode_field` plus ``meta``, a JSON string)."""
        import numpy as np

        out_dir = Path(directory) if directory is not None else self.workdir / "cavity_modes"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = self.cavity_modes()
        paths = []
        for k in (range(len(rows)) if modes is None else modes):
            row = rows[k]
            F = self.cavity_mode_field(row, subdiv=subdiv)
            meta = {"design": self.design.name, "symmetry": "xsec", "row": row,
                    "normalization": "int eps_r |E|^2 dA = 1 over the cross-section (E in m^-1); H physical: "
                                     "int mu_r |H|^2 dA = 1/Z0^2 at resonance",
                    "phase": "complex fields E_c = i E, H_c = H (e^{-i omega t})", "subdiv": subdiv}
            arrays = {kk: vv for kk, vv in F.items() if isinstance(vv, np.ndarray)}
            p = out_dir / f"{row['family']}_{row['index']}.npz"
            np.savez_compressed(p, meta=np.array(json.dumps(meta, default=float)), **arrays)
            paths.append(p)
        return paths

    def geometry(self, subdiv: int = 2) -> dict:
        """All elements on a lattice (``x``, ``y``, ``tris``, ``element_region``, ``element_cavity``), region names
        and kinds, and the cavity walls as polylines (``walls``: body, sheet, points)."""
        import numpy as np

        g = json.loads(self._model.geometry(int(subdiv)))
        for k in ("x", "y"):
            g[k] = np.asarray(g[k])
        g["tris"] = np.asarray(g["tris"], dtype=np.int64).reshape(-1, 3)
        return g

    def plot_mode(self, i=0, path: str | Path | None = None, family: str | None = None, *, quantity: str = "Emag",
                  subdiv: int = 3, ax=None, cmap: str = "viridis", colorbar: bool = True, title: str | None = None,
                  dpi: int = 150):
        """Map of a mode on the cross-section (matplotlib): ``quantity`` Emag (default), Hmag, Ez, Ex, Ey, Hx, Hy, Hz
        or "absEz". Conductors are drawn grey, dielectrics outlined, walls black. With ``ax`` draws into it and
        returns the artist; otherwise saves a PNG (default ``<workdir>/<name>_mode_<FAM><index>_<quantity>.png``)
        and returns its path."""
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.tri import Triangulation

        row = self._cavity_mode_row(i, family)
        F = self.cavity_mode_field(row, subdiv=subdiv)
        key = {"Ez": "E_z", "Ex": "E_x", "Ey": "E_y", "Hx": "H_x", "Hy": "H_y", "Hz": "H_z"}.get(quantity, quantity)
        v = np.abs(F["E_z"]) if quantity == "absEz" else F[key]
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
        tri = Triangulation(F["x"], F["y"], F["tris"])
        signed = quantity in ("Ez", "Ex", "Ey", "Hx", "Hy", "Hz")
        vm = float(np.nanmax(np.abs(v))) or 1.0
        art = ax.tripcolor(tri, v, shading="gouraud", cmap=("RdBu_r" if signed and cmap == "viridis" else cmap),
                           vmin=-vm if signed else 0.0, vmax=vm, rasterized=True)
        self._draw_geometry(ax)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        lab = f"{row['family']}_z[{row['index']}] f = {row['f_Hz'] / 1e9:.4f} GHz"
        if row.get("C") is not None and row["family"] == "TM":
            lab += f", C = {row['C']:.3f}"
        ax.set_title(title or lab, fontsize=9)
        if colorbar:
            plt.colorbar(art, ax=ax, shrink=0.85, label=f"{quantity} (normalized)")
        if not own:
            return art
        p = Path(path) if path else self.workdir / f"{self.design.name}_mode_{row['family']}{row['index']}_{quantity}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        return p

    def _draw_geometry(self, ax) -> None:
        import numpy as np
        from matplotlib.collections import LineCollection
        from matplotlib.tri import Triangulation

        g = self.geometry(subdiv=2)
        names, kinds = g["regions"], g["region_kind"]
        er = np.asarray(g["element_region"])
        cav = np.asarray(g["element_cavity"])
        tris = g["tris"].reshape(len(er), -1, 3)
        for ridx, name in enumerate(names):
            kind = kinds.get(name, "vacuum")
            sel = np.where((er == ridx) & (cav == 0))[0]
            if kind == "conductor" and sel.size:
                t = tris[sel].reshape(-1, 3)
                ax.tripcolor(Triangulation(g["x"], g["y"], t), np.zeros(len(g["x"])), cmap="Greys", vmin=-1.0,
                             vmax=1.6, shading="flat", rasterized=True)
        segs = [np.asarray(w["points"]) for w in g["walls"]]
        ax.add_collection(LineCollection(segs, colors="k", linewidths=0.8))

    def summary(self) -> str:
        lines = [f"cross-section cavity {self.design.name!r}: A = {self.fullwave.get('cavity', {}).get('area_m2')} m^2"]
        for r in self.cavity_modes():
            q = r.get("Q0")
            lines.append(f"  {r['family']}_z[{r['index']}] f = {r['f_Hz']:.6e} Hz, C = {r['C']:.4f}"
                         + (f", Q0 = {q:.4g}" if q else ""))
        for w in self.data.get("warnings", []):
            lines.append(f"  warning: {w}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------------------------------- builders
def skin_rf(sigma_S_m: float, T_K: float = 300.0) -> dict:
    """``rf`` block of a normal metal with the classical skin effect (R_s = sqrt(pi f mu0 / sigma))."""
    return {"model": "skin", "rho_ohm_m": 1.0 / sigma_S_m, "T_K": T_K}


def xsec_design(regions: list, *, name: str = "xsec", materials: dict | None = None, curves: list | None = None,
                wall_material: str | None = None, length: float | None = None, end_cap_material: str | None = None,
                modes: int = 6, families: Sequence[str] = ("TM",), target_f_Hz: float | None = None,
                interior_point: Sequence[float] | None = None, h: float | None = None,
                surface_size: float | None = None, order: int = 2, Bz: float | None = None,
                parameters: dict | None = None, losses: bool = True, allow_lossless_walls: bool = False):
    """A ``"symmetry": "xsec"`` design document from regions (shapes in (x, y)), materials and the cavity options.
    ``h``/``surface_size``: element size inside / at the walls (defaults: extent/30, h/2).
    ``allow_lossless_walls``: walls without rf data are lossless on purpose (cavity.allow_lossless_walls)."""
    from .design import Design

    fw: dict = {"modes": int(modes), "families": list(families)}
    if target_f_Hz:
        fw["target_f_Hz"] = float(target_f_Hz)
    cav: dict = {"losses": bool(losses)}
    if interior_point is not None:
        cav["interior_point"] = [float(interior_point[0]), float(interior_point[1])]
    if wall_material:
        cav["wall_material"] = wall_material
    if length:
        cav["length"] = float(length)
    if end_cap_material:
        cav["end_cap_material"] = end_cap_material
    if allow_lossless_walls:
        cav["allow_lossless_walls"] = True
    mesh: dict = {"order": int(order)}
    if h:
        mesh["size"] = float(h)
    if surface_size:
        mesh["surface_size"] = float(surface_size)
    doc: dict = {"schema": "axionfem.design/1", "name": name, "symmetry": "xsec", "materials": dict(materials or {}),
                 "regions": regions, "fullwave": fw, "cavity": cav, "mesh": mesh}
    if curves:
        doc["curves"] = curves
    if parameters:
        doc["parameters"] = dict(parameters)
    if Bz:
        doc["background"] = {"type": "uniform", "Bz": float(Bz)}
    return Design(doc)


def _rod_shape(rod: dict) -> dict:
    a = rod["radius"]
    if "center" in rod:
        return {"disk": {"center": list(rod["center"]), "radius": a}}
    b = {"arm": rod["arm"], "angle_deg": rod["angle_deg"], "radius": a}
    if rod.get("pivot") is not None:
        b["pivot"] = list(rod["pivot"])
    return {"disk": b}


def cylinder_rods_design(R: float, rods: Sequence[dict] = (), *, name: str = "cylinder_rods",
                         wall_rf: dict | None = None, rod_material: dict | None = None, length: float | None = None,
                         modes: int = 6, families: Sequence[str] = ("TM",), target_f_Hz: float | None = None,
                         h: float | None = None, surface_size: float | None = None, order: int = 2,
                         fill: dict | None = None, Bz: float | None = None, interior_point=None):
    """A cylinder of radius R (centred on the origin) with rods spanning its full length.

    ``rods``: dicts with ``radius`` and either ``center`` [x, y] or ``arm``, ``angle_deg`` and an optional
    ``pivot`` [x, y] (centre = pivot + arm (cos, sin)(angle); default pivot = the axis); optional ``material``
    (a material dict, e.g. ``{"kind": "linear", "eps_r": 9.4, "tan_delta": 1e-5}`` for a dielectric rod; default
    ``rod_material``, else the wall's: a conductor) and ``name``. ``wall_rf``: the walls' ``rf`` block (e.g.
    :func:`skin_rf`); None: lossless (Q not reported). ``length``: end-cap losses (end caps of the wall material).
    """
    wall = {"kind": "conductor"}
    if wall_rf:
        wall["rf"] = dict(wall_rf)
    mats: dict = {"wall_cu": wall, "rod_default": copy.deepcopy(rod_material) if rod_material else dict(wall)}
    if fill is not None:
        mats["fill"] = dict(fill)
    regions = [{"name": "cavity", "shape": {"disk": {"center": [0.0, 0.0], "radius": R}},
                "material": "fill" if fill is not None else "vacuum"}]
    for k, rod in enumerate(rods):
        mname = "rod_default"
        if rod.get("material") is not None:
            mname = f"rod{k}_mat"
            mats[mname] = copy.deepcopy(rod["material"])
        regions.append({"name": rod.get("name", f"rod{k}"), "shape": _rod_shape(rod), "material": mname})
    return xsec_design(regions, name=name, materials=mats, wall_material="wall_cu", length=length, modes=modes,
                       families=families, target_f_Hz=target_f_Hz, h=h or R / 25, surface_size=surface_size, order=order,
                       Bz=Bz, interior_point=interior_point, allow_lossless_walls=not wall_rf)


def sector_cell_design(Rc: float, N: int, *, full: bool = False, gap: float = 0.0, name: str | None = None,
                       wall_rf: dict | None = None, length: float | None = None, modes: int = 4,
                       h: float | None = None, surface_size: float | None = None, order: int = 2):
    """CAPP-style multi-cell cavity of radius Rc with N >= 2 cells (zero-thickness PEC partitions on the radii at
    angles 2 pi k / N). ``full=False``: one sector cell (opening angle 2 pi / N) as a region of its own; ``full=True``:
    the full disk with the partitions as conducting sheets (curves) from radius ``gap`` (centre gap R_g; 0: they meet
    on the axis) to the wall; with gap = 0 the cells are separate cavities and cavity.interior_point selects one."""
    th = 2 * math.pi / N
    wall = {"kind": "conductor"}
    if wall_rf:
        wall["rf"] = dict(wall_rf)
    mats = {"wall_cu": wall}
    curves = None
    if not full:
        # disk minus the complement of the wedge [0, th] (a polygon on a circle of radius 4 Rc, steps <= 90 deg)
        n = max(2, int(math.ceil((2 * math.pi - th) / (math.pi / 2))))
        pts = [[0.0, 0.0]] + [[4 * Rc * math.cos(th + (2 * math.pi - th) * j / n),
                                4 * Rc * math.sin(th + (2 * math.pi - th) * j / n)] for j in range(n + 1)]
        shape = {"difference": [{"disk": {"center": [0.0, 0.0], "radius": Rc}}, {"polygon": {"points": pts}}]}
        ip = None
    else:
        shape = {"disk": {"center": [0.0, 0.0], "radius": Rc}}
        curves = [{"name": f"partition{k}", "material": "wall_cu",
                   "path": {"segment": [[gap * math.cos(k * th), gap * math.sin(k * th)],
                                        [Rc * math.cos(k * th), Rc * math.sin(k * th)]]}} for k in range(N)]
        ip = (0.5 * Rc * math.cos(th / 2), 0.5 * Rc * math.sin(th / 2))
    regions = [{"name": "cell", "shape": shape, "material": "vacuum"}]
    return xsec_design(regions, name=name or f"sector_{N}{'_full' if full else ''}", materials=mats, curves=curves,
                       wall_material="wall_cu", length=length, modes=modes, h=h or Rc / 20,
                       surface_size=surface_size, order=order, interior_point=ip, allow_lossless_walls=not wall_rf)


# ---------------------------------------------------------------------------------------------------- sweeps
def sweep(build: Callable[[Any], Any], values: Sequence[Any], *, workdir: str | Path = "runs/xsec_sweep",
          track: str = "C", verbose: int = 0, keep: bool = False, overlap_grid: int = 48) -> list[dict]:
    """Solve ``build(v)`` (a Design) for each value and collect the modes: one dict per value with ``value``,
    ``modes`` (the :meth:`XsecResults.cavity_modes` rows) and ``tracked``, the TM mode chosen by ``track``:

    - ``"C"`` (default): the TM mode with the largest C at every step (the haloscope mode; it jumps branches where
      another mode overtakes it in C, e.g. at an avoided crossing, which is usually what is wanted);
    - ``"overlap"``: start from the largest C, then follow the mode whose E_z (sampled on a fixed
      ``overlap_grid`` x ``overlap_grid`` grid over the first cross-section, zero in metal) has the largest
      |normalized overlap| with the previous tracked mode; robust at true crossings (different symmetry), follows
      the adiabatic branch through avoided crossings. ``tracked_overlap`` holds that overlap; below 0.5 the step is
      flagged in ``warnings`` (steps too coarse to identify the mode);
    - ``"nearest"``: the mode nearest in frequency to the previous tracked one (fragile: it jumps to the other mode
      at a crossing whenever the step moves the tracked mode further than the gap; prefer "overlap").

    ``keep``: also return the XsecResults (``result``)."""
    import numpy as np

    if track not in ("C", "overlap", "nearest"):
        raise ValueError(f"track must be 'C', 'overlap' or 'nearest', not {track!r}")
    out = []
    prev = None
    prev_vec = None
    grid = None

    def vec(res, row):
        v = np.nan_to_num(np.asarray(res.cavity_mode_sample(row, grid[0], grid[1])["E_z"], dtype=float)).ravel()
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    for k, v in enumerate(values):
        res = build(v).solve(workdir=Path(workdir) / f"step{k:03d}", verbose=verbose)
        rows = res.cavity_modes()
        tm = [r for r in rows if r["family"] == "TM"]
        tr = None
        warn = list(res.data.get("warnings", []))
        ov = None
        if tm:
            if track == "nearest" and prev is not None:
                tr = min(tm, key=lambda r: abs(r["f_Hz"] - prev))
            elif track == "overlap" and prev_vec is not None:
                ovs = [abs(float(prev_vec @ vec(res, r))) for r in tm]
                i = int(np.argmax(ovs))
                tr, ov = tm[i], ovs[i]
                if ov < 0.5:
                    warn.append(f"sweep: tracked-mode overlap {ov:.2f} < 0.5 at value {v!r}: the mode identification is "
                                "ambiguous (refine the steps)")
            else:
                tr = max(tm, key=lambda r: r["C"] or 0.0)
            prev = tr["f_Hz"]
            if track == "overlap":
                if grid is None:
                    g = res.geometry(subdiv=1)
                    xs = np.linspace(float(np.min(g["x"])), float(np.max(g["x"])), int(overlap_grid))
                    ys = np.linspace(float(np.min(g["y"])), float(np.max(g["y"])), int(overlap_grid))
                    grid = np.meshgrid(xs, ys)
                prev_vec = vec(res, tr)
        d = {"value": v, "modes": rows, "tracked": tr, "warnings": warn}
        if track == "overlap":
            d["tracked_overlap"] = ov
        if keep:
            d["result"] = res
        out.append(d)
    return out


def rod_angle_sweep(R: float, rods: Sequence[dict], angles_deg: Sequence[float], *, moving: Sequence[int] | None = None,
                    workdir: str | Path = "runs/rod_sweep", track: str = "C", verbose: int = 0, keep: bool = False,
                    **design_kw) -> list[dict]:
    """Rotate the rods ``moving`` (indices into ``rods``; default all rods given by arm/angle) to ``angle_deg + a0``
    (a0 = each rod's own ``angle_deg``, so the list holds offsets) for each angle in ``angles_deg``, solving
    :func:`cylinder_rods_design` each time; returns :func:`sweep` rows (value = the angle)."""
    mv = list(range(len(rods))) if moving is None else list(moving)

    def build(a):
        rr = [dict(r) for r in rods]
        for i in mv:
            rr[i]["angle_deg"] = rods[i].get("angle_deg", 0.0) + a
        return cylinder_rods_design(R, rr, **design_kw)

    return sweep(build, angles_deg, workdir=workdir, track=track, verbose=verbose, keep=keep)


def plot_mode_map(rows: list[dict], path: str | Path, *, xlabel: str = "rod angle [deg]", families=("TM", "TE"),
                  title: str | None = None) -> Path:
    """Mode map of a :func:`sweep`: f of every mode vs the swept value, TM coloured by C, TE grey; the tracked mode
    as a line."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    xs, fs, cs = [], [], []
    for r in rows:
        for m in r["modes"]:
            if m["family"] == "TM" and "TM" in families:
                xs.append(r["value"]); fs.append(m["f_Hz"] / 1e9); cs.append(m["C"] or 0.0)
            elif m["family"] == "TE" and "TE" in families:
                ax.plot(r["value"], m["f_Hz"] / 1e9, ".", color="0.6", ms=3)
    sc = ax.scatter(xs, fs, c=cs, cmap="viridis", vmin=0, vmax=0.7, s=12)
    tr = [(r["value"], r["tracked"]["f_Hz"] / 1e9) for r in rows if r["tracked"]]
    if tr:
        ax.plot(*zip(*tr), "-", color="C3", lw=1, label="tracked (max C)")
        ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="C (TM_z)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("f [GHz]")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    p = Path(path)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------------------------------- closed forms
def disk_tm_modes(R: float, f_max: float, eps_r: float = 1.0) -> list[dict]:
    """TM_mn0 modes of an empty (or uniformly filled) disk cross-section below f_max: k = j_mn / R; C = 4/j_0n^2 for
    m = 0 (0 otherwise); G_2d = Z0 j_mn / 2 (barrel per unit length, vacuum; any m); degeneracy 2 for m > 0."""
    from scipy.special import jn_zeros

    kmax = 2 * math.pi * f_max * math.sqrt(eps_r) / C_LIGHT
    out = []
    for m in range(0, int(kmax * R) + 2):
        for n, x in enumerate(jn_zeros(m, int(kmax * R / math.pi) + 3), start=1):
            if x / R > kmax:
                break
            out.append({"m": m, "n": n, "chi": float(x), "f_Hz": C_LIGHT * x / (2 * math.pi * R * math.sqrt(eps_r)),
                        "C": 4 / x ** 2 if m == 0 else 0.0, "G2d_ohm": Z0 * x / 2, "degeneracy": 1 if m == 0 else 2})
    out.sort(key=lambda r: r["f_Hz"])
    return out


def disk_te_modes(R: float, f_max: float) -> list[dict]:
    """TE_mn (z-independent H_z) modes of an empty disk: k = j'_mn / R (zeros of J_m'), C = 0,
    G_2d = Z0 j'_mn (1 - m^2/j'_mn^2) / 2."""
    from scipy.special import jnp_zeros

    kmax = 2 * math.pi * f_max / C_LIGHT
    out = []
    for m in range(0, int(kmax * R) + 2):
        for n, x in enumerate(jnp_zeros(m, int(kmax * R / math.pi) + 3), start=1):
            if x / R > kmax:
                break
            out.append({"m": m, "n": n, "chi": float(x), "f_Hz": C_LIGHT * x / (2 * math.pi * R), "C": 0.0,
                        "G2d_ohm": Z0 * x * (1 - m * m / (x * x)) / 2, "degeneracy": 1 if m == 0 else 2})
    out.sort(key=lambda r: r["f_Hz"])
    return out


def coax_tm_G2d(a: float, b: float) -> dict:
    """Lowest TM_z mode of a coaxial cross-section (rod a, wall b): k, f, C (:func:`axionfem.cavity.coax_tm_mode`)
    and the barrel geometry factor per unit length G_2d = omega mu0 int H^2 dA / oint H_t^2 dl, split into the rod
    and the outer wall (G = omega mu0 int Z1^2 r dr / (a Z1(ka)^2 + b Z1(kb)^2), Z1 the radial derivative profile)."""
    from scipy.integrate import quad
    from scipy.special import j0, j1, y0, y1

    from .cavity import coax_tm_mode

    m = coax_tm_mode(a, b)
    k = m["k"]
    z1 = lambda r: j1(k * r) * y0(k * a) - y1(k * r) * j0(k * a)  # noqa: E731  (E_z = Z0 profile, H_phi ~ Z1)
    I = quad(lambda r: z1(r) ** 2 * r, a, b, epsabs=0, epsrel=1e-13, limit=200)[0]
    w = C_LIGHT * k
    Ga = w * MU0 * I / (a * z1(a) ** 2)
    Gb = w * MU0 * I / (b * z1(b) ** 2)
    return {**m, "G2d_ohm": 1 / (1 / Ga + 1 / Gb), "G2d_rod_ohm": Ga, "G2d_wall_ohm": Gb}


def sector_cell_mode(N: int, Rc: float, L: float | None = None) -> dict:
    """Closed-form TM_{nu,1} mode of an ideal PEC sector cell (N cells, opening 2 pi/N, nu = N/2; N = 0: the empty
    disk), Ez = J_nu(k r) sin(nu phi), k = j_{nu,1}/Rc (as benchmarks/cavity/refs/digitize/capp_sector_analytic.py):
    f, C, G_2d (= Q R_s without end caps) and, with L, G with end caps (1/G = 1/G_2d + 2/(omega mu0 L))."""
    import numpy as np
    from scipy import integrate, optimize, special as sp

    if N == 0:
        nu, th = 0.0, 2 * math.pi
    else:
        th = 2 * math.pi / N
        nu = math.pi / th
    xs = np.linspace(max(0.5, nu), nu + 6 + 2 * nu ** (1 / 3), 4000)
    v = sp.jv(nu, xs)
    i = int(np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0][0])
    x = optimize.brentq(lambda t: sp.jv(nu, t), xs[i], xs[i + 1], xtol=1e-15)
    k = x / Rc
    I2 = integrate.quad(lambda r: sp.jv(nu, k * r) ** 2 * r, 0, Rc, epsabs=0, epsrel=1e-13, limit=200)[0]
    I1 = integrate.quad(lambda r: sp.jv(nu, k * r) * r, 0, Rc, epsabs=0, epsrel=1e-13, limit=200)[0]
    ang2 = th / 2 if N else 2 * math.pi
    ang1 = 2 / nu if N else 2 * math.pi
    E2 = ang2 * I2
    arc = k ** 2 * sp.jvp(nu, x) ** 2 * Rc * ang2
    sides = 0.0
    if N:
        sides = 2 * integrate.quad(lambda r: (nu / r) ** 2 * sp.jv(nu, k * r) ** 2 if r > 0 else 0.0, 0, Rc,
                                   epsabs=0, epsrel=1e-13, limit=200)[0]
    w = C_LIGHT * k
    A = th * Rc ** 2 / 2 if N else math.pi * Rc ** 2
    G2 = w * MU0 * k ** 2 * E2 / (arc + sides)
    out = {"N": N, "nu": nu, "chi": x, "f_Hz": w / (2 * math.pi), "C": (ang1 * I1) ** 2 / (A * E2), "G2d_ohm": G2,
           "A_m2": A}
    if L:
        out["G_ohm"] = 1 / (1 / G2 + 2 / (w * MU0 * L))
    return out


def z_dependent_spectrum(rows: list[dict], length: float, p_max: int = 3) -> list[dict]:
    """The z-dependent modes of a capped cavity of the given length built from the cross-section modes, valid for a
    VACUUM cross-section (separable): TM_mnp f = sqrt(f_c^2 + (p c / 2L)^2), p = 0..p_max; TE_mnp, p = 1..p_max.
    ``rows``: :meth:`XsecResults.cavity_modes` (both families for the full spectrum). With dielectrics the p >= 1
    modes are hybrid and this list is not valid (use the 3D solver)."""
    out = []
    for r in rows:
        for p in range(0 if r["family"] == "TM" else 1, p_max + 1):
            f = math.hypot(r["f_Hz"], p * C_LIGHT / (2 * length))
            out.append({"family": r["family"], "index": r["index"], "p": p, "f_Hz": f,
                        "C": r["C"] if (r["family"] == "TM" and p == 0) else 0.0})
    out.sort(key=lambda m: m["f_Hz"])
    return out


def eccentric_rod_tm_mode(R: float, a: float, d: float, k_bracket: tuple[float, float] | None = None, *,
                          n_outer: int = 36, n_rod: int = 24, n_pts: int = 160) -> dict:
    """Independent reference (no FEM) for an off-centre PEC rod: lowest TM_z mode symmetric about the x axis of a
    disk of radius R with a rod of radius a centred at (d, 0), by the method of particular solutions (Betcke &
    Trefethen, SIAM Rev. 47 (2005) 469): E_z = sum A_n J_n(k r) cos(n theta) + sum B_n Y_n(k rho) cos(n phi) ((r, theta)
    about the axis, (rho, phi) about the rod centre); k minimizes the smallest singular value of the boundary block of
    the orthonormalized basis (interior points prevent the trivial solution). C and G_2d follow from boundary integrals
    of the normal derivative (exact identities for a Dirichlet eigenfunction of -Laplacian):
    int u dA = -(1/k^2) oint du/dn dl, int u^2 dA = (1/(2 k^2)) oint (x.n) (du/dn)^2 dl (Rellich),
    G_2d = omega mu0 k^2 int u^2 dA / oint (du/dn)^2 dl (split into wall and rod). ``k_bracket``: search interval
    (default: around the empty-disk TM010). Accuracy ~1e-9 in k for d + a < R - 0.1 R."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    from scipy.special import jv, jvp, yv, yvp

    if not (0 < a and d >= 0 and d + a < R):
        raise ValueError("need 0 < a, d >= 0 and d + a < R (rod inside the disk)")
    nO, nR = np.arange(n_outer + 1), np.arange(n_rod + 1)
    tb = np.linspace(0, np.pi, n_pts)  # half circles suffice (even modes); endpoints included
    Pout = np.c_[R * np.cos(tb), R * np.sin(tb)]
    Prod = np.c_[d + a * np.cos(tb), a * np.sin(tb)]
    rng = np.random.default_rng(1)
    pts = []
    while len(pts) < 2 * n_pts:  # interior points in the upper half
        p = rng.uniform([-R, 0], [R, R])
        if np.hypot(*p) < 0.97 * R and np.hypot(p[0] - d, p[1]) > 1.03 * a:
            pts.append(p)
    Pin = np.array(pts)

    def basis(P, k, deriv=False):
        r, th = np.hypot(P[:, 0], P[:, 1]), np.arctan2(P[:, 1], P[:, 0])
        rho, ph = np.hypot(P[:, 0] - d, P[:, 1]), np.arctan2(P[:, 1], P[:, 0] - d)
        if not deriv:
            return np.c_[jv(nO[None], k * r[:, None]) * np.cos(nO[None] * th[:, None]),
                         yv(nR[None], k * rho[:, None]) * np.cos(nR[None] * ph[:, None])]
        out = []
        for (rr, tt, nn, f, fp) in ((r, th, nO, jv, jvp), (rho, ph, nR, yv, yvp)):
            dr = k * fp(nn[None], k * rr[:, None]) * np.cos(nn[None] * tt[:, None])
            dt = -nn[None] * f(nn[None], k * rr[:, None]) * np.sin(nn[None] * tt[:, None]) / rr[:, None]
            c, s = np.cos(tt)[:, None], np.sin(tt)[:, None]
            out.append((c * dr - s * dt, s * dr + c * dt))
        return np.c_[out[0][0], out[1][0]], np.c_[out[0][1], out[1][1]]

    def smin(k):
        A = np.r_[basis(Pout, k), basis(Prod, k), basis(Pin, k)]
        Q, _ = np.linalg.qr(A)
        return np.linalg.svd(Q[: 2 * n_pts], compute_uv=False)[-1]

    k0 = 2.404825557695773 / R
    lo, hi = k_bracket or (0.6 * k0, 1.8 * k0)
    ks = np.linspace(lo, hi, 161)
    s = np.array([smin(k) for k in ks])
    # the lowest eigenvalue in the bracket: the first local minimum of sigma_min that refines to ~0
    res = None
    for i in range(len(ks)):
        if (i > 0 and s[i] > s[i - 1]) or (i + 1 < len(ks) and s[i] > s[i + 1]):
            continue
        r = minimize_scalar(smin, bounds=(ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]), method="bounded",
                            options={"xatol": 1e-13 * k0})
        if r.fun < 1e-5:
            res = r
            break
    if res is None:
        raise ValueError(f"no eigenvalue found in k = [{lo}, {hi}] (smallest sigma_min {s.min():.2e})")
    k = float(res.x)
    A = np.r_[basis(Pout, k), basis(Prod, k), basis(Pin, k)]
    Q, Rm = np.linalg.qr(A)
    v = np.linalg.svd(Q[: 2 * n_pts])[2][-1]
    coef = np.linalg.solve(Rm, v)
    # boundary integrals on the full circles (trapezoid, periodic)
    t = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    acc = {}
    for name, cx_, rad, sgn in (("wall", 0.0, R, 1.0), ("rod", d, a, -1.0)):
        P = np.c_[cx_ + rad * np.cos(t), rad * np.sin(t)]
        gx, gy = basis(P, k, deriv=True)
        nx, ny = sgn * np.cos(t), sgn * np.sin(t)  # outward normal of the cavity domain
        dn = (gx @ coef) * nx + (gy @ coef) * ny
        dl = rad * 2 * np.pi / t.size
        acc[name] = (np.sum(dn) * dl, np.sum((P[:, 0] * nx + P[:, 1] * ny) * dn ** 2) * dl, np.sum(dn ** 2) * dl)
    I1 = -(acc["wall"][0] + acc["rod"][0]) / k ** 2
    I2 = (acc["wall"][1] + acc["rod"][1]) / (2 * k ** 2)
    A_ = math.pi * (R * R - a * a)
    w = C_LIGHT * k
    Gw = w * MU0 * k ** 2 * I2 / acc["wall"][2]
    Gr = w * MU0 * k ** 2 * I2 / acc["rod"][2]
    bnd = float(np.max(np.abs(np.r_[basis(Pout, k), basis(Prod, k)] @ coef)) / np.max(np.abs(basis(Pin, k) @ coef)))
    return {"k": k, "kR": k * R, "f_Hz": w / (2 * math.pi), "C": I1 ** 2 / (A_ * I2), "C_full": I1 ** 2 / (math.pi * R * R * I2),
            "G2d_ohm": 1 / (1 / Gw + 1 / Gr), "G2d_wall_ohm": Gw, "G2d_rod_ohm": Gr, "boundary_residual": bnd,
            "sigma_min": float(res.fun)}


def eccentric_dielectric_rod_tm_mode(R: float, a: float, d: float, eps_r: float,
                                     k_bracket: tuple[float, float] | None = None, *, n_outer: int = 40,
                                     n_rod: int = 30, n_pts: int = 200) -> dict:
    """Independent reference (no FEM) for an off-centre DIELECTRIC rod (relative permittivity eps_r, mu_r = 1,
    radius a, centre (d, 0)) in a PEC disk of radius R: lowest TM_z mode symmetric about the x axis by the method of
    particular solutions. Outside the rod E_z = sum A_n J_n(k r) cos(n theta) + sum B_n
    Y_n(k rho) cos(n phi), inside sum C_n J_n(k sqrt(eps_r) rho) cos(n phi) ((r, theta) about the axis, (rho, phi)
    about the rod); E_z = 0 on the wall, E_z and dE_z/drho continuous on the rod surface. Area integrals by spectral
    quadrature in polar coordinates about the rod centre. Returns k, f_Hz, C (= |int E_z dA|^2 / (pi R^2 int eps_r
    E_z^2 dA)), fe_rod (electric energy fraction in the rod), G2d_wall_ohm, boundary_residual."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    from scipy.special import jv, jvp, yv, yvp

    if not (0 < a and d >= 0 and d + a < R and eps_r >= 1):
        raise ValueError("need 0 < a, d >= 0, d + a < R (rod inside the disk) and eps_r >= 1")
    nr = math.sqrt(eps_r)
    nO, nR = np.arange(n_outer + 1), np.arange(n_rod + 1)
    tb = np.linspace(0, np.pi, n_pts)
    Pw = np.c_[R * np.cos(tb), R * np.sin(tb)]
    Pr = np.c_[d + a * np.cos(tb), a * np.sin(tb)]
    rng = np.random.default_rng(3)
    Po, Pi = [], []
    while len(Po) < n_pts:
        p = rng.uniform([-R, 0], [R, R])
        if np.hypot(*p) < 0.97 * R and np.hypot(p[0] - d, p[1]) > 1.03 * a:
            Po.append(p)
    while len(Pi) < n_pts // 2:
        p = rng.uniform([d - a, 0], [d + a, a])
        if np.hypot(p[0] - d, p[1]) < 0.97 * a:
            Pi.append(p)
    Po, Pi = np.array(Po), np.array(Pi)

    def cols(P, k, c, ns, f):
        rho, ph = np.hypot(P[:, 0] - c, P[:, 1]), np.arctan2(P[:, 1], P[:, 0] - c)
        return f(ns[None], k * rho[:, None]) * np.cos(ns[None] * ph[:, None])

    def grad(P, k, c, ns, f, fp):
        rho, ph = np.hypot(P[:, 0] - c, P[:, 1]), np.arctan2(P[:, 1], P[:, 0] - c)
        dr = k * fp(ns[None], k * rho[:, None]) * np.cos(ns[None] * ph[:, None])
        dt = -ns[None] * f(ns[None], k * rho[:, None]) * np.sin(ns[None] * ph[:, None]) / rho[:, None]
        cs, sn = np.cos(ph)[:, None], np.sin(ph)[:, None]
        return cs * dr - sn * dt, sn * dr + cs * dt

    no, nn = n_outer + 1, n_rod + 1
    nrm = np.c_[np.cos(tb), np.sin(tb)]

    def rows(k):
        dn = lambda g: g[0] * nrm[:, :1] + g[1] * nrm[:, 1:]  # noqa: E731
        wall = np.c_[cols(Pw, k, 0.0, nO, jv), cols(Pw, k, d, nR, yv), np.zeros((n_pts, nn))]
        cont = np.c_[cols(Pr, k, 0.0, nO, jv), cols(Pr, k, d, nR, yv), -cols(Pr, k * nr, d, nR, jv)]
        flux = np.c_[dn(grad(Pr, k, 0.0, nO, jv, jvp)), dn(grad(Pr, k, d, nR, yv, yvp)),
                     -dn(grad(Pr, k * nr, d, nR, jv, jvp))] / k
        inner = np.r_[np.c_[cols(Po, k, 0.0, nO, jv), cols(Po, k, d, nR, yv), np.zeros((len(Po), nn))],
                      np.c_[np.zeros((len(Pi), no + nn)), cols(Pi, k * nr, d, nR, jv)]]
        return np.r_[wall, cont, flux], inner

    def smin(k):
        B, I = rows(k)
        Q, _ = np.linalg.qr(np.r_[B, I])
        return np.linalg.svd(Q[: len(B)], compute_uv=False)[-1]

    k0 = 2.404825557695773 / R
    lo, hi = k_bracket or (0.3 * k0, 1.5 * k0)
    ks = np.linspace(lo, hi, 161)
    s = np.array([smin(k) for k in ks])
    res = None
    for i in range(len(ks)):
        if (i > 0 and s[i] > s[i - 1]) or (i + 1 < len(ks) and s[i] > s[i + 1]):
            continue
        r = minimize_scalar(smin, bounds=(ks[max(i - 1, 0)], ks[min(i + 1, len(ks) - 1)]), method="bounded",
                            options={"xatol": 1e-14 * k0})
        if r.fun < 1e-6:
            res = r
            break
    if res is None:
        raise ValueError(f"no eigenvalue found in k = [{lo}, {hi}] (smallest sigma_min {s.min():.2e})")
    k = float(res.x)
    B, I = rows(k)
    Q, Rm = np.linalg.qr(np.r_[B, I])
    c = np.linalg.solve(Rm, np.linalg.svd(Q[: len(B)])[2][-1])
    cA, cB, cC = c[:no], c[no:no + nn], c[no + nn:]
    xg, wg = np.polynomial.legendre.leggauss(80)
    nph = 512
    I1 = I2o = I2i = 0.0
    for p in np.linspace(0, 2 * np.pi, nph, endpoint=False):
        rw = -d * math.cos(p) + math.sqrt(R * R - (d * math.sin(p)) ** 2)  # rod centre to the wall along phi
        for r0, r1, inside in ((0.0, a, True), (a, rw, False)):
            rr = r0 + 0.5 * (r1 - r0) * (xg + 1)
            P = np.c_[d + rr * math.cos(p), rr * math.sin(p)]
            u = (cols(P, k * nr, d, nR, jv) @ cC if inside
                 else cols(P, k, 0.0, nO, jv) @ cA + cols(P, k, d, nR, yv) @ cB)
            w = 0.5 * (r1 - r0) * wg * rr * (2 * np.pi / nph)
            I1 += float(np.sum(w * u))
            if inside:
                I2i += float(np.sum(w * u * u))
            else:
                I2o += float(np.sum(w * u * u))
    Ie = I2o + eps_r * I2i
    t = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    P = np.c_[R * np.cos(t), R * np.sin(t)]
    g1, g2 = grad(P, k, 0.0, nO, jv, jvp), grad(P, k, d, nR, yv, yvp)
    dn = (g1[0] @ cA + g2[0] @ cB) * np.cos(t) + (g1[1] @ cA + g2[1] @ cB) * np.sin(t)
    S = float(np.sum(dn ** 2)) * R * 2 * np.pi / t.size
    w = C_LIGHT * k
    bnd = float(np.max(np.abs(B @ c)) / np.max(np.abs(I @ c)))
    return {"k": k, "kR": k * R, "f_Hz": w / (2 * math.pi), "C": I1 ** 2 / (math.pi * R * R * Ie),
            "fe_rod": eps_r * I2i / Ie, "G2d_wall_ohm": w * MU0 * k * k * Ie / S, "boundary_residual": bnd,
            "sigma_min": float(res.fun)}
