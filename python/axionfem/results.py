"""Solved designs: numbers, summaries, scenes, field samples."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from . import _core

MU0 = 1.25663706212e-6


def requirement(benchmark: str | dict = "optimistic", **scan: Any) -> dict:
    """F_req for a benchmark name ("optimistic", "conservative", "conservative_solenoid")
    or a dict {name, Q, T, eta_A}; keyword arguments override scan settings
    (f_min, f_max, live_time [yr], model, C_QCD, rho_DM, Q_axion, Theta95)."""
    b = benchmark if isinstance(benchmark, dict) else benchmark
    return json.loads(_core.requirement(json.dumps(b), json.dumps(scan)))


def conductors() -> dict:
    """Built-in superconductor critical surfaces ("nbti", "nb3sn", "rebco") with their parameters
    and a few Jc values in a perpendicular field (docs/validation/hts.md)."""
    return json.loads(_core.conductors())


def jc(conductor: str | dict, B_perp, B_par=None, T=None):
    """Critical current density Jc [A/m^2] of a conductor (a built-in name or a `conductors` entry
    of the design format) at fields B_perp (normal to the tape) and B_par [T] and temperature T [K].
    Scalars or sequences; B_par defaults to 0 and T to the conductor's T_op."""
    import numpy as np

    bp = np.atleast_1d(np.asarray(B_perp, dtype=float))
    bt = np.zeros_like(bp) if B_par is None else np.broadcast_to(np.asarray(B_par, dtype=float), bp.shape)
    tt = [] if T is None else list(np.broadcast_to(np.asarray(T, dtype=float), bp.shape))
    out = _core.conductor_jc(json.dumps(conductor), list(bp), list(bt), tt)
    return out if np.ndim(B_perp) else float(out[0])


def _b64(a, dtype: str) -> dict:
    """A typed array of the scene format (docs/scene_format.md)."""
    import base64

    import numpy as np

    arr = np.ascontiguousarray(np.asarray(a), dtype=dtype)
    shape = list(arr.shape) if arr.ndim > 1 else [arr.size, 1]
    return {"dtype": dtype, "shape": shape, "data": base64.b64encode(arr.tobytes()).decode("ascii")}


def _contour_segments(x, y, v, tris, levels):
    """Marching triangles: line segments (n, 4) = (x0, y0, x1, y1) of the level sets of the piecewise-linear v."""
    import numpy as np

    tris = np.asarray(tris)
    P = np.column_stack([x, y])
    out = []
    for lev in levels:
        s = v[tris] - lev                                 # (nt, 3)
        pts, hits = [], []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            sa, sb = s[:, a], s[:, b]
            cross = sa * sb < 0
            t = np.where(cross, sa / np.where(cross, sa - sb, 1.0), 0.0)
            pa, pb = P[tris[:, a]], P[tris[:, b]]
            pts.append(pa + t[:, None] * (pb - pa))
            hits.append(cross)
        H = np.column_stack(hits)
        two = H.sum(axis=1) == 2
        if not two.any():
            continue
        pts = np.stack(pts, axis=1)[two]                  # (m, 3, 2)
        H = H[two]
        first = np.argmax(H, axis=1)
        last = 2 - np.argmax(H[:, ::-1], axis=1)
        idx = np.arange(len(H))
        out.append(np.column_stack([pts[idx, first], pts[idx, last]]))
    return np.concatenate(out) if out else np.zeros((0, 4))


def _fmt_location(loc) -> str:
    """'  at r = 1.200 m, z = 0.087 m (winding)' from a location dict of the results (coordinates *_m, region)."""
    if not isinstance(loc, dict) or not loc:
        return ""
    xs = [f"{k[:-2]} = {v:.3f} m" for k, v in loc.items() if k.endswith("_m") and isinstance(v, (int, float))]
    reg = loc.get("region")
    return "  at " + ", ".join(xs) + (f" ({reg})" if reg else "") if xs or reg else ""


def _fmt(x: float, unit: str = "", digits: int = 4) -> str:
    if x is None:
        return "-"
    if x == 0:
        return f"0 {unit}".strip()
    return f"{x:.{digits}g} {unit}".strip()


class Results:
    """Results of a solved design.

    ``res.data`` is the full results document (also written to
    ``<workdir>/results.json``).  Convenience properties give the most used
    numbers in SI units.
    """

    def __init__(self, model: "_core.Model", design, workdir: Path):
        self._model = model
        self.design = design
        self.workdir = Path(workdir)
        self.data: dict = json.loads(model.results())
        self.workdir.mkdir(parents=True, exist_ok=True)
        (self.workdir / "results.json").write_text(json.dumps(self.data, indent=2))

    # ------------------------------------------------------------ numbers
    @property
    def coupling(self) -> dict:
        return self.data.get("coupling", {})

    @property
    def forces(self) -> dict:
        """Lorentz / Maxwell forces.

        Per winding: ``Fz_N`` and ``radial_force_per_radian_N`` (the hoop load); for a powered
        azimuthal winding also ``Fz_self_N`` (the numerical self-force, exactly zero in theory, so
        the error bar on ``Fz_N``) and ``Fz_excluding_self_N`` (the force from all other sources --
        the number to quote for a linear design).  Per body: ``superconductor:<name>`` /
        ``iron:<name>`` Maxwell pressure forces.  ``closure`` is the Newton's-third-law sum over all
        bodies, which must vanish.  ``distributions`` holds the distributed load of each body; plot
        it with :func:`axionfem.plots.force_distribution`.
        """
        return self.data.get("forces", {})

    @property
    def Psi(self) -> float:
        """Signal coupling Psi = (1/I) int A_I . B_lab d^3x  [H T m^2] (SI)."""
        return self.coupling.get("Psi_SI_H_T_m2", 0.0)

    @property
    def L_p(self) -> float:
        """Pickup inductance [H]."""
        return self.coupling.get("L_p_H", 0.0)

    @property
    def F(self) -> float:
        """Scan figure of merit Psi^4 / L_p^2 [T^4 H^2 m^8]."""
        return self.coupling.get("F_SI_T4_H2_m8", 0.0)

    @property
    def W(self) -> float:
        """Stored magnetic energy [J]."""
        return self.coupling.get("W_mag_J", 0.0)

    @property
    def g(self) -> float:
        """Geometric factor g = Psi^2/(L_p W R^2) (natural units)."""
        return self.coupling.get("g_factor", 0.0)

    @property
    def fullwave(self) -> dict:
        """Full-wave analysis beyond MQS (present when the design has a `fullwave` section): `sweep` rows
        with Psi(omega), Z(omega), L(omega), R_rad, Q_rad, F(omega); `eigenmodes`."""
        return self.data.get("fullwave", {})

    def fullwave_losses(self, port: int = 0) -> dict:
        """The surface losses of the full-wave sweep per solved frequency as arrays: ``f_Hz``, ``Gamma`` and
        ``R_eff_ohm`` per surface, ``R_loss_ohm``, ``Q_loss``, ``Q_total``, ``T_over_Q_K`` and, from the 3D engine,
        ``L_eff_H``, ``Q_loss_tuned``, ``Q_total_tuned``, ``T_over_Q_tuned_K``, ``R_loss_over_abs_X``
        (:func:`axionfem.circuit.fullwave_losses`; definitions in docs/physics/fullwave_surface_losses.md). The
        summary of a 3D sweep is ``fullwave["surface_losses"]``. The R_s models behind ``R_loss`` are beta."""
        from . import circuit as _ck
        return _ck.fullwave_losses(self.data, port)

    def fullwave_dense(self) -> dict:
        """Beta (the adaptive sweep). The dense curve of an adaptive 3D full-wave sweep (``fullwave.adaptive``) as
        arrays: ``f_Hz``, ``Z``, ``L_H``, ``Psi`` (port 0), ``Z_matrix``, ``Psi_vector``, ``dZ_domega_H``,
        ``estimated_relative_error`` (:func:`axionfem.circuit.fullwave_dense`; :func:`axionfem.circuit.fullwave_model`
        evaluates the model at any frequency)."""
        from . import circuit as _ck
        return _ck.fullwave_dense(self.data)

    def write_touchstone(self, path: str | Path | None = None, kind: str = "S", z0: float = 50.0,
                         dense: bool = False, **kw) -> dict:
        """Write the full-wave port parameters as a Touchstone v1 file (``.sNp``; ``kind`` "S", "Z" or "Y",
        reference resistance ``z0`` ohm, ``dense``: the adaptive model's dense curve instead of the solved rows;
        ``surface_losses=True`` adds the perturbative conductor loss).  The values are in the engineering
        convention exp(+j omega t): the engine's exp(-i omega t) Z is conjugated.  ``path`` defaults to
        ``<workdir>/<design>[_dense]_<kind>.s<N>p`` (the design name sanitised; an existing file is overwritten, see
        ``overwrote`` in the report).  Returns the report (sigma_max of S, passivity violations, reciprocity error,
        warnings) of :func:`axionfem.touchstone.write_touchstone`; docs/physics/touchstone.md."""
        import warnings as _w

        from . import touchstone as _ts
        if path is None:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                n = _ts.port_parameters(self.data, dense=dense, surface_losses=kw.get("surface_losses", False))["Z"]
            stem = _ts.safe_stem(self.design.name)
            path = self.workdir / f"{stem}{'_dense' if dense else ''}_{kind}.s{n.shape[-1]}p"
        data = dict(self.data)
        data.setdefault("design", {"name": self.design.name})
        return _ts.write_touchstone(data, path, kind, z0, dense, **kw)

    def cavity_modes(self, family: str | None = None) -> list[dict]:
        """Eigenmodes of a cavity design (design section ``cavity``) as flat rows sorted by
        frequency: ``family`` ("TE"/"TM"; "3D" for 3D designs), ``index`` (within the family), ``f_Hz``, ``C`` (B0 at the reference
        point), ``C_rms``, ``V_m3``, ``G_ohm`` (geometry factor Q R_s of all walls), ``Q_wall``,
        ``Q_dielectric``, ``Q0``, ``T_over_Q_K``, ``residual_true``, ``energy_fraction``, ``walls`` (per body:
        G_ohm, Rs_ohm, Q, P_frac). Missing quantities are None. Empty for designs without a cavity section."""
        fw = self.fullwave
        if "cavity" not in fw:
            return []
        rows = []
        eig = fw.get("eigenmodes", {})
        blocks = {"3D": eig} if isinstance(eig.get("modes"), list) else eig  # 3D: one block, family "3D"
        for fam, block in blocks.items():
            if family is not None and fam != family:
                continue
            for i, m in enumerate(block.get("modes", [])):
                ff, lo = m.get("form_factor", {}), m.get("losses", {})
                rows.append({"family": fam, "index": i, "f_Hz": m["f_Hz"], "C": ff.get("C"), "C_rms": ff.get("C_rms"),
                             "V_m3": ff.get("V_m3"), "G_ohm": lo.get("G_ohm"), "Q_wall": lo.get("Q_wall"),
                             "Q_dielectric": lo.get("Q_dielectric"), "Q0": lo.get("Q0"),
                             "T_over_Q_K": lo.get("T_over_Q_K"), "residual_true": m.get("residual_true"),
                             "energy_fraction": m.get("energy_fraction", {}), "walls": lo.get("walls", {})})
        rows.sort(key=lambda r: r["f_Hz"])
        return rows

    # ---------------------------------------------- cavity eigenmode fields (2D) -- begin
    def _cavity_mode_row(self, i, family: str | None) -> dict:
        """The cavity_modes() row of mode ``i``: a row dict, an index into ``cavity_modes()`` (all families, sorted
        by frequency) or, with ``family``, the index within that family (the order of the results file)."""
        rows = self.cavity_modes()
        if rows and rows[0]["family"] == "3D":
            raise NotImplementedError("cavity mode fields on the mesh are available for axisymmetric and cross-section "
                                      "designs only; 3D cavity mode fields are not exported yet (planned)")
        if isinstance(i, dict):
            return i
        if not rows:
            raise ValueError("this result has no cavity eigenmodes (the design needs a `cavity` section and "
                             "fullwave.modes > 0)")
        if family is None:
            if not -len(rows) <= int(i) < len(rows):
                raise IndexError(f"cavity mode {i} out of range: {len(rows)} modes (Results.cavity_modes())")
            return rows[int(i)]
        fam = [r for r in rows if r["family"] == family]
        hit = [r for r in fam if r["index"] == int(i)]
        if not hit:
            raise IndexError(f"no {family} cavity mode with index {i}: the {family} indices are "
                             f"{sorted(r['index'] for r in fam)}")
        return hit[0]

    @staticmethod
    def _mode_arrays(d: dict, row: dict) -> dict:
        import numpy as np

        out = {k: (np.asarray(v) if not isinstance(v, (str, int, float)) else v) for k, v in d.items()}
        out["elem"] = np.asarray(out["elem"]).astype(np.int64)
        if "tris" in out:
            out["tris"] = np.asarray(out["tris"]).astype(np.int64).reshape(-1, 3)
        out["Emag"] = np.sqrt(out["E_r"] ** 2 + out["E_phi"] ** 2 + out["E_z"] ** 2)
        out["Hmag"] = np.sqrt(out["H_r"] ** 2 + out["H_phi"] ** 2 + out["H_z"] ** 2)
        out["mode"] = {k: row.get(k) for k in ("family", "index", "f_Hz", "C", "C_rms", "V_m3", "G_ohm",
                                                   "Q_wall", "Q_dielectric", "Q0")}
        return out

    def cavity_mode_field(self, i=0, family: str | None = None, *, subdiv: int = 2) -> dict:
        """Fields of a cavity eigenmode on the cavity mesh (2D designs with a ``cavity`` section).

        ``i`` indexes :meth:`cavity_modes` (all families, sorted by frequency); with ``family`` ("TE"/"TM") it is
        the index within that family (``row["index"]``); a row of :meth:`cavity_modes` also works.
        Returns numpy arrays over a lattice of points per cavity element (``subdiv`` = 2: vertices and edge
        midpoints; points on element edges repeat with each element's value): ``r``, ``z`` [m], ``tris`` (n, 3)
        sub-triangles, ``E_r``, ``E_phi``, ``E_z``, ``H_r``, ``H_phi``, ``H_z``, ``Emag``, ``Hmag``, ``flux``,
        ``eps_r``, ``mu_r``, ``elem`` (element number), and ``mode`` (f_Hz, C, G_ohm, Q0, ... of the row).

        Normalization: int eps_r |E|^2 dV = 1 over the cavity (E in m^-3/2, the normalization of the form factor);
        H is the matching physical field, so int mu_r |H|^2 dV = eps0/mu0 = 1/Z0^2 at resonance. The arrays are
        real amplitudes: the complex fields are E_c = i E, H_c = H (e^{-i omega t}, i.e. E(t) = E sin(omega t),
        H(t) = H cos(omega t)). TM modes have (E_r, E_z, H_phi), TE modes (E_phi, H_r, H_z); the other components
        are zero. The sign is fixed so that the largest E component at the point of largest |E| is positive (a
        convention: for modes whose |E| maxima are mirror images with opposite signs, e.g. TM011, it may differ
        between meshes). ``flux`` = r H_phi (TM; E
        field lines are its contours) or r E_phi/omega (TE; H field lines)."""
        row = self._cavity_mode_row(i, family)
        d = self._model.cavity_mode_mesh(row["family"], int(row["index"]), int(subdiv))
        return self._mode_arrays(d, row)

    def cavity_mode_sample(self, i, r, z, family: str | None = None) -> dict:
        """Fields of a cavity eigenmode at points (r, z) [m] (same keys, normalization and mode selection as
        :meth:`cavity_mode_field`, without ``tris``); NaN outside the cavity (a point exactly on the wall may fall
        in the wall element: sample just inside)."""
        import numpy as np

        row = self._cavity_mode_row(i, family)
        rr, zz = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(z, dtype=float))
        d = self._model.cavity_mode_sample(row["family"], int(row["index"]), [float(x) for x in rr.ravel()],
                                           [float(x) for x in zz.ravel()])
        out = self._mode_arrays(d, row)
        for k, v in list(out.items()):
            if isinstance(v, np.ndarray) and v.size == rr.size:
                out[k] = v.reshape(rr.shape)
        return out

    def cavity_mode_probe(self, i, r: tuple, z: tuple, family: str | None = None) -> dict:
        """Fields of a cavity eigenmode on a regular (r, z) grid: ``r = (r0, r1, nr)``, ``z = (z0, z1, nz)``
        (the design's ``cavity.probe_grid``, evaluated on demand). Arrays have shape
        (nz, nr); ``R``, ``Z`` hold the grid."""
        import numpy as np

        R, Z = np.meshgrid(np.linspace(r[0], r[1], int(r[2])), np.linspace(z[0], z[1], int(z[2])))
        out = self.cavity_mode_sample(i, R, Z, family)
        out["R"], out["Z"] = R, Z
        return out

    def save_cavity_modes(self, directory: str | Path | None = None, modes=None, *, subdiv: int = 2) -> list[Path]:
        """Write each cavity eigenmode's fields (:meth:`cavity_mode_field`) to ``<directory>/<FAM>_<index>.npz``
        (default directory ``<workdir>/cavity_modes``; ``modes``: indices into :meth:`cavity_modes`, default all).
        Every file holds the arrays plus ``meta`` (a JSON string: the cavity_modes() row, the normalization and the
        phase convention). Load with ``numpy.load(path)``."""
        import numpy as np

        out_dir = Path(directory) if directory is not None else self.workdir / "cavity_modes"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = self.cavity_modes()
        sel = range(len(rows)) if modes is None else modes
        paths = []
        for k in sel:
            row = rows[k]
            F = self.cavity_mode_field(row, subdiv=subdiv)
            meta = {"design": self.design.name, "row": {kk: vv for kk, vv in row.items()},
                    "normalization": "int eps_r |E|^2 dV = 1 over the cavity (E in m^-3/2); H physical (A/m per "
                                     "the same scale): int mu_r |H|^2 dV = 1/Z0^2 at resonance",
                    "phase": "complex fields E_c = i E, H_c = H (e^{-i omega t}): E(t) = E sin(omega t), H(t) = H cos(omega t)",
                    "flux": "r H_phi (TM) or r E_phi / omega (TE)", "subdiv": subdiv}
            arrays = {kk: vv for kk, vv in F.items() if isinstance(vv, np.ndarray)}
            p = out_dir / f"{row['family']}_{row['index']}.npz"
            np.savez_compressed(p, meta=np.array(json.dumps(meta, default=float)), **arrays)
            paths.append(p)
        return paths

    def mode_scene(self, i=0, family: str | None = None, *, subdiv: int = 2, field_lines: bool = True,
                   contours: int = 14, view_scale: float = 1.3, label: str | None = None) -> dict:
        """Viewer/renderer scene of a cavity eigenmode: the design's regions with the mode's normalized fields
        (quantities Emag, Hmag, Er, Ez, Ephi, Hr, Hz, Hphi; see :meth:`cavity_mode_field`) on the cavity, and
        field lines (TM: E lines, contours of r H_phi; TE: H lines, contours of r E_phi) when ``field_lines``.
        Render it with :func:`axionfem.render.save_png` or :meth:`plot_mode`."""
        import numpy as np

        row = self._cavity_mode_row(i, family)
        F = self.cavity_mode_field(row, subdiv=subdiv)
        sc = self.scene(fields=False, view_scale=view_scale)
        ex = sc["extent"]  # window: the device box enlarged by view_scale in r and z separately (not square)
        zc, hz = 0.5 * (ex["z_min"] + ex["z_max"]), 0.5 * (ex["z_max"] - ex["z_min"])
        sc["view"] = {"r_max": view_scale * ex["r_max"], "z_min": zc - view_scale * hz, "z_max": zc + view_scale * hz}
        V = np.column_stack([F["r"], F["z"]])
        q = {"Emag": F["Emag"], "Hmag": F["Hmag"], "Er": F["E_r"], "Ez": F["E_z"], "Ephi": F["E_phi"],
             "Hr": F["H_r"], "Hz": F["H_z"], "Hphi": F["H_phi"]}
        uE, uH = "m^{-3/2}", "m^{-3/2} Ω^{-1}"
        sc["field"] = {"vertices": _b64(V, "float32"), "indices": _b64(F["tris"], "uint32"),
                       "quantities": {k: _b64(v, "float32") for k, v in q.items()},
                       "ranges": {k: [float(np.nanmin(v)), float(np.nanmax(v))] for k, v in q.items()},
                       "units": {k: (uE if k.startswith("E") else uH) for k in q}}
        segs = np.zeros((0, 4))
        if field_lines and contours > 0:
            fl = F["flux"]
            lo, hi = float(np.nanmin(fl)), float(np.nanmax(fl))
            if hi > lo:
                lev = lo + (hi - lo) * (np.arange(1, contours + 1) - 0.5) / contours
                segs = _contour_segments(F["r"], F["z"], fl, F["tris"], lev)
        sc["field_lines"] = {"static": _b64(segs.reshape(-1, 2), "float32"), "pickup": _b64(np.zeros((0, 2)), "float32"),
                             "static_label": "E lines (r H_{φ} contours)" if row["family"] == "TM"
                             else "H lines (r E_{φ} contours)"}
        lab = label or f"{row['family']}[{row['index']}]"
        sc["summary"] = {"cavity_mode": {"label": lab, **{k: row.get(k) for k in ("f_Hz", "C", "G_ohm", "Q0", "V_m3")
                                                         if row.get(k) is not None}}}
        sc["name"] = f"{self.design.name} {lab}"
        return sc

    def plot_mode(self, i=0, path: str | Path | None = None, family: str | None = None, *, quantity: str = "Emag",
                  subdiv: int = 2, field_lines: bool = True, contours: int = 14, view_scale: float = 1.3,
                  label: str | None = None, **options) -> Path:
        """PNG of a cavity eigenmode on the meridional section, mirrored about the axis (C++ renderer;
        :meth:`mode_scene`). ``quantity``: Emag (default), Hmag, Er, Ez, Ephi, Hr, Hz, Hphi. Other options go to
        :func:`axionfem.render.render_png` (width, height, theme, annotate, title, log_scale, ...)."""
        from .render import save_png

        row = self._cavity_mode_row(i, family)
        sc = self.mode_scene(row, subdiv=subdiv, field_lines=field_lines, contours=contours, view_scale=view_scale,
                             label=label)
        options.setdefault("view", "section")
        options["field_lines"] = field_lines
        p = Path(path) if path else self.workdir / f"{self.design.name}_mode_{row['family']}{row['index']}_{quantity}.png"
        return save_png(sc, p, quantity=quantity, **options)
    # ---------------------------------------------- cavity eigenmode fields (2D) -- end

    # -- electrostatics -----------------------------------------------------
    @property
    def electrostatics(self) -> dict:
        """Electrostatics (present when the design has an `electrostatics` section): `capacitance_F`
        (the Maxwell matrix, rows in the order of `bodies`), `capacitance_to_ground_F`, `pairs` with the
        mutual capacitances, and `checks`."""
        return self.data.get("electrostatics", {})

    @property
    def C_matrix(self) -> list:
        """Maxwell capacitance matrix [F]; rows follow ``electrostatics['bodies']``."""
        return self.electrostatics.get("capacitance_F", [])

    @property
    def port_capacitance(self) -> dict:
        """Readout-gap capacitance of a poloidal (port) pickup: `C_gap_F` (with fringing),
        `C_gap_port_region_F` (gap region only, no fringing) and the self-resonance estimates.

        Neither capacitance is a proven bound on the other: the rigorous relation, exact for the
        discrete problems, is `checks.dirichlet_bound`. `self_resonance_Hz` = 1/(2 pi sqrt(L_p C_gap))
        is an estimate of the first self-resonance, not a bound."""
        return (self.data.get("pickup") or {}).get("port_capacitance", {})
    # ---------------------------------------------------------------------------

    @property
    def conductors(self) -> dict:
        """Critical-current margins of every winding made of a named conductor: operating point
        (engineering and superconductor current density, peak |B|, field angle to the tape), Jc at the
        worst point of the winding, the load-line fraction and short-sample scale, and the temperature
        margin.  The same block appears per benchmark (`conductor_margins`) at the required amplitude
        and at the cost-optimal size."""
        return self.data.get("conductors", {})

    @property
    def hts(self) -> dict:
        """Critical-state (screening-current) analysis (present when the design has an `hts` section):
        `steps` (screening current, moment, max |J|/Jc, SCIF at the field-of-view points, Psi, energy
        balance), `final` (field quality, trapped flux, current profiles), `ac_loss_per_cycle_J`,
        `psi` (Psi with and without screening currents)."""
        return self.data.get("hts", {})

    def hts_sample(self, r, z) -> dict:
        """Screening-current quantities of the final critical-state state at points (r, z) [m]:
        engineering current density J [A/m^2], local engineering Jc, azimuthal electric field E [V/m],
        the total field (Br, Bz) [T], A_phi and `in_hts`."""
        return self._model.hts_sample(list(map(float, r)), list(map(float, z)))

    @property
    def B_max(self) -> float:
        return self.data.get("fields", {}).get("B_max_T", 0.0)

    def benchmark(self, name: str) -> dict:
        for b in self.data.get("benchmarks", []):
            if b["name"] == name:
                return b
        names = [b["name"] for b in self.data.get("benchmarks", [])]
        raise KeyError(f"no benchmark {name!r}; available: {names}")

    # ------------------------------------------------------------ text
    def summary(self) -> str:
        c, f = self.coupling, self.data.get("fields", {})
        lines = [f"design {self.design.name!r}",
                 f"  Psi      = {_fmt(c.get('Psi_SI_H_T_m2'), 'H T m^2')}  ({_fmt(c.get('Psi_natural_T_m3'), 'T m^3')} natural)",
                 f"  L_p      = {_fmt(c.get('L_p_H'), 'H')}",
                 f"  W_mag    = {_fmt(c.get('W_mag_J'), 'J')}",
                 f"  F        = {_fmt(c.get('F_SI_T4_H2_m8'), 'T^4 H^2 m^8')}",
                 f"  g        = {_fmt(c.get('g_factor'))}",
                 f"  |B|max   = {_fmt(f.get('B_max_T'), 'T')}{_fmt_location(f.get('B_max_location'))}"]
        for k, v in f.get("superconductor_surface_T", {}).items():
            lines.append(f"  SC surface peak [{k}] = {_fmt(v, 'T')}")
        for b in self.data.get("benchmarks", []):
            lines.append(f"  benchmark {b['name']}: F/F_req = {_fmt(b.get('F_over_F_req'))}, "
                         f"amplitude x{_fmt(b.get('amplitude_scale'))}, W_req = {_fmt(b.get('W_mag_at_req_J'), 'J')}, "
                         f"B_max = {_fmt(b.get('B_max_at_req_T'), 'T')}, cost = {_fmt(b.get('cost', {}).get('total_MCHF'), 'MCHF')}")
            opt = b.get("cost_optimal_size")
            if opt:
                lines.append(f"      cost-optimal size: x{_fmt(opt['length_scale'])} -> R = {_fmt(opt['device_radius_m'], 'm')}, "
                             f"B_max = {_fmt(opt['B_max_T'], 'T')}, cost = {_fmt(opt['cost']['total_MCHF'], 'MCHF')}")
        for k, v in self.data.get("conductors", {}).items():
            lines.append(f"  conductor [{k}] {v['conductor']}: J_sc = {_fmt(v['J_superconductor_A_m2'], 'A/m^2')}, "
                         f"|B| = {_fmt(v['worst_point']['B_T'], 'T')}, Jc = {_fmt(v['Jc_A_m2'], 'A/m^2')}, "
                         f"I/Ic = {_fmt(v['load_line_fraction'])}, dT = {_fmt(v['temperature_margin_K'], 'K')}")
        h = self.data.get("hts", {})
        if h:
            fin = h.get("final", {})
            for rg in fin.get("regions", []):
                lines.append(f"  critical state [{rg['region']}]: I_screen = {_fmt(rg['screening_current_A'], 'A')}, "
                             f"m = {_fmt(rg['moment_Am2'], 'A m^2')}, max|J|/Jc = {_fmt(rg['max_J_over_Jc'])}")
            lines.append(f"  critical state: loss = {_fmt(h.get('total_loss_J'), 'J')}"
                         + (f", per cycle = {_fmt(h.get('ac_loss_per_cycle_J'), 'J')}" if "ac_loss_per_cycle_J" in h else "")
                         + (f", max SCIF = {_fmt(fin.get('field_quality', {}).get('max_scif_Bz_T'), 'T')}"
                            if "field_quality" in fin else ""))
            if "relative_change" in h.get("psi", {}):
                lines.append(f"      Psi with screening currents: {_fmt(h['psi']['Psi_critical_state_SI_H_T_m2'])}"
                             f" ({h['psi']['relative_change']:+.3%} vs the ideal model)")
        fw = self.data.get("fullwave", {})
        for row in fw.get("sweep", []):
            lines.append(f"  full-wave f = {_fmt(row['f_Hz'], 'Hz')} (kR = {_fmt(row['k_R'])}): "
                         f"|Psi|/Psi_MQS = {_fmt(row.get('Psi_over_MQS'))}, L/L_MQS = {_fmt(row.get('L_over_MQS'))}, "
                         f"R_rad = {_fmt(row.get('R_rad_ohm'), 'ohm')}, Q_rad = {_fmt(row.get('Q_rad'))}"
                         + (" (Re Z unresolved)" if "Re_Z_unresolved" in row else "")
                         + (f", R_loss = {_fmt(row['R_loss_ohm'], 'ohm')}, "
                            + (f"Q (tuned) = {_fmt(row.get('Q_loss_tuned'))}, Q_total (tuned) = "
                               f"{_fmt(row.get('Q_total_tuned'))}" if "L_eff_H" in row
                               else f"Q_loss = {_fmt(row.get('Q_loss'))}")
                            if "R_loss_ohm" in row else ""))
        for fam, e in fw.get("eigenmodes", {}).items():
            lines.append(f"  {fam} eigenmodes [Hz]: " + ", ".join(_fmt(m['f_Hz']) for m in e["modes"]))
        # electrostatics ------------------------------------------------------
        es = self.electrostatics
        if es:
            names = [b["name"] for b in es.get("bodies", [])]
            for i, row in enumerate(es.get("capacitance_F", [])):
                lines.append(f"  C[{names[i]}] = " + ", ".join(_fmt(x, 'F') for x in row))
            for p in es.get("pairs", []):
                lines.append(f"  C_mutual({p['a']}, {p['b']}) = {_fmt(p['C_mutual_others_grounded_F'], 'F')} "
                             f"(others grounded), {_fmt(p['C_mutual_others_floating_F'], 'F')} (others floating)")
        pc = self.port_capacitance
        if pc.get("C_gap_F"):
            lines.append(f"  readout gap C = {_fmt(pc['C_gap_F'], 'F')} "
                         f"(gap region only {_fmt(pc.get('C_gap_port_region_F'), 'F')}), "
                         f"f_SR ~ {_fmt(pc.get('self_resonance_Hz'), 'Hz')}")
        # -------------------------------------------------------------------
        for w in self.data.get("warnings", []):
            lines.append(f"  warning: {w}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # ------------------------------------------------------------ sensitivities
    def sensitivities(self, params: Sequence[str] | None = None, objectives: Sequence[str] | None = None,
                      method: str = "adjoint", step: float = 1e-4, steps: dict | None = None,
                      benchmark: str = "", order: int = 0) -> dict:
        """Derivatives of the figures of merit with respect to design parameters.

        ``params`` defaults to every entry of the design's ``parameters``, ``objectives`` to
        Psi, L_p, W_mag, F and the cost of the first benchmark
        (:func:`axionfem.objectives` lists them all; cost quantities accept a benchmark,
        e.g. ``"cost:optimistic"``).  ``method``: "adjoint" (discrete adjoint on a morphed
        mesh, the default), "fd_morph" (central differences with mesh morphing) or
        "fd_remesh" (central differences with a new mesh per evaluation; the only method for
        nonlinear iron or for parameters that change the mesh topology).  ``step`` is the
        relative finite-difference step, ``steps`` gives absolute per-parameter overrides.

        The returned dict holds ``values``, ``gradient[objective][parameter]`` (SI),
        ``relative[objective][parameter]`` (= d log J / d log p), the primitive quantities and
        diagnostics.
        """
        from .sensitivity import _sensitivities

        return _sensitivities(self._model, params, objectives, method, step, steps, benchmark, self.workdir, order)

    # ------------------------------------------------------------ signal reach
    def reach(self, benchmark: str = "optimistic", amplitude: str = "required", **kw) -> dict:
        """Expected 95% reach g_95(m_a) of this design for a noise benchmark.
        amplitude='required': field scaled so that F = F_req (reaches the QCD target);
        'as_designed': the ampere-turns given in the design.  Extra keywords go to scan.reach."""
        from . import scan

        Psi, L = self.Psi, self.L_p
        if amplitude == "required":
            s = self.benchmark(benchmark)["amplitude_scale"]
            Psi = Psi * s
        elif amplitude != "as_designed":
            raise ValueError("amplitude must be 'required' or 'as_designed'")
        return scan.reach(Psi, L, benchmark, **kw)

    # ----------------------------------------------- readout circuit (circuit.py) -- begin
    def readout(self, f_Hz: float | None = None, benchmark: str | dict | None = None,
                amplitude: str = "as_designed", model: str = "mqs", pickup_Q: float | None = None,
                R_loss_ohm: float | None = None, parasitic: bool = False,
                analyze: bool = True, **kw) -> dict:
        """Readout-circuit analysis of this solved design.

        Builds a DMRadio-style network -- pickup in series with a tuning capacitor, the SQUID
        input coil and the leads -- from this design's ``Psi``, ``L_p`` and losses, tunes it to
        ``f_Hz`` (default: the middle of the design's scan band in log frequency) and returns
        the analysis: resonance, loaded Q, referred ``T_eff``, the amplifier coupling ``xi``,
        the signal flux into the SQUID, the Johnson/amplifier noise budget, the bandwidths and
        the equivalent optimal-scan inputs ``(Psi, L_eff, Q, T, eta_A)``.

        ``model="mqs"``: ``Z_p = R_loss - i omega L_p`` with ``R_loss`` from ``R_loss_ohm``, from
        ``pickup_Q``, or from the design's RF surface losses.  ``model="fullwave"``: ``Z_p(omega)``
        interpolated from ``results["fullwave"]["sweep"]``.
        ``benchmark`` sets ``T_K`` and ``eta_A`` from ``scan.BENCHMARKS`` (and, when no loss data
        and no ``pickup_Q`` are available, the pickup ``Q`` too).
        ``amplitude="required"`` scales ``Psi`` so that ``F = F_req`` for that benchmark.
        ``parasitic=True`` puts this design's readout-port gap capacitance
        (``pickup/port_capacitance/C_gap_F``, only the gap's share of the distributed
        capacitance) across the
        pickup, so that the self-resonance it implies is part of the circuit.
        Extra keywords go to :func:`axionfem.circuit.series_readout` (``L_in_H``, ``R_in_ohm``,
        ``lead_length_m``, ``L_per_m``, ``R_per_m``, ``tan_delta``, ``M_H``, ``xi``, ...).
        The returned dict carries the live :class:`~axionfem.circuit.Readout` under
        ``"readout"``; it holds complex impedances, so it is not JSON-serialisable.
        """
        from . import circuit as _ck, scan as _scan

        b = _scan.BENCHMARKS[benchmark] if isinstance(benchmark, str) else (benchmark or {})
        if f_Hz is None:
            s = self.data.get("scan", {})
            lo, hi = s.get("f_min"), s.get("f_max")
            f_Hz = (lo * hi) ** 0.5 if (lo and hi) else 1.209e5
        Psi = self.Psi
        if amplitude == "required":
            if benchmark is None or not isinstance(benchmark, str):
                raise ValueError("Results.readout: amplitude='required' needs a benchmark name")
            Psi = Psi * self.benchmark(benchmark)["amplitude_scale"]
        elif amplitude != "as_designed":
            raise ValueError("Results.readout: amplitude must be 'as_designed' or 'required'")

        kw.setdefault("T_K", b.get("T", 0.02))
        kw.setdefault("eta_A", b.get("eta_A", 1.0))
        if parasitic:
            Cg = self.data.get("pickup", {}).get("port_capacitance", {}).get("C_gap_F", 0.0)
            if not Cg:
                raise ValueError(
                    "Results.readout(parasitic=True): this design has no "
                    "pickup/port_capacitance/C_gap_F (it is computed for poloidal port "
                    "pickups only).  Pass C_par_F=... explicitly instead.")
            kw.setdefault("C_par_F", Cg)
        if model == "fullwave":
            Z_p = _ck.pickup_impedance(self.data, model="fullwave")
            r = _ck.series_readout(Z_p=Z_p, L_p_H=self.L_p, Psi_SI=Psi, f_Hz=f_Hz,
                                   label=f"{self.design.name} full-wave readout", **kw)
        else:
            has_loss = bool(self.data.get("pickup", {}).get("losses", {})
                            .get("T_over_Q_total_K"))
            if R_loss_ohm is None and pickup_Q is None and not has_loss:
                pickup_Q = b.get("Q")
                if pickup_Q is None:
                    raise ValueError(
                        "Results.readout: this design has no RF loss data, so the pickup Q is "
                        "unknown.  Pass pickup_Q=..., R_loss_ohm=..., or a benchmark name whose "
                        "Q is used instead.")
            Z_p = _ck.pickup_impedance(self.data, L_p_H=self.L_p, R_loss_ohm=R_loss_ohm,
                                       Q=pickup_Q, f_ref_Hz=f_Hz if pickup_Q else None)
            r = _ck.series_readout(Z_p=Z_p, L_p_H=self.L_p, Psi_SI=Psi, f_Hz=f_Hz,
                                   label=f"{self.design.name} readout", **kw)
        if not analyze:
            return {"readout": r}
        out = r.analyze(f_Hz)
        out["readout"] = r
        out["design"] = self.design.name
        out["model"] = model
        return out

    def draw_readout(self, annotate: bool = True, **kw):
        """Effective-circuit schematic of this design's readout: the network that
        :meth:`readout` builds (same keywords: ``f_Hz``, ``benchmark``, ``model``, ``pickup_Q``,
        ``parasitic``, ``L_in_H``, ...), drawn by :meth:`axionfem.circuit.Readout.draw` with the
        values at ``f_Hz``.  Needs ``schemdraw`` (``axionfem[draw]``)."""
        kw["analyze"] = False
        r = self.readout(**kw)["readout"]
        return r.draw(annotate=annotate, f_Hz=kw.get("f_Hz"), title=r.label if annotate else None)
    # ----------------------------------------------- readout circuit (circuit.py) -- end

    # ------------------------------------------------------------ fields and scenes
    def sample(self, r: Sequence[float], z: Sequence[float]) -> dict:
        """Fields at points (r, z) [m]: Br, Bz, Bphi [T], psi [Wb/rad], pickup fields per ampere."""
        return self._model.sample(list(map(float, r)), list(map(float, z)))

    # ---- mechanics / thermal ----
    @property
    def mechanics(self) -> dict:
        """Elasticity results (present when the design has a `mechanics` section): per-region
        max |u|, von Mises, hoop/radial/axial stress; support reactions; equilibrium closure."""
        return self.data.get("mechanics", {})

    @property
    def thermal(self) -> dict:
        """Steady-conduction results (present when the design has a `thermal` section):
        per-region T min/max and the heat-flow balance."""
        return self.data.get("thermal", {})

    def sample_mech(self, r: Sequence[float], z: Sequence[float]) -> dict:
        """Displacement (u_r, u_z [m]), stress (sigma_rr, sigma_zz, sigma_hoop, sigma_rz,
        von_mises [Pa]) and temperature (T_K) at points (r, z) [m]. `in_structure` /
        `in_thermal_domain` flag the points that lie inside each analysis domain."""
        return self._model.sample_mech(list(map(float, r)), list(map(float, z)))
    # ---- end mechanics / thermal ----

    # ---- vibrations: axisymmetric m = 0 torsional dynamics (beta) ----
    def torsion(self, nmodes: int = 0, load=None, freqs_Hz=None, eta: float = 1e-4,
                shapes: bool = False, regions: dict | None = None, dofs: bool = False,
                lanczos_extra: int = 24, verbose: int = 0, loads: list | None = None,
                band: dict | None = None, solver: dict | None = None,
                mesh_check: str = "error") -> dict:
        """Axisymmetric m = 0 torsional modal / harmonic analysis (beta; physics: docs/physics/mechanics_thermal.md
        and docs/physics/torsion_highfreq.md; validation: docs/validation/torsion.md and
        docs/validation/torsion_highfreq.md).

        Loads (they add up):
        - `load`: one azimuthal body-force value per degree of freedom of the space (use ``dofs=True``
          once to get `dof_r`, `dof_z`); the FEM integrates its P_p interpolant, and the response
          reports what was applied (`load.force_N`, `load.torque_N_m`);
        - `loads`: a list of named loads, integrated exactly at quadrature points:
          ``{"region": R, "g_N_m3": v}`` (body force in region R),
          ``{"surface": {"on": R, "from": S or "vacuum"}, "traction_Pa": v}`` (azimuthal traction on the
          faces R shares with S), ``{"curve": C, "traction_Pa": v}`` (on a zero-thickness curve); v is a
          number, an expression in r and z [m] (e.g. ``"exp(-z^2/(2*0.005^2))"``) or one value per DOF.
          Expressions use standard precedence: ``exp(-z^2/(2*0.005^2))`` is a Gaussian. Check
          ``loads.total.force_N`` against the load you meant.
        With `freqs_Hz`, or `band` = ``{"f_Hz": [f0, f1], "n": N, "length_m": L}`` (N odd; `length_m`
        optional), the response returns the generalised mobility Y = -i omega q of the conjugate pair
        (load scale, q = int g u_phi dV + int tau u_phi dA), time dependence e^{-i omega t};
        S_q = 4 kB T Re[Y]/omega^2. `harmonic.band` holds the band mean of Re Y, its quadrature
        uncertainty, a ripple diagnostic and, with `length_m`, a conservative scale of the finite-length
        deviation (`edge_estimate`).
        Single frequencies are not certified by the mesh guard (it is calibrated on band means).
        `solver` = ``{"backend": "auto"|"eigen"|"strumpack", "parallel_frequencies": p | "auto"}``.
        `mesh_check` ("error", "warn", "off") guards the elements per shear wavelength at the highest
        frequency (`harmonic.resolution`).
        """
        req: dict = {"nmodes": int(nmodes), "shapes": bool(shapes), "dofs": bool(dofs),
                     "eta": float(eta), "lanczos_extra": int(lanczos_extra),
                     "verbose": int(verbose), "mesh_check": str(mesh_check)}
        if load is not None:
            req["load"] = [float(x) for x in load]
        if loads is not None:
            req["loads"] = [_torsion_load_entry(x) for x in loads]
        if freqs_Hz is not None:
            req["freqs_Hz"] = [float(x) for x in freqs_Hz]
        if band is not None:
            req["band"] = dict(band)
        if solver is not None:
            req["solver"] = dict(solver)
        if regions:
            req["regions"] = regions
        return json.loads(self._model.torsion(json.dumps(req)))

    def torsion_band(self, f_min: float, f_max: float, loads=None, load=None, eta: float = 1e-3, n: int = 201,
                     solver: dict | None = None, mesh_check: str = "error", regions: dict | None = None,
                     verbose: int = 0, length_m: float | None = None) -> dict:
        """Band-averaged torsional mobility over [f_min, f_max] (n frequencies, n odd): ``harmonic.band``
        holds the mean of Re Y, its quadrature uncertainty, a ripple diagnostic (`ripple_sem`) and, with
        `length_m` (the structure's axial length), `edge_estimate`, a conservative scale of how far the band mean
        of the finite structure may sit from a smooth value (docs/physics/torsion_highfreq.md). The default solver is
        serial STRUMPACK (serial Eigen SparseLU in a build without STRUMPACK): robust in memory and never far from
        the fastest choice. ``solver={"parallel_frequencies": "auto"}`` opts in to the measured choice between
        concurrent Eigen solves and serial STRUMPACK (faster on thin walls; it costs one STRUMPACK and possibly one
        Eigen probe solve)."""
        if solver is None:
            from . import build_info

            solver = {"backend": "strumpack" if build_info().get("strumpack") else "eigen",
                      "parallel_frequencies": 1}
        band = {"f_Hz": [float(f_min), float(f_max)], "n": int(n)}
        if length_m is not None:
            band["length_m"] = float(length_m)
        return self.torsion(load=load, loads=loads, eta=eta, band=band,
                            solver=solver, mesh_check=mesh_check, regions=regions, verbose=verbose)
    # ---- end torsional dynamics ----

    def scene(self, subdiv: int = 1, contours: int = 28, view_scale: float = 1.3, fields: bool = True) -> dict:
        """Viewer scene."""
        return json.loads(self._model.scene(subdiv, contours, view_scale, fields))

    def save_scene(self, path: str | Path | None = None, **kw) -> Path:
        p = Path(path) if path else self.workdir / "scene.json"
        p.write_text(json.dumps(self.scene(**kw)))
        return p

    def write_vtu(self, path: str | Path | None = None, subdiv: int = 2) -> Path:
        p = Path(path) if path else self.workdir / "fields.vtu"
        self._model.write_vtu(str(p), subdiv)
        return p

    def png(self, path: str | Path | None = None, **options) -> Path:
        """PNG snapshot (C++ renderer). Options: view='section'|'3d', quantity, log_scale, theme, ...
        With ``mode=i`` (and optionally ``family``), a cavity eigenmode (:meth:`plot_mode`; quantity Emag by
        default)."""
        from .render import save_png

        if "mode" in options:
            i = options.pop("mode")
            options.setdefault("quantity", "Emag")
            return self.plot_mode(i, path, options.pop("family", None), **options)

        view = options.get("view", "section")
        q = options.get("quantity", "Bmag")
        p = Path(path) if path else self.workdir / f"{self.design.name}_{view}_{q}.png"
        return save_png(self.scene(), p, **options)

    def export_html(self, path: str | Path | None = None, title: str | None = None, **scene_kw) -> Path:
        """Self-contained interactive 3D viewer page."""
        from .viewer import export_html

        p = Path(path) if path else self.workdir / f"{self.design.name}.html"
        export_html(self.scene(**scene_kw), str(p), title=title or self.design.name)
        return p

    # --------------------------------------------- Jupyter widget -- begin
    def show(self, quantity: str | None = None, **kw):
        """Show this result in a Jupyter notebook and return the widget.

        The same self-contained page as :meth:`export_html`, in an iframe, with
        a selector for the plotted quantity (``w.quantity = "Bz"`` also works).
        Keyword arguments go to :class:`axionfem.widget.SceneWidget` (``mode``,
        ``scale``, ``height``, ``options``, ``backend``); ``subdiv``,
        ``contours``, ``view_scale`` and ``fields`` go to :meth:`scene`.
        """
        from .widget import SceneWidget

        scene_kw = {k: kw.pop(k) for k in ("subdiv", "contours", "view_scale", "fields") if k in kw}
        return SceneWidget(
            self.scene(**scene_kw), title=self.design.name, quantity=quantity, **kw
        ).show()

    # ----------------------------------------------- Jupyter widget -- end


class Results3D(Results):
    """Results of a solved 3D design (``"symmetry": "3d"``).  Same numbers and benchmarks as
    :class:`Results`; fields are sampled at (x, y, z) and scenes carry region surfaces plus a slice."""

    def sample(self, x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> dict:  # type: ignore[override]
        """Static field Bx, By, Bz [T] and pickup field per ampere at points (x, y, z) [m]; NaN outside."""
        return self._model.sample(list(map(float, x)), list(map(float, y)), list(map(float, z)))

    def scene(self, fields: bool = True, **_ignored) -> dict:  # type: ignore[override]
        """3D viewer scene: region surfaces (with surface |B|) and a field slice."""
        return json.loads(self._model.scene(fields))

    def torsion(self, *args, **kw) -> dict:  # type: ignore[override]
        """Not available for 3D designs (the torsional analysis is axisymmetric, m = 0)."""
        raise NotImplementedError("Results.torsion is the axisymmetric m = 0 torsional analysis; it has no 3D "
                                  "counterpart. Solve the axisymmetric design (docs/design_format.md, "
                                  "Torsional dynamics)")

    def sensitivities(self, *args, **kw) -> dict:  # type: ignore[override]
        """Not implemented for 3D designs."""
        raise NotImplementedError("shape sensitivities are implemented for axisymmetric designs only; "
                                  "differentiate the axisymmetric model and cross-check in 3D")

    def save(self, path: str | Path | None = None) -> Path:
        """Write the solved fields (mesh, A_lab, A_pickup, state) for :func:`load_solution`."""
        p = Path(path) if path else self.workdir / "fields"
        self._model.save(str(p))
        return p

    def write_vtu(self, path: str | Path | None = None, subdiv: int = 2) -> Path:  # type: ignore[override]
        """ParaView output (high-order VTU collection) of A_lab and the pickup potential."""
        p = Path(path) if path else self.workdir / "paraview"
        self._model.write_paraview(str(p))
        return p

    def png(self, path: str | Path | None = None, **options) -> Path:
        """PNG snapshot of the 3D scene (C++ renderer): surfaces cut open at the slice plane."""
        from .render import save_png

        options.setdefault("view", "3d")
        q = options.get("quantity", "Bmag")
        p = Path(path) if path else self.workdir / f"{self.design.name}_3d_{q}.png"
        return save_png(self.scene(), p, **options)


def load_solution(path: str | Path) -> Results3D:
    """Load a saved 3D solution (``axf solve --save`` or :meth:`Results3D.save`): results, sampling and
    scenes without solving again."""
    from .design import Design

    p = Path(path)
    model = _core.Model3D.load(str(p))
    st = json.loads((p / "state.json").read_text())
    return Results3D(model, Design(st["design"]), p.parent if p.name == "fields" else p)


def _torsion_load_entry(x: dict) -> dict:
    """A `loads` entry for Results.torsion, with array values converted to plain floats."""
    import numpy as np

    out = dict(x)
    for key in ("g_N_m3", "traction_Pa"):
        v = out.get(key)
        if v is None or isinstance(v, str):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) or np.ndim(v) == 0:
            out[key] = float(v)
        else:
            out[key] = [float(t) for t in np.asarray(v).ravel()]
    return out
