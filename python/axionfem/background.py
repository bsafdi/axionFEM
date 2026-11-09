"""Background (externally given) static fields for axisymmetric designs.

A design's ``background`` block describes a field produced by sources that are
not part of the finite-element model (typically a magnet computed elsewhere,
e.g. exported from COMSOL).  The solver computes the reaction of
superconductors and persistent circuits to it.  The grid form is::

    {"type": "grid", "r": [...], "z": [...], "Br": [...], "Bz_grid": [...]}

with ``r[0] == 0``, strictly increasing (possibly non-uniform) ``r`` and ``z``
node lists and the two components stored row-major with z outer, i.e. value
``[iz * nr + ir]``.  How the C++ core uses it (``cpp/src/background.cpp``):

* ``Br``/``Bz`` are interpolated bilinearly in (r, z) for reported fields and
  Lorentz forces;
* the flux function psi = r A_phi is built from ``Bz_grid`` alone by trapezoid
  integration of ``Bz r`` along each row from the axis, and is interpolated
  bilinearly in (r^2, z).  psi sets the superconductor ties and the background
  flux linkage of persistent circuits;
* outside the grid the background is taken as zero (the solver warns when a
  superconductor or winding reaches outside it).

Consequences for users: a coarse table interpolated bilinearly has
derivative kinks at every grid line, and a superconducting surface converts
such kinks into spurious surface-current spikes as the mesh is refined.  Near superconductors, supply a fine grid
sampled from a smooth (preferably source-free) reconstruction; the helpers
here build such grids and report how well the grid, *as the solver will
interpolate it*, reproduces the raw samples.

Main entry points
-----------------
``read_comsol_text``     parse a COMSOL text/spreadsheet export (``%`` header)
``read_xyz``             parse an ``r z value`` interpolation table
``regularize``           scattered (r, z, values) -> regular tensor grid
``GriddedField``         linear / cubic interpolation of a regular grid
``HarmonicField``        source-free least-squares fit (axisymmetric solid harmonics)
``PatchedField``         one reconstruction inside a box, another outside
``RadialBlend``          inner/outer reconstructions joined with a flux-consistent radial blend
``graded_nodes``         non-uniform node lists (fine zones, anchored nodes)
``grid_block``           sample any field callable into the design block
``from_comsol_table``    one-call COMSOL export -> design block
``evaluate_block``       Br, Bz, psi exactly as the solver interpolates the block
``accuracy_report``      block (as interpolated) versus raw samples

SI units throughout (metres, tesla, Wb/rad for psi).
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

FieldFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

__all__ = [
    "read_comsol_text",
    "read_xyz",
    "regularize",
    "GriddedField",
    "HarmonicField",
    "PatchedField",
    "RadialBlend",
    "graded_nodes",
    "grid_block",
    "from_comsol_table",
    "evaluate_block",
    "flux_function",
    "accuracy_report",
]


# ---------------------------------------------------------------- reading exports
def read_comsol_text(path_or_text: str | Path | bytes) -> dict:
    """Parse a COMSOL text export (header lines start with ``%``).

    Returns ``{"columns": [names], "data": (N, ncol) array, "header": {key: value}}``.
    Column names come from the last header line (e.g. ``r z mf.Br (T) mf.Bz (T)``);
    units in parentheses are kept attached to the preceding name.
    """
    if isinstance(path_or_text, bytes):
        text = path_or_text.decode()
    elif isinstance(path_or_text, Path) or (isinstance(path_or_text, str) and "\n" not in path_or_text):
        text = Path(path_or_text).read_text()
    else:
        text = str(path_or_text)
    header, rows = {}, []
    last = ""
    for line in text.splitlines():
        if line.startswith("%"):
            body = line[1:].strip()
            if ":" in body and not body.startswith(("r ", "x ", "R ")):
                k, v = body.split(":", 1)
                header[k.strip()] = v.strip()
            last = body
        elif line.strip():
            rows.append(line)
    data = np.loadtxt(io.StringIO("\n".join(rows)), ndmin=2)
    cols = _split_columns(last, data.shape[1])
    return {"columns": cols, "data": data, "header": header}


def _split_columns(line: str, n: int) -> list[str]:
    toks = line.split()
    names: list[str] = []
    for t in toks:
        if t.startswith("(") and names:
            names[-1] += " " + t
        else:
            names.append(t)
    if len(names) != n:
        names = [f"c{i}" for i in range(n)]
    return names


def read_xyz(path_or_bytes: str | Path | bytes) -> np.ndarray:
    """Read a whitespace table ``r z value`` (COMSOL interpolation-function file)."""
    if isinstance(path_or_bytes, bytes):
        return np.loadtxt(io.BytesIO(path_or_bytes), ndmin=2)
    return np.loadtxt(Path(path_or_bytes), ndmin=2)


def regularize(r: Sequence[float], z: Sequence[float], *values: Sequence[float],
               decimals: int = 9) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Scattered samples on a complete tensor grid -> (r_nodes, z_nodes, [V[nz, nr], ...]).

    Raises ValueError (saying which nodes) for duplicates or missing nodes.
    Coordinates are rounded to ``decimals`` before matching.
    """
    r = np.round(np.asarray(r, float), decimals)
    z = np.round(np.asarray(z, float), decimals)
    rn, zn = np.unique(r), np.unique(z)
    ir, iz = np.searchsorted(rn, r), np.searchsorted(zn, z)
    key = iz * len(rn) + ir
    uniq, counts = np.unique(key, return_counts=True)
    if np.any(counts > 1):
        k = uniq[counts > 1][0]
        raise ValueError(f"duplicate sample at r={rn[k % len(rn)]}, z={zn[k // len(rn)]} "
                         f"({int(np.sum(counts > 1))} duplicated nodes); filter the export first")
    if len(uniq) != len(rn) * len(zn):
        raise ValueError(f"samples do not fill a tensor grid: {len(uniq)} of {len(rn)} x {len(zn)} nodes present")
    out = []
    for v in values:
        g = np.empty(len(rn) * len(zn))
        g[key] = np.asarray(v, float)
        out.append(g.reshape(len(zn), len(rn)))
    return rn, zn, out


# ---------------------------------------------------------------- reconstructions
class GriddedField:
    """Interpolation of a regular (r, z) table of Br, Bz (arrays [nz, nr]).

    ``method='linear'`` is bilinear in (r, z) (as scipy RegularGridInterpolator);
    ``method='cubic'`` is a componentwise interpolating bicubic spline
    (scipy RectBivariateSpline, s=0).  Neither enforces div B = curl B = 0.
    """

    def __init__(self, r: Sequence[float], z: Sequence[float], Br: np.ndarray, Bz: np.ndarray):
        from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

        self.r = np.asarray(r, float)
        self.z = np.asarray(z, float)
        self.Br = np.asarray(Br, float).reshape(len(self.z), len(self.r))
        self.Bz = np.asarray(Bz, float).reshape(len(self.z), len(self.r))
        self._lin = [RegularGridInterpolator((self.z, self.r), g, bounds_error=False, fill_value=np.nan)
                     for g in (self.Br, self.Bz)]
        self._cub = [RectBivariateSpline(self.z, self.r, g, s=0) for g in (self.Br, self.Bz)]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return float(self.r[0]), float(self.r[-1]), float(self.z[0]), float(self.z[-1])

    def inside(self, r, z) -> np.ndarray:
        r, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(z, float))
        return (r >= self.r[0]) & (r <= self.r[-1]) & (z >= self.z[0]) & (z <= self.z[-1])

    def field(self, r, z, method: str = "linear") -> tuple[np.ndarray, np.ndarray]:
        r, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(z, float))
        if not np.all(self.inside(r, z)):
            raise ValueError("points outside the tabulated domain "
                             f"r in [{self.r[0]}, {self.r[-1]}], z in [{self.z[0]}, {self.z[-1]}]")
        if method == "linear":
            pts = np.stack([z.ravel(), r.ravel()], axis=-1)
            return tuple(f(pts).reshape(r.shape) for f in self._lin)  # type: ignore[return-value]
        if method == "cubic":
            return tuple(s.ev(z.ravel(), r.ravel()).reshape(r.shape) for s in self._cub)  # type: ignore[return-value]
        raise ValueError("method must be 'linear' or 'cubic'")

    def callable(self, method: str = "linear") -> FieldFn:
        return lambda r, z: self.field(r, z, method)

    def psi(self, r, z, method: str = "linear") -> np.ndarray:
        """Flux function int_0^r Bz r' dr' of the interpolant, exact for both methods
        (3-point Gauss per table cell; r*Bz is at most quartic in r within a cell)."""
        r, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(z, float))
        x, w = np.polynomial.legendre.leggauss(3)
        out = np.empty(r.size)
        for k, (rr, zz) in enumerate(zip(r.ravel(), z.ravel())):
            g = np.r_[self.r[self.r < rr], rr] if rr > 0 else np.array([0.0])
            if len(g) < 2:
                out[k] = 0.0
                continue
            h = np.diff(g) / 2
            sp = (g[:-1] + h)[:, None] + h[:, None] * x
            bz = self.field(sp.ravel(), np.full(sp.size, zz), method)[1].reshape(sp.shape)
            out[k] = np.sum(h[:, None] * w * sp * bz)
        return out.reshape(r.shape)

    def cell_ampere_turns(self) -> np.ndarray:
        """Azimuthal current in each grid cell implied by the table, (curl B)_phi / mu0 * area [A].

        Array [nz-1, nr-1].  Large values flag magnet conductors inside the table:
        no smooth source-free reconstruction is valid across those cells.
        """
        dz = np.diff(self.z)[:, None]
        dr = np.diff(self.r)[None, :]
        dBr_dz = 0.5 * ((self.Br[1:, :-1] - self.Br[:-1, :-1]) + (self.Br[1:, 1:] - self.Br[:-1, 1:])) / dz
        dBz_dr = 0.5 * ((self.Bz[:-1, 1:] - self.Bz[:-1, :-1]) + (self.Bz[1:, 1:] - self.Bz[1:, :-1])) / dr
        return (dBr_dz - dBz_dr) / (4e-7 * math.pi) * dz * dr


def _harmonic_basis(r: np.ndarray, z: np.ndarray, degree: int, z0: float, scale: float) -> np.ndarray:
    """B = -grad(rho^n P_n(cos theta)) for n = 1..degree in scaled coordinates; shape (N, 2, degree).

    Same recursion for the regular axisymmetric solid harmonics h_n(r, z):
    n h_n = (2n-1) z h_{n-1} - (n-1)(r^2+z^2) h_{n-2}, h_0 = 1, h_1 = z.
    Columns are fields per unit coefficient in tesla when coefficients are in tesla
    (the potential is scaled by ``scale``).
    """
    rr = np.asarray(r, float).ravel() / scale
    zz = (np.asarray(z, float).ravel() - z0) / scale
    h0, h1 = np.ones_like(rr), zz.copy()
    dr0, dr1 = np.zeros_like(rr), np.zeros_like(rr)
    dz0, dz1 = np.zeros_like(rr), np.ones_like(rr)
    cols = [np.stack([-dr1, -dz1], axis=1)]
    s2 = rr * rr + zz * zz
    for n in range(2, degree + 1):
        h = ((2 * n - 1) * zz * h1 - (n - 1) * s2 * h0) / n
        dr = ((2 * n - 1) * zz * dr1 - (n - 1) * (2 * rr * h0 + s2 * dr0)) / n
        dz = ((2 * n - 1) * (h1 + zz * dz1) - (n - 1) * (2 * zz * h0 + s2 * dz0)) / n
        cols.append(np.stack([-dr, -dz], axis=1))
        h0, h1, dr0, dr1, dz0, dz1 = h1, h, dr1, dr, dz1, dz
    return np.stack(cols, axis=-1)


class HarmonicField:
    """Source-free axisymmetric field B = -grad sum_n c_n rho^n P_n, fitted by least squares.

    Satisfies div B = curl B = 0 identically, so a superconducting surface sees no
    fictitious volume currents.  Valid only inside a source-free region containing
    the fit points (the series diverges outside); ``domain`` records the fit box and
    ``metrics`` the fit and hold-out residuals.
    """

    def __init__(self, r, z, Br, Bz, degree: int = 30, z0: float | None = None,
                 scale: float | None = None, holdout_every: int = 3, rcond: float = 1e-12,
                 holdout_mask: Sequence[bool] | None = None):
        r = np.asarray(r, float).ravel()
        z = np.asarray(z, float).ravel()
        Br = np.asarray(Br, float).ravel()
        Bz = np.asarray(Bz, float).ravel()
        self.degree = int(degree)
        self.z0 = float(0.5 * (z.min() + z.max()) if z0 is None else z0)
        self.scale = float(scale if scale is not None else max(r.max(), 0.5 * (z.max() - z.min())))
        A = _harmonic_basis(r, z, self.degree, self.z0, self.scale).reshape(-1, self.degree)
        y = np.stack([Br, Bz], axis=1).ravel()
        self.coeff, _, rank, sv = np.linalg.lstsq(A, y, rcond=rcond)
        err = (A @ self.coeff - y).reshape(-1, 2)
        # hold-out: drop every k-th point (pattern over the sample order), refit, test on the dropped points
        idx = np.arange(len(r))
        if holdout_mask is not None:
            hold = np.asarray(holdout_mask, bool).ravel()
        else:
            hold = (idx % holdout_every) == 0 if holdout_every > 1 else np.zeros(len(r), bool)
        hm = {}
        if hold.any() and (~hold).sum() * 2 > self.degree:
            At = _harmonic_basis(r[~hold], z[~hold], self.degree, self.z0, self.scale).reshape(-1, self.degree)
            ct = np.linalg.lstsq(At, np.stack([Br[~hold], Bz[~hold]], axis=1).ravel(), rcond=rcond)[0]
            Ah = _harmonic_basis(r[hold], z[hold], self.degree, self.z0, self.scale).reshape(-1, self.degree)
            eh = (Ah @ ct - np.stack([Br[hold], Bz[hold]], axis=1).ravel()).reshape(-1, 2)
            hm = {"holdout_rms_component_T": float(np.sqrt(np.mean(eh ** 2))),
                  "holdout_max_vector_T": float(np.linalg.norm(eh, axis=1).max()),
                  "holdout_points": int(hold.sum())}
        self.domain = (float(r.min()), float(r.max()), float(z.min()), float(z.max()))
        self.metrics = {"degree": self.degree, "rank": int(rank), "z0_m": self.z0, "scale_m": self.scale,
                        "training_points": int(len(r)),
                        "fit_rms_component_T": float(np.sqrt(np.mean(err ** 2))),
                        "fit_max_vector_T": float(np.linalg.norm(err, axis=1).max()),
                        "condition": float(sv[0] / sv[-1]), **hm,
                        "domain_m": {"r": [self.domain[0], self.domain[1]], "z": [self.domain[2], self.domain[3]]}}

    def field(self, r, z) -> tuple[np.ndarray, np.ndarray]:
        r, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(z, float))
        b = _harmonic_basis(r, z, self.degree, self.z0, self.scale) @ self.coeff
        return b[:, 0].reshape(r.shape), b[:, 1].reshape(r.shape)

    def psi(self, r, z) -> np.ndarray:
        """Flux function int_0^r Bz r' dr' (Gauss-Legendre, exact for the polynomial)."""
        r, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(z, float))
        x, w = np.polynomial.legendre.leggauss(self.degree // 2 + 3)
        rs = r.ravel()[:, None] * (x + 1) / 2
        _, bz = self.field(rs, np.broadcast_to(z.ravel()[:, None], rs.shape))
        return (0.5 * r.ravel() * np.sum(w * rs * bz, axis=1)).reshape(r.shape)

    def __call__(self, r, z):
        return self.field(r, z)


class PatchedField:
    """Use ``inner`` inside the box r0<=r<=r1, z0<=z<=z1 and ``outer`` elsewhere.

    Typical use: a source-free harmonic fit around a superconductor, the raw table
    (linear or cubic) for windings further away.  The two reconstructions differ at
    the patch boundary by the fit residual; keep superconductors well inside it.
    """

    def __init__(self, inner: FieldFn, outer: FieldFn, box: tuple[float, float, float, float]):
        self.inner, self.outer, self.box = inner, outer, tuple(map(float, box))

    def __call__(self, r, z):
        r, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(z, float))
        r0, r1, z0, z1 = self.box
        m = (r >= r0) & (r <= r1) & (z >= z0) & (z <= z1)
        Br, Bz = np.empty(r.shape), np.empty(r.shape)
        if m.any():
            a, b = self.inner(r[m], z[m])
            Br[m], Bz[m] = a, b
        if (~m).any():
            a, b = self.outer(r[~m], z[~m])
            Br[~m], Bz[~m] = a, b
        return Br, Bz


class RadialBlend:
    """``inner`` for r <= r_a, ``outer`` for r >= r_b, joined in between, inside z0 <= z <= z1
    (``outer`` everywhere else).  The join is flux consistent: with a smooth step s(r),

        psi   = (1 - s) psi_in + s psi_out
        Bz    = (1/r) dpsi/dr = (1 - s) Bz_in + s Bz_out + s'(r) (psi_out - psi_in) / r
        Br    = -(1/r) dpsi/dz = (1 - s) Br_in + s Br_out

    so the solver's flux function (integrated from Bz) equals psi_in near the axis and
    exactly psi_out beyond r_b.  Typical use: a source-free fit on a superconductor near the
    axis and the raw table for windings further out, each winding seeing exactly the
    outer reconstruction's flux.  ``psi_in``/``psi_out`` are callables (r, z) -> psi.
    """

    def __init__(self, inner: FieldFn, outer: FieldFn, psi_in: Callable, psi_out: Callable,
                 r_a: float, r_b: float, z_range: tuple[float, float]):
        if not 0 < r_a < r_b:
            raise ValueError("need 0 < r_a < r_b")
        self.inner, self.outer, self.psi_in, self.psi_out = inner, outer, psi_in, psi_out
        self.r_a, self.r_b, self.z_range = float(r_a), float(r_b), tuple(map(float, z_range))

    def _s(self, r):
        t = np.clip((r - self.r_a) / (self.r_b - self.r_a), 0.0, 1.0)
        s = t * t * t * (10 - 15 * t + 6 * t * t)            # C2 smoothstep
        ds = 30 * t * t * (1 - t) ** 2 / (self.r_b - self.r_a)
        return s, ds

    def __call__(self, r, z):
        r, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(z, float))
        Br, Bz = np.empty(r.shape), np.empty(r.shape)
        zin = (z >= self.z_range[0]) & (z <= self.z_range[1])
        a = zin & (r <= self.r_a)
        o = ~zin | (r >= self.r_b)
        m = zin & (r > self.r_a) & (r < self.r_b)
        if a.any():
            Br[a], Bz[a] = self.inner(r[a], z[a])
        if o.any():
            Br[o], Bz[o] = self.outer(r[o], z[o])
        if m.any():
            rm, zm = r[m], z[m]
            s, ds = self._s(rm)
            bi, bo = self.inner(rm, zm), self.outer(rm, zm)
            dpsi = self.psi_out(rm, zm) - self.psi_in(rm, zm)
            Br[m] = (1 - s) * bi[0] + s * bo[0]
            Bz[m] = (1 - s) * bi[1] + s * bo[1] + ds * dpsi / rm
        return Br, Bz


# ---------------------------------------------------------------- grids and blocks
def graded_nodes(start: float, stop: float, h: float,
                 fine: Iterable[tuple[float, float, float]] = (),
                 anchors: Iterable[float] = (), decimals: int = 10) -> np.ndarray:
    """Increasing node list from ``start`` to ``stop`` with spacing ``h``, spacing ``hf`` on
    each fine zone ``(a, b, hf)``, and every value in ``anchors`` (e.g. raw table nodes,
    so that a linear table is reproduced exactly) inserted."""
    pts = [np.arange(start, stop + 0.5 * h, h)]
    for a, b, hf in fine:
        a, b = max(a, start), min(b, stop)
        if b > a:
            pts.append(np.arange(a, b + 0.5 * hf, hf))
    anchors = np.asarray(list(anchors), float)
    pts.append(anchors[(anchors >= start) & (anchors <= stop)])
    pts.append(np.array([start, stop]))
    x = np.unique(np.round(np.concatenate(pts), decimals))
    x = x[(x >= start - 1e-15) & (x <= stop + 1e-15)]
    # drop nodes closer than 1e-3 of the local spacing to a neighbour (round-off duplicates)
    keep = np.r_[True, np.diff(x) > 1e-9]
    return x[keep]


def grid_block(field: FieldFn, r_nodes: Sequence[float], z_nodes: Sequence[float],
               source: str = "", notes: str = "", scale: float = 1.0, digits: int = 9,
               interpolation: str | None = None) -> dict:
    """Sample ``field(r, z) -> (Br, Bz)`` on the tensor grid and return the design block.

    ``r_nodes[0]`` must be 0 (the flux function is integrated from the axis).
    Values are rounded to ``digits`` significant digits to keep the JSON compact.
    ``interpolation``: ``"hermite"`` (solver default: bicubic psi(r^2, z), O(h^3) fields) or
    ``"bilinear"`` (legacy scheme); omitted means the solver default.
    """
    r = np.asarray(r_nodes, float)
    z = np.asarray(z_nodes, float)
    if abs(r[0]) > 1e-12:
        raise ValueError("r_nodes must start at r = 0 (the solver integrates the flux function from the axis)")
    if np.any(np.diff(r) <= 0) or np.any(np.diff(z) <= 0):
        raise ValueError("r_nodes and z_nodes must be strictly increasing")
    Br, Bz = np.empty((len(z), len(r))), np.empty((len(z), len(r)))
    rows = max(1, int(2e5 // len(r)))  # evaluate in row chunks (keeps fit bases small)
    for j in range(0, len(z), rows):
        R, Z = np.meshgrid(r, z[j:j + rows])
        a, b = field(R, Z)
        Br[j:j + rows], Bz[j:j + rows] = np.asarray(a, float), np.asarray(b, float)
    if not (np.all(np.isfinite(Br)) and np.all(np.isfinite(Bz))):
        raise ValueError("the field is not finite on every grid node (outside its domain?)")
    Br[:, 0] = 0.0  # exact on the axis
    fmt = lambda a: [float(f"{v:.{digits}g}") for v in a.ravel()]  # noqa: E731
    block = {"type": "grid", "r": [float(v) for v in r], "z": [float(v) for v in z],
             "Br": fmt(Br), "Bz_grid": fmt(Bz)}
    if scale != 1.0:
        block["scale"] = float(scale)
    if interpolation:
        block["interpolation"] = interpolation
    if source:
        block["source"] = source
    if notes:
        block["notes"] = notes
    return block


def from_comsol_table(table, r_nodes: Sequence[float] | None = None, z_nodes: Sequence[float] | None = None,
                      method: str = "linear", columns: tuple[str, str] = ("Br", "Bz"), **kw) -> dict:
    """COMSOL (r, z, Br, Bz) export -> design ``background`` block.

    ``table`` is a path to a COMSOL text export, a dict from ``read_comsol_text``, or a
    tuple of arrays ``(r, z, Br, Bz)``.  The samples must fill a tensor grid.  Without
    ``r_nodes``/``z_nodes`` the raw nodes are used; otherwise the table is interpolated
    (``method`` 'linear' or 'cubic') onto the given nodes.  Prefer fine nodes and a
    smooth reconstruction near superconductors (see the module docstring).
    """
    if isinstance(table, (str, Path)):
        table = read_comsol_text(table)
    if isinstance(table, dict):
        cols = table["columns"]
        data = table["data"]

        def col(name):
            for i, c in enumerate(cols):
                if c.split()[0].lower().endswith(name.lower()):
                    return data[:, i]
            raise ValueError(f"no column ending in '{name}' among {cols}")

        r, z, Br, Bz = data[:, 0], data[:, 1], col(columns[0]), col(columns[1])
    else:
        r, z, Br, Bz = map(np.asarray, table)
    rn, zn, (gBr, gBz) = regularize(r, z, Br, Bz)
    g = GriddedField(rn, zn, gBr, gBz)
    if r_nodes is None and z_nodes is None and method == "linear":
        return grid_block(lambda rr, zz: g.field(rr, zz, "linear"), rn, zn, **kw)
    return grid_block(g.callable(method), rn if r_nodes is None else r_nodes,
                      zn if z_nodes is None else z_nodes, **kw)


# ---------------------------------------------------------------- what the solver sees
def flux_function(block: dict) -> np.ndarray:
    """psi[nz, nr] exactly as the C++ core builds it (trapezoid of Bz r from the axis)."""
    r = np.asarray(block["r"], float)
    z = np.asarray(block["z"], float)
    s = float(block.get("scale", 1.0))
    Bz = s * np.asarray(block["Bz_grid"], float).reshape(len(z), len(r))
    f = Bz * r[None, :]
    psi = np.zeros_like(Bz)
    psi[:, 1:] = np.cumsum(0.5 * (f[:, 1:] + f[:, :-1]) * np.diff(r)[None, :], axis=1)
    return psi


def evaluate_block(block: dict, r, z) -> dict:
    """Br, Bz (bilinear) and psi (bilinear in r^2, z) as the solver interpolates ``block``.

    Points outside the grid get zeros, as in the solver.
    """
    rn = np.asarray(block["r"], float)
    zn = np.asarray(block["z"], float)
    nr, nz = len(rn), len(zn)
    s = float(block.get("scale", 1.0))
    Br = s * np.asarray(block["Br"], float).reshape(nz, nr)
    Bz = s * np.asarray(block["Bz_grid"], float).reshape(nz, nr)
    P = flux_function(block)
    r, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(z, float))
    rf, zf = r.ravel(), z.ravel()
    inside = (rf >= rn[0]) & (rf <= rn[-1]) & (zf >= zn[0]) & (zf <= zn[-1])
    rc, zc = np.clip(rf, rn[0], rn[-1]), np.clip(zf, zn[0], zn[-1])
    i = np.clip(np.searchsorted(rn, rc, side="right") - 1, 0, nr - 2)
    j = np.clip(np.searchsorted(zn, zc, side="right") - 1, 0, nz - 2)
    tr = (rc - rn[i]) / (rn[i + 1] - rn[i])
    tz = (zc - zn[j]) / (zn[j + 1] - zn[j])
    r0, r1 = rn[i] ** 2, rn[i + 1] ** 2
    ts = np.clip(np.where(r1 > r0, (rc * rc - r0) / np.where(r1 > r0, r1 - r0, 1), tr), 0, 1)

    def bil(G, t):
        return (1 - tz) * ((1 - t) * G[j, i] + t * G[j, i + 1]) + tz * ((1 - t) * G[j + 1, i] + t * G[j + 1, i + 1])

    out = {"Br": bil(Br, tr), "Bz": bil(Bz, tr), "psi": bil(P, ts)}
    for k in out:
        out[k] = np.where(inside, out[k], 0.0).reshape(r.shape)
    return out


def accuracy_report(block: dict, r, z, Br, Bz, reference_psi: Callable | None = None,
                    region: tuple[float, float, float, float] | None = None) -> dict:
    """Compare the block, as interpolated by the solver, with raw samples (r, z, Br, Bz).

    Optionally restrict to ``region`` = (r0, r1, z0, z1) and compare psi with
    ``reference_psi(r, z)``.  Also reports the Maxwell consistency of the block:
    the radial field implied by its flux function, -(1/r) dpsi/dz, versus its own Br
    (they differ when Bz and Br are not a divergence-free pair, or from interpolation).
    """
    r, z, Br, Bz = (np.asarray(a, float).ravel() for a in (r, z, Br, Bz))
    if region is not None:
        m = (r >= region[0]) & (r <= region[1]) & (z >= region[2]) & (z <= region[3])
        r, z, Br, Bz = r[m], z[m], Br[m], Bz[m]
    e = evaluate_block(block, r, z)
    d = np.hypot(e["Br"] - Br, e["Bz"] - Bz)
    rep = {"points": int(len(r)), "max_vector_error_T": float(d.max()) if len(d) else 0.0,
           "rms_vector_error_T": float(np.sqrt(np.mean(d ** 2))) if len(d) else 0.0,
           "max_sample_field_T": float(np.hypot(Br, Bz).max()) if len(d) else 0.0}
    if len(d):
        k = int(np.argmax(d))
        rep["worst_point_m"] = [float(r[k]), float(z[k])]
    if reference_psi is not None and len(r):
        p = reference_psi(r, z)
        rep["max_psi_error_Wb_per_rad"] = float(np.max(np.abs(e["psi"] - p)))
        rep["max_psi_relative"] = float(np.max(np.abs(e["psi"] - p)) / max(np.max(np.abs(p)), 1e-300))
    zn = np.asarray(block["z"], float)
    m = (r > 1e-3) & (z - 1e-5 >= zn[0]) & (z + 1e-5 <= zn[-1])
    if m.any():
        h = 1e-5
        dp = (evaluate_block(block, r[m], z[m] + h)["psi"] - evaluate_block(block, r[m], z[m] - h)["psi"]) / (2 * h)
        rep["max_Br_from_psi_minus_Br_T"] = float(np.max(np.abs(-dp / r[m] - e["Br"][m])))
    return rep
