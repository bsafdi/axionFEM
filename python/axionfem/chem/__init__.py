"""Chemistry for axionFEM (beta): the steady primary and secondary current distribution in an electrolyte (2D planar
or axisymmetric, 3D on MPI ranks, a 1D porous electrode) and, in :mod:`axionfem.chem.kinetics`, 0D stiff chemical
kinetics (reactors, equilibrium, implicit time integrators).

Beta: the interface may change in a later release, and the verification covers manufactured, closed-form and
cross-code problems only (docs/guides/chemistry.md, docs/validation/chemistry.md).  The first import of
``axionfem.chem`` (or of one of its submodules) in a process emits one :class:`axionfem.BetaWarning`; silence it with
``warnings.simplefilter("ignore", axionfem.BetaWarning)``.

    from axionfem.chem import CurrentDistribution, Reaction, meshes
    m = meshes.rect(16, x1=0.01, y1=0.01)
    cd = CurrentDistribution(m, order=2, axisymmetric=False)
    cd.set_conductivity({1: 1.0})
    cd.add_electrode("anode", [1], reactions=[Reaction("bv", i0=10, E_eq=0.05)], phi_s=0.6)
    cd.add_electrode("cathode", [2], reactions=[Reaction("bv", i0=10, E_eq=-0.02)], phi_s=0.0)
    report = cd.solve()

Conventions: SI units; anodic currents positive; eta = phi_s - phi - E_eq;
axisymmetric meshes live in the meridian (r, z), every integral carries r ("per radian"), galvanostatic currents are
3D totals.  The C++ solver is ``axf::chem::CurrentDistribution`` (cpp/src/chem); this module converts inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .._beta import warn_beta
from . import meshes, meshes3d
from .meshes import TriMesh

warn_beta("axionfem.chem")

FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618

__all__ = ["CurrentDistribution", "CurrentDistribution3D", "PorousElectrode", "Reaction", "ConvergenceError", "meshes", "meshes3d", "TriMesh", "FARADAY", "GAS_CONSTANT", "NEWTON_DEFAULTS"]

# the Newton criterion of the current-distribution verification cases
NEWTON_DEFAULTS = dict(stag=1e-13, max_iter=60, gate=1e-12, limiter_rt_f=10.0, ls_min=1.0 / 64.0, clip=200.0,
                       temperature=0.0, refinement_steps=1)   # temperature 0: the problem's T (set_temperature)


class ConvergenceError(RuntimeError):
    """Newton did not reach the gate; the message says what stalled and how to proceed."""


@dataclass
class Reaction:
    """One electrode reaction: kind 'bv' (Butler-Volmer), 'linear' (linearized BV), 'tafel_a' (i0 10^(eta/A), A > 0)
    or 'tafel_c' (-i0 10^(eta/A), A < 0)."""
    kind: str
    i0: float
    E_eq: float = 0.0
    alpha_a: float = 0.5
    alpha_c: float = 0.5
    A: float = 0.0

    def as_tuple(self):
        if self.kind not in ("bv", "linear", "tafel_a", "tafel_c"):
            raise ValueError(f"Reaction kind {self.kind!r}: use 'bv', 'linear', 'tafel_a' or 'tafel_c'")
        if self.kind.startswith("tafel") and self.A == 0.0:
            raise ValueError("Tafel reaction needs its slope A (V/decade; > 0 anodic, < 0 cathodic)")
        return (self.kind, float(self.i0), float(self.E_eq), float(self.alpha_a), float(self.alpha_c), float(self.A))

    @classmethod
    def from_dict(cls, d: dict) -> "Reaction":
        def num(v, default=0.0):
            if v is None:
                return default
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            if "/" in s:
                a, b = s.split("/")
                return float(a) / float(b)
            return float(s)
        return cls(kind=d["kind"], i0=num(d.get("i0")), E_eq=num(d.get("E_eq")), alpha_a=num(d.get("alpha_a"), 0.5),
                   alpha_c=num(d.get("alpha_c"), 0.5), A=num(d.get("A")))


class CurrentDistribution:
    """Electrolyte potential with conductivity and electrode kinetics on a 2D triangle mesh (planar or axisymmetric
    meridian), Lagrange P1/P2; P2 on a mapped mesh is isoparametric (the mesh's edge-node rule).

    quadrature: {"domain": degree, "boundary": degree}; default 2k + 4 for both (the rule of the verification cases)."""

    def __init__(self, mesh: TriMesh, order: int = 1, axisymmetric: bool = False, quadrature: Optional[dict] = None,
                 curved: bool = True):
        from .._core import chem as _c
        self.mesh = mesh
        q = quadrature or {}
        dd = int(q.get("domain", 2 * order + 4))
        bd = int(q.get("boundary", 2 * order + 4))
        if order == 2 and curved:
            ev, exy = mesh.edge_nodes()
        else:
            ev, exy = np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2))
        self._cd = _c.CurrentDistribution(
            np.ascontiguousarray(mesh.vertices(), float), np.ascontiguousarray(mesh.T, np.int32),
            [int(a) for a in mesh.eattr], np.ascontiguousarray(mesh.B, np.int32), [int(a) for a in mesh.battr],
            np.ascontiguousarray(ev, np.int32).reshape(-1, 2), np.ascontiguousarray(exy, float).reshape(-1, 2),
            int(order), bool(axisymmetric), dd, bd)
        self.order, self.axisymmetric = order, axisymmetric
        self.names: list[str] = []
        self.report: Optional[dict] = None

    # ---------------------------------------------------------------- set-up
    def domain_points(self) -> np.ndarray:
        return self._cd.domain_points()

    def boundary_points(self, attrs: Sequence[int]) -> np.ndarray:
        return self._cd.boundary_points([int(a) for a in attrs])

    def set_conductivity(self, kappa):
        """kappa: {element attribute: value} or a callable X (n, 2) -> (n,) evaluated at the quadrature points."""
        if isinstance(kappa, dict):
            self._cd.set_conductivity_by_attribute({int(a): float(v) for a, v in kappa.items()})
        else:
            self._cd.set_conductivity_at_points(np.asarray(kappa(self.domain_points()), float))

    def set_source(self, f):
        self._cd.set_source_at_points(np.asarray(f(self.domain_points()), float))

    def set_insulator_flux(self, attr: int, q):
        """kappa grad phi . n_out = q(X) on boundary attribute attr (default: 0, insulating)."""
        self._cd.set_insulator_flux(int(attr), np.asarray(q(self.boundary_points([attr])), float))

    def add_electrode(self, name: str, attrs: Sequence[int], *, reactions: Sequence[Reaction] = (),
                      primary: bool = False, E_eq: float = 0.0, phi_s: float = 0.0, galvanostatic: bool = False,
                      current: float = 0.0, g=None) -> int:
        """Kinetic (reactions) or primary electrode.  Potentiostatic: phi_s is the metal potential.  Galvanostatic:
        `current` (3D total in axisymmetry, per unit depth in plane; anodic > 0) and phi_s the initial guess.
        g: optional manufactured boundary source (MMS), callable on boundary_points(attrs)."""
        if primary and reactions:
            raise ValueError(f"electrode {name!r}: a primary electrode has no reactions (give E_eq)")
        gv = [] if g is None else [float(v) for v in np.asarray(g(self.boundary_points(attrs)), float)]
        k = self._cd.add_electrode(name, [int(a) for a in attrs], bool(primary), float(E_eq),
                                   [r.as_tuple() for r in reactions], bool(galvanostatic), float(phi_s),
                                   float(current), gv)
        self.names.append(name)
        return k

    def set_exterior_halfspace(self, attr: int, Rb: float, modes: int, kappa: float):
        """Half-space even-mode DtN on the outer arc rho = Rb of an axisymmetric quarter disk (insulating z = 0)."""
        self._cd.set_exterior_halfspace(int(attr), float(Rb), int(modes), float(kappa))

    # ---------------------------------------------------------------- solve and results
    def solve(self, check: bool = True, **newton) -> dict:
        """Newton (the criterion of the verification cases by default).  initial='zero' (phi = 0), 'state' (warm start from
        set_state, e.g. the previous solution of a continuation) or 'primary' (eta = 0 on every electrode).
        check=True raises ConvergenceError when the solve does not converge; check=False returns the report."""
        o = dict(NEWTON_DEFAULTS)
        o.update(newton)
        self.report = self._cd.solve(**o)
        if check and not self.report["converged"]:
            r = self.report
            raise ConvergenceError(
                f"Newton did not converge: stop '{r['stop']}' after {r['iterations']} iterations, relres "
                f"{r['relative_residual']:.3e} (gate {o['gate']:g}), {r['limited_steps']} limited steps, exponent clip "
                f"active: {r['clamp_active']}, initial guess '{r['initial']}'.  For large cell voltages start from "
                "initial='primary', or continue in the applied potential (set_control + initial='state', see "
                "solve_continuation); raise max_iter only if the history still falls.")
        if self.report["unresolved_currents"]:
            import warnings
            warnings.warn(f"currents of {self.report['unresolved_currents']} are below the round-off of the field "
                          "residual: they are not resolved (a blocked electrode)", RuntimeWarning, stacklevel=2)
        return self.report

    def solve_continuation(self, electrode: int, values, **newton) -> dict:
        """Continuation in one electrode's control value (phi_s or current): solve at values[0] with the given
        initial guess, then at each further value from the previous solution (initial='state')."""
        rep = None
        for i, v in enumerate(values):
            self._cd.set_control(int(electrode), float(v))
            kw = dict(newton)
            if i > 0:
                kw["initial"] = "state"
            rep = self.solve(**kw)
        return rep

    def electrode_attrs(self, e: int):
        return self._cd.electrode_attrs(int(e))

    def __getattr__(self, name):          # state, residual, jacobian, quadratures, ... of the C++ object
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._cd, name)

    def electrode_index(self, name: str) -> int:
        return self.names.index(name)

    def mesh_stats(self) -> dict:
        ne = self._cd.num_elements()
        return dict(h_nominal=float(self.mesh.h_nominal), h_max=float(self._cd.h_max()),
                    h_eff=float((self._cd.area() / ne) ** 0.5), n_elements=int(ne), dofs_field=int(self._cd.num_dofs()),
                    dofs_total=int(self._cd.num_unknowns()))


# ------------------------------------------------------------------------------------------------ 3D
class CurrentDistribution3D:
    """3D current distribution on MPI ranks: tetrahedral Lagrange P1/P2 (isoparametric P2 on mapped meshes),
    the PDE, kinetics and Newton criterion of CurrentDistribution; CG + BoomerAMG.  Run under an MPI launcher
    (``mpirun -n 4 python script.py``, or ``srun``): every rank builds the same ``meshes3d.TetMesh``; the solver
    partitions it (METIS) and every array it returns is the rank's local part (local elements, local boundary
    elements, owned true dofs).

        from axionfem.chem import CurrentDistribution3D, Reaction, meshes3d
        cd = CurrentDistribution3D(meshes3d.box(8), order=2)
        cd.set_conductivity(1.0)
        cd.add_electrode("anode", [1], reactions=[Reaction("bv", i0=10, E_eq=0.05)], phi_s=0.6)
        cd.add_electrode("cathode", [2], reactions=[Reaction("bv", i0=10, E_eq=-0.02)], phi_s=0.0)
        cd.solve()

    Large overpotentials: Newton starts by default from phi_l = 0, so the whole applied potential first sits on the
    interface; far above the solution each Newton step on the exponential branch lowers alpha_a f eta by about one
    (eta by RT/(alpha_a F)), so a few volts need many iterations.  The step limiter (|d eta| <= 10 RT/F per step) is
    not the cause: it caps the large steps Newton takes when it approaches the solution from below, and it applies to
    every kinetic electrode, linear ones included.  ``initial="primary"`` is not available in 3D; use continuation:
    change the control after set-up with
    ``set_control(electrode, value)`` (phi_s of a potentiostatic electrode, the current of a galvanostatic one) and
    solve again with ``initial="state"`` (warm start from the previous solution), or call ``solve_continuation``.
    """

    def __init__(self, mesh, order: int = 1, quadrature: Optional[dict] = None):
        from .._core import chem as _c
        if not hasattr(_c, "CurrentDistribution3D"):
            raise RuntimeError("CurrentDistribution3D needs the MPI build of axionFEM (this core was built without MPI)")
        q = quadrature or {}
        dd = int(q.get("domain", 2 * order + 4))
        bd = int(q.get("boundary", 2 * order + 4))
        ev = mesh.edge_v if order == 2 else np.zeros((0, 2), np.int32)
        exy = mesh.edge_xyz if order == 2 else np.zeros((0, 3))
        self.mesh = mesh
        self._cd = _c.CurrentDistribution3D(
            np.ascontiguousarray(mesh.vertices, float), np.ascontiguousarray(mesh.T, np.int32),
            [int(a) for a in mesh.eattr], np.ascontiguousarray(mesh.B, np.int32), [int(a) for a in mesh.battr],
            np.ascontiguousarray(ev, np.int32).reshape(-1, 2), np.ascontiguousarray(exy, float).reshape(-1, 3),
            int(order), dd, bd)
        self.order = order
        self.names: list[str] = []
        self.report: Optional[dict] = None

    def set_conductivity(self, kappa):
        """kappa: a number or a callable X (n, 3) -> (n,) evaluated at the local quadrature points."""
        if callable(kappa):
            v = np.asarray(kappa(self._cd.domain_points()), float)
            self._cd.set_conductivity_at_points(v)
        else:
            self._cd.set_conductivity_constant(float(kappa))

    def set_source(self, f):
        self._cd.set_source_at_points(np.asarray(f(self._cd.domain_points()), float))

    def set_insulator_flux(self, attr: int, q):
        self._cd.set_insulator_flux(int(attr), np.asarray(q(self._cd.boundary_points([int(attr)])), float))

    def add_electrode(self, name: str, attrs: Sequence[int], *, reactions: Sequence[Reaction] = (),
                      primary: bool = False, E_eq: float = 0.0, phi_s: float = 0.0, galvanostatic: bool = False,
                      current: float = 0.0, g=None) -> int:
        """As CurrentDistribution.add_electrode; `current` is the 3D total (A), reduced over all ranks."""
        gv = [] if g is None else [float(v) for v in np.asarray(g(self._cd.boundary_points([int(a) for a in attrs])),
                                                                 float)]
        k = self._cd.add_electrode(name, [int(a) for a in attrs], bool(primary), float(E_eq),
                                   [r.as_tuple() for r in reactions], bool(galvanostatic), float(phi_s),
                                   float(current), gv)
        self.names.append(name)
        return k

    def solve(self, check: bool = True, **newton) -> dict:
        o = dict(NEWTON_DEFAULTS)
        o.update(newton)
        self.report = self._cd.solve(**o)
        if check and not self.report["converged"]:
            r = self.report
            raise ConvergenceError(f"Newton did not converge: stop '{r['stop']}' after {r['iterations']} iterations, "
                                   f"relres {r['relative_residual']:.3e}, exponent clip active: {r['clamp_active']}")
        return self.report

    def solve_continuation(self, electrode: int, values, **newton) -> dict:
        """Continuation in one electrode's control value: solve at values[0] (with the given initial guess), then at
        each further value warm-started from the previous solution (initial='state')."""
        rep = None
        for i, v in enumerate(values):
            self._cd.set_control(int(electrode), float(v))
            kw = dict(newton)
            if i > 0:
                kw["initial"] = "state"
            rep = self.solve(**kw)
        return rep

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._cd, name)


class PorousElectrode:
    """1D porous electrode: solid and electrolyte potentials on [0, l], exchange j = a_v i(phi1 - phi2 - E_eq),
    collector at x = 0 (phi1 = 0), separator at x = l (cell current I, or phi2(l) = -V).  The exchange must be a
    ``Reaction("linear", ...)`` (linearized Butler-Volmer): that is what the verification covers; ``bv`` and Tafel
    exchange are refused (nonlinear porous exchange is not implemented)."""

    def __init__(self, length: float, n_elements: int, order: int = 1, degree: Optional[int] = None):
        from .._core import chem as _c
        self._p = _c.PorousElectrode1D(float(length), int(n_elements), int(order),
                                       int(degree if degree is not None else 2 * order + 4))
        self.report: Optional[dict] = None

    def setup(self, sigma: float, kappa: float, a_v: float, reaction: Reaction, I: Optional[float] = None,
              V: Optional[float] = None):
        self._p.set_conductivities(float(sigma), float(kappa))
        self._p.set_exchange(float(a_v), reaction.as_tuple())
        if (I is None) == (V is None):
            raise ValueError("give exactly one of I (galvanostatic) or V (potentiostatic)")
        if I is not None:
            self._p.set_galvanostatic(float(I))
        else:
            self._p.set_potentiostatic(float(V))

    def solve(self, check: bool = True, **newton) -> dict:
        o = dict(NEWTON_DEFAULTS)
        o.update(newton)
        self.report = self._p.solve(**o)
        if check and not self.report["converged"]:
            raise ConvergenceError(f"porous electrode: Newton did not converge ({self.report['stop']})")
        return self.report

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._p, name)
