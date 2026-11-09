"""axionFEM fluids (beta): incompressible flow on MFEM.  Taylor–Hood Stokes (solve_stokes: planar and axisymmetric
with swirl; solve_stokes_3d), steady Navier–Stokes (solve_navier_stokes, with inductionless MHD in 2.5D), transient
Navier–Stokes (TransientNavierStokes), stabilized equal-order P1/P1 (gls_problem), steady scalar transport and
Boussinesq natural convection (solve_heat, solve_boussinesq; the builder axionfem.transport).

Beta: the interface may change in a later release, there is no design-file (spec/CLI) interface, and the
verification covers manufactured, closed-form and cross-code problems only (docs/guides/fluids.md,
docs/validation/fluids.md).  The first import of ``axionfem.fluids`` in a process emits one
:class:`axionfem.BetaWarning`; silence it with ``warnings.simplefilter("ignore", axionfem.BetaWarning)``.

Built only in MPI builds, as the submodule `axionfem._core._fluids`; importing this package in a build without it
raises ImportError.

    from axionfem import fluids
    mesh = fluids.Mesh.unit_square(16)
    sol = fluids.solve_stokes(mesh, nu=1.0, force=lambda x, y: (0.0, 0.0),
                              bcs=[{"attribute": a, "kind": "dirichlet", "value": g} for a in (1, 2, 3, 4)])
    sol.diagnostics["converged"], sol.diagnostics["relative_residual"], sol.diagnostics["flux_defect"]
    u, grad_u = sol.velocity(elements, ref_points)

Boundary conditions: exactly one per boundary attribute, a dict {"attribute", "kind", ...} with kind
- "dirichlet" (value: velocity; P2 nodal interpolation), "do_nothing", "traction" (value: g = sigma_form n),
- "slip" / "symmetry" (u.n = 0 strongly, zero tangential traction of the form; the nodal normal is the consistent
  (mass-conserving) normal of the discrete boundary, or the optional "normal" coefficient, which may break the
  discrete mass balance of an enclosed flow on non-symmetric curved meshes; corners are found geometrically),
- "periodic" ({"partner": attribute, "translation": (dx, dy)} with x_partner = x + translation, and optionally
  "pressure_drop" = p(x) - p(x + translation) or "flow_rate" = Q, the period-averaged section flux (equal to the
  flux through a section when the other boundaries are impermeable); the pressure the solution returns is then the
  periodic part, the gradient G is diagnostics["periodic_pressure_gradient"]),
- "axis" (axisymmetric: u_r = u_theta = 0 strongly at every velocity node on r = 0; u_z and p natural).
Forces and torques: `sol.reaction(attributes, center=(), residual_form="solved")` (Babuska-Miller, kinematic,
exerted by the fluid on the body; axisymmetric values include 2 pi). The default "solved" is the PHYSICAL force on any
wall (at rest, moving, slip), grad-div included; for form="laplacian" it adds the null-Lagrangian correction that turns
the pseudo-traction into the Cauchy traction, and needs a constant viscosity (refused otherwise). residual_form=
"laplacian" is the raw pseudo-traction nu du/dn - p n: the force only on no-slip walls AT REST (a rotating sphere:
2/3 of the torque); "stress" for a Laplacian-form solve is not Galerkin-consistent. Both are diagnostics.
Caveat: a Laplacian-form slip (or traction) wall imposes zero PSEUDO-traction, not free slip (zero Cauchy shear), so
on curved slip or traction walls the solved problem itself is not the physical one: solve with form="stress" there.
A traction may depend on the outward unit normal of the (possibly curved) boundary: {"kind": "traction",
"value": g, "normal_dependent": True} with g(x, y, n_x, n_y) (vectorized: arrays), evaluated at the boundary
quadrature points with the geometric normal of the isoparametric boundary.
Axisymmetric (geometry="axisymmetric"): the mesh is the meridional half plane (x, y) = (r, z), r >= 0; three
velocity components (u_r, u_z, u_theta); every integral carries the weight r, continuity is tested with r q; grad-div
is gamma (div_r u, div_r v)_r with div_r u = d_r u_r + u_r/r + d_z u_z, on the same rule as the viscous terms.
`pressure_point={"x0": (x, y), "value": p0}` prescribes p at a pressure node and returns the raw pressure; without
it an enclosed flow's pressure has zero mean (in axisymmetric geometry the r-weighted, i.e. 3D, mean). `grad_div=gamma` adds gamma (div u, div v), gamma in viscosity units.
Viscosity, force and data are numbers or callables; with `vectorized=True` callables take arrays X, Y and return
arrays ((n,) for scalars, (n, vdim) or a sequence of vdim arrays for vectors), and the viscosity and force are
tabulated once on the quadrature rules the solver uses. Point evaluation, boundary_nodes() and blocks() need one
MPI rank. Units are kinematic (divided by rho).
"""
from __future__ import annotations

import numpy as np

from .. import _core
from .._beta import warn_beta

try:
    _f = _core._fluids
except AttributeError as e:  # a sequential build or a wheel
    raise ImportError("axionfem.fluids needs an MPI build of axionfem (the _core._fluids submodule is missing)") from e

warn_beta("axionfem.fluids")

Mesh = _f.Mesh
StokesSolution = _f.StokesSolution
FluidsError = _core.AxionFEMError  # one exception type for the whole library
lu_backend_available = _f.lu_backend_available


def default_lu() -> str:
    """STRUMPACK when the build has it, else Eigen's SparseLU."""
    return "strumpack" if _f.lu_backend_available("strumpack") else "eigen"


def _vectorize(fn, vdim):
    """Normalize a vectorized callable f(X, Y) to a contiguous float64 array (n,) or (n, vdim)."""
    if fn is None or not callable(fn):
        return fn

    def g(X, Y):
        X = np.asarray(X, float)
        n = X.shape[0]
        r = fn(X, np.asarray(Y, float))
        if vdim == 1:
            a = np.broadcast_to(np.asarray(r, float), (n,))
        else:
            if isinstance(r, (tuple, list)):
                r = np.stack([np.broadcast_to(np.asarray(c, float), (n,)) for c in r], axis=-1)
            a = np.asarray(r, float)
            if a.shape == (vdim, n) and n != vdim:
                a = a.T
            a = np.broadcast_to(a, (n, vdim))
        return np.ascontiguousarray(a, dtype=float)
    return g


def _vectorize_normal(fn, vdim):
    """Normalize a vectorized normal-dependent traction g(X, Y, NX, NY) to a contiguous (n, vdim) array."""
    def g(X, Y, NX, NY):
        X = np.asarray(X, float)
        n = X.shape[0]
        r = fn(X, np.asarray(Y, float), np.asarray(NX, float), np.asarray(NY, float))
        if isinstance(r, (tuple, list)):
            r = np.stack([np.broadcast_to(np.asarray(c, float), (n,)) for c in r], axis=-1)
        a = np.asarray(r, float)
        if a.shape == (vdim, n) and n != vdim:
            a = a.T
        return np.ascontiguousarray(np.broadcast_to(a, (n, vdim)), dtype=float)
    return g


def _solve(mesh, nu, force, bcs, form, geometry, solver, lu, nullspace, pressure_point, order, rtol, direct_tol,
           grad_div, pressure_order, allow_unstable_pair, vectorized, min_refine_steps, solve,
           lu_threads=1, slip_corner_angle_deg=30.0, refuse_velocity_nullspace=True, velocity_nullspace_tol=1e-3):
    vdim = 3 if geometry == "axisymmetric" else 2
    bcs = [dict(b) for b in bcs]
    if vectorized:
        nu = _vectorize(nu, 1)
        force = _vectorize(force, vdim)
        for b in bcs:
            for k in ("value", "normal"):
                if callable(b.get(k)):
                    b[k] = (_vectorize_normal(b[k], vdim) if k == "value" and b.get("normal_dependent")
                            else _vectorize(b[k], vdim))
    return _f.solve_stokes(mesh, nu, force, bcs, form, geometry, solver, lu or default_lu(), nullspace,
                           pressure_point, order, rtol, direct_tol, float(grad_div), int(pressure_order),
                           bool(allow_unstable_pair), bool(vectorized), int(min_refine_steps), bool(solve),
                           int(lu_threads), float(slip_corner_angle_deg), bool(refuse_velocity_nullspace),
                           float(velocity_nullspace_tol))


def solve_stokes(mesh, nu, force=None, bcs=(), *, form="stress", geometry="planar", solver="direct", lu=None,
                 nullspace="auto", pressure_point=None, order=2, rtol=1e-10, direct_tol=1e-10,
                 grad_div=0.0, pressure_order=-1, allow_unstable_pair=False, vectorized=False, min_refine_steps=0,
                 lu_threads=1, slip_corner_angle_deg=30.0, refuse_velocity_nullspace=True, velocity_nullspace_tol=1e-3):
    """Solve Taylor–Hood Stokes; see the module docstring. form: "stress" (2 nu eps:eps) or "laplacian"
    (nu grad:grad), constant or variable nu > 0 in both. direct_tol: the gate of the direct path (relative residual
    of the consistent system); ill-scaled problems (large viscosity contrast, high aspect ratio) may need a looser
    one, they are reported as not converged, never thrown. min_refine_steps: extra LU refinement steps (a tightened
    direct solve). lu_threads: threads of the sparse LU (1, the default, is deterministic). slip_corner_angle_deg:
    slip nodes whose adjacent walls turn by more than this are corners (u = 0); a real corner turning by less is
    averaged and leaks up to sin(angle/2) of the tangential velocity (diagnostics["slip_max_averaged_kink_deg"]).
    A velocity nullspace of the continuous problem (a rigid motion with zero viscous energy in the chosen form -- the
    rotation only in the stress form -- whose rms misfit with the homogeneous boundary conditions on the curved
    boundary is <= velocity_nullspace_tol, a geometric, mesh-independent tolerance) is refused unless
    refuse_velocity_nullspace=False (then only reported in diagnostics["velocity_nullspace"]); near-nullspaces are
    solved, and diagnostics["velocity_nullspace_residual"] reports the misfit of the least constrained rigid motion. Never raises on non-convergence."""
    return _solve(mesh, nu, force, bcs, form, geometry, solver, lu, nullspace, pressure_point, order, rtol,
                  direct_tol, grad_div, pressure_order, allow_unstable_pair, vectorized, min_refine_steps,
                  True, lu_threads, slip_corner_angle_deg, refuse_velocity_nullspace, velocity_nullspace_tol)


def assemble_stokes(mesh, nu, bcs=(), *, form="stress", order=2, pressure_order=-1, allow_unstable_pair=False,
                    grad_div=0.0, lu=None):
    """Assemble without solving (blocks() for inf-sup estimates; an unstable pair may be singular)."""
    return _solve(mesh, nu, None, bcs, form, "planar", "direct", lu, "auto", None, order, 1e-10, 1e-10,
                  grad_div, pressure_order, allow_unstable_pair, False, 0, False)


NavierStokesSolution = _f.NavierStokesSolution


def solve_navier_stokes(mesh, mu, rho, force=None, bcs=(), *, form="stress", convection="standard",
                        geometry="planar", order=2, lu=None, nullspace="auto", strategy="newton", n_picard=3,
                        line_search=True, newton_rtol=1e-12, newton_maxit=30, picard_maxit=200, ptc_dt0=0.05,
                        ptc_release_dt=1e12, ptc_maxit=200, continuation_re=None, re=None, viscosity_scales=None,
                        quad_order_convection=0, direct_tol=1e-10, lu_threads=1, vectorized=False, mhd=None):
    """Steady incompressible Navier-Stokes: rho (u.grad)u - div sigma_form + grad p
    = f, div u = 0, in DYNAMIC units (mu the dynamic viscosity, number or callable; rho a constant density; p the
    physical pressure; force a force density).  form "stress" | "laplacian"; convection "standard" (default) |
    "skew" (Temam: + rho/2 (div u) u; no boundary term, so a traction BC prescribes sigma_form n = g for both forms);
    geometry "planar" | "axisymmetric" (with swirl: three components (u_r, u_z, u_theta) in data, force and output).
    Boundary conditions: dirichlet, axis, traction, do_nothing (slip, symmetry, periodic, a pressure point and
    grad-div are refused).  Direct sparse LU only.

    Nonlinear solver: a true Stokes start, then strategy "newton" (n_picard Picard steps, Newton with the exact
    Jacobian, Armijo line search on the dual-norm merit if line_search; with continuation_re=[Re_1, ..., Re_m] one
    Newton stage per Re, mu_j = (Re_m / Re_j) mu, or viscosity_scales directly), "picard" (Oseen iteration only) or
    "ptc" (pseudo-transient continuation from the Stokes solution: dt0, switched evolution relaxation, release to
    Newton once dt > ptc_release_dt).  Every residual is the steady relative residual in the discrete dual norm; the
    solve stops at newton_rtol.  Never raises on non-convergence: check diagnostics["converged"]; the fields are refused
    then, the history and the free-DOF API (state, residual, jacobian_action) are not.

    Inductionless MHD: geometry="planar25d" (z-invariant, three velocity
    components (u_x, u_y, u_z), data and output in that order, Laplacian form) together with
    mhd={"sigma": s(x, y), "B": (B_x, B_y, B_z)(x, y), "E_z": E_z, "q": q(x, y) (optional current source),
    "electric": [{"attribute": a, "kind": "insulating" | "normal_current" | "potential", "value": ...}, ...] (one per
    boundary attribute; normal_current: j_n with the outward normal; potential: phi_D), "potential_order": 2}.  The
    unknowns are (u, p, phi); J = sigma(-grad phi + E_z e_z + u x B), the force is J x B with the same J; the axial
    pressure gradient G_z is part of `force`.  rho = 0 (Stokes-like) and rho > 0 (Newton) are offered.  The solution
    has potential(elements, ref_points) and potential_boundary_nodes(attributes); the potential has zero mean when it
    is defined up to a constant (diagnostics["phi_nullspace_detected"]).  One MPI rank only."""
    vdim = 2 if geometry == "planar" else 3
    if mhd is not None:
        mhd = dict(mhd)
        if vectorized:
            for k in ("sigma", "q"):
                if callable(mhd.get(k)):
                    mhd[k] = _vectorize(mhd[k], 1)
            if callable(mhd.get("B")):
                mhd["B"] = _vectorize(mhd["B"], 3)
            mhd["electric"] = [dict(e, value=_vectorize(e["value"], 1)) if callable(e.get("value")) else dict(e)
                               for e in mhd.get("electric", [])]
    bcs = [dict(b) for b in bcs]
    if vectorized:
        mu = _vectorize(mu, 1)
        force = _vectorize(force, vdim)
        for b in bcs:
            if callable(b.get("value")):
                b["value"] = (_vectorize_normal(b["value"], vdim) if b.get("normal_dependent")
                              else _vectorize(b["value"], vdim))
    if re is not None and continuation_re is None:
        raise ValueError("re= is only meaningful with continuation_re (the last continuation Re must equal re); "
                         "to solve at a Reynolds number, set mu = rho U L / Re")
    if viscosity_scales is not None and continuation_re is None:
        vs = [float(x) for x in viscosity_scales]
        if vs and abs(vs[-1] - 1.0) > 1e-12:
            raise ValueError(f"viscosity_scales must end at 1 (the last stage solves the given mu), got {vs[-1]}")
    if continuation_re is not None:
        if viscosity_scales is not None:
            raise ValueError("give continuation_re or viscosity_scales, not both")
        res = [float(r) for r in continuation_re]
        if not res or any(not (r > 0) for r in res):
            raise ValueError("continuation_re must be a non-empty list of positive Reynolds numbers")
        if re is not None and abs(res[-1] - float(re)) > 1e-12 * abs(float(re)):
            raise ValueError(f"the last continuation Re ({res[-1]}) must be re ({re})")
        final = res[-1]
        viscosity_scales = [final / r for r in res]
    return _f.solve_navier_stokes(mesh, mu, float(rho), force, bcs, form, convection, geometry, int(order),
                                  lu or default_lu(), nullspace, strategy, int(n_picard), bool(line_search),
                                  float(newton_rtol), int(newton_maxit), int(picard_maxit), float(ptc_dt0),
                                  float(ptc_release_dt), int(ptc_maxit),
                                  [float(s) for s in (viscosity_scales or [])], int(quad_order_convection),
                                  float(direct_tol), int(lu_threads), bool(vectorized), mhd, True)


from .gls import GlsSolution, gls_problem, solve_gls  # noqa: E402  (stabilized P1/P1 GLS)
from .transient import TransientNavierStokes  # noqa: E402  (transient Navier-Stokes, BDF1/BDF2)

__all__ = ["gls_problem", "solve_gls", "GlsSolution", "Mesh", "StokesSolution", "NavierStokesSolution", "FluidsError", "solve_stokes", "assemble_stokes",
           "solve_navier_stokes", "default_lu", "lu_backend_available",
           "TransientNavierStokes"]

# 3D Taylor–Hood Stokes on a partitioned mesh (stokes3d.py)
from .stokes3d import Mesh3D, StokesSolution3D, assemble_stokes_3d, default_lu_3d, solve_stokes_3d  # noqa: E402

__all__ += ["Mesh3D", "StokesSolution3D", "solve_stokes_3d", "assemble_stokes_3d", "default_lu_3d"]
# ======================================================================================= transport, Boussinesq
ScalarTransportSolution = _f.ScalarTransportSolution
BoussinesqSolution = _f.BoussinesqSolution


def _v1(fn):
    """A vectorized scalar callable f(X, Y) normalized to a contiguous float64 array (n,)."""
    if fn is None or not callable(fn):
        return fn

    def g(X, Y):
        X = np.asarray(X, float)
        return np.ascontiguousarray(np.broadcast_to(np.asarray(fn(X, np.asarray(Y, float)), float), (X.shape[0],)),
                                    dtype=float)
    return g


def _v2(fn):
    """A vectorized 2-vector callable f(X, Y) (an (n, 2) array, a pair of arrays or a (2, n) array) as an (n, 2) array."""
    if fn is None:
        return None
    if not callable(fn):
        c = np.asarray(fn, float)
        return lambda X, Y: np.ascontiguousarray(np.broadcast_to(c, (np.asarray(X).shape[0], 2)), dtype=float)

    def g(X, Y):
        X = np.asarray(X, float)
        n = X.shape[0]
        r = fn(X, np.asarray(Y, float))
        if isinstance(r, (tuple, list)):
            r = np.stack([np.broadcast_to(np.asarray(c, float), (n,)) for c in r], axis=-1)
        a = np.asarray(r, float)
        if a.shape == (2, n) and n != 2:
            a = a.T
        return np.ascontiguousarray(np.broadcast_to(a, (n, 2)), dtype=float)
    return g


def _tbcs(bcs):
    """Transport/temperature boundary conditions: dicts {"attribute"|"attributes", "kind", "value", "h", "T_ext"};
    callables are vectorized f(X, Y) -> (n,), numbers are constants.  Flux data are INWARD: g = n . k grad T."""
    out = []
    for b in bcs:
        d = dict(b)
        for key in ("value", "h", "T_ext"):
            if callable(d.get(key)):
                d[key] = _v1(d[key])
        out.append(d)
    return out


def _check_tau(supg, tau_rule):
    if supg and tau_rule != "shakib_hK":
        raise ValueError(f"tau_rule '{tau_rule}' is not offered: the only SUPG rule is 'shakib_hK'")


def solve_heat(mesh, kappa, *, capacity=1.0, velocity=None, source=None, bcs=(), geometry="planar", order=2,
               form="convective", supg=False, tau_rule="shakib_hK", grad_kappa=None, div_velocity=None, shift=0.0,
               lu=None, rtol=1e-12, maxit=4, solve=True):
    """Steady scalar (heat) transport  a b.grad T - div(kappa grad T) = s:
    the shared axf::transport::ScalarTransport operator in heat vocabulary.  ONE MPI rank only: on a mesh partitioned
    over more than one rank it is refused with an error (it would otherwise solve one independent problem per partition).
    capacity a = rho c_p (1 for species); the advective flux velocity is beta = a b for a callable b(X, Y) -> (n, 2)
    (vectorized; NOTE the argument is b, the velocity, not beta) or beta = a u_h for a NavierStokesSolution velocity (same
    mesh, no interpolation; a BoussinesqSolution velocity is not accepted).  kappa, source: numbers or
    vectorized callables; with a variable kappa, grad_kappa(X, Y) -> (n, 2) is required for SUPG (the strong residual
    carries grad kappa . grad T).  form: "convective" (default) | "conservative_flux" (total-flux natural datum) |
    "conservative_divergence" (b.grad T + (div b) T, diffusive flux natural; div_velocity needed for a callable b).
    order 1 | 2.  supg: Shakib tau with the longest edge, C = 1, pointwise (the only rule).  bcs: dicts with kind
    "dirichlet" (value), "flux" (value g = INWARD n.kappa grad T; refused with conservative_flux), "robin" (h, T_ext:
    n.kappa grad T = h (T_ext - T) with the convective and conservative_divergence forms; with conservative_flux it is
    the TOTAL-flux Robin j_out = h (T - T_ext)), "total_flux" (value = INWARD total flux
    n.(kappa grad T - beta T); conservative_flux only), "zero" (insulation, symmetry, outflow; with conservative_flux it
    is a zero TOTAL-flux condition, i.e. a wall or a symmetry line with beta.n = 0, so kind "outflow" is refused there),
    "axis" (axisymmetric r = 0); one condition per boundary attribute.  A node shared by two Dirichlet parts takes the
    value of the later bcs entry (no check that the data agree; continuous data are the caller's job).  geometry
    "planar" | "axisymmetric" ((r, z) = mesh (x, y); weight r).
    Never raises on non-convergence: diagnostics["converged"]."""
    _check_tau(supg, tau_rule)
    vsol = velocity if hasattr(velocity, "velocity") and not callable(velocity) else None
    vfn = None if vsol is not None else _v2(velocity)
    return _f.solve_scalar_transport(
        mesh, geometry, int(order), form, bool(supg), tau_rule, float(capacity), _v1(kappa), _v2(grad_kappa), vfn,
        _v1(div_velocity), float(capacity), vsol, _v1(source), _tbcs(bcs), float(shift), lu or default_lu(),
        float(rtol), int(maxit), bool(solve))


def natconv_cavity_parameters(Ra, Pr, L=1.0, dT=1.0):
    """The de Vahl Davis scaling of the differentially heated cavity (velocities in alpha/L):
    rho0 = c_p = k = 1, mu = Pr, beta g = Ra Pr / (dT L^3).  A convenience only: the core is dimensional."""
    return {"rho0": 1.0, "cp": 1.0, "k": 1.0, "mu": float(Pr), "beta_g": float(Ra) * float(Pr) / (float(dT) * L ** 3)}


def solve_boussinesq(mesh, mu, rho0, cp, k, beta_g, *, T_ref=None, gravity="-y", velocity_bcs=(), temperature_bcs=(),
                     force=None, heat_source=None, form="stress", convection="standard", geometry="planar",
                     supg=False, line_search=False, newton_rtol=1e-12, newton_maxit=30, continuation_ra=None, ra=None,
                     buoyancy_scales=None, lu=None, nullspace="auto", grad_k=None, strategy="newton", n_picard=0,
                     start="lift", transient=False, quad_order_convection=0, direct_tol=1e-10, lu_threads=1,
                     t_order=2, solve=True):
    """Steady Boussinesq natural convection, DIMENSIONAL:
        rho0 (u.grad)u - div sigma_form + grad p = f + rho0 beta_g (T - T_ref) e_up,   div u = 0,
        rho0 c_p u.grad T - div(k grad T) = Q,
    gravity "-y" (buoyancy along +y; axisymmetric "-z", the mesh y) or "+y"; gravity along x / r is refused.
    T_ref is REQUIRED (it only shifts the pressure, p is relative to the reference hydrostatic state).  Taylor-Hood
    P2/P1 with P2 temperature; supg=True stabilizes the energy equation (Shakib tau, exact Jacobian).  One monolithic
    Newton solve on (u, p, T) from the Dirichlet lift; continuation_ra=[Ra_1, ..., Ra_m] with ra=Ra_m (or
    buoyancy_scales directly) scales beta_g stage by stage, a schedule must end at 1.  Segregated / Picard / PTC, P1
    temperature, conservative forms, T-dependent properties and a transient solve are refused.  velocity_bcs:
    the Navier-Stokes boundary conditions (dirichlet, axis, traction, do_nothing); temperature_bcs: see solve_heat.  mu, k,
    heat_source, force are numbers or vectorized callables (force: (n, vdim)); grad_k(X, Y) -> (n, 2) is needed for
    SUPG with a variable k.  Never raises on non-convergence."""
    if T_ref is None:
        raise ValueError("T_ref is required (no default): it fixes the pressure relative to the reference hydrostatic "
                         "state")
    if transient:
        raise ValueError("a transient Boussinesq solve is not offered (steady only)")
    if t_order != 2:
        raise ValueError("the coupled solver refuses P1 temperature (t_order must be 2)")
    g = str(gravity).replace(" ", "")
    ups = {"-y": 1, "+y": -1, "-z": 1, "+z": -1}
    if g not in ups:
        raise ValueError(f"gravity '{gravity}' is refused: gravity must act along the vertical axis y (planar) or z "
                         "(axisymmetric, the mesh y); gravity along x / r breaks the geometry")
    # accepted spellings: planar "-y" / "+y"; axisymmetric "-z" / "+z" (the mesh y) (alias "-y" / "+y" refused)
    want = "y" if geometry == "planar" else "z"
    if g[1] != want:
        raise ValueError(f"{geometry} Boussinesq: gravity is '-{want}' or '+{want}' (got '{gravity}')")
    if strategy != "newton":
        raise ValueError(f"strategy '{strategy}' is refused for Boussinesq: one monolithic Newton solve (continuation "
                         "scales the buoyancy)")
    if start != "lift":
        raise ValueError("the only start is 'lift'")
    if ra is not None and continuation_ra is None:
        raise ValueError("ra= is only meaningful with continuation_ra (the last continuation Ra must equal ra)")
    if buoyancy_scales is not None and continuation_ra is not None:
        raise ValueError("give continuation_ra or buoyancy_scales, not both")
    scales = None
    if buoyancy_scales is not None:
        scales = [float(x) for x in buoyancy_scales]
        if scales and abs(scales[-1] - 1.0) > 1e-12:
            raise ValueError(f"buoyancy_scales must end at 1 (the last stage solves the problem asked for), got {scales[-1]}")
    if continuation_ra is not None:
        ras = [float(r) for r in continuation_ra]
        if not ras or any(not (r > 0) for r in ras):
            raise ValueError("continuation_ra must be a non-empty list of positive Rayleigh numbers")
        if ra is None:
            raise ValueError("continuation_ra needs ra= (the final Rayleigh number; the last stage must equal it)")
        if abs(ras[-1] - float(ra)) > 1e-12 * abs(float(ra)):
            raise ValueError(f"the last continuation Ra ({ras[-1]}) must be ra ({ra})")
        scales = [r / ras[-1] for r in ras]
    _check_tau(supg, "shakib_hK")
    vdim = 3 if geometry == "axisymmetric" else 2
    vbcs = [dict(b) for b in velocity_bcs]
    for b in vbcs:
        if callable(b.get("value")):
            b["value"] = _vectorize(b["value"], vdim)
    return _f.solve_boussinesq(
        mesh, _vectorize(mu, 1), float(rho0), _vectorize(force, vdim), vbcs, form, convection, geometry, 2,
        lu or default_lu(), nullspace, int(newton_maxit), float(newton_rtol), bool(line_search),
        int(quad_order_convection), float(direct_tol), int(lu_threads), True, float(cp), float(beta_g), float(T_ref),
        ups[g], _v1(k), _v2(grad_k), _v1(heat_source), _tbcs(temperature_bcs), bool(supg), 2, scales or [], strategy,
        int(n_picard), start, False, bool(solve))


__all__ += ["ScalarTransportSolution", "BoussinesqSolution", "solve_heat", "solve_boussinesq",
            "natconv_cavity_parameters"]
