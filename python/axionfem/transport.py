"""axionfem.transport (beta): a builder for the scalar-transport operator axf::transport::ScalarTransport
(cpp/src/numerics/transport/), the operator behind ``axionfem.fluids.solve_heat`` and the energy equation of
``axionfem.fluids.solve_boussinesq``.  Steady advection-diffusion with SUPG; formulation: docs/guides/fluids.md.

Beta: the interface may change in a later release.  The first import of ``axionfem.transport`` in a process emits one
:class:`axionfem.BetaWarning` (and importing it imports ``axionfem.fluids``, which emits its own).

    op = axionfem.transport.ScalarTransport(mesh, order=2, geometry="planar")
    op.set_capacity(a); op.set_diffusivity(kappa, grad=...); op.set_velocity(b, form="convective", divergence=None)  # b = the velocity; the advective flux velocity is a b
    op.add_bc(attrs, "dirichlet" | "flux" | "robin" | "total_flux" | "zero" | "axis", **data)
    op.set_stabilization("supg", tau_rule="shakib_hK")
    sol = op.solve(criterion="dual_norm", rtol=1e-11)
    sol.scalar(elements, ref); sol.residual_full(); sol.boundary_weights(attrs); sol.tau_samples()

Flux data are INWARD (g = n . kappa grad T).  Only the dual-norm convergence criterion is offered.  Callables are
vectorized f(X, Y).  Needs an MPI build; ONE rank only (more is refused).
"robin" with form="conservative_flux" is the total-flux Robin; "outflow" is refused with that form (see solve_heat).
"""
from __future__ import annotations

from . import fluids as _fl
from ._beta import warn_beta

warn_beta("axionfem.transport")


class ScalarTransport:
    def __init__(self, mesh, order=2, geometry="planar", quadrature=None):
        if quadrature not in (None, 2 * order + 4):
            raise ValueError(f"quadrature degree {quadrature} is refused: the operator integrates at 2p + 4 = "
                             f"{2 * order + 4}")
        self.mesh, self.order, self.geometry = mesh, order, geometry
        self._cap, self._kappa, self._gk, self._beta, self._div, self._form = 1.0, None, None, None, None, "convective"
        self._bcs, self._src, self._supg, self._tau, self._shift = [], None, False, "shakib_hK", 0.0

    def set_capacity(self, a):
        self._cap = float(a)

    def set_diffusivity(self, kappa, grad=None):
        self._kappa, self._gk = kappa, grad

    def set_velocity(self, b, form="convective", divergence=None):
        """b: the VELOCITY callable b(X, Y) -> (n, 2) (or a NavierStokesSolution); the operator multiplies it by the
        capacity, so the advective flux velocity is beta = a b."""
        self._beta, self._form, self._div = b, form, divergence

    def set_source(self, s):
        self._src = s

    def set_shift(self, s):
        self._shift = float(s)

    def add_bc(self, attrs, kind, **data):
        attrs = [attrs] if isinstance(attrs, int) else list(attrs)
        self._bcs.append(dict(attributes=attrs, kind=kind, **data))

    def set_stabilization(self, kind="supg", tau_rule="shakib_hK"):
        if kind not in ("supg", None, "none"):
            raise ValueError("the only stabilization is 'supg'")
        self._supg, self._tau = kind == "supg", tau_rule

    def solve(self, criterion="dual_norm", rtol=1e-11, lu=None):
        if criterion != "dual_norm":
            raise ValueError("only criterion='dual_norm' is offered by the operator")
        return _fl.solve_heat(self.mesh, self._kappa, capacity=self._cap, velocity=self._beta, source=self._src,
                              bcs=self._bcs, geometry=self.geometry, order=self.order, form=self._form,
                              supg=self._supg, tau_rule=self._tau, grad_kappa=self._gk, div_velocity=self._div,
                              shift=self._shift, lu=lu, rtol=rtol)


__all__ = ["ScalarTransport"]
