// Shape sensitivities of the figures of merit with respect to design parameters
// (axisymmetric, linear media).
//
// Method: semi-analytic discrete adjoint on a morphed mesh.  Write the discrete problem as
// R(y, p) = 0 with y the reduced poloidal unknowns (see axi.hpp) and J(y, p) a figure of merit.
// Then
//     dJ/dp = partial J/partial p + lambda . partial R/partial p,   S lambda = -partial J/partial y,
// with S the (symmetric) tangent operator.  The partial derivatives at frozen y come from central
// differences of ASSEMBLY AND POST-PROCESSING ONLY on a mesh morphed to the perturbed geometry
// (morph.hpp), so no field is re-solved; the adjoint solve reuses one factorization per problem.
// The toroidal (G) problems are re-solved at every perturbation instead: they are local to the
// winding/port regions plus a handful of cavity levels, so they cost almost nothing, and their
// solution then enters the partial derivative exactly.
//
// Figures of merit are built from six primitive quantities (Psi, L_p, W_mag, A_cold, iron mass,
// device radius); F, g, the energy at the requirement and the benchmark costs follow by the chain
// rule through closed forms that are checked against the solver's own post-processing.
//
// Alternatives for designs the adjoint does not cover (nonlinear iron, parameters that change the
// mesh topology): method "fd_morph" (central differences with mesh morphing, full re-solves) and
// "fd_remesh" (central differences with a new mesh for every evaluation).
#pragma once

#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionfem/model.hpp"

namespace axf::sens {

using json = nlohmann::json;

struct Options {
  std::string method = "adjoint";        // "adjoint" | "fd_morph" | "fd_remesh"
  double step = 1e-4;                    // relative finite-difference step on each parameter
  std::map<std::string, double> steps;   // absolute step overrides per parameter
  std::vector<std::string> params;       // empty: every design parameter
  std::vector<std::string> objectives;   // empty: Psi, L_p, W_mag, F, cost
  std::string benchmark;                 // default benchmark for cost objectives (empty: the first)
  SolveOptions solve;                    // workdir/order/verbosity of the perturbed evaluations
};

// Names of the quantities that can be differentiated, with a one-line meaning each.
json objective_catalog();

// Derivatives of `o.objectives` with respect to `o.params` for a solved model.
json sensitivities(const AxiModel& m, const Options& o);

}  // namespace axf::sens
