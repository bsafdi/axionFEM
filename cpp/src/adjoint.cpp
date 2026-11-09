#include "axionfem/adjoint.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>

#include "axionfem/core.hpp"
#include "axionfem/cost.hpp"
#include "axionfem/morph.hpp"
#include "axionfem/scan.hpp"
#include "axionfem/units.hpp"

namespace axf::sens {

using units::mu0;

namespace {

// The quantities every figure of merit is built from.  Psi, L_p and W_mag depend on the poloidal
// state and therefore need an adjoint; the geometric ones do not.  The peak fields are piecewise
// smooth (the location of the maximum jumps), so their derivatives are only as good as finite
// differences of a maximum; they are computed on request.
enum PrimId { kPsi = 0, kLp, kW, kAcold, kIron, kRdev, kScPeak, kBPeak, kNPrim };
const char* kPrimName[kNPrim] = {"Psi", "L_p", "W_mag", "A_cold", "iron_mass", "device_radius",
                                 "sc_surface_peak", "B_peak"};

struct Prims {
  double q[kNPrim] = {};
};

Prims primitives_of(const json& R) {
  Prims p;
  const json& c = R.at("coupling");
  p.q[kPsi] = c.value("Psi_SI_H_T_m2", 0.0);
  p.q[kLp] = c.value("L_p_H", 0.0);
  p.q[kW] = c.value("W_mag_J", 0.0);
  p.q[kRdev] = c.value("device_radius_m", 0.0);
  const json& g = R.at("geometry");
  p.q[kAcold] = g.value("A_cold_m2", 0.0);
  p.q[kIron] = g.value("iron_mass_kg", 0.0);
  if (R.contains("fields")) {
    const json& f = R.at("fields");
    p.q[kBPeak] = f.value("B_max_T", 0.0);
    if (f.contains("superconductor_surface_regular_T")) {
      const json& sr = f.at("superconductor_surface_regular_T");
      for (auto it = sr.begin(); it != sr.end(); ++it) p.q[kScPeak] = std::max(p.q[kScPeak], it->get<double>());
    }
  }
  return p;
}

const spec::Benchmark* find_benchmark(const spec::Design& d, const std::string& name) {
  if (d.benchmarks.empty()) return nullptr;
  if (name.empty()) return &d.benchmarks.front();
  for (auto& b : d.benchmarks) if (b.name == name) return &b;
  std::string have;
  for (auto& b : d.benchmarks) have += (have.empty() ? "" : ", ") + b.name;
  throw Error("no benchmark '" + name + "' in this design", "/benchmarks", "Benchmarks present: " + have + ".");
}

// Closed forms of the figures of merit in terms of the primitives, mirroring the solver's
// post-processing (model.cpp).  Checked against it at the nominal point.
double objective_value(const std::string& name, const Prims& p, const spec::Design& d,
                       const std::string& default_bench) {
  std::string base = name, bench = default_bench;
  const size_t colon = name.find(':');
  if (colon != std::string::npos) {
    base = name.substr(0, colon);
    bench = name.substr(colon + 1);
  }
  for (int i = 0; i < kNPrim; ++i) if (base == kPrimName[i]) return p.q[i];
  const double Psi = p.q[kPsi], Lp = p.q[kLp], W = p.q[kW], A = p.q[kAcold], Fe = p.q[kIron], Rd = p.q[kRdev];
  const double F = Lp > 0 ? std::pow(Psi, 4) / (Lp * Lp) : 0.0;
  if (base == "F") return F;
  if (base == "g_factor") return (Lp > 0 && W > 0 && Rd > 0) ? Psi * Psi / (Lp * W * Rd * Rd) / (mu0 * mu0) : 0.0;
  if (base == "cost" || base == "cost_optimal" || base == "W_at_req" || base == "F_over_F_req" ||
      base == "amplitude_scale" || base == "sc_surface_at_req" || base == "B_peak_at_req") {
    const spec::Benchmark* b = find_benchmark(d, bench);
    AXF_REQUIRE(b != nullptr, "the design has no benchmarks, so '" + name + "' cannot be evaluated", "/benchmarks");
    const scan::Requirement req = scan::F_required(*b, d.scan);
    if (base == "F_over_F_req") return F / req.F_SI;
    AXF_REQUIRE(F > 0, "the design has F = 0 (no coupling or no pickup), so '" + name + "' cannot be evaluated");
    const double s = std::pow(req.F_SI / F, 0.25);
    if (base == "amplitude_scale") return s;
    if (base == "sc_surface_at_req") return s * p.q[kScPeak];
    if (base == "B_peak_at_req") return s * p.q[kBPeak];
    const double Wr = s * s * W;
    if (base == "W_at_req") return Wr;
    if (base == "cost") return cost::evaluate(d.cost, Wr, A, Fe).total;
    // cost at the cost-optimal size: only defined for iron-free designs, as in post()
    AXF_REQUIRE(Fe == 0.0, "'cost_optimal' is reported only for iron-free designs");
    const double l = cost::optimal_scale(d.cost, Wr, A);
    return cost::evaluate(d.cost, Wr / (l * l), A * l * l, 0).total;
  }
  std::string known;
  const json cat = objective_catalog();
  for (auto& k : cat.items()) known += (known.empty() ? "" : ", ") + k.key();
  throw Error("unknown objective '" + name + "'", "", "Available: " + known +
                                                          ". Cost objectives accept a benchmark, e.g. 'cost:optimistic'.");
}

// Sensitivity of an objective to each primitive, by central differences of the closed form.
void objective_chain(const std::string& name, const Prims& p, const spec::Design& d, const std::string& bench,
                     double dJ[kNPrim]) {
  std::fill(dJ, dJ + kNPrim, 0.0);
  const std::string base = name.substr(0, name.find(':'));
  for (int i = 0; i < kNPrim; ++i) if (base == kPrimName[i]) { dJ[i] = 1.0; return; }
  if (base == "g_factor")
    AXF_REQUIRE(p.q[kLp] > 0 && p.q[kW] > 0 && p.q[kRdev] > 0,
                "g_factor derivative requires positive L_p, W_mag and device_radius", "/sensitivities/objectives");
  for (int i = 0; i < kNPrim; ++i) {
    // Absent iron and zero peaks still have a finite partial derivative. Other zero
    // physical scales can make a composite objective singular: reject, never return null.
    if (p.q[i] == 0.0) {
      if (i == kIron) {
        if (base == "cost") dJ[i] = d.cost.iron_cost_per_kg;
        continue;  // cost_optimal is defined on the iron-free subspace only
      }
      if (i == kScPeak || i == kBPeak) {
        Prims a = p; a.q[i] = 1.0;
        dJ[i] = objective_value(name, a, d, bench) - objective_value(name, p, d, bench);
        continue;
      }
      if (base == "F" || base == "g_factor" || base == "F_over_F_req") {
        if (i == kPsi) continue; // both powers have zero derivative at Psi = 0
      }
      AXF_REQUIRE(i == kAcold || i == kRdev || i == kW || i == kLp || i == kPsi,
                  "internal: unknown zero primitive");
    }
    const double h = 1e-6 * (p.q[i] != 0.0 ? std::fabs(p.q[i]) : 1.0);
    Prims a = p, b = p;
    a.q[i] += h; b.q[i] -= h;
    dJ[i] = (objective_value(name, a, d, bench) - objective_value(name, b, d, bench)) / (2 * h);
    // A singular partial matters only if this primitive actually changes; checked after dq is known.
  }
}

// One evaluation of the design at perturbed parameters: primitives, and (for the adjoint) the
// reduced residual of the frozen poloidal states.
struct Eval {
  Prims p;
  std::vector<double> res_lab, res_lab_c, res_pick, res_pick_c;
  json morph;
  double seconds = 0;
};

spec::Design perturbed(const spec::Design& d0, const std::string& name, double value) {
  json doc = d0.source;
  AXF_REQUIRE(doc.contains("parameters") && doc["parameters"].contains(name),
              "the design has no parameter '" + name + "'", "/parameters/" + name,
              "Sensitivities are taken with respect to named entries of \"parameters\".");
  doc["parameters"][name] = value;
  return spec::parse_design(doc);
}

Eval evaluate(const AxiModel& base, const morph::Morpher* morpher, const spec::Design& d1, const Options& o,
              bool frozen, bool remesh, bool fields, const std::string& tag) {
  Timer t;
  Eval ev;
  AxiModel m(d1);
  SolveOptions so = o.solve;
  so.verbose = 0;
  so.minimal_post = !fields;
  so.winding_self_force = false;  // no figure of merit uses the forces, and it costs a solve per winding
  // the perturbed evaluations are an implementation detail: no progress messages
  struct Quiet {
    int saved;
    Quiet() : saved(log_level()) { log_level() = 0; }
    ~Quiet() { log_level() = saved; }
  } quiet;
  if (remesh) {
    so.workdir = o.solve.workdir + "/sens_" + tag;
    m.run(so);
  } else {
    morph::Report rep;
    axi::MeshBundle mb = morpher->to(d1, &rep);
    ev.morph = rep.to_json();
    m.set_mesh(std::move(mb), o.solve.order);
    if (frozen) {
      const axi::ASolution& la = base.poloidal_lab_solution();
      const axi::ASolution& pi = base.poloidal_pickup_solution();
      m.freeze_poloidal(la.y, la.coil_current, pi.y, pi.coil_current);
    }
    m.solve(so);
  }
  ev.p = primitives_of(m.results());
  if (frozen) {
    if (m.poloidal_lab()) {
      ev.res_lab = m.poloidal_lab_solution().residual_y;
      ev.res_lab_c = m.poloidal_lab_solution().residual_circuit;
    }
    if (m.poloidal_pickup()) {
      ev.res_pick = m.poloidal_pickup_solution().residual_y;
      ev.res_pick_c = m.poloidal_pickup_solution().residual_circuit;
    }
  }
  ev.seconds = t.seconds();
  return ev;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

}  // namespace

json objective_catalog() {
  return {{"Psi", "signal coupling Psi = (1/I) int A_I . B_lab dV [H T m^2]"},
          {"L_p", "pickup inductance [H]"},
          {"W_mag", "stored magnetic energy of the design as drawn [J]"},
          {"F", "scan figure of merit Psi^4 / L_p^2 [T^4 H^2 m^8]"},
          {"g_factor", "g = Psi^2 / (L_p W R^2) in natural units"},
          {"A_cold", "cold-envelope area [m^2]"},
          {"iron_mass", "iron mass [kg]"},
          {"device_radius", "outer radius of the device [m]"},
          {"F_over_F_req", "F / F_req of a benchmark"},
          {"amplitude_scale", "field amplitude scale s needed to reach F_req"},
          {"W_at_req", "stored energy at the requirement, s^2 W [J]"},
          {"sc_surface_peak", "largest superconductor surface field away from sharp corners [T]"},
          {"B_peak", "largest |B| anywhere [T]"},
          {"sc_surface_at_req", "superconductor surface field at the requirement, s * peak [T]"},
          {"B_peak_at_req", "largest |B| at the requirement, s * peak [T]"},
          {"cost", "cost metric at the requirement [MCHF]"},
          {"cost_optimal", "cost at the cost-optimal size (iron-free designs) [MCHF]"}};
}

json sensitivities(const AxiModel& m0, const Options& o_in) {
  Timer timer;
  AXF_REQUIRE(m0.solved(), "solve the design before asking for sensitivities");
  Options opt = o_in;
  if (opt.solve.order <= 0) opt.solve.order = m0.space().order;  // the perturbed meshes must match
  const spec::Design& d0 = m0.design();
  AXF_REQUIRE(d0.symmetry == "axisymmetric", "sensitivities are implemented for axisymmetric designs",
              "/symmetry", "3D shape sensitivities are not implemented yet.");
  const json& R0 = m0.results();
  const std::string method = opt.method;
  AXF_REQUIRE(method == "adjoint" || method == "fd_morph" || method == "fd_remesh",
              "method must be 'adjoint', 'fd_morph' or 'fd_remesh'");
  const bool adjoint = method == "adjoint";
  if (adjoint) {
    AXF_REQUIRE(R0.value("linear", true),
                "the adjoint needs a linear problem; this design has nonlinear iron", "",
                "Use method='fd_remesh' (finite differences with remeshing) instead.");
  }

  // ------------------------------------------------------------ parameters and objectives
  std::vector<std::string> params = opt.params;
  if (params.empty()) {
    if (d0.source.contains("parameters"))
      for (auto it = d0.source["parameters"].begin(); it != d0.source["parameters"].end(); ++it)
        params.push_back(it.key());
    AXF_REQUIRE(!params.empty(), "the design has no parameters to differentiate", "/parameters",
                "Add named entries to \"parameters\" and use them in the geometry expressions.");
  }
  const Prims p0 = primitives_of(R0);
  std::vector<std::string> objectives = opt.objectives;
  if (objectives.empty()) {
    objectives = {"Psi", "L_p", "W_mag", "F"};
    // the cost is defined through the amplitude that reaches F_req, so it needs a coupling
    if (!d0.benchmarks.empty() && p0.q[kPsi] != 0.0 && p0.q[kLp] != 0.0) objectives.push_back("cost");
  }
  const std::string bench = opt.benchmark;

  // Peak-field objectives need the full post-processing of every perturbed evaluation.
  bool want_fields = false;
  for (auto& n : objectives)
    if (n.rfind("sc_surface", 0) == 0 || n.rfind("B_peak", 0) == 0) want_fields = true;
  AXF_REQUIRE(!want_fields || R0.contains("fields"),
              "peak-field objectives need a solve with the full post-processing", "",
              "Solve the design without minimal_post before asking for sc_surface/B_peak sensitivities.");

  // ------------------------------------------------------------ nominal values and self-check
  json values = json::object(), check = json::object();
  for (auto& name : objectives) values[name] = objective_value(name, p0, d0, bench);
  {
    // the closed forms must reproduce the solver's own post-processing
    auto rel = [](double a, double b) { return std::fabs(a - b) / std::max({std::fabs(a), std::fabs(b), 1e-300}); };
    double worst = 0;
    const json& c = R0.at("coupling");
    worst = std::max(worst, rel(objective_value("F", p0, d0, bench), c.value("F_SI_T4_H2_m8", 0.0)));
    worst = std::max(worst, rel(objective_value("g_factor", p0, d0, bench), c.value("g_factor", 0.0)));
    for (auto& b : R0.value("benchmarks", json::array())) {
      if (!b.contains("cost")) continue;
      const std::string nm = "cost:" + b["name"].get<std::string>();
      worst = std::max(worst, rel(objective_value(nm, p0, d0, bench), b["cost"].value("total_MCHF", 0.0)));
      worst = std::max(worst, rel(objective_value("W_at_req:" + b["name"].get<std::string>(), p0, d0, bench),
                                  b.value("W_mag_at_req_J", 0.0)));
      if (b.contains("cost_optimal_size"))
        worst = std::max(worst, rel(objective_value("cost_optimal:" + b["name"].get<std::string>(), p0, d0, bench),
                                    b["cost_optimal_size"]["cost"].value("total_MCHF", 0.0)));
    }
    check["closed_form_vs_post_processing"] = worst;
    AXF_REQUIRE(worst < 1e-9,
                strcat("internal: the sensitivity closed forms disagree with the post-processing by ", worst));
  }

  // ------------------------------------------------------------ adjoint multipliers
  // lambda solves S lambda = -dq/dy for each state-dependent primitive and each poloidal problem.
  std::vector<double> lam_lab[kNPrim], lam_lab_c[kNPrim], lam_pick[kNPrim], lam_pick_c[kNPrim];
  json adj = json::object();
  if (adjoint) {
    const axi::PoloidalProblem* pl = m0.poloidal_lab();
    const axi::PoloidalProblem* pp = m0.poloidal_pickup();
    const double I = m0.pickup_current();
    auto adjoint_solve = [](const axi::PoloidalProblem* p, std::vector<double> g, std::vector<double> gc,
                            std::vector<double>& lam, std::vector<double>& lam_c) {
      for (double& v : g) v = -v;
      for (double& v : gc) v = -v;
      p->solve_tangent(g, gc, lam, lam_c);
    };
    // Psi is linear in the poloidal potential, with the other problem's field as the weight.
    if (pl && m0.toroidal_pickup()) {
      const axi::ToroidalProblem* hp = m0.toroidal_pickup();
      const axi::HSolution& hs = m0.toroidal_pickup_solution();
      auto w = [hp, &hs](int e, const axi::PointEval& pe) {
        double H = 0, mu = 0;
        hp->field(hs, e, pe, H, mu);
        return mu * H;
      };
      std::vector<double> g;
      pl->linear_functional(m0.poloidal_lab_solution().y, w, &g);
      for (double& v : g) v /= I;
      adjoint_solve(pl, g, std::vector<double>(m0.poloidal_lab_solution().coil_current.size(), 0.0), lam_lab[kPsi],
                    lam_lab_c[kPsi]);
    }
    if (pp && m0.toroidal_lab()) {
      const axi::ToroidalProblem* hl = m0.toroidal_lab();
      const axi::HSolution& hs = m0.toroidal_lab_solution();
      auto w = [hl, &hs](int e, const axi::PointEval& pe) {
        double H = 0, mu = 0;
        hl->field(hs, e, pe, H, mu);
        return mu * H;
      };
      std::vector<double> g;
      pp->linear_functional(m0.poloidal_pickup_solution().y, w, &g);
      for (double& v : g) v /= I;
      adjoint_solve(pp, g, std::vector<double>(m0.poloidal_pickup_solution().coil_current.size(), 0.0), lam_pick[kPsi],
                    lam_pick_c[kPsi]);
    }
    if (pl) {  // W_mag
      std::vector<double> g, gc;
      pl->energy_gradient(m0.poloidal_lab_solution().y, g, gc);
      adjoint_solve(pl, g, gc, lam_lab[kW], lam_lab_c[kW]);
    }
    if (pp) {  // L_p = 2 W_pickup / I^2
      std::vector<double> g, gc;
      pp->energy_gradient(m0.poloidal_pickup_solution().y, g, gc);
      for (double& v : g) v *= 2.0 / (I * I);
      for (double& v : gc) v *= 2.0 / (I * I);
      adjoint_solve(pp, g, gc, lam_pick[kLp], lam_pick_c[kLp]);
    }
    adj["poloidal_problems"] = (pl ? 1 : 0) + (pp ? 1 : 0);
    int nsolve = 0;
    for (int k = 0; k < kNPrim; ++k) nsolve += (!lam_lab[k].empty()) + (!lam_pick[k].empty());
    adj["adjoint_solves"] = nsolve;
  }

  // ------------------------------------------------------------ parameter loop
  json grad = json::object(), prim_grad = json::object(), steps = json::object(), diag = json::object();
  for (auto& name : objectives) grad[name] = json::object();
  for (int k = 0; k < kNPrim; ++k) prim_grad[kPrimName[k]] = json::object();
  json morph_diag = json::object(), warn = json::array();
  double eval_seconds = 0;
  // one mesh velocity machinery for all parameters: the node-to-primitive map and the factorized
  // Laplace extension depend on the nominal mesh only
  std::unique_ptr<morph::Morpher> morpher;
  if (method != "fd_remesh") {
    Timer tm;
    morpher = std::make_unique<morph::Morpher>(m0.mesh_bundle(), d0);
    diag["morph_setup_seconds"] = tm.seconds();
  }
  std::map<std::string, std::array<double, kNPrim>> chain;  // dJ/d(primitive) for each objective
  for (auto& oname : objectives) {
    std::array<double, kNPrim> c{};
    if (adjoint) objective_chain(oname, p0, d0, bench, c.data());
    chain[oname] = c;
  }

  for (const std::string& pname : params) {
    auto it = d0.parameters.find(pname);
    if (it == d0.parameters.end()) {
      std::string have;
      for (auto& [k, v] : d0.parameters) have += (have.empty() ? "" : ", ") + k;
      throw Error("the design has no parameter '" + pname + "'", "/parameters/" + pname,
                  "Parameters of this design: " + have + ".");
    }
    const double v0 = it->second;
    double h = 0;
    auto sit = opt.steps.find(pname);
    if (sit != opt.steps.end()) h = sit->second;
    else if (v0 != 0.0) h = opt.step * std::fabs(v0);
    else {
      h = opt.step;
      warn.push_back("parameter '" + pname + "' is zero: the finite-difference step is absolute (" +
                     std::to_string(h) + ") and the derivative is two-sided across zero");
    }
    AXF_REQUIRE(std::isfinite(h) && h > 0, "sensitivity step must be finite and positive",
                sit == opt.steps.end() ? "/sensitivities/step" : "/sensitivities/steps/" + pname);
    steps[pname] = h;

    const spec::Design dp = perturbed(d0, pname, v0 + h);
    const spec::Design dm = perturbed(d0, pname, v0 - h);
    const bool remesh = method == "fd_remesh";
    if (!remesh) {
      // At a fixed mesh the element sizes are frozen, so a parameter that also sets them has no
      // derivative through them.  (The domain radius is an exception: the morph moves the outer
      // boundary with it.)
      bool sizes = dp.mesh.size != d0.mesh.size || dp.mesh.surface_size != d0.mesh.surface_size ||
                   dp.mesh.order != d0.mesh.order || dp.mesh.refinements != d0.mesh.refinements;
      AXF_REQUIRE(dp.regions.size() == d0.regions.size() && dm.regions.size() == d0.regions.size(), "topology changes need fd_remesh", "/parameters/" + pname);
      for (size_t i = 0; i < d0.regions.size(); ++i)
        if (dp.regions[i].mesh_size != d0.regions[i].mesh_size) sizes = true;
      if (sizes)
        warn.push_back("'" + pname + "' also sets mesh sizes; the derivative is taken at a fixed mesh, so the "
                       "geometry part is complete but the change in discretization is not included");
    }
    Eval ep = evaluate(m0, morpher.get(), dp, opt, adjoint, remesh, want_fields, pname + "_plus");
    Eval em = evaluate(m0, morpher.get(), dm, opt, adjoint, remesh, want_fields, pname + "_minus");
    eval_seconds += ep.seconds + em.seconds;
    if (!ep.morph.is_null()) morph_diag[pname] = ep.morph;

    // partial derivative of each primitive at frozen state, and the residual derivative
    double dq[kNPrim];
    for (int k = 0; k < kNPrim; ++k) dq[k] = (ep.p.q[k] - em.p.q[k]) / (2 * h);
    if (adjoint) {
      std::vector<double> dR_lab, dR_lab_c, dR_pick, dR_pick_c;
      auto diff = [&](const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) {
        AXF_REQUIRE(a.size() == b.size(), "internal: residual sizes differ between perturbations");
        out.assign(a.size(), 0.0);
        for (size_t i = 0; i < a.size(); ++i) out[i] = (a[i] - b[i]) / (2 * h);
      };
      diff(ep.res_lab, em.res_lab, dR_lab);
      diff(ep.res_lab_c, em.res_lab_c, dR_lab_c);
      diff(ep.res_pick, em.res_pick, dR_pick);
      diff(ep.res_pick_c, em.res_pick_c, dR_pick_c);
      for (int k = 0; k < kNPrim; ++k) {
        if (!lam_lab[k].empty()) dq[k] += dot(lam_lab[k], dR_lab) + dot(lam_lab_c[k], dR_lab_c);
        if (!lam_pick[k].empty()) dq[k] += dot(lam_pick[k], dR_pick) + dot(lam_pick_c[k], dR_pick_c);
      }
    }
    if (adjoint && want_fields) {
      Eval fp = evaluate(m0, morpher.get(), dp, opt, false, false, true, pname + "_peak_plus");
      Eval fm = evaluate(m0, morpher.get(), dm, opt, false, false, true, pname + "_peak_minus");
      eval_seconds += fp.seconds + fm.seconds;
      dq[kScPeak] = (fp.p.q[kScPeak] - fm.p.q[kScPeak]) / (2*h);
      dq[kBPeak] = (fp.p.q[kBPeak] - fm.p.q[kBPeak]) / (2*h);
    }
    for (int k = 0; k < kNPrim; ++k) prim_grad[kPrimName[k]][pname] = dq[k];

    // chain rule to the requested objectives
    for (auto& oname : objectives) {
      double total = (objective_value(oname, p0, dp, bench) - objective_value(oname, p0, dm, bench)) / (2*h);
      if (adjoint) for (int k = 0; k < kNPrim; ++k) if (dq[k] != 0.0) {
        AXF_REQUIRE(std::isfinite(chain[oname][k]), "objective has a singular partial derivative in a changing primitive",
                    "/sensitivities/objectives", "Choose a nonsingular design or a primitive objective.");
        total += chain[oname][k] * dq[k];
      }
      if (!adjoint) total = (objective_value(oname, ep.p, dp, bench) -
                             objective_value(oname, em.p, dm, bench)) / (2*h);
      AXF_REQUIRE(std::isfinite(total), "requested derivative is not finite", "/sensitivities/objectives",
                  "Check the objective's domain and use a finite positive step.");
      grad[oname][pname] = total;
    }
  }

  // ------------------------------------------------------------ output
  json rel = json::object();
  for (auto& oname : objectives) {
    rel[oname] = json::object();
    const double J = values[oname].get<double>();
    for (const std::string& pname : params) {
      const double v0 = d0.parameters.at(pname);
      rel[oname][pname] = (J != 0.0 && v0 != 0.0) ? grad[oname][pname].get<double>() * v0 / J : 0.0;
    }
  }
  json pvals = json::object();
  for (const std::string& pname : params) pvals[pname] = d0.parameters.at(pname);
  json pv = json::object();
  for (int k = 0; k < kNPrim; ++k) pv[kPrimName[k]] = p0.q[k];
  diag["checks"] = check;
  diag["adjoint"] = adj;
  diag["morph"] = morph_diag;
  diag["evaluation_seconds"] = eval_seconds;
  diag["seconds"] = timer.seconds();
  return {{"ok", true},
          {"method", method},
          {"params", params},
          {"parameter_values", pvals},
          {"objectives", objectives},
          {"values", values},
          {"gradient", grad},
          {"relative", rel},
          {"primitives", {{"values", pv}, {"gradient", prim_grad}}},
          {"step", steps},
          {"benchmark", bench.empty() ? (d0.benchmarks.empty() ? "" : d0.benchmarks.front().name) : bench},
          {"warnings", warn},
          {"diagnostics", diag},
          {"note", "gradient[objective][parameter] = dJ/dp in SI units; relative[...] = d log J / d log p"}};
}

}  // namespace axf::sens
