// nanobind bindings of the kinetics kernel, the reactors and the axf::time core into _core.chem; included by
// bindings_chem.cpp (one include, one call) so this surface stays in its own file.  The documented Python API is
// python/axionfem/chem/kinetics.py; this layer only converts arrays and callables.
#pragma once

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "axionfem/chem/kinetics.hpp"
#include "axionfem/chem/reactors.hpp"
#include "axionfem/chem/time_core.hpp"

namespace axf_c1_bind {
namespace nb = nanobind;
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using axf::time::NewtonOptions;
using axf::time::Problem;
using axf::time::Trajectory;

inline nb::dict newton_dict(const axf::time::NewtonResult& r) {
  nb::dict d;
  d["converged"] = r.converged;
  d["stagnated"] = r.stagnated;
  d["relative_residual"] = r.relres;
  d["iterations"] = r.iters;
  d["jac_evals"] = r.jac_evals;
  return d;
}

inline nb::dict stats_dict(const axf::time::Stats& s) {
  nb::dict d;
  d["converged"] = s.converged;
  d["relative_residual"] = s.max_relres;
  d["newton_iters"] = s.newton_iters;
  d["solves"] = s.solves;
  d["steps"] = s.steps;
  d["rejected"] = s.rejected;
  d["newton_failures"] = s.newton_failures;
  d["jac_evals"] = s.jac_evals;
  d["stagnated"] = s.stagnated;
  return d;
}

inline NewtonOptions nopt(double rtol, int max_iters) {
  NewtonOptions o;
  o.rtol = rtol;
  o.max_iters = max_iters;
  return o;
}

inline Mat rows(const std::vector<Vec>& v) {
  if (v.empty()) return Mat();
  Mat m(v.size(), v[0].size());
  for (size_t i = 0; i < v.size(); ++i) m.row(i) = v[i].transpose();
  return m;
}

// A Python callable owned through Problem::held (so tp_traverse can show it to the GC: a callable that refers back to
// the problem, e.g. through module globals, forms a collectable cycle instead of a leak at exit).
struct PyRef {
  nb::object obj;
};
using PyRefPtr = std::shared_ptr<PyRef>;

inline PyRefPtr pyref(nb::object o) { return std::make_shared<PyRef>(PyRef{std::move(o)}); }

// the callable of r; refuses (instead of calling a null handle) once tp_clear has released it
inline const nb::object& pyref_callable(const PyRefPtr& r) {
  if (!r->obj.is_valid())
    throw std::runtime_error("chem: the Python callable of this problem was released by the garbage collector "
                             "(its owner is being collected); keep the owning object alive");
  return r->obj;
}

inline std::function<Vec(double)> wrap_source(const PyRefPtr& r) {
  return [r](double t) { return nb::cast<Vec>(pyref_callable(r)(t)); };
}

// the Problem's Python callables for the GC
inline int problem_traverse_held(const Problem& p, visitproc visit, void* arg) {
  for (const auto& h : p.held) {
    auto r = std::static_pointer_cast<PyRef>(h);
    if (r && r->obj.ptr()) Py_VISIT(r->obj.ptr());
  }
  return 0;
}
inline void problem_clear_held(Problem& p) {
  for (const auto& h : p.held) {
    auto r = std::static_pointer_cast<PyRef>(h);
    if (r) r->obj = nb::object();
  }
}

inline int timeproblem_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  return problem_traverse_held(*nb::inst_ptr<Problem>(self), visit, arg);
}
inline int timeproblem_tp_clear(PyObject* self) {
  if (nb::inst_ready(self)) problem_clear_held(*nb::inst_ptr<Problem>(self));
  return 0;
}
inline int eqsys_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  return problem_traverse_held(nb::inst_ptr<axf::chem::EquilibriumSystem>(self)->problem(), visit, arg);
}
inline int eqsys_tp_clear(PyObject* self) {
  if (nb::inst_ready(self))
    problem_clear_held(const_cast<Problem&>(nb::inst_ptr<axf::chem::EquilibriumSystem>(self)->problem()));
  return 0;
}
inline PyType_Slot timeproblem_slots[] = {{Py_tp_traverse, (void*)timeproblem_tp_traverse},
                                          {Py_tp_clear, (void*)timeproblem_tp_clear}, {0, nullptr}};
inline PyType_Slot eqsys_slots[] = {{Py_tp_traverse, (void*)eqsys_tp_traverse},
                                    {Py_tp_clear, (void*)eqsys_tp_clear}, {0, nullptr}};

// reactor builders: wrap the optional Python source and keep it in the returned problem
template <typename Build>
Problem with_source(nb::object src, Build&& build) {
  if (src.is_none()) return build(axf::chem::SourceFn());
  PyRefPtr r = pyref(src);
  Problem p = build(axf::chem::SourceFn(wrap_source(r)));
  p.held.push_back(r);
  return p;
}

inline void bind(nb::module_& m) {
  using namespace axf::chem;
  m.attr("GAS_CONSTANT_EXACT") = kGasConstantExact;

  nb::class_<KineticReaction>(m, "KineticReaction")
      .def(nb::init<>())
      .def_rw("reac", &KineticReaction::reac)
      .def_rw("nu_r", &KineticReaction::nu_r)
      .def_rw("prod", &KineticReaction::prod)
      .def_rw("nu_p", &KineticReaction::nu_p)
      .def_prop_rw("kf", [](const KineticReaction& r) { return std::make_tuple(r.kf.A, r.kf.n, r.kf.Ea); },
                   [](KineticReaction& r, std::tuple<double, double, double> v) {
                     r.kf = {std::get<0>(v), std::get<1>(v), std::get<2>(v)};
                   })
      .def_rw("reversible", &KineticReaction::reversible)
      .def_rw("has_Kc", &KineticReaction::has_Kc)
      .def_rw("Kc", &KineticReaction::Kc)
      .def_rw("has_kr", &KineticReaction::has_kr)
      .def_prop_rw("kr", [](const KineticReaction& r) { return std::make_tuple(r.kr.A, r.kr.n, r.kr.Ea); },
                   [](KineticReaction& r, std::tuple<double, double, double> v) {
                     r.kr = {std::get<0>(v), std::get<1>(v), std::get<2>(v)};
                   })
      .def_rw("order_idx", &KineticReaction::order_idx)
      .def_rw("order_val", &KineticReaction::order_val)
      .def_rw("dH", &KineticReaction::dH);

  nb::class_<Mechanism>(m, "Mechanism")
      .def("__init__",
           [](Mechanism* self, int ns, std::vector<KineticReaction> rx, const std::string& conc, const std::string& time,
              const std::string& energy, bool allow) {
             new (self) Mechanism(ns, std::move(rx), Units::from_strings(conc, time, energy), allow);
           },
           nb::arg("n_species"), nb::arg("reactions"), nb::arg("concentration") = "mol/m^3", nb::arg("time") = "s",
           nb::arg("energy") = "J/mol", nb::arg("allow_nonequilibrium_cycle") = false)
      .def("n_species", &Mechanism::n_species)
      .def("n_reactions", &Mechanism::n_reactions)
      .def("stoichiometry", &Mechanism::stoichiometry)
      .def("concentration_to_si", &Mechanism::concentration_to_si)
      .def("units", [](const Mechanism& mm) { return std::make_tuple(mm.units().conc, mm.units().time, mm.units().energy); })
      .def("reactions_si", &Mechanism::reactions_si)
      .def("rates",
           [](const Mechanism& mm, const Vec& c, double T) {
             Vec w, dT;
             Mat J;
             mm.rates(c, T, w, &J, &dT);
             return std::make_tuple(w, J, dT);
           },
           nb::arg("c"), nb::arg("T") = std::numeric_limits<double>::quiet_NaN())
      .def("progress",
           [](const Mechanism& mm, const Vec& c, double T) {
             Vec q, dT;
             Mat J;
             mm.progress(c, T, q, &J, &dT);
             return std::make_tuple(q, J, dT);
           },
           nb::arg("c"), nb::arg("T") = std::numeric_limits<double>::quiet_NaN())
      .def("rate_constants", [](const Mechanism& mm, int j, double T) {
        double kf, kr, dkf, dkr;
        mm.rate_constants(j, T, kf, kr, &dkf, &dkr);
        return std::make_tuple(kf, kr, dkf, dkr);
      });

  nb::class_<Problem>(m, "TimeProblem", nb::type_slots(timeproblem_slots))
      .def("__init__",
           [](Problem* self, int n, nb::object f, nb::object jac, nb::object mass, nb::object source) {
             new (self) Problem();
             self->n = n;
             if (!PyCallable_Check(f.ptr()) || !PyCallable_Check(jac.ptr()))
               throw nb::type_error("TimeProblem: f and jac must be callables f(t, y), jac(t, y)");
             PyRefPtr rf = pyref(f), rj = pyref(jac);
             self->f = [rf](double t, const Vec& y) { return nb::cast<Vec>(pyref_callable(rf)(t, y)); };
             self->jac = [rj](double t, const Vec& y) { return nb::cast<Mat>(pyref_callable(rj)(t, y)); };
             self->held = {rf, rj};
             if (!mass.is_none()) self->mass = nb::cast<Mat>(mass);
             if (!source.is_none()) {
               PyRefPtr rs = pyref(source);
               self->source = wrap_source(rs);
               self->held.push_back(rs);
             }
             self->validate();
           },
           nb::arg("n"), nb::arg("f"), nb::arg("jac"), nb::arg("mass") = nb::none(), nb::arg("source") = nb::none())
      .def_ro("n", &Problem::n)
      .def("rhs", &Problem::rhs)
      .def("jacobian", &Problem::jacobian)
      .def("mass_matrix", &Problem::mass_matrix)
      .def("has_source", [](const Problem& p) { return static_cast<bool>(p.source); });

  m.def("batch_problem",
        [](const Mechanism& mm, double T, nb::object s) {
          return with_source(s, [&](axf::chem::SourceFn f) { return batch_problem(mm, T, f); });
        },
        nb::arg("mechanism"), nb::arg("T") = std::numeric_limits<double>::quiet_NaN(), nb::arg("source") = nb::none());
  m.def("cstr_problem",
        [](const Mechanism& mm, double T, double tau, const Vec& cin, nb::object s) {
          return with_source(s, [&](axf::chem::SourceFn f) { return cstr_problem(mm, T, tau, cin, f); });
        },
        nb::arg("mechanism"), nb::arg("T"), nb::arg("tau"), nb::arg("c_in"), nb::arg("source") = nb::none());
  m.def("semibatch_problem",
        [](const Mechanism& mm, double T, double q, const Vec& cf, nb::object s) {
          return with_source(s, [&](axf::chem::SourceFn f) { return semibatch_problem(mm, T, q, cf, f); });
        },
        nb::arg("mechanism"), nb::arg("T"), nb::arg("q"), nb::arg("c_feed"), nb::arg("source") = nb::none());
  m.def("const_p_gas_problem",
        [](const Mechanism& mm, double T, double P, nb::object s) {
          return with_source(s, [&](axf::chem::SourceFn f) { return const_p_gas_problem(mm, T, P, f); });
        },
        nb::arg("mechanism"), nb::arg("T"), nb::arg("P"), nb::arg("source") = nb::none());
  m.def("const_p_gas_volume", &const_p_gas_volume, nb::arg("T"), nb::arg("P"), nb::arg("N"));
  m.def("adiabatic_liquid_problem",
        [](const Mechanism& mm, double rhocp, nb::object s) {
          return with_source(s, [&](axf::chem::SourceFn f) { return adiabatic_liquid_problem(mm, rhocp, f); });
        },
        nb::arg("mechanism"), nb::arg("rho_cp"), nb::arg("source") = nb::none());
  m.def("pfr_problem", &pfr_problem, nb::arg("mechanism"), nb::arg("T"), nb::arg("v0"), nb::arg("F_tot0"));

  nb::class_<EquilibriumReaction>(m, "EquilibriumReaction")
      .def(nb::init<>())
      .def_rw("reac", &EquilibriumReaction::reac)
      .def_rw("prod", &EquilibriumReaction::prod)
      .def_rw("nu_r", &EquilibriumReaction::nu_r)
      .def_rw("nu_p", &EquilibriumReaction::nu_p)
      .def_rw("K", &EquilibriumReaction::K);
  nb::class_<EquilibriumSystem>(m, "EquilibriumSystem", nb::type_slots(eqsys_slots))
      .def("__init__",
           [](EquilibriumSystem* self, const Mechanism& kin, std::vector<EquilibriumReaction> eq, double T,
              nb::object s) {
             if (s.is_none()) {
               new (self) EquilibriumSystem(kin, std::move(eq), T, nullptr);
             } else {
               PyRefPtr r = pyref(s);
               new (self) EquilibriumSystem(kin, std::move(eq), T, axf::chem::SourceFn(wrap_source(r)));
               self->hold(r);
             }
           },
           nb::arg("kinetic"), nb::arg("equilibria"), nb::arg("T") = std::numeric_limits<double>::quiet_NaN(),
           nb::arg("source") = nb::none())
      // A copy of the system's problem.  Its source closure shares the system's PyRef, so the copy
      // must not list that PyRef in its own `held` (the callable would be visited by two tp_traverse slots for one
      // reference, and tp_clear of either would empty the other's callable).  Instead the copy holds one fresh,
      // GC-visible reference to the system itself: the system lives while the copy does, the GC sees the edge
      // copy -> system (a cycle through both is collected), and tp_clear of the copy only drops that reference.
      .def("problem",
           [](nb::pointer_and_handle<EquilibriumSystem> self) {
             const Problem& P = self.p->problem();
             Problem q = P;
             q.held.clear();
             if (!P.held.empty()) q.held.push_back(pyref(nb::borrow<nb::object>(self.h)));
             return q;
           })
      .def("W", &EquilibriumSystem::W)
      .def("consistent_init", &EquilibriumSystem::consistent_init)
      .def("constraint", &EquilibriumSystem::constraint)
      .def("dae", &EquilibriumSystem::dae);

  nb::class_<axf::time::Tableau>(m, "Tableau")
      .def("__init__",
           [](axf::time::Tableau* self, const std::string& name, const Mat& A, const Vec& b, const Vec& c, nb::object bhat,
              int order, int embedded_order) {
             new (self) axf::time::Tableau();
             self->name = name;
             self->A = A;
             self->b = b;
             self->c = c;
             self->order = order;
             if (!bhat.is_none()) {
               self->bhat = nb::cast<Vec>(bhat);
               self->has_embedded = true;
               self->embedded_order = embedded_order;
             }
             self->validate();
           },
           nb::arg("name"), nb::arg("A"), nb::arg("b"), nb::arg("c"), nb::arg("bhat") = nb::none(),
           nb::arg("order") = 0, nb::arg("embedded_order") = 0)
      .def_ro("name", &axf::time::Tableau::name)
      .def_ro("A", &axf::time::Tableau::A)
      .def_ro("b", &axf::time::Tableau::b)
      .def_ro("bhat", &axf::time::Tableau::bhat)
      .def_ro("c", &axf::time::Tableau::c)
      .def_ro("order", &axf::time::Tableau::order)
      .def_ro("embedded_order", &axf::time::Tableau::embedded_order)
      .def_ro("has_embedded", &axf::time::Tableau::has_embedded);
  m.def("tableau_esdirk32", &axf::time::tableau_esdirk32);
  m.def("tableau_theta_half", &axf::time::tableau_theta_half);

  nb::class_<Trajectory>(m, "Trajectory")
      .def_ro("t", &Trajectory::t)
      .def_prop_ro("y", [](const Trajectory& tr) { return rows(tr.y); })
      .def_prop_ro("yp", [](const Trajectory& tr) { return rows(tr.yp); })
      .def_ro("order", &Trajectory::order)
      .def_ro("kind", &Trajectory::kind)
      .def_prop_ro("stats", [](const Trajectory& tr) { return stats_dict(tr.stats); })
      .def_prop_ro("events",
                   [](const Trajectory& tr) {
                     nb::list l;
                     for (const auto& e : tr.events) l.append(nb::make_tuple(e.index, e.t, e.y));
                     return l;
                   })
      .def("dense", &Trajectory::dense);

  m.def("consistent_derivative", &axf::time::consistent_derivative);
  m.def("bdf_coefficients", &axf::time::bdf_coefficients);
  m.def("bdf_advance",
        [](const Problem& P, const std::vector<double>& th, const std::vector<Vec>& yh, double h, int k, double rtol,
           int max_iters, nb::object yp) {
          Vec ypv;
          const Vec* pyp = nullptr;
          if (!yp.is_none()) { ypv = nb::cast<Vec>(yp); pyp = &ypv; }
          auto S = axf::time::bdf_advance(P, th, yh, h, k, nopt(rtol, max_iters), pyp);
          nb::dict d;
          d["y"] = S.y;
          d["est"] = S.est;
          d["y_pred"] = S.y_pred;
          d["ok"] = S.ok;
          d["newton"] = newton_dict(S.newton);
          return d;
        },
        nb::arg("problem"), nb::arg("t_hist"), nb::arg("y_hist"), nb::arg("h"), nb::arg("order"),
        nb::arg("newton_rtol") = 1e-10, nb::arg("newton_max_iters") = 25, nb::arg("yp") = nb::none());
  m.def("rk_advance",
        [](const Problem& P, const axf::time::Tableau& tb, double t, const Vec& y, const Vec& yp, double h, double rtol,
           int max_iters) {
          auto R = axf::time::rk_advance(P, tb, t, y, yp, h, nopt(rtol, max_iters));
          nb::dict d;
          d["y"] = R.y;
          d["est"] = R.est;
          d["yp_end"] = R.yp_end;
          d["y_embedded"] = R.y_embedded;
          d["ok"] = R.ok;
          d["advances_higher"] = R.advances_higher;
          d["newton"] = newton_dict(R.worst);
          d["newton_iters"] = R.newton_iters;
          d["solves"] = R.solves;
          return d;
        },
        nb::arg("problem"), nb::arg("tableau"), nb::arg("t"), nb::arg("y"), nb::arg("yp"), nb::arg("h"),
        nb::arg("newton_rtol") = 1e-10, nb::arg("newton_max_iters") = 25);
  m.def("integrate_grid_bdf",
        [](const Problem& P, int k, const std::vector<double>& times, const std::vector<Vec>& ys, double rtol,
           int max_iters) { return axf::time::integrate_grid_bdf(P, k, times, ys, nopt(rtol, max_iters)); },
        nb::arg("problem"), nb::arg("order"), nb::arg("times"), nb::arg("y_start"), nb::arg("newton_rtol") = 1e-10,
        nb::arg("newton_max_iters") = 25);
  m.def("integrate_grid_rk",
        [](const Problem& P, const axf::time::Tableau& tb, const std::vector<double>& times, const Vec& y0, double rtol,
           int max_iters) { return axf::time::integrate_grid_rk(P, tb, times, y0, nopt(rtol, max_iters)); },
        nb::arg("problem"), nb::arg("tableau"), nb::arg("times"), nb::arg("y0"), nb::arg("newton_rtol") = 1e-10,
        nb::arg("newton_max_iters") = 25);
  m.def("integrate_adaptive",
        [](const Problem& P, const std::string& method, double t0, const Vec& y0, const std::vector<double>& t_out,
           double rtol, const Vec& atol, double h0, int max_order, nb::object tableau,
           std::vector<std::function<double(double, const Vec&)>> events, double newton_rtol, int newton_max_iters,
           double safety, bool pi) {
          axf::time::AdaptiveOptions o;
          o.method = method;
          o.rtol = rtol;
          o.atol = atol;
          o.h0 = h0;
          o.max_order = max_order;
          o.tableau = tableau.is_none() ? axf::time::tableau_esdirk32() : nb::cast<axf::time::Tableau>(tableau);
          o.newton = nopt(newton_rtol, newton_max_iters);
          o.safety = safety;
          o.pi_control = pi;
          std::vector<axf::time::EventSpec> ev;
          for (auto& g : events) ev.push_back({g});
          return axf::time::integrate_adaptive(P, o, t0, y0, t_out, ev);
        },
        nb::arg("problem"), nb::arg("method"), nb::arg("t0"), nb::arg("y0"), nb::arg("t_out"), nb::arg("rtol"),
        nb::arg("atol"), nb::arg("h0") = 0.0, nb::arg("max_order") = 5, nb::arg("tableau") = nb::none(),
        nb::arg("events") = std::vector<std::function<double(double, const Vec&)>>(), nb::arg("newton_rtol") = 1e-10,
        nb::arg("newton_max_iters") = 25, nb::arg("safety") = 0.9, nb::arg("pi_control") = true);
}

}  // namespace axf_c1_bind
