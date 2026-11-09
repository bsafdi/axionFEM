// nanobind bindings of the chemistry module (beta): _core.chem (current distribution).  The documented Python API is
// python/axionfem/chem; this layer only converts arrays.
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "axionfem/chem/current_distribution.hpp"
#include "axionfem/chem/current_distribution_3d.hpp"
#include "axionfem/chem/porous_electrode.hpp"
#include "bindings_chem_c1.hpp"   // kinetics, reactors, axf::time: its own file

namespace nb = nanobind;
using namespace axf::chem;

namespace {
using RMatD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RMatI = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::vector<double> flat(const RMatD& m) { return {m.data(), m.data() + m.size()}; }
std::vector<int> flat(const RMatI& m) { return {m.data(), m.data() + m.size()}; }

NewtonOptions newton_opts(double stag, int max_iter, double gate, double limiter, double ls_min, double clip, double T,
                          int refine, const std::string& initial) {
  NewtonOptions o;
  o.stag = stag, o.max_iter = max_iter, o.gate = gate, o.limiter_rt_f = limiter, o.ls_min = ls_min, o.clip = clip;
  o.temperature = T, o.refinement_steps = refine, o.initial = initial;
  return o;
}
using RxTuple = std::tuple<std::string, double, double, double, double, double>;
Reaction to_rx(const RxTuple& t) {
  auto& [k, i0, e, aa, ac, A] = t;
  return {reaction_kind_from_string(k), i0, e, aa, ac, A};
}

nb::dict report_dict(const SolveReport& r, const char* linear = "eigen-simplicial-ldlt+refinement") {
  nb::dict d;
  d["converged"] = r.converged;
  d["relative_residual"] = r.relative_residual;
  d["iterations"] = r.iterations;
  d["residual_history"] = r.residual_history;
  d["step_lengths"] = r.step_lengths;
  d["stop"] = r.stop;
  d["limited_steps"] = r.limited_steps;
  d["line_search_halvings"] = r.line_search_halvings;
  d["clamp_active"] = r.clamp_active;
  d["field_residual"] = r.field_residual;
  d["constraint_residual"] = r.constraint_residual;
  d["unknowns"] = r.unknowns;
  d["factor_nonzeros"] = r.factor_nonzeros;
  d["seconds"] = r.seconds;
  d["linear_solver"] = linear;
  d["initial"] = r.initial;
  d["unresolved_currents"] = r.unresolved_currents;
  return d;
}
}  // namespace

void axf_bind_chem(nb::module_& parent) {
  nb::module_ m = parent.def_submodule("chem", "chemistry (beta): current distribution, kinetics, reactors");
  axf_c1_bind::bind(m);   // kinetics, reactors, time core
  m.attr("FARADAY") = kFaraday;
  m.attr("GAS_CONSTANT") = kGasConstant;

  m.def(
      "reaction_current",
      [](const std::string& kind, double i0, double E_eq, double aa, double ac, double A, double eta, double T) {
        Reaction r{reaction_kind_from_string(kind), i0, E_eq, aa, ac, A};
        double di = 0;
        const double i = reaction_current(r, eta, kFaraday / (kGasConstant * T), 200.0, &di, nullptr);
        return std::make_tuple(i, di);
      },
      nb::arg("kind"), nb::arg("i0"), nb::arg("E_eq"), nb::arg("alpha_a"), nb::arg("alpha_c"), nb::arg("A"),
      nb::arg("eta"), nb::arg("T") = 298.15, "(i, di/deta) of one reaction at overpotential eta");

  nb::class_<CurrentDistribution>(m, "CurrentDistribution")
      .def(
          "__init__",
          [](CurrentDistribution* self, const RMatD& V, const RMatI& T, const std::vector<int>& eattr, const RMatI& B,
             const std::vector<int>& battr, const RMatI& edge_v, const RMatD& edge_xy, int order, bool axi,
             int ddeg, int bdeg) {
            MeshInput in;
            in.vertices = flat(V);
            in.triangles = flat(T);
            in.element_attr = eattr;
            in.boundary = flat(B);
            in.boundary_attr = battr;
            in.edge_node_vertices = flat(edge_v);
            in.edge_node_xy = flat(edge_xy);
            new (self) CurrentDistribution(in, order, axi, ddeg, bdeg);
          },
          nb::arg("vertices"), nb::arg("triangles"), nb::arg("element_attr"), nb::arg("boundary"),
          nb::arg("boundary_attr"), nb::arg("edge_node_vertices"), nb::arg("edge_node_xy"), nb::arg("order"),
          nb::arg("axisymmetric"), nb::arg("domain_degree"), nb::arg("boundary_degree"))
      .def("domain_points", &CurrentDistribution::domain_points)
      .def("boundary_points", &CurrentDistribution::boundary_points)
      .def("set_conductivity_by_attribute", &CurrentDistribution::set_conductivity_by_attribute)
      .def("set_conductivity_at_points", &CurrentDistribution::set_conductivity_at_points)
      .def("set_source_at_points", &CurrentDistribution::set_source_at_points)
      .def("set_insulator_flux", &CurrentDistribution::set_insulator_flux)
      .def(
          "add_electrode",
          [](CurrentDistribution& cd, const std::string& name, const std::vector<int>& attrs, bool primary,
             double E_eq, const std::vector<std::tuple<std::string, double, double, double, double, double>>& rx,
             bool galv, double phi_s, double current, const std::vector<double>& g) {
            ElectrodeSpec s;
            s.name = name;
            s.attrs = attrs;
            s.primary = primary;
            s.E_eq = E_eq;
            for (auto& [k, i0, e, aa, ac, A] : rx) s.reactions.push_back({reaction_kind_from_string(k), i0, e, aa, ac, A});
            s.galvanostatic = galv;
            s.phi_s = phi_s;
            s.current = current;
            s.g = g;
            return cd.add_electrode(s);
          },
          nb::arg("name"), nb::arg("attrs"), nb::arg("primary"), nb::arg("E_eq"), nb::arg("reactions"),
          nb::arg("galvanostatic"), nb::arg("phi_s"), nb::arg("current"), nb::arg("g"))
      .def("set_exterior_halfspace", &CurrentDistribution::set_exterior_halfspace, nb::arg("attr"), nb::arg("Rb"),
           nb::arg("modes"), nb::arg("kappa"))
      .def(
          "solve",
          [](CurrentDistribution& cd, double stag, int max_iter, double gate, double limiter, double ls_min,
             double clip, double T, int refine, const std::string& initial) {
            NewtonOptions o;
            o.stag = stag;
            o.max_iter = max_iter;
            o.gate = gate;
            o.limiter_rt_f = limiter;
            o.ls_min = ls_min;
            o.clip = clip;
            o.temperature = T;
            o.refinement_steps = refine;
            o.initial = initial;
            SolveReport r;
            {
              nb::gil_scoped_release rel;
              r = cd.solve(o);
            }
            return report_dict(r);
          },
          nb::arg("stag") = 1e-13, nb::arg("max_iter") = 60, nb::arg("gate") = 1e-12, nb::arg("limiter_rt_f") = 10.0,
          nb::arg("ls_min") = 1.0 / 64.0, nb::arg("clip") = 200.0, nb::arg("temperature") = 0.0,
          nb::arg("refinement_steps") = 1, nb::arg("initial") = "zero")
      .def("set_temperature", &CurrentDistribution::set_temperature)
      .def("set_control", &CurrentDistribution::set_control, nb::arg("electrode"), nb::arg("value"))
      .def("temperature", &CurrentDistribution::temperature)
      .def("electrode_attrs", &CurrentDistribution::electrode_attrs)
      .def("num_unknowns", &CurrentDistribution::num_unknowns)
      .def("state", &CurrentDistribution::state)
      .def("set_state", &CurrentDistribution::set_state)
      .def("initial_state", &CurrentDistribution::initial_state)
      .def("residual", &CurrentDistribution::residual)
      .def("jacobian", &CurrentDistribution::jacobian)
      .def("newton_update", &CurrentDistribution::newton_update)
      .def("relres", &CurrentDistribution::relres)
      .def("phi_dofs", &CurrentDistribution::phi_dofs)
      .def("phi_s", &CurrentDistribution::phi_s)
      .def("electrode_names", &CurrentDistribution::electrode_names)
      .def("domain_quadrature",
           [](const CurrentDistribution& cd, int degree) {
             DomainQuadrature q = cd.domain_quadrature(degree);
             nb::dict d;
             d["x"] = q.x, d["w"] = q.w, d["phi"] = q.phi, d["grad"] = q.grad, d["element"] = q.element;
             return d;
           })
      .def("boundary_quadrature",
           [](const CurrentDistribution& cd, const std::vector<int>& attrs, int degree) {
             BoundaryQuadrature q = cd.boundary_quadrature(attrs, degree);
             nb::dict d;
             d["x"] = q.x, d["w"] = q.w, d["phi"] = q.phi, d["grad"] = q.grad, d["normal"] = q.normal;
             d["bdr_element"] = q.bdr_element;
             return d;
           })
      .def("residual_full", &CurrentDistribution::residual_full, nb::arg("exclude") = -1)
      .def("attr_dofs", &CurrentDistribution::attr_dofs)
      .def("electrode_dofs", &CurrentDistribution::electrode_dofs)
      .def("boundary_mass", &CurrentDistribution::boundary_mass)
      .def("mesh_vertices", &CurrentDistribution::mesh_vertices)
      .def("mesh_elements", &CurrentDistribution::mesh_elements)
      .def("dof_coordinates", &CurrentDistribution::dof_coordinates)
      .def("num_elements", &CurrentDistribution::num_elements)
      .def("num_dofs", &CurrentDistribution::num_dofs)
      .def("h_max", &CurrentDistribution::h_max)
      .def("area", &CurrentDistribution::area)
      .def("diagnostics", &CurrentDistribution::diagnostics)
      .def_prop_ro("order", &CurrentDistribution::order)
      .def_prop_ro("axisymmetric", &CurrentDistribution::axisymmetric)
      .def_prop_ro("domain_degree", &CurrentDistribution::domain_degree)
      .def_prop_ro("boundary_degree", &CurrentDistribution::boundary_degree);

  // ---------------------------------------------------------------- 1D porous electrode and 3D
  nb::class_<PorousElectrode1D>(m, "PorousElectrode1D")
      .def(nb::init<double, int, int, int>(), nb::arg("length"), nb::arg("n_elements"), nb::arg("order"),
           nb::arg("degree"))
      .def("set_conductivities", &PorousElectrode1D::set_conductivities, nb::arg("sigma"), nb::arg("kappa"))
      .def("set_exchange", [](PorousElectrode1D& p, double a_v, const RxTuple& r) { p.set_exchange(a_v, to_rx(r)); },
           nb::arg("a_v"), nb::arg("reaction"))
      .def("set_galvanostatic", &PorousElectrode1D::set_galvanostatic, nb::arg("I"))
      .def("set_potentiostatic", &PorousElectrode1D::set_potentiostatic, nb::arg("V"))
      .def(
          "solve",
          [](PorousElectrode1D& p, double stag, int max_iter, double gate, double limiter, double ls_min, double clip,
             double T, int refine, const std::string& initial) {
            SolveReport r;
            {
              nb::gil_scoped_release rel;
              r = p.solve(newton_opts(stag, max_iter, gate, limiter, ls_min, clip, T, refine, initial));
            }
            return report_dict(r, "eigen-dense-lu+refinement");
          },
          nb::arg("stag") = 1e-13, nb::arg("max_iter") = 60, nb::arg("gate") = 1e-12, nb::arg("limiter_rt_f") = 10.0,
          nb::arg("ls_min") = 1.0 / 64.0, nb::arg("clip") = 200.0, nb::arg("temperature") = 0.0,
          nb::arg("refinement_steps") = 1, nb::arg("initial") = "zero")
      .def("num_elements", &PorousElectrode1D::num_elements)
      .def("num_dofs", &PorousElectrode1D::num_dofs)
      .def("state", &PorousElectrode1D::state)
      .def("node_x", &PorousElectrode1D::node_x)
      .def("quadrature", &PorousElectrode1D::quadrature)
      .def("residual_full", &PorousElectrode1D::residual_full)
      .def("collector_row", &PorousElectrode1D::collector_row)
      .def("separator_row", &PorousElectrode1D::separator_row)
      .def("abs_action_sum", &PorousElectrode1D::abs_action_sum)
      .def("h", &PorousElectrode1D::h)
      .def_prop_ro("order", &PorousElectrode1D::order)
      .def_prop_ro("degree", &PorousElectrode1D::degree);

#ifndef AXF_NO_MPI
  nb::class_<CurrentDistribution3D>(m, "CurrentDistribution3D")
      .def(
          "__init__",
          [](CurrentDistribution3D* self, const RMatD& V, const RMatI& T, const std::vector<int>& eattr, const RMatI& B,
             const std::vector<int>& battr, const RMatI& edge_v, const RMatD& edge_xyz, int order, int ddeg, int bdeg) {
            Mesh3DInput in;
            in.vertices = flat(V);
            in.tets = flat(T);
            in.element_attr = eattr;
            in.boundary = flat(B);
            in.boundary_attr = battr;
            in.edge_node_vertices = flat(edge_v);
            in.edge_node_xyz = flat(edge_xyz);
            nb::gil_scoped_release rel;
            new (self) CurrentDistribution3D(in, order, ddeg, bdeg);
          },
          nb::arg("vertices"), nb::arg("tets"), nb::arg("element_attr"), nb::arg("boundary"), nb::arg("boundary_attr"),
          nb::arg("edge_node_vertices"), nb::arg("edge_node_xyz"), nb::arg("order"), nb::arg("domain_degree"),
          nb::arg("boundary_degree"))
      .def("rank", &CurrentDistribution3D::rank)
      .def("ranks", &CurrentDistribution3D::ranks)
      .def("domain_points", &CurrentDistribution3D::domain_points)
      .def("domain_point_elements", &CurrentDistribution3D::domain_point_elements)
      .def("element_centroids", &CurrentDistribution3D::element_centroids)
      .def("boundary_points", &CurrentDistribution3D::boundary_points)
      .def("set_conductivity_constant", &CurrentDistribution3D::set_conductivity_constant)
      .def("set_conductivity_at_points", &CurrentDistribution3D::set_conductivity_at_points)
      .def("set_source_at_points", &CurrentDistribution3D::set_source_at_points)
      .def("set_insulator_flux", &CurrentDistribution3D::set_insulator_flux)
      .def(
          "add_electrode",
          [](CurrentDistribution3D& cd, const std::string& name, const std::vector<int>& attrs, bool primary,
             double E_eq, const std::vector<RxTuple>& rx, bool galv, double phi_s, double current,
             const std::vector<double>& g) {
            ElectrodeSpec s;
            s.name = name, s.attrs = attrs, s.primary = primary, s.E_eq = E_eq;
            for (auto& r : rx) s.reactions.push_back(to_rx(r));
            s.galvanostatic = galv, s.phi_s = phi_s, s.current = current, s.g = g;
            return cd.add_electrode(s);
          },
          nb::arg("name"), nb::arg("attrs"), nb::arg("primary"), nb::arg("E_eq"), nb::arg("reactions"),
          nb::arg("galvanostatic"), nb::arg("phi_s"), nb::arg("current"), nb::arg("g"))
      .def("set_linear_tolerance", &CurrentDistribution3D::set_linear_tolerance, nb::arg("rel"), nb::arg("max_iter"))
      .def("set_control", &CurrentDistribution3D::set_control, nb::arg("electrode"), nb::arg("value"))
      .def(
          "solve",
          [](CurrentDistribution3D& cd, double stag, int max_iter, double gate, double limiter, double ls_min,
             double clip, double T, int refine, const std::string& initial) {
            SolveReport r;
            {
              nb::gil_scoped_release rel;
              r = cd.solve(newton_opts(stag, max_iter, gate, limiter, ls_min, clip, T, refine, initial));
            }
            nb::dict d = report_dict(r, "hypre CG + BoomerAMG (bordered rows by Schur complement)");
            d["cg_iterations_max"] = r.factor_nonzeros;
            return d;
          },
          nb::arg("stag") = 1e-13, nb::arg("max_iter") = 60, nb::arg("gate") = 1e-12, nb::arg("limiter_rt_f") = 10.0,
          nb::arg("ls_min") = 1.0 / 64.0, nb::arg("clip") = 200.0, nb::arg("temperature") = 0.0,
          nb::arg("refinement_steps") = 1, nb::arg("initial") = "zero")
      .def("state", &CurrentDistribution3D::state)
      .def("set_state", &CurrentDistribution3D::set_state)
      .def("phi_s", &CurrentDistribution3D::phi_s)
      .def("newton_update", &CurrentDistribution3D::newton_update)
      .def("residual", &CurrentDistribution3D::residual)
      .def("jacobian_apply", &CurrentDistribution3D::jacobian_apply)
      .def("num_true_dofs", &CurrentDistribution3D::num_true_dofs)
      .def("global_true_dofs", &CurrentDistribution3D::global_true_dofs)
      .def("domain_quadrature",
           [](const CurrentDistribution3D& cd, int degree, int e0, int e1) {
             DomainQuadrature3D q = cd.domain_quadrature(degree, e0, e1);
             nb::dict d;
             d["x"] = q.x, d["w"] = q.w, d["phi"] = q.phi, d["grad"] = q.grad;
             return d;
           },
           nb::arg("degree"), nb::arg("elem_begin") = 0, nb::arg("elem_end") = -1)
      .def("boundary_quadrature",
           [](const CurrentDistribution3D& cd, const std::vector<int>& attrs, int degree) {
             BoundaryQuadrature3D q = cd.boundary_quadrature(attrs, degree);
             nb::dict d;
             d["x"] = q.x, d["w"] = q.w, d["phi"] = q.phi, d["grad"] = q.grad, d["normal"] = q.normal;
             return d;
           })
      .def("residual_full", &CurrentDistribution3D::residual_full, nb::arg("exclude") = -1)
      .def("attr_true_dofs", &CurrentDistribution3D::attr_true_dofs)
      .def("local_elements", &CurrentDistribution3D::local_elements)
      .def("global_elements", &CurrentDistribution3D::global_elements)
      .def("owned_cells_per_rank", &CurrentDistribution3D::owned_cells_per_rank)
      .def("h_max", &CurrentDistribution3D::h_max)
      .def("volume", &CurrentDistribution3D::volume)
      .def("mesh_vertices", &CurrentDistribution3D::mesh_vertices)
      .def("mesh_elements", &CurrentDistribution3D::mesh_elements)
      .def("geometry_nodes", &CurrentDistribution3D::geometry_nodes)
      .def("diagnostics", &CurrentDistribution3D::diagnostics)
      .def_prop_ro("order", &CurrentDistribution3D::order)
      .def_prop_ro("domain_degree", &CurrentDistribution3D::domain_degree)
      .def_prop_ro("boundary_degree", &CurrentDistribution3D::boundary_degree);
#endif
}
