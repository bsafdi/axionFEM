// nanobind module axionfem._core: thin bindings over the C++ library.
// JSON documents cross the boundary as strings; the Python package converts
// them to dicts.  Field samples are returned as NumPy arrays.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include "axionfem/omp_shim.hpp"

#include "axionfem/adjoint.hpp"
#include "axionfem/core.hpp"
#include "axionfem/device.hpp"
#include "axionfem/model.hpp"
#include "axionfem/model3d.hpp"
#include "axionfem/render.hpp"
#include "axionfem/scan.hpp"
#include "axionfem/spec.hpp"
#include "axionfem/sym_solver.hpp"
#include "axionfem/torsion.hpp"
#include "axionfem/templates.hpp"
#include "axionfem/xsec.hpp"
#include "axionfem/fw2d.hpp"
#include "axionfem/build_provenance.hpp"

#if __has_include("axionfem/build_git.hpp")
#include "axionfem/build_git.hpp"   // generated on every build by cmake/axf_gitinfo.cmake
#else
#define AXF_BUILD_GIT_SHA "unknown"
#define AXF_BUILD_GIT_DIRTY 1
#define AXF_BUILD_GIT_DIRTY_FILES ""
#endif
#ifndef AXF_BUILD_SOURCE_DIRTY
#define AXF_BUILD_SOURCE_DIRTY "unknown"
#endif

namespace nb = nanobind;
using json = nlohmann::json;

// chemistry bindings (beta; cpp/src/chem/python/bindings_chem.cpp)
void axf_bind_chem(nb::module_& m);

namespace {
json parse(const std::string& s, const char* what) {
  try {
    return json::parse(s);
  } catch (json::parse_error& e) {
    throw axf::Error(std::string("invalid JSON for ") + what + ": " + e.what());
  }
}

nb::ndarray<nb::numpy, double> to_numpy(std::vector<double> v) {
  auto* data = new std::vector<double>(std::move(v));
  nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  return nb::ndarray<nb::numpy, double>(data->data(), {data->size()}, owner);
}

// cavity eigenmode fields (AxiModel::CavityModeField) as a dict of numpy arrays
nb::dict mode_field_dict(axf::AxiModel::CavityModeField F) {
  nb::dict d;
  d["family"] = F.family;
  d["index"] = F.index;
  d["f_Hz"] = F.f_Hz;
  d["norm"] = F.norm;
  d["r"] = to_numpy(std::move(F.r));
  d["z"] = to_numpy(std::move(F.z));
  const char* c[3] = {"r", "phi", "z"};
  for (int k = 0; k < 3; ++k) {
    d[(std::string("E_") + c[k]).c_str()] = to_numpy(std::move(F.E[k]));
    d[(std::string("H_") + c[k]).c_str()] = to_numpy(std::move(F.H[k]));
  }
  d["flux"] = to_numpy(std::move(F.flux));
  d["eps_r"] = to_numpy(std::move(F.eps_r));
  d["mu_r"] = to_numpy(std::move(F.mu_r));
  d["elem"] = to_numpy(std::vector<double>(F.elem.begin(), F.elem.end()));
  if (!F.tris.empty()) d["tris"] = to_numpy(std::vector<double>(F.tris.begin(), F.tris.end()));
  return d;
}

// cross-section cavity mode fields (XsecModel::ModeField) as a dict of numpy arrays
nb::dict xsec_field_dict(axf::XsecModel::ModeField F) {
  nb::dict d;
  d["family"] = F.family;
  d["index"] = F.index;
  d["f_Hz"] = F.f_Hz;
  d["norm"] = F.norm;
  d["x"] = to_numpy(std::move(F.x));
  d["y"] = to_numpy(std::move(F.y));
  const char* c[3] = {"x", "y", "z"};
  for (int k = 0; k < 3; ++k) {
    d[(std::string("E_") + c[k]).c_str()] = to_numpy(std::move(F.E[k]));
    d[(std::string("H_") + c[k]).c_str()] = to_numpy(std::move(F.H[k]));
  }
  d["eps_r"] = to_numpy(std::move(F.eps_r));
  d["mu_r"] = to_numpy(std::move(F.mu_r));
  d["elem"] = to_numpy(std::vector<double>(F.elem.begin(), F.elem.end()));
  if (!F.tris.empty()) d["tris"] = to_numpy(std::vector<double>(F.tris.begin(), F.tris.end()));
  return d;
}

// ------------------------------------------------------------ scan helpers (Secs. 5-6)
axf::scan::ScanParams scan_params(double Q_axion, double Theta, double rho, double v0, double vobs,
                                  bool exact_Gbar) {
  axf::scan::ScanParams p;
  p.Q_axion = Q_axion;
  p.Theta = Theta;
  p.rho_GeV_cm3 = rho;
  p.v0_kms = v0;
  p.vobs_kms = vobs;
  p.exact_Gbar = exact_Gbar;
  return p;
}

axf::scan::Stat parse_stat(const std::string& s) {
  if (s == "weak") return axf::scan::Stat::Weak;
  if (s == "limit") return axf::scan::Stat::Limit;
  if (s == "discovery") return axf::scan::Stat::Discovery;
  throw axf::Error("unknown statistic '" + s + "'; use 'weak', 'limit' or 'discovery'");
}

nb::dict band_dict(const axf::scan::ScanBand& b) {
  nb::dict d;
  d["m1_eV"] = b.m1_eV;
  d["m2_eV"] = b.m2_eV;
  d["f_min_Hz"] = b.f_min_Hz;
  d["f_max_Hz"] = b.f_max_Hz;
  d["live_time_s"] = b.live_time_s;
  d["live_time_yr"] = b.live_time_s / (365.25 * 86400.0);
  d["t_req_s"] = b.t_req_s;
  d["allocation"] = b.allocation;
  d["n_settings"] = b.n_settings;
  d["n_targets"] = b.n_targets;
  d["statistic"] = b.stat;
  d["Psi_SI_H_T_m2"] = b.Psi_SI_T_H_m2;
  d["L_H"] = b.L_H;
  d["Q"] = b.Q;
  d["T_K"] = b.T_K;
  d["eta_A"] = b.eta_A;
  d["F_SI_T4_H2_m8"] = b.F_SI_T4_H2_m8;
  d["Q_axion"] = b.Q_axion;
  d["Theta"] = b.Theta;
  d["rho_GeV_cm3"] = b.rho_GeV_cm3;
  d["v0_kms"] = b.v0_kms;
  d["vobs_kms"] = b.vobs_kms;
  d["exact_Gbar"] = b.exact_Gbar;
  return d;
}

std::vector<axf::scan::NoiseCurve> noise_curves(const std::vector<std::vector<double>>& omega,
                                                const std::vector<std::vector<double>>& N_V) {
  if (omega.size() != N_V.size())
    throw axf::Error("noise curves: " + std::to_string(omega.size()) + " frequency grids but " +
                     std::to_string(N_V.size()) + " N_V grids (one of each per setting)");
  std::vector<axf::scan::NoiseCurve> out(omega.size());
  for (size_t i = 0; i < omega.size(); ++i) {
    out[i].omega_rad_s = omega[i];
    out[i].N_V_V2_s = N_V[i];
  }
  return out;
}

nb::dict reach_dict(const axf::scan::ReachCurve& rc) {
  nb::dict d;
  d["m_eV"] = to_numpy(rc.m_eV);
  d["f_Hz"] = to_numpy(rc.f_Hz);
  d["g95_GeV"] = to_numpy(rc.g95_GeV);
  d["tau_s_per_rad_s"] = to_numpy(rc.tau_s_per_rad_s);
  d["s_max"] = to_numpy(rc.s_max);
  d["band"] = band_dict(rc.band);
  return d;
}
}  // namespace

#ifdef AXF_WITH_FLUIDS
// fluids Python boundary (beta; cpp/src/fluids/python/fluids_bindings.cpp): the submodule _core._fluids, MPI
// builds only
namespace axf::fluids {
void register_python(nb::module_& core);
}  // namespace axf::fluids
#endif

NB_MODULE(_core, m) {
  m.doc() = "axionFEM C++ core (see the axionfem package for the documented API)";

  static nb::exception<axf::Error> exc(m, "AxionFEMError");
  (void)exc;

  // $AXF_DEVICE ("cpu", or a GPU for hypre in a build configured with -DAXF_GPU); MFEM itself stays on the CPU.
  axf::configure_device();

  m.def("version", []() { return std::string(axf::kVersion); });
  m.def("device", []() { return axf::configure_device(); },
        "Where hypre runs: $AXF_DEVICE, \"cpu\" or a GPU (\"cuda\", \"cuda:N\"); MFEM always runs on the CPU "
        "(docs/install.md, GPU builds).");
  m.def(
      "build_info",
      []() {
        json b = {{"version", axf::kVersion},
                  {"mfem", MFEM_VERSION_STRING},
                  {"device", axf::configure_device()},
                  {"gpu_backend", axf::gpu_backend()},
#ifdef AXF_NO_MPI
                  {"mpi", false},
                  {"engine_3d", false},
#else
                  {"mpi", true},
                  {"engine_3d", true},
#endif
#ifdef AXF_HAVE_OPENMP
                  {"openmp", true},
#else
                  {"openmp", false},
#endif
                  // optional sparse-solver backends of the axisymmetric engine
                  {"strumpack", axf::have_strumpack()},
                  {"strumpack_version", axf::strumpack_version()},
                  // "concurrent" (8.0.x), "serialized" (other versions: one STRUMPACK call at a time), "" without
                  {"strumpack_guard", axf::strumpack_guard_mode()},
#ifdef AXF_STRUMPACK_MPI
                  {"strumpack_mpi", true},   // distributed 3D direct factorization (AXF_FW_STRUMPACK_DIST)
#else
                  {"strumpack_mpi", false},
#endif
                  {"strumpack_min_dofs", axf::SymSolver::threshold()},
                  {"metis", axf::have_metis()},
                  {"threads", omp_get_max_threads()},
                  // build provenance, all from the header cmake/axf_gitinfo.cmake writes on every build: the commit
                  // the compiled sources came from, and whether cpp/, cmake/ or CMakeLists.txt differed from it
                  // (tracked or untracked) when the build ran
                  {"git_sha", AXF_BUILD_GIT_SHA},
                  {"git_dirty", AXF_BUILD_GIT_DIRTY != 0},
                  {"git_dirty_files", AXF_BUILD_GIT_DIRTY_FILES},
                  // the same commit under the name the benchmark records use, and whether python/, benchmarks/ or
                  // tests/ also differed ("true", "false" or "unknown")
                  {"source_commit", AXF_BUILD_GIT_SHA},
                  {"source_dirty", AXF_BUILD_SOURCE_DIRTY},
                  // the same, compiled into libaxionfem: a library rebuilt from other sources disagrees
                  {"library", json::parse(axf::library_build_info())}};
#ifdef AXF_WITH_FLUIDS
        // fluids (beta)
        b["fluids"] = true;
#else
        b["fluids"] = false;
#endif
        return b.dump();
      },
      "JSON description of this build: MPI, the 3D engine, OpenMP, the GPU backend, MFEM, and the\n"
      "optional sparse-solver backends (STRUMPACK, METIS) of the axisymmetric engine.");
  m.def("set_log_level", [](int lvl) { axf::log_level() = lvl; });
#ifdef AXF_WITH_FLUIDS
  axf::fluids::register_python(m);
#endif
  m.def("set_threads", [](int n) { omp_set_num_threads(n > 0 ? n : 1); });
  m.def("get_threads", []() { return omp_get_max_threads(); });
  // the guard around every call into STRUMPACK (sym_solver.cpp), for tests/test_strumpack_threads.py
  // None where the guard does not read STRUMPACK's timer lists (the serialized guard), as in strumpack_guard_stats
  m.def("strumpack_timer_entries", []() -> nb::object {
    const auto st = axf::strumpack_guard_stats(false);
    if (axf::have_strumpack() && !st.internals_known) return nb::none();
    return nb::cast(st.timer_entries);
  }, "Entries left in STRUMPACK's task-timer lists (None when the guard does not read them; 0 without STRUMPACK).");
  m.def("strumpack_guard_stats", [](bool reset) {
    const auto st = axf::strumpack_guard_stats(reset);
    // STRUMPACK's internals are read by the concurrent guard only: null with the serialized guard
    auto internal = [&](auto v) { return st.internals_known ? axf::json(v) : axf::json(nullptr); };
    return axf::json{{"mode", axf::strumpack_guard_mode()},
                     {"timer_entries", internal(st.timer_entries)}, {"timer_lists", internal(st.timer_lists)},
                     {"max_same_thread_number", st.max_same_thread_number},
                     {"max_concurrent", st.max_concurrent},
                     {"max_concurrent_during_life", st.max_concurrent_during_life},
                     {"num_threads", internal(st.num_threads)},
                     {"task_recursion_cutoff_level", internal(st.task_recursion_cutoff_level)}}.dump();
  }, nb::arg("reset") = false, "JSON: the STRUMPACK guard's mode, its timer lists and the most calls seen at once.");
  m.def("strumpack_guard_test_hold", [](int ms) { axf::strumpack_guard_test_hold(ms); },
        "Testing only: every guarded STRUMPACK call holds its locks ms milliseconds longer (0: off).");
  // torsion: the auto mode's memory reader and decision, exposed for tests (JSON in, JSON out)
  m.def("torsion_available_memory", [](const std::string& opts) {
    return axf::axi::torsion_available_memory(axf::json::parse(opts)).dump();
  });
  m.def("torsion_auto_choice", [](const std::string& in) {
    return axf::axi::torsion_auto_choice(axf::json::parse(in)).dump();
  });

  m.def("validate", [](const std::string& doc) {
    auto d = axf::spec::parse_design(parse(doc, "design"));
    json out = {{"ok", true}, {"name", d.name}, {"parameters", d.parameters}};
    return out.dump();
  });
  m.def("resolve", [](const std::string& doc) {
    return axf::spec::resolved_geometry(axf::spec::parse_design(parse(doc, "design"))).dump();
  });
  m.def("mesh", [](const std::string& doc, const std::string& workdir, int verbose, const std::string& mesher) {
    axf::SolveOptions o;
    o.workdir = workdir;
    o.verbose = verbose;
    o.mesher = mesher;
    axf::log_level() = verbose;
    nb::gil_scoped_release release;
    return axf::mesh_any(parse(doc, "design"), o).dump();
  }, nb::arg("design"), nb::arg("workdir"), nb::arg("verbose") = 1, nb::arg("mesher") = "");
  m.def("schema", []() { return axf::spec::design_schema().dump(); });
  m.def("conductors", []() {
    // Built-in critical surfaces with a few Jc values of each (A/m^2) for orientation.
    json out = json::object();
    for (auto& [name, c] : axf::spec::builtin_conductors()) {
      auto law = axf::make_jc(c);
      json j = {{"law", c.law}, {"T_op_K", c.T_op}, {"fill_factor", c.fill_factor},
                {"n_value", c.n_value}, {"E0_V_m", c.E0}, {"anisotropic", c.anisotropic},
                {"tape_normal", {c.normal_r, c.normal_z}}, {"B_min_T", c.B_min},
                {"Tc_at_zero_field_K", law->Tc_at(0.0)}};
      json tab = json::array();
      for (double B : {0.1, 1.0, 3.0, 5.0, 10.0}) tab.push_back({{"B_T", B}, {"Jc_A_m2", law->Jc(B, 0.0, c.T_op)}});
      j["Jc_perpendicular"] = tab;
      out[name] = j;
    }
    return out.dump();
  });
  // Jc(B_perp, B_par, T) of a conductor definition (a built-in name or a design `conductors` entry).
  m.def("conductor_jc", [](const std::string& doc, std::vector<double> B_perp, std::vector<double> B_par,
                           std::vector<double> T) {
    json j = parse(doc, "conductor");
    axf::spec::Conductor c;
    if (j.is_string()) {
      auto all = axf::spec::builtin_conductors();
      auto it = all.find(j.get<std::string>());
      if (it == all.end()) throw axf::Error("unknown built-in conductor '" + j.get<std::string>() + "'");
      c = it->second;
    } else {
      // route through the design parser so that validation and defaults are the same
      json design = {{"name", "jc"},
                     {"regions", json::array({{{"name", "x"}, {"shape", {{"rect", {{"r", {0, 1}}, {"z", {0, 1}}}}}}}})},
                     {"conductors", {{"c", j}}}};
      c = axf::spec::parse_design(design).conductors.at("c");
    }
    auto law = axf::make_jc(c);
    const size_t n = B_perp.size();
    if (B_par.size() != n) throw axf::Error("B_perp and B_par must have the same length");
    if (!T.empty() && T.size() != n) throw axf::Error("T must be empty (use the conductor's T_op) or as long as B");
    std::vector<double> jc(n);
    for (size_t i = 0; i < n; ++i) jc[i] = law->Jc(B_perp[i], B_par[i], T.empty() ? c.T_op : T[i]);
    return to_numpy(jc);
  }, nb::arg("conductor"), nb::arg("B_perp"), nb::arg("B_par"), nb::arg("T"));
  // R_s(f) [ohm] of an `rf` block of the design format (validated by the design parser), and the flux_flow part.
  m.def("surface_resistance", [](const std::string& rf, std::vector<double> f) {
    json design = {{"name", "rs"},
                   {"regions", json::array({{{"name", "x"}, {"shape", {{"rect", {{"r", {0, 1}}, {"z", {0, 1}}}}}}}})},
                   {"materials", {{"wall", {{"kind", "conductor"}, {"rf", parse(rf, "rf")}}}}}};
    const auto d = axf::spec::parse_design(design);
    const auto& r = d.materials.at("wall").rf;
    std::vector<double> Rs(f.size()), ff(f.size());
    for (size_t i = 0; i < f.size(); ++i) {
      Rs[i] = r.Rs(f[i]);
      ff[i] = r.Rs_flux_flow(f[i]);
    }
    return nb::make_tuple(to_numpy(Rs), to_numpy(ff));
  }, nb::arg("rf"), nb::arg("f_Hz"));
  m.def("objectives", []() { return axf::sens::objective_catalog().dump(); });
  m.def("templates", []() { return axf::templates::list().dump(); });
  m.def("template", [](const std::string& name, const std::string& overrides) {
    return axf::templates::make(name, parse(overrides, "overrides")).dump();
  });
  m.def("requirement", [](const std::string& benchmark, const std::string& scan) {
    json b = parse(benchmark, "benchmark"), s = parse(scan, "scan");
    json doc = {{"name", "requirement"},
                {"regions", json::array({{{"name", "x"}, {"shape", {{"rect", {{"r", {0, 1}}, {"z", {0, 1}}}}}}}})},
                {"benchmarks", json::array({b})}};
    if (!s.is_null() && !s.empty()) doc["scan"] = s;
    auto d = axf::spec::parse_design(doc);
    auto r = axf::scan::F_required(d.benchmarks.at(0), d.scan);
    return json{{"F_req_SI", r.F_SI}, {"F_req_natural_T4_m10", r.F_nat}, {"F_req_eV-2", r.F_nat_eV},
                {"g_min_GeV", r.g_min_GeV}}
        .dump();
  });

  // ------------------------------------------------------------ rendering
  m.def(
      "render_png",
      [](const std::string& scene, const std::string& options) {
        auto png = axf::render::render_png(parse(scene, "scene"), axf::render::options_from_json(parse(options, "options")));
        return nb::bytes(reinterpret_cast<const char*>(png.data()), png.size());
      },
      nb::arg("scene"), nb::arg("options") = "{}");
  m.def("run_3d", [](const std::string& doc, const std::string& workdir, int order, int verbose) {
    axf::SolveOptions o;
    o.workdir = workdir;
    o.order = order;
    o.verbose = verbose;
    axf::log_level() = verbose;
    nb::gil_scoped_release release;
    return axf::run_any(parse(doc, "design"), o).dump();
  }, nb::arg("design"), nb::arg("workdir"), nb::arg("order") = 0, nb::arg("verbose") = 1);

  // ------------------------------------------------------------ signal / scan
  m.def(
      "reach_local",
      [](double psi_SI, double L_H, double Q, double T_K, double eta, double m1, double m2, double live_time_s,
         const std::string& kind, int n, double E_over_N, double C_gamma, double ma_fa, double Q_axion, double Theta,
         double rho) {
        auto det = axf::scan::Detector::from_SI(psi_SI, L_H, Q, T_K, eta);
        axf::scan::ScanParams p;
        p.Q_axion = Q_axion;
        p.Theta = Theta;
        p.rho_GeV_cm3 = rho;
        auto k = kind == "log_uniform" ? axf::scan::Allocation::Kind::LogUniform : axf::scan::Allocation::Kind::Target;
        auto rc = axf::scan::reach_local(det, m1, m2, live_time_s, k, n, axf::scan::qcd_line(E_over_N, C_gamma, ma_fa), p);
        nb::dict d;
        d["m_eV"] = to_numpy(rc.m_eV);
        d["f_Hz"] = to_numpy(rc.f_Hz);
        d["g95_GeV"] = to_numpy(rc.g95_GeV);
        d["tau_s_per_rad_s"] = to_numpy(rc.tau_s_per_rad_s);
        std::vector<double> gq;
        for (double mm : rc.m_eV) gq.push_back(axf::scan::g_qcd_GeV(mm, E_over_N, C_gamma, ma_fa));
        d["g_target_GeV"] = to_numpy(gq);
        nb::dict b = band_dict(rc.band);
        b["model_E_over_N"] = E_over_N;
        b["C_gamma"] = C_gamma;
        b["ma_fa_GeV2"] = ma_fa;
        d["band"] = b;
        return d;
      },
      nb::arg("psi_SI"), nb::arg("L_H"), nb::arg("Q"), nb::arg("T_K"), nb::arg("eta_A"), nb::arg("m1_eV"),
      nb::arg("m2_eV"), nb::arg("live_time_s"), nb::arg("allocation") = "target", nb::arg("n") = 200,
      nb::arg("E_over_N") = 8.0 / 3.0, nb::arg("C_gamma") = 1.77, nb::arg("ma_fa") = 5.70e-3,
      nb::arg("Q_axion") = 6.8e5, nb::arg("Theta") = 2.71, nb::arg("rho") = 0.3);
  m.def(
      "psd_at_tuning",
      [](double psi_SI, double L_H, double Q, double T_K, double eta, double f0, double m_a, double g,
         std::vector<double> f, double rho) {
        auto det = axf::scan::Detector::from_SI(psi_SI, L_H, Q, T_K, eta);
        auto c = axf::scan::psd_at_tuning(det, f0, m_a, g, f, -1.0, rho);
        nb::dict d;
        d["f_Hz"] = to_numpy(c.f_Hz);
        d["S_Va_V2_s"] = to_numpy(c.S_Va_V2_s);
        d["N_V_V2_s"] = to_numpy(c.N_V_V2_s);
        d["lambda_a_A2_s"] = to_numpy(c.lambda_a_A2_s);
        d["lambda_B_A2_s"] = to_numpy(c.lambda_B_A2_s);
        d["SV_th_V2_s"] = to_numpy(c.SV_th_V2_s);
        d["SV_ba_V2_s"] = to_numpy(c.SV_ba_V2_s);
        d["SI_imp_A2_s"] = to_numpy(c.SI_imp_A2_s);
        d["s"] = to_numpy(c.s);
        d["xi"] = c.xi;
        d["Phi_a_Wb"] = c.Phi_a_Wb;
        return d;
      },
      nb::arg("psi_SI"), nb::arg("L_H"), nb::arg("Q"), nb::arg("T_K"), nb::arg("eta_A"), nb::arg("f0_Hz"),
      nb::arg("m_a_eV"), nb::arg("g_GeV"), nb::arg("f_Hz"), nb::arg("rho") = 0.3);
  // ------------------------------------------------------------ Sec. 5-6: kernels, LP, reach
  // Finite-band kernel of the reduced single-pole detector (S6.1-S6.2).
  m.def(
      "kernel_row",
      [](double psi_SI, double L_H, double Q, double T_K, double eta, double m_a, double g,
         std::vector<double> omega0, std::vector<double> xi, const std::string& stat, int n_v, double R_exp,
         double Q_axion, double Theta, double rho, double v0, double vobs) {
        auto det = axf::scan::Detector::from_SI(psi_SI, L_H, Q, T_K, eta);
        det.R_exp = R_exp;
        const auto p = scan_params(Q_axion, Theta, rho, v0, vobs, true);
        return to_numpy(axf::scan::kernel_row(det, m_a, g, omega0, xi, parse_stat(stat), p, n_v));
      },
      nb::arg("psi_SI"), nb::arg("L_H"), nb::arg("Q"), nb::arg("T_K"), nb::arg("eta_A"), nb::arg("m_a_eV"),
      nb::arg("g_GeV"), nb::arg("omega0_rad_s"), nb::arg("xi") = std::vector<double>(),
      nb::arg("stat") = "weak", nb::arg("n_v") = 96, nb::arg("R_exp") = 0.0, nb::arg("Q_axion") = 6.8e5,
      nb::arg("Theta") = 2.71, nb::arg("rho") = 0.3, nb::arg("v0_kms") = 220.0, nb::arg("vobs_kms") = 232.0);
  m.def(
      "kernel_narrow_line",
      [](double psi_SI, double L_H, double Q, double T_K, double eta, double m_a, double g, double omega0,
         double xi, double R_exp, double Q_axion, double Theta, double rho) {
        auto det = axf::scan::Detector::from_SI(psi_SI, L_H, Q, T_K, eta);
        det.R_exp = R_exp;
        const auto p = scan_params(Q_axion, Theta, rho, 220.0, 232.0, true);
        return axf::scan::kernel_narrow_line(det, m_a, g, omega0, xi, p);
      },
      nb::arg("psi_SI"), nb::arg("L_H"), nb::arg("Q"), nb::arg("T_K"), nb::arg("eta_A"), nb::arg("m_a_eV"),
      nb::arg("g_GeV"), nb::arg("omega0_rad_s"), nb::arg("xi"), nb::arg("R_exp") = 0.0,
      nb::arg("Q_axion") = 6.8e5, nb::arg("Theta") = 2.71, nb::arg("rho") = 0.3);
  // Finite-band kernel from the EXACT circuit noise N_V(omega) of each setting (C4.5).
  m.def(
      "kernel_row_noise",
      [](double psi_SI, double m_a, double g, std::vector<std::vector<double>> omega,
         std::vector<std::vector<double>> N_V, const std::string& stat, int n_v, double Q_axion, double Theta,
         double rho, double v0, double vobs) {
        const auto nc = noise_curves(omega, N_V);
        const auto p = scan_params(Q_axion, Theta, rho, v0, vobs, true);
        const auto st = parse_stat(stat);
        std::vector<double> K;
        {
          nb::gil_scoped_release release;
          K = axf::scan::kernel_row_noise(psi_SI, m_a, g, nc, st, p, n_v);
        }
        return to_numpy(std::move(K));
      },
      nb::arg("psi_SI"), nb::arg("m_a_eV"), nb::arg("g_GeV"), nb::arg("omega_rad_s"), nb::arg("N_V_V2_s"),
      nb::arg("stat") = "weak", nb::arg("n_v") = 96, nb::arg("Q_axion") = 6.8e5, nb::arg("Theta") = 2.71,
      nb::arg("rho") = 0.3, nb::arg("v0_kms") = 220.0, nb::arg("vobs_kms") = 232.0);
  m.def(
      "kernel_narrow_line_noise",
      [](std::vector<double> psi_SI, std::vector<double> m_a, std::vector<double> g, std::vector<double> N_V,
         double Q_axion, double rho) {
        const size_t n = m_a.size();
        if (psi_SI.size() != n || g.size() != n || N_V.size() != n)
          throw axf::Error("kernel_narrow_line_noise: psi_SI, m_a_eV, g_GeV and N_V must have the same length");
        const auto p = scan_params(Q_axion, 2.71, rho, 220.0, 232.0, true);
        std::vector<double> K(n);
        for (size_t i = 0; i < n; ++i)
          K[i] = axf::scan::kernel_narrow_line_noise(psi_SI[i], m_a[i], g[i], N_V[i], p);
        return to_numpy(K);
      },
      nb::arg("psi_SI"), nb::arg("m_a_eV"), nb::arg("g_GeV"), nb::arg("N_V_V2_s"), nb::arg("Q_axion") = 6.8e5,
      nb::arg("rho") = 0.3);
  // Minimum-total-time schedule (S6.4).
  m.def(
      "lp_schedule",
      [](std::vector<std::vector<double>> K, double Theta) {
        axf::scan::LPSchedule s;
        {
          nb::gil_scoped_release release;
          s = axf::scan::lp_schedule(K, Theta);
        }
        nb::dict d;
        d["t_s"] = to_numpy(s.t_s);
        d["t_total_s"] = s.t_total_s;
        d["min_coverage"] = s.min_coverage;
        d["pivots"] = s.pivots;
        return d;
      },
      nb::arg("K"), nb::arg("Theta") = 2.71);
  // Reach of an exact-noise finite-band schedule (S6.1/S6.4 at each target).
  m.def(
      "reach_band",
      [](std::vector<double> psi_SI, std::vector<double> m_eV, std::vector<std::vector<double>> omega,
         std::vector<std::vector<double>> N_V, std::vector<double> t_s, const std::string& stat, int n_v,
         double Q_axion, double Theta, double rho, double v0, double vobs) {
        const auto nc = noise_curves(omega, N_V);
        const auto p = scan_params(Q_axion, Theta, rho, v0, vobs, true);
        const auto st = parse_stat(stat);
        axf::scan::ReachCurve rc;
        {
          nb::gil_scoped_release release;
          rc = axf::scan::reach_band(psi_SI, m_eV, nc, t_s, st, p, n_v);
        }
        return reach_dict(rc);
      },
      nb::arg("psi_SI"), nb::arg("m_eV"), nb::arg("omega_rad_s"), nb::arg("N_V_V2_s"), nb::arg("t_s"),
      nb::arg("stat") = "limit", nb::arg("n_v") = 96, nb::arg("Q_axion") = 6.8e5, nb::arg("Theta") = 2.71,
      nb::arg("rho") = 0.3, nb::arg("v0_kms") = 220.0, nb::arg("vobs_kms") = 232.0);
  // Per-mass detector curves: the local scan rate (S5.2) and g95 (S5.8) with Psi, L, Q, T,
  // eta_A all functions of the tuning -- the beyond-MQS / dispersive-readout reach.
  m.def(
      "scan_rate_curve",
      [](std::vector<double> psi_SI, std::vector<double> L_H, std::vector<double> Q, std::vector<double> T_K,
         std::vector<double> eta, std::vector<double> m_eV, std::vector<double> g_GeV, double Q_axion,
         double Theta, double rho, bool exact_Gbar) {
        const size_t n = m_eV.size();
        if (psi_SI.size() != n || L_H.size() != n || Q.size() != n || T_K.size() != n || eta.size() != n ||
            g_GeV.size() != n)
          throw axf::Error("scan_rate_curve: every array must have the same length as m_eV");
        const auto p = scan_params(Q_axion, Theta, rho, 220.0, 232.0, exact_Gbar);
        std::vector<double> r(n);
        for (size_t i = 0; i < n; ++i) {
          auto det = axf::scan::Detector::from_SI(psi_SI[i], L_H[i], Q[i], T_K[i], eta[i]);
          r[i] = axf::scan::scan_rate(det, m_eV[i], g_GeV[i], p);
        }
        return to_numpy(r);
      },
      nb::arg("psi_SI"), nb::arg("L_H"), nb::arg("Q"), nb::arg("T_K"), nb::arg("eta_A"), nb::arg("m_eV"),
      nb::arg("g_GeV"), nb::arg("Q_axion") = 6.8e5, nb::arg("Theta") = 2.71, nb::arg("rho") = 0.3,
      nb::arg("exact_Gbar") = true);
  m.def(
      "g95_curve",
      [](std::vector<double> psi_SI, std::vector<double> L_H, std::vector<double> Q, std::vector<double> T_K,
         std::vector<double> eta, std::vector<double> m_eV, std::vector<double> tau, double Q_axion,
         double Theta, double rho, bool exact_Gbar) {
        const size_t n = m_eV.size();
        if (psi_SI.size() != n || L_H.size() != n || Q.size() != n || T_K.size() != n || eta.size() != n ||
            tau.size() != n)
          throw axf::Error("g95_curve: every array must have the same length as m_eV");
        const auto p = scan_params(Q_axion, Theta, rho, 220.0, 232.0, exact_Gbar);
        std::vector<double> g(n);
        for (size_t i = 0; i < n; ++i) {
          auto det = axf::scan::Detector::from_SI(psi_SI[i], L_H[i], Q[i], T_K[i], eta[i]);
          g[i] = axf::scan::g95_local(det, m_eV[i], tau[i], p);
        }
        return to_numpy(g);
      },
      nb::arg("psi_SI"), nb::arg("L_H"), nb::arg("Q"), nb::arg("T_K"), nb::arg("eta_A"), nb::arg("m_eV"),
      nb::arg("tau_s_per_rad_s"), nb::arg("Q_axion") = 6.8e5, nb::arg("Theta") = 2.71, nb::arg("rho") = 0.3,
      nb::arg("exact_Gbar") = true);
  m.def(
      "bandwidths",
      [](double omega0, double Q, double T_K, double eta_A, double xi) {
        const double n = axf::scan::n_bose(omega0, T_K);
        const auto b = axf::scan::sensitivity_bandwidths(omega0, Q, n, eta_A, xi);
        nb::dict d;
        d["linewidth"] = b.linewidth;
        d["noise_doubling"] = b.noise_doubling;
        d["thermal_gt_squid"] = b.thermal_gt_squid;
        d["equivalent"] = b.equivalent;
        d["xi"] = b.xi;
        d["n"] = n;
        return d;
      },
      nb::arg("omega0_rad_s"), nb::arg("Q"), nb::arg("T_K"), nb::arg("eta_A"), nb::arg("xi") = -1.0);
  m.def("xi_opt", [](double n, double eta_A) { return axf::scan::xi_opt(n, eta_A); }, nb::arg("n"),
        nb::arg("eta_A"));
  m.def("G_bar_opt", [](double n, double eta_A) { return axf::scan::G_bar(n, eta_A); }, nb::arg("n"),
        nb::arg("eta_A"));
  m.def("g_qcd_GeV", [](std::vector<double> m_eV, double E_over_N, double C_gamma, double ma_fa) {
    std::vector<double> g;
    for (double x : m_eV) g.push_back(axf::scan::g_qcd_GeV(x, E_over_N, C_gamma, ma_fa));
    return to_numpy(g);
  }, nb::arg("m_eV"), nb::arg("E_over_N") = 8.0 / 3.0, nb::arg("C_gamma") = 1.77, nb::arg("ma_fa") = 5.70e-3);
  m.def("Q_axion_shm", [](double v0, double vobs) { return axf::scan::Q_axion_shm(v0, vobs); },
        nb::arg("v0_kms") = 220.0, nb::arg("vobs_kms") = 232.0);
  m.def("mass_to_freq", &axf::scan::mass_to_freq);
  m.def("freq_to_mass", &axf::scan::freq_to_mass);

  nb::class_<axf::Model3D>(m, "Model3D")
      .def("__init__",
           [](axf::Model3D* self, const std::string& doc) {
             new (self) axf::Model3D(axf::spec::parse_design(parse(doc, "design")));
           })
      .def(
          "run",
          [](axf::Model3D& mdl, const std::string& workdir, int order, int verbose, const std::string& mesher) {
            axf::SolveOptions o;
            o.workdir = workdir;
            o.order = order;
            o.verbose = verbose;
            o.mesher = mesher;
            axf::log_level() = verbose;
            nb::gil_scoped_release release;
            mdl.run(o);
          },
          nb::arg("workdir"), nb::arg("order") = 0, nb::arg("verbose") = 1, nb::arg("mesher") = "")
      .def("results", [](const axf::Model3D& mdl) { return mdl.results().dump(); })
      .def(
          "scene",
          [](const axf::Model3D& mdl, bool fields) {
            axf::SceneOptions o;
            o.fields = fields;
            nb::gil_scoped_release release;
            return mdl.scene(o).dump();
          },
          nb::arg("fields") = true)
      .def("write_paraview", [](const axf::Model3D& mdl, const std::string& dir) { mdl.write_paraview(dir); })
      .def("save", [](const axf::Model3D& mdl, const std::string& dir) { mdl.save(dir); })
      .def_static("load", [](const std::string& dir) { return axf::Model3D::load(dir); })
      .def("sample", [](const axf::Model3D& mdl, std::vector<double> x, std::vector<double> y, std::vector<double> z) {
        auto s = mdl.sample(x, y, z);
        std::vector<double> B[3], pB[3], reg;
        for (auto& v : s) {
          for (int k = 0; k < 3; ++k) {
            B[k].push_back(v.B[k]);
            pB[k].push_back(v.pB[k]);
          }
          reg.push_back(v.region);
        }
        nb::dict d;
        d["Bx"] = B[0];
        d["By"] = B[1];
        d["Bz"] = B[2];
        d["pickup_Bx"] = pB[0];
        d["pickup_By"] = pB[1];
        d["pickup_Bz"] = pB[2];
        d["region_attr"] = reg;
        return d;
      });

  // cross-section cavity eigensolver ("symmetry": "xsec"; beta)
  nb::class_<axf::XsecModel>(m, "XsecModel")
      .def("__init__",
           [](axf::XsecModel* self, const std::string& doc) {
             new (self) axf::XsecModel(axf::spec::parse_design(parse(doc, "design")));
           })
      .def(
          "run",
          [](axf::XsecModel& mdl, const std::string& workdir, int order, int verbose, const std::string& mesher) {
            axf::SolveOptions o;
            o.workdir = workdir;
            o.order = order;
            o.verbose = verbose;
            o.mesher = mesher;
            axf::log_level() = verbose;
            nb::gil_scoped_release release;
            mdl.run(o);
          },
          nb::arg("workdir"), nb::arg("order") = 0, nb::arg("verbose") = 1, nb::arg("mesher") = "")
      .def("results", [](const axf::XsecModel& mdl) { return mdl.results().dump(); })
      .def("solved", &axf::XsecModel::solved)
      .def("mode_count", &axf::XsecModel::mode_count)
      .def(
          "mode_mesh",
          [](const axf::XsecModel& mdl, const std::string& family, int index, int subdiv) {
            return xsec_field_dict(mdl.mode_mesh(family, index, subdiv));
          },
          nb::arg("family"), nb::arg("index"), nb::arg("subdiv") = 2)
      .def("mode_sample",
           [](const axf::XsecModel& mdl, const std::string& family, int index, std::vector<double> x,
              std::vector<double> y) { return xsec_field_dict(mdl.mode_sample(family, index, x, y)); })
      .def("geometry", [](const axf::XsecModel& mdl, int subdiv) { return mdl.geometry(subdiv).dump(); },
           nb::arg("subdiv") = 2);

  // 2D full wave on an imported mesh (planar / axisymmetric; COMSOL 2D import, beta)
  nb::class_<axf::Fw2DModel>(m, "Fw2DModel")
      .def("__init__",
           [](axf::Fw2DModel* self, const std::string& doc) {
             new (self) axf::Fw2DModel(axf::spec::parse_design(parse(doc, "design")));
           })
      .def(
          "run",
          [](axf::Fw2DModel& mdl, const std::string& workdir, int order, int verbose) {
            axf::SolveOptions o;
            o.workdir = workdir;
            o.order = order;
            o.verbose = verbose;
            axf::log_level() = verbose;
            nb::gil_scoped_release release;
            mdl.run(o);
          },
          nb::arg("workdir"), nb::arg("order") = 0, nb::arg("verbose") = 1)
      .def("results", [](const axf::Fw2DModel& mdl) { return mdl.results().dump(); })
      .def("solved", &axf::Fw2DModel::solved)
      .def("sample",
           [](const axf::Fw2DModel& mdl, const std::string& what, const std::string& family, int index,
              std::vector<double> x1, std::vector<double> x2) { return mdl.sample(what, family, index, x1, x2).dump(); },
           nb::arg("what"), nb::arg("family"), nb::arg("index"), nb::arg("x1"), nb::arg("x2"));

  nb::class_<axf::AxiModel>(m, "Model")
      .def("__init__",
           [](axf::AxiModel* self, const std::string& doc) {
             new (self) axf::AxiModel(axf::spec::parse_design(parse(doc, "design")));
           })
      .def(
          "run",
          [](axf::AxiModel& mdl, const std::string& workdir, int order, int verbose, const std::string& mesher,
             bool self_force, int force_samples) {
            axf::SolveOptions o;
            o.workdir = workdir;
            o.order = order;
            o.verbose = verbose;
            o.mesher = mesher;
            o.winding_self_force = self_force;
            o.force_samples = force_samples;
            axf::log_level() = verbose;
            nb::gil_scoped_release release;
            mdl.run(o);
          },
          nb::arg("workdir"), nb::arg("order") = 0, nb::arg("verbose") = 1, nb::arg("mesher") = "",
          nb::arg("self_force") = true, nb::arg("force_samples") = 200)
      .def("results", [](const axf::AxiModel& mdl) { return mdl.results().dump(); })
      .def("solved", &axf::AxiModel::solved)
      .def(
          "sensitivities",
          [](const axf::AxiModel& mdl, std::vector<std::string> params, std::vector<std::string> objectives,
             const std::string& method, double step, const std::string& steps, const std::string& benchmark,
             const std::string& workdir, int order) {
            axf::sens::Options so;
            so.params = std::move(params);
            so.objectives = std::move(objectives);
            so.method = method;
            so.step = step;
            so.benchmark = benchmark;
            so.solve.workdir = workdir;
            so.solve.order = order;
            json st = parse(steps, "steps");
            if (st.is_object())
              for (auto it = st.begin(); it != st.end(); ++it) so.steps[it.key()] = it.value().get<double>();
            nb::gil_scoped_release release;
            return axf::sens::sensitivities(mdl, so).dump();
          },
          nb::arg("params") = std::vector<std::string>(), nb::arg("objectives") = std::vector<std::string>(),
          nb::arg("method") = "adjoint", nb::arg("step") = 1e-4, nb::arg("steps") = "{}", nb::arg("benchmark") = "",
          nb::arg("workdir") = "axf_run", nb::arg("order") = 0)
      .def(
          "scene",
          [](const axf::AxiModel& mdl, int subdiv, int contours, double view_scale, bool fields) {
            axf::SceneOptions o;
            o.subdiv = subdiv;
            o.contours = contours;
            o.view_scale = view_scale;
            o.fields = fields;
            return mdl.scene(o).dump();
          },
          nb::arg("subdiv") = 1, nb::arg("contours") = 28, nb::arg("view_scale") = 1.3, nb::arg("fields") = true)
      .def(
          "write_vtu",
          [](const axf::AxiModel& mdl, const std::string& path, int subdiv) {
            axf::SceneOptions o;
            o.subdiv = subdiv;
            mdl.write_vtu(path, o);
          },
          nb::arg("path"), nb::arg("subdiv") = 2)
      .def("cavity_mode_count", &axf::AxiModel::cavity_mode_count)
      .def(
          "cavity_mode_mesh",
          [](const axf::AxiModel& mdl, const std::string& family, int index, int subdiv) {
            return mode_field_dict(mdl.cavity_mode_mesh(family, index, subdiv));
          },
          nb::arg("family"), nb::arg("index"), nb::arg("subdiv") = 2)
      .def("cavity_mode_sample",
           [](const axf::AxiModel& mdl, const std::string& family, int index, std::vector<double> r,
              std::vector<double> z) { return mode_field_dict(mdl.cavity_mode_sample(family, index, r, z)); })
      .def("has_hts", &axf::AxiModel::has_hts)
      .def("hts_sample", [](const axf::AxiModel& mdl, std::vector<double> r, std::vector<double> z) {
        auto s = mdl.hts_sample(r, z);
        std::vector<double> J, Jc, E, Br, Bz, A, hts, reg;
        for (auto& x : s) {
          J.push_back(x.J);
          Jc.push_back(x.Jc);
          E.push_back(x.E);
          Br.push_back(x.Br);
          Bz.push_back(x.Bz);
          A.push_back(x.A);
          hts.push_back(x.hts ? 1.0 : 0.0);
          reg.push_back(x.region);
        }
        nb::dict d;
        d["J"] = to_numpy(J);
        d["Jc"] = to_numpy(Jc);
        d["E"] = to_numpy(E);
        d["Br"] = to_numpy(Br);
        d["Bz"] = to_numpy(Bz);
        d["A"] = to_numpy(A);
        d["in_hts"] = to_numpy(hts);
        d["region_attr"] = to_numpy(reg);
        return d;
      })
      // ---- torsional dynamics (vibrations) ----
      .def("torsion", [](const axf::AxiModel& mdl, const std::string& request) {
        return mdl.torsion(axf::json::parse(request)).dump();
      }, nb::arg("request"),
        "Axisymmetric m = 0 torsional modal / harmonic analysis; JSON request in, JSON out.")
      // ---- mechanics/thermal ----
      .def("sample_mech", [](const axf::AxiModel& mdl, std::vector<double> r, std::vector<double> z) {
        auto s = mdl.sample_mech(r, z);
        std::vector<double> ur, uz, srr, szz, stt, srz, vm, T, solid, therm, reg;
        for (auto& x : s) {
          ur.push_back(x.ur);
          uz.push_back(x.uz);
          srr.push_back(x.s_rr);
          szz.push_back(x.s_zz);
          stt.push_back(x.s_tt);
          srz.push_back(x.s_rz);
          vm.push_back(x.von_mises);
          T.push_back(x.T_K);
          solid.push_back(x.solid ? 1.0 : 0.0);
          therm.push_back(x.thermal ? 1.0 : 0.0);
          reg.push_back(x.region);
        }
        nb::dict d;
        d["u_r"] = to_numpy(ur);
        d["u_z"] = to_numpy(uz);
        d["sigma_rr"] = to_numpy(srr);
        d["sigma_zz"] = to_numpy(szz);
        d["sigma_hoop"] = to_numpy(stt);
        d["sigma_rz"] = to_numpy(srz);
        d["von_mises"] = to_numpy(vm);
        d["T_K"] = to_numpy(T);
        d["in_structure"] = to_numpy(solid);
        d["in_thermal_domain"] = to_numpy(therm);
        d["region_attr"] = to_numpy(reg);
        return d;
      })
      // ---- end mechanics/thermal ----
      .def("sample", [](const axf::AxiModel& mdl, std::vector<double> r, std::vector<double> z) {
        auto s = mdl.sample(r, z);
        std::vector<double> Br, Bz, Bp, psi, pBr, pBz, pBp, ppsi, reg;
        for (auto& x : s) {
          Br.push_back(x.Br);
          Bz.push_back(x.Bz);
          Bp.push_back(x.Bphi);
          psi.push_back(x.psi);
          pBr.push_back(x.pBr);
          pBz.push_back(x.pBz);
          pBp.push_back(x.pBphi);
          ppsi.push_back(x.ppsi);
          reg.push_back(x.region);
        }
        nb::dict d;
        d["Br"] = to_numpy(Br);
        d["Bz"] = to_numpy(Bz);
        d["Bphi"] = to_numpy(Bp);
        d["psi"] = to_numpy(psi);
        d["pickup_Br"] = to_numpy(pBr);
        d["pickup_Bz"] = to_numpy(pBz);
        d["pickup_Bphi"] = to_numpy(pBp);
        d["pickup_psi"] = to_numpy(ppsi);
        d["region_attr"] = to_numpy(reg);
        return d;
      });

  // chemistry track: submodule _core.chem
  axf_bind_chem(m);
}
