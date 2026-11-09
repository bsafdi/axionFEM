// axf: command-line interface to axionFEM.
//
//   axf solve design.json [-o DIR] [--order P] [-v N]   mesh + solve, write DIR/results.json
//                                                       (3D designs: srun/mpirun -n N axf solve ... runs on N MPI ranks)
//     --no-self-force: skip the per-winding vacuum solve that removes the self-force floor
//     --status-file F: write the summary to F instead of DIR/axf_status.json
//   axf sensitivities design.json [--param P]... [--objective Q]...   adjoint or finite-difference shape sensitivities
//   axf validate design.json                            check a design, print errors as JSON
//   axf mesh design.json [-o DIR] [--mesher M] [-v N]   generate the mesh only (DIR/mesh.msh or mesh3d.msh)
//   axf resolve design.json                             print the geometry with every expression evaluated
//   axf schema                                          print the design JSON Schema
//   axf requirement [--benchmark NAME]                  print F_req for the built-in benchmarks
//   axf render scene.json -o out.png [--view section|3d]  render a field image from a scene file
//   axf templates                                       list parametric templates
//   axf template NAME [--set key=value ...]             print a design from a template
//
// Errors are reported on stdout as {"ok": false, "error": ..., "where": ..., "hint": ...}
// so that agents can act on them. `axf solve` also writes that summary (or the success summary) to
// DIR/axf_status.json, or to the file named by --status-file, from rank 0: MPI launchers may split or interleave long stdout lines (srun forwards output in
// 1 KiB messages), and Design.solve(mpi_ranks=N) reads the file first.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "axionfem/adjoint.hpp"
#include "axionfem/core.hpp"
#include "axionfem/device.hpp"
#include "axionfem/model.hpp"
#include "axionfem/model3d.hpp"
#include "axionfem/xsec.hpp"
#include "axionfem/fw2d.hpp"
#include "axionfem/render.hpp"
#include "axionfem/scan.hpp"
#include "axionfem/spec.hpp"
#include "axionfem/templates.hpp"

using json = nlohmann::json;

namespace {
using axf::mpi_rank;
// Rank and size of this process as set by the launcher (srun / mpirun), before or without MPI initialization.
int launcher_rank(int* size) {
  auto env_int = [](std::initializer_list<const char*> names, int dflt) {
    for (const char* n : names)
      if (const char* v = std::getenv(n)) return std::atoi(v);
    return dflt;
  };
  if (size) *size = env_int({"OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "SLURM_STEP_NUM_TASKS"}, 1);
  return env_int({"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID"}, 0);
}
// DIR/axf_status.json of the running `axf solve` (empty before its options are read)
std::string& status_file() {
  static std::string path;
  return path;
}

// Print the summary on stdout (rank 0; others on stderr) and write it to the status file (rank 0, atomically).
void emit(const json& j, bool root) {
  (root ? std::cout : std::cerr) << j.dump(2) << "\n";
  if (!root || launcher_rank(nullptr) != 0 || status_file().empty()) return;
  try {
    const std::filesystem::path path(status_file());
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    const std::string tmp = status_file() + ".tmp" + std::to_string(::getpid());
    std::ofstream f(tmp);
    f << j.dump(2) << "\n";
    f.close();  // a write error at close (a full quota) shows here, before the rename
    std::error_code ec;
    if (!f) {
      std::filesystem::remove(tmp, ec);
      return;
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) std::filesystem::remove(tmp, ec);
  } catch (const std::exception&) {  // the stdout copy remains
  }
}

json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw axf::Error("cannot open '" + path + "'");
  try {
    return json::parse(f);
  } catch (json::parse_error& e) {
    throw axf::Error(std::string("invalid JSON in '") + path + "': " + e.what(), "",
                     "Check for trailing commas and unquoted names.");
  }
}

int usage() {
  std::cerr << "usage:\n"
               "  axf solve design.json [-o DIR] [--order P] [-v N] [--scene] [--vtu] [--save] [--no-self-force]\n"
               "                        [--force-samples N] [--status-file F]\n"
               "  axf sensitivities design.json [--param P]... [--objective Q]... [--method adjoint|fd_morph|fd_remesh]\n"
               "                                [--step S] [--benchmark NAME] [-o DIR] [--order P]\n"
               "  axf validate design.json\n"
               "  axf mesh design.json [-o DIR] [--mesher M] [-v N]\n"
               "  axf resolve design.json\n"
               "  axf schema\n"
               "  axf requirement [--benchmark NAME]\n"
               "  axf render scene.json -o out.png [--view section|3d] [--quantity Q] [--log] [--theme dark]\n"
               "  axf templates\n"
               "  axf template NAME [--set key=value ...]\n";
  return 2;
}
}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> a(argv + 1, argv + argc);
  if (a.empty()) return usage();
  try {
    axf::configure_device();  // $AXF_DEVICE: hypre on a GPU in a -DAXF_GPU build; MFEM always on the CPU
    const std::string cmd = a[0];
    if (cmd == "schema") {
      std::cout << axf::spec::design_schema().dump(2) << "\n";
      return 0;
    }
    if (cmd == "requirement") {
      std::vector<std::string> names = {"optimistic", "conservative", "conservative_solenoid"};
      for (size_t i = 1; i + 1 < a.size(); ++i) if (a[i] == "--benchmark") names = {a[i + 1]};
      json out = json::array();
      axf::spec::ScanSpec s;
      for (auto& n : names) {
        auto b = axf::spec::builtin_benchmark(n);
        auto r = axf::scan::F_required(b, s);
        out.push_back({{"benchmark", n}, {"F_req_SI_T4_H2_m8", r.F_SI}, {"F_req_natural_T4_m10", r.F_nat},
                       {"F_req_eV-2", r.F_nat_eV}, {"g_min_GeV-1", r.g_min_GeV}});
      }
      std::cout << out.dump(2) << "\n";
      return 0;
    }
    if (cmd == "render") {
      // axf render scene.json -o out.png [--view section|3d] [--quantity Q] [--log] [--theme dark]
      if (a.size() < 2) return usage();
      json scene = read_json(a[1]);
      json opt = json::object();
      std::string out = "render.png";
      for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "-o" && i + 1 < a.size()) out = a[++i];
        else if (a[i] == "--view" && i + 1 < a.size()) opt["view"] = a[++i];
        else if (a[i] == "--quantity" && i + 1 < a.size()) opt["quantity"] = a[++i];
        else if (a[i] == "--theme" && i + 1 < a.size()) opt["theme"] = a[++i];
        else if (a[i] == "--log") opt["log_scale"] = true;
        else return usage();
      }
      axf::render::write_png(out, scene, axf::render::options_from_json(opt));
      std::cout << json{{"ok", true}, {"png", out}}.dump() << "\n";
      return 0;
    }
    if (cmd == "templates") {
      std::cout << axf::templates::list().dump(2) << "\n";
      return 0;
    }
    if (cmd == "template") {
      if (a.size() < 2) return usage();
      json ov = json::object();
      for (size_t i = 2; i + 1 < a.size(); ++i) {
        if (a[i] != "--set") return usage();
        std::string kv = a[++i];
        auto eq = kv.find('=');
        if (eq == std::string::npos) return usage();
        std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        try {
          size_t pos;
          double x = std::stod(v, &pos);
          if (pos == v.size()) ov[k] = x;
          else ov[k] = v;
        } catch (...) {
          ov[k] = v;
        }
      }
      std::cout << axf::templates::make(a[1], ov).dump(2) << "\n";
      return 0;
    }
    if (a.size() < 2) return usage();
    json doc = read_json(a[1]);
    if (cmd == "validate") {
      auto d = axf::spec::parse_design(doc);
      std::cout << json{{"ok", true}, {"name", d.name}, {"regions", d.regions.size()}, {"curves", d.curves.size()},
                        {"windings", d.windings.size()}}
                       .dump(2)
                << "\n";
      return 0;
    }
    if (cmd == "resolve") {
      std::cout << axf::spec::resolved_geometry(axf::spec::parse_design(doc)).dump(2) << "\n";
      return 0;
    }
    if (cmd == "mesh") {
      axf::SolveOptions o;
      o.workdir = "axf_mesh";
      o.verbose = 0;
      for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "-o" && i + 1 < a.size()) o.workdir = a[++i];
        else if (a[i] == "--mesher" && i + 1 < a.size()) o.mesher = a[++i];
        else if (a[i] == "-v" && i + 1 < a.size()) o.verbose = std::stoi(a[++i]);
        else return usage();
      }
      axf::log_level() = o.verbose;
      json out = axf::mesh_any(doc, o);
      if (mpi_rank() == 0) std::cout << out.dump(2) << "\n";
      return 0;
    }
    if (cmd == "solve") {
      axf::SolveOptions o;
      o.workdir = "axf_run";
      bool want_scene = false, want_vtu = false, want_save = false;
      for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "-o" && i + 1 < a.size()) o.workdir = a[++i];
        else if (a[i] == "--order" && i + 1 < a.size()) o.order = std::stoi(a[++i]);
        else if (a[i] == "-v" && i + 1 < a.size()) o.verbose = std::stoi(a[++i]);
        else if (a[i] == "--mesher" && i + 1 < a.size()) o.mesher = a[++i];
        else if (a[i] == "--scene") want_scene = true;
        else if (a[i] == "--vtu") want_vtu = true;
        else if (a[i] == "--save") want_save = true;
        else if (a[i] == "--no-self-force") o.winding_self_force = false;
        else if (a[i] == "--force-samples" && i + 1 < a.size()) o.force_samples = std::stoi(a[++i]);
        else if (a[i] == "--status-file" && i + 1 < a.size()) status_file() = a[++i];
        else return usage();
      }
      axf::log_level() = o.verbose;
      if (status_file().empty()) status_file() = o.workdir + "/axf_status.json";
      if (doc.value("symmetry", std::string("axisymmetric")) != "3d" && doc.value("symmetry", std::string()) != "xsec" &&
          doc.contains("mesh") && doc["mesh"].contains("file")) {  // 2D full wave on an imported mesh
        if (launcher_rank(nullptr) > 0) return 0;
        axf::Fw2DModel model(axf::spec::parse_design(doc));
        model.run(o);
        const json& res = model.results();
        emit(json{{"ok", true}, {"results", o.workdir + "/results.json"}, {"warnings", res["warnings"]}}, true);
        return 0;
      }
      if (doc.value("symmetry", std::string("axisymmetric")) == "xsec") {  // z-uniform cavity cross-section
        if (launcher_rank(nullptr) > 0) return 0;
        axf::XsecModel model(axf::spec::parse_design(doc));
        model.run(o);
        const json& res = model.results();  // also written to workdir/results.json
        emit(json{{"ok", true}, {"results", o.workdir + "/results.json"}, {"warnings", res["warnings"]}}, true);
        return 0;
      }
      if (doc.value("symmetry", std::string("axisymmetric")) == "3d") {
        // MPI: every rank solves its part (srun/mpirun -n N axf solve ...); rank 0 writes the outputs
        axf::Model3D model(axf::spec::parse_design(doc));
        model.run(o);
        const json& res = model.results();
        const bool root = mpi_rank() == 0;
        if (root) std::ofstream(o.workdir + "/results.json") << res.dump(2);
        if (want_scene) {
          json sc = model.scene();  // gathered fields: identical on every rank
          if (root) std::ofstream(o.workdir + "/scene.json") << sc.dump();
        }
        if (want_vtu) model.write_paraview(o.workdir + "/paraview");  // collective, one piece per rank
        if (want_save) model.save(o.workdir + "/fields");            // rank 0 writes the gathered fields
        if (root)
          emit(json{{"ok", true}, {"results", o.workdir + "/results.json"}, {"coupling", res["coupling"]},
                    {"benchmarks", res["benchmarks"]}, {"warnings", res["warnings"]}}, true);
        return 0;
      }
      // axisymmetric designs run on one process: under a multi-rank launch only rank 0 works (the others would
      // write the same files)
      int nlaunch = 1;
      if (const int r = launcher_rank(&nlaunch); r > 0) {
        std::cerr << json{{"ok", true}, {"idle_rank", r},
                          {"note", "axisymmetric designs run on rank 0 only; this rank did nothing"}}.dump() << "\n";
        return 0;
      }
      axf::AxiModel model(axf::spec::parse_design(doc));
      model.run(o);
      json res = model.results();
      if (nlaunch > 1)
        res["warnings"].push_back("axisymmetric designs run on a single process: " + std::to_string(nlaunch - 1) +
                                  " of the launched ranks stayed idle");
      std::ofstream(o.workdir + "/results.json") << res.dump(2);
      if (want_scene) std::ofstream(o.workdir + "/scene.json") << model.scene().dump();
      if (want_vtu) model.write_vtu(o.workdir + "/fields.vtu");
      json summary = {{"ok", true}, {"results", o.workdir + "/results.json"}, {"coupling", res["coupling"]},
                      {"benchmarks", res["benchmarks"]}, {"warnings", res["warnings"]}};
      emit(summary, true);
      return 0;
    }
    if (cmd == "sensitivities") {
      axf::sens::Options so;
      so.solve.workdir = "axf_run";
      for (size_t i = 2; i < a.size(); ++i) {
        if (a[i] == "-o" && i + 1 < a.size()) so.solve.workdir = a[++i];
        else if (a[i] == "--order" && i + 1 < a.size()) so.solve.order = std::stoi(a[++i]);
        else if (a[i] == "--param" && i + 1 < a.size()) so.params.push_back(a[++i]);
        else if (a[i] == "--objective" && i + 1 < a.size()) so.objectives.push_back(a[++i]);
        else if (a[i] == "--method" && i + 1 < a.size()) so.method = a[++i];
        else if (a[i] == "--step" && i + 1 < a.size()) so.step = std::stod(a[++i]);
        else if (a[i] == "--benchmark" && i + 1 < a.size()) so.benchmark = a[++i];
        else if (a[i] == "--mesher" && i + 1 < a.size()) so.solve.mesher = a[++i];
        else if (a[i] == "-v" && i + 1 < a.size()) so.solve.verbose = std::stoi(a[++i]);
        else return usage();
      }
      axf::log_level() = so.solve.verbose;
      AXF_REQUIRE(doc.value("symmetry", std::string("axisymmetric")) != "3d",
                  "sensitivities are implemented for axisymmetric designs", "/symmetry",
                  "Differentiate the axisymmetric model and cross-check the optimum in 3D.");
      axf::AxiModel model(axf::spec::parse_design(doc));
      axf::SolveOptions o = so.solve;
      model.run(o);
      json out = axf::sens::sensitivities(model, so);
      std::ofstream(o.workdir + "/sensitivities.json") << out.dump(2);
      out["file"] = o.workdir + "/sensitivities.json";
      out["diagnostics"].erase("morph");
      std::cout << out.dump(2) << "\n";
      return 0;
    }
    return usage();
  } catch (axf::Error& e) {
    // MPI: the JSON error goes to stdout from rank 0 only (other ranks report on stderr)
    emit(json{{"ok", false}, {"error", e.message()}, {"where", e.where()}, {"hint", e.hint()}}, mpi_rank() == 0);
    return 1;
  } catch (std::exception& e) {
    emit(json{{"ok", false}, {"error", e.what()}}, mpi_rank() == 0);
    return 1;
  }
}
