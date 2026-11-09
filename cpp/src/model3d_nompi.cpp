// Stand-in for cpp/src/model3d.cpp in builds without MPI.
//
// model3d.cpp (with model3d_fullwave.cpp, model3d_electro.cpp, model3d_mech.cpp and model3d_impl.hpp) is the only code in
// axionFEM that uses MPI or hypre: the 3D
// H(curl) engine is built on mfem::ParMesh, mfem::ParFiniteElementSpace and
// hypre's AMS/BoomerAMG preconditioners.  None of those types exist in an MFEM
// configured with MFEM_USE_MPI=NO, which is what the sequential build (and the planned wheels) use, because
// a wheel on PyPI cannot require a system MPI (scripts/build_deps_wheel.sh).
//
// CMake compiles this file instead when AXF_MPI=OFF, so that the axisymmetric
// engine, the full-wave solver, the scan/cost layer, the viewer and the whole
// Python API are unaffected and only the 3D entry points fail -- with an error
// that says what is missing and how to get a build that has it.  The exported
// surface is exactly the one declared in axionfem/model3d.hpp.
//
// See docs/install.md ("build matrix").
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "axionfem/core.hpp"
#include "axionfem/model3d.hpp"
#include "axionfem/fw2d.hpp"

namespace axf {

namespace {

[[noreturn]] void no_3d_engine(const char* what) {
  throw Error(
      std::string("the 3D engine is not available in this build of axionFEM (needed for ") + what + ")",
      "/symmetry",
      "This build has no MPI: it was configured with AXF_MPI=OFF against an MFEM built with "
      "MFEM_USE_MPI=NO (the sequential configuration, also that of the planned wheels, which cannot "
      "require a system MPI). The axisymmetric and full-wave engines are unaffected. For 3D "
      "(\"symmetry\": \"3d\", axionfem.convert.to_3d), use the MPI source build: see "
      "docs/guides/install.md, section \"MPI source build\".");
}

}  // namespace

// The real engine keeps the mesh, spaces and solutions here; this build has no
// state, but the type must be complete for unique_ptr<Impl> in the header.
struct Model3D::Impl {};

Model3D::Model3D(spec::Design d) : d_(std::move(d)) {
  no_3d_engine("Model3D");
}

Model3D::~Model3D() = default;

void Model3D::run(const SolveOptions&) { no_3d_engine("Model3D::run"); }

std::vector<Model3D::Sample> Model3D::sample(const std::vector<double>&, const std::vector<double>&,
                                             const std::vector<double>&) const {
  no_3d_engine("Model3D::sample");
}

json Model3D::scene(const SceneOptions&) const { no_3d_engine("Model3D::scene"); }

void Model3D::write_paraview(const std::string&) const { no_3d_engine("Model3D::write_paraview"); }

void Model3D::save(const std::string&) const { no_3d_engine("Model3D::save"); }

std::unique_ptr<Model3D> Model3D::load(const std::string&) { no_3d_engine("Model3D::load"); }

void Model3D::fullwave_analysis(const SolveOptions&) { no_3d_engine("Model3D::fullwave_analysis"); }

void Model3D::electrostatics_analysis(const SolveOptions&) { no_3d_engine("Model3D::electrostatics_analysis"); }

void Model3D::mechanics_thermal_analysis(const SolveOptions&) {
  no_3d_engine("Model3D::mechanics_thermal_analysis");
}

void Model3D::run_once(const SolveOptions&) { no_3d_engine("Model3D::run_once"); }

std::vector<double> Model3D::error_indicators() const { no_3d_engine("Model3D::error_indicators"); }

int mpi_rank() { return 0; }

// Same dispatch as the real file, except that "3d" designs are rejected before
// any meshing happens, so the error names the design, not a missing symbol.
json mesh_any(const json& doc, const SolveOptions& o) {
  auto d = spec::parse_design(doc);
  if (d.symmetry != "3d" && d.symmetry != "xsec" && !d.mesh.file.empty())  // 2D imported mesh: nothing to mesh
    return {{"ok", true}, {"symmetry", d.symmetry}, {"mesh", d.mesh.file}, {"mesher_info", d.mesh.info_file},
            {"note", "imported mesh (mesh.file): no meshing"}};
  if (d.symmetry == "3d") no_3d_engine("meshing a design with \"symmetry\": \"3d\"");
  AxiModel m(std::move(d));
  m.mesh(o);
  return {{"ok", true}, {"symmetry", "axisymmetric"}, {"mesh", o.workdir + "/mesh.msh"},
          {"mesher_input", o.workdir + "/mesh_in.json"}, {"mesher_info", o.workdir + "/mesh_info.json"}};
}

json run_any(const json& doc, const SolveOptions& o) {
  auto d = spec::parse_design(doc);
  if (d.symmetry != "3d" && d.symmetry != "xsec" && !d.mesh.file.empty()) {  // 2D full wave, imported mesh
    Fw2DModel m(std::move(d));
    m.run(o);
    return m.results();
  }
  if (d.symmetry == "3d") no_3d_engine("a design with \"symmetry\": \"3d\"");
  AxiModel m(std::move(d));
  m.run(o);
  return m.results();
}

}  // namespace axf
