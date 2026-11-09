// The mesh check appended to solver-failure messages (axionfem/mesh_quality.hpp): on deliberately bad
// meshes it must find the bad element, name its region and centroid, measure it, and give the right remedy.
//   - a straight sliver tetrahedron among good ones (flat: aspect > 1e3);
//   - a curved (order-2) tetrahedron with a mid-edge node pulled through the element (inverted);
//   - a curved triangle folded the same way (2D remedies, (r, z) coordinates);
//   - a clean mesh (no remedy beyond "not the mesh");
//   - pack/unpack and merge, which the 3D engine uses to reduce the worst element over MPI ranks;
//   - a P2 triangle folded only near a vertex (found at the vertices), the remedy for an imported
//     mesh, the pointer for a thrown error, and every spec key a remedy names accepted by spec::parse_design.
#include <cmath>
#include <cstdio>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "axionfem/core.hpp"
#include "axionfem/mesh_quality.hpp"
#include "axionfem/spec.hpp"

namespace mq = axf::mesh_quality;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAILED: %s\n", what.c_str());
    ++failures;
  }
}

bool contains(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

std::string region(int attr) {
  if (attr == 1) return "air";
  if (attr == 2) return "gap";
  return "attribute " + std::to_string(attr);
}

// two tetrahedra sharing the face (v1, v2, v3): a regular one (attribute 1) and one of height h above it
mfem::Mesh two_tets(double h) {
  mfem::Mesh m(3, 5, 2, 0, 3);
  m.AddVertex(0, 0, 0);
  m.AddVertex(1, 0, 0);
  m.AddVertex(0, 1, 0);
  m.AddVertex(0, 0, 1);
  const double c = 1.0 / 3.0 + h / std::sqrt(3.0);
  m.AddVertex(c, c, c);  // h along the outward normal of the face x + y + z = 1 from its centre
  m.AddTet(0, 1, 2, 3, 1);
  m.AddTet(1, 2, 3, 4, 2);
  m.FinalizeTetMesh(1, 1, true);
  return m;
}

// move the high-order node at the midpoint of the element's edge (a, b) to x
void move_edge_node(mfem::Mesh& m, int e, const double* a, const double* b, const double* x) {
  mfem::GridFunction& nodes = *m.GetNodes();
  const mfem::FiniteElementSpace* nfes = nodes.FESpace();
  mfem::Array<int> vd;
  nfes->GetElementVDofs(e, vd);
  const int nd = nfes->GetFE(e)->GetDof();
  const int dim = m.SpaceDimension();
  for (int i = 0; i < nd; ++i) {
    bool here = true;
    for (int c = 0; c < dim; ++c) {
      const int dof = vd[c * nd + i];
      here = here && std::fabs(nodes(dof >= 0 ? dof : -1 - dof) - 0.5 * (a[c] + b[c])) < 1e-12;
    }
    if (!here) continue;
    for (int c = 0; c < dim; ++c) {
      const int dof = vd[c * nd + i];
      nodes(dof >= 0 ? dof : -1 - dof) = x[c];
    }
    return;
  }
  check(false, "edge node found");
}

// a 2D mesh of two P2 triangles; the second (attribute `attr`) has the midpoint of its edge (a, b) moved to x
mfem::Mesh folded_triangles(int attr, const double* a, const double* b, const double* x) {
  mfem::Mesh m(2, 4, 2, 0, 2);
  m.AddVertex(0, 0);
  m.AddVertex(1, 0);
  m.AddVertex(0, 1);
  m.AddVertex(1, 1);
  m.AddTriangle(0, 1, 2, attr);
  m.AddTriangle(1, 3, 2, 2);
  m.FinalizeTriMesh(1, 1, true);
  m.SetCurvature(2, false, 2, mfem::Ordering::byVDIM);
  move_edge_node(m, 0, a, b, x);
  return m;
}

// every spec key a remedy names must be accepted by the parser ("mesh.mesh_size", for example, is no key)
void check_named_keys(const std::string& text, bool three_d) {
  nlohmann::json base = three_d
      ? nlohmann::json::parse(R"({"schema": "axionfem.design/1", "name": "k", "symmetry": "3d",
          "regions": [{"name": "gap", "solid": {"box": {"min": [0, 0, 0], "max": [0.1, 0.1, 0.1]}},
                       "material": "copper"}], "mesh": {"order": 2, "size": 0.05}})")
      : nlohmann::json::parse(R"({"schema": "axionfem.design/1", "name": "k",
          "regions": [{"name": "gap", "shape": {"rect": {"r": [0.5, 0.6], "z": [-0.1, 0.1]}},
                       "material": "copper"}], "mesh": {"order": 2, "size": 0.05}})");
  auto value = [](const std::string& key) -> nlohmann::json {
    if (key == "geometry_order" || key == "order") return 1;
    if (key == "curvature_points") return 10;
    if (key == "optimize_netgen") return true;
    return 0.01;
  };
  const std::regex global(R"(mesh\.([a-z_]+))"), regional(R"(the ([a-z_]+) of region '([^']+)')");
  int named = 0;
  for (std::sregex_iterator it(text.begin(), text.end(), global), end; it != end; ++it) {
    const std::string key = (*it)[1];
    if (key == "info") continue;  // mesh.info.worst_elements: a results key
    nlohmann::json doc = base;
    doc["mesh"][key] = value(key);
    try {
      axf::spec::parse_design(doc);
      ++named;
    } catch (const std::exception& e) {
      check(false, "mesh." + key + " is accepted: " + e.what() + " (named in: " + text + ")");
    }
  }
  for (std::sregex_iterator it(text.begin(), text.end(), regional), end; it != end; ++it) {
    const std::string key = (*it)[1];
    nlohmann::json doc = base;
    doc["regions"][0][key] = value(key);
    try {
      axf::spec::parse_design(doc);
      ++named;
    } catch (const std::exception& e) {
      check(false, "region key " + key + " is accepted: " + e.what() + " (named in: " + text + ")");
    }
  }
  check(named > 0 || !contains(text, "Fix in the design"), "a remedy names at least one key: " + text);
  // negative control: the parser does reject a key that does not exist
  nlohmann::json bad = base;
  bad["mesh"]["mesh_size"] = 0.01;
  bool rejected = false;
  try {
    axf::spec::parse_design(bad);
  } catch (const std::exception&) {
    rejected = true;
  }
  check(rejected, "mesh.mesh_size is rejected by the parser");
}

}  // namespace

int main() {
  mq::DescribeOptions o3;
  mq::DescribeOptions o2;
  o2.three_d = false;
  {  // straight sliver among good elements
    mfem::Mesh m = two_tets(1e-5);
    const mq::Summary s = mq::scan(m, 5);
    check(s.elements == 2 && s.worst.size() == 2, "sliver: two elements scanned");
    const mq::ElementQuality& w = s.worst.front();
    check(w.index == 1 && w.attribute == 2, "sliver: the flat element is the worst");
    check(!w.inverted() && !w.curved, "sliver: straight and not inverted");
    check(w.aspect > 1e3 && std::isfinite(w.aspect), "sliver: aspect ratio above 1e3");
    check(std::fabs(m.GetElementSize(1, 2) / m.GetElementSize(1, 1) - w.aspect) < 1e-9 * w.aspect,
          "sliver: aspect ratio as Mesh::GetElementSize measures it");
    check(s.flat == 1 && s.inverted == 0 && s.distorted == 0, "sliver: counts");
    check(std::fabs(w.centroid[0] - (1.0 + 1.0 / 3.0) / 4.0) < 1e-4, "sliver: centroid");
    const std::string t = mq::describe(s, region, o3);
    check(contains(t, "element 1 in region 'gap' (attribute 2)"), "sliver: names element and region: " + t);
    check(contains(t, "centred at (0.333"), "sliver: centroid in the text: " + t);
    check(contains(t, "flat (sliver)") && contains(t, "mesh.optimize_netgen: true") &&
              contains(t, "the surface_size of region 'gap'"),
          "sliver: remedy in the design: " + t);
    const auto j = mq::to_json(s, region);
    check(j["worst"][0]["region"] == "gap" && j["flat"] == 1, "sliver: json");
  }
  {  // curved tetrahedron folded by a mid-edge node pulled through it
    mfem::Mesh m = two_tets(1.0);
    m.SetCurvature(2, false, 3, mfem::Ordering::byVDIM);
    // the midpoint of (v1, v2) moved three times its distance to the opposite edge (v3, v4), through the element
    const double a[3] = {1, 0, 0}, b[3] = {0, 1, 0}, x[3] = {0.36, 0.36, 2.9};
    move_edge_node(m, 1, a, b, x);
    const mq::Summary s = mq::scan(m, 5);
    const mq::ElementQuality& w = s.worst.front();
    check(w.curved && w.inverted() && w.det_ratio < 0, "curved: inverted element detected");
    check(!std::isfinite(w.aspect), "curved: infinite aspect where det J <= 0");
    check(s.inverted >= 1, "curved: inverted count");
    const std::string t = mq::describe(s, region, o3);
    check(contains(t, "INVERTED or degenerate (det J <= 0), curved") && contains(t, "mesh.geometry_order to 1"),
          "curved: inversion and its remedy: " + t);
    // the order-2 mesh before the fold has no bad element and says so
    mfem::Mesh clean = two_tets(1.0);
    clean.SetCurvature(2, false, 3, mfem::Ordering::byVDIM);
    const mq::Summary c = mq::scan(clean, 5);
    check(c.inverted == 0 && c.flat == 0 && c.distorted == 0, "clean: no bad element");
    check(contains(mq::describe(c, region, o3), "probably not the cause"), "clean: not the mesh");
  }
  {  // 2D: a folded curved triangle
    mfem::Mesh m(2, 4, 2, 0, 2);
    m.AddVertex(0, 0);
    m.AddVertex(1, 0);
    m.AddVertex(0, 1);
    m.AddVertex(1, 1);
    m.AddTriangle(0, 1, 2, 1);
    m.AddTriangle(1, 3, 2, 2);
    m.FinalizeTriMesh(1, 1, true);
    m.SetCurvature(2, false, 2, mfem::Ordering::byVDIM);
    const double a[2] = {1, 0}, b[2] = {1, 1}, x[2] = {-1.0, 0.5};
    move_edge_node(m, 1, a, b, x);
    const mq::Summary s = mq::scan(m, 5);
    const mq::ElementQuality& w = s.worst.front();
    check(w.index == 1 && w.inverted(), "2D: folded triangle found");
    const std::string t = mq::describe(s, region, o2);
    check(contains(t, "centred at (r, z) = (") && contains(t, "region 'gap'") && contains(t, "mesh.order to 1"),
          "2D: coordinates and remedy: " + t);
  }
  {  // a P2 triangle folded only near a vertex (its mid-edge node slid along the edge past 1/4):
     // det J < 0 at the vertex, positive at every interior quadrature point
    const double a[2] = {0, 0}, b[2] = {1, 0};
    for (double slide : {0.26, 0.30}) {
      const double x[2] = {0.5 + slide, 0.0};
      mfem::Mesh m = folded_triangles(2, a, b, x);
      const mq::Summary s = mq::scan(m, 5);
      check(s.inverted == 1 && s.worst.front().index == 0 && s.worst.front().inverted(),
            "vertex fold found at slide " + std::to_string(slide));
      const std::string t = mq::describe(s, region, o2);
      check(contains(t, "vertices, edge midpoints") && !contains(t, "probably not the cause"),
            "vertex fold: text: " + t);
    }
    const double x[2] = {0.6, 0.0};  // slid 0.1: valid
    mfem::Mesh m = folded_triangles(2, a, b, x);
    check(mq::scan(m, 5).inverted == 0, "slide 0.1 is valid");
  }
  {  // an imported mesh gets a remedy about the file, not about the mesher's settings
    mfem::Mesh m = two_tets(1e-5);
    const mq::Summary s = mq::scan(m, 5);
    mq::DescribeOptions imp = o3;
    imp.imported_file = "/data/cad.msh";
    const std::string t = mq::describe(s, region, imp);
    check(contains(t, "the mesh is imported (mesh.file '/data/cad.msh')") &&
              contains(t, "Repair or regenerate it in the tool that wrote it") && !contains(t, "optimize_netgen") &&
              !contains(t, "surface_size"),
          "imported: remedy: " + t);
    mq::DescribeOptions imp2 = o2;
    imp2.imported_file = "/data/comsol.mphtxt.msh";
    check(contains(mq::describe(s, region, imp2), "Repair or regenerate"), "imported 2D: remedy");
    // a moderate worst aspect ratio of an imported mesh: no size field named either
    mfem::Mesh moderate = two_tets(1e-2);
    const mq::Summary sm = mq::scan(moderate, 5);
    check(sm.worst.front().aspect > 100 && sm.worst.front().aspect < 1e3, "moderate: aspect between 100 and 1e3");
    const std::string tm = mq::describe(sm, region, imp);
    check(contains(tm, "the mesh is imported") && contains(tm, "probably not needed") && !contains(tm, "surface_size"),
          "imported moderate: remedy: " + tm);
    // a clean imported mesh: the usual "not the cause"
    mfem::Mesh c = two_tets(1.0);
    check(contains(mq::describe(mq::scan(c, 5), region, imp), "probably not the cause"), "imported clean");
  }
  {  // a thrown error writes no results, so the text does not point at them
    mfem::Mesh m = two_tets(1e-5);
    mq::DescribeOptions thrown = o2;
    thrown.listed_in.clear();
    const std::string t = mq::describe(mq::scan(m, 5), region, thrown);
    check(!contains(t, "worst_elements"), "thrown: no pointer: " + t);
    mq::DescribeOptions rejected = o3;
    rejected.listed_in = "fullwave_rejected.json";
    check(contains(mq::describe(mq::scan(m, 5), region, rejected), "worst_elements in fullwave_rejected.json"),
          "rejected: pointer");
  }
  {  // every key named by any remedy, for a design region and for air, 2D and 3D
    for (int attr : {1, 2}) {
      // flat (straight sliver), 3D
      mfem::Mesh flat3 = two_tets(1e-5);
      flat3.SetAttribute(1, attr);
      flat3.SetAttributes();
      for (bool three_d : {true, false}) {
        mq::DescribeOptions o = three_d ? o3 : o2;
        mq::Summary s = mq::scan(flat3, 5);
        check_named_keys(mq::describe(s, region, o), three_d);
        // folded curved
        mfem::Mesh curved = two_tets(1.0);
        curved.SetAttribute(1, attr);
        curved.SetAttributes();
        curved.SetCurvature(2, false, 3, mfem::Ordering::byVDIM);
        const double a[3] = {1, 0, 0}, b[3] = {0, 1, 0}, x[3] = {0.36, 0.36, 2.9};
        move_edge_node(curved, 1, a, b, x);
        check_named_keys(mq::describe(mq::scan(curved, 5), region, o), three_d);
        // moderate aspect
        mfem::Mesh moderate = two_tets(1e-2);
        moderate.SetAttribute(1, attr);
        moderate.SetAttributes();
        check_named_keys(mq::describe(mq::scan(moderate, 5), region, o), three_d);
      }
    }
    // the 2D flat remedy names mesh.size for air, not mesh.mesh_size
    mfem::Mesh flat_air = two_tets(1e-5);
    flat_air.SetAttribute(1, 1);
    flat_air.SetAttributes();
    const std::string t = mq::describe(mq::scan(flat_air, 5), region, o2);
    check(contains(t, "mesh.size") && !contains(t, "mesh.mesh_size"), "2D air remedy: " + t);
  }
  {  // MPI reduction building blocks: records survive packing, and merging keeps the worst over the parts
    mfem::Mesh m = two_tets(1e-5);
    mq::Summary s = mq::scan(m, 5);
    double rec[mq::kRecordDoubles];
    mq::pack(s.worst.front(), rec);
    const mq::ElementQuality u = mq::unpack(rec);
    check(u.index == 1 && u.attribute == 2 && u.aspect == s.worst.front().aspect &&
              u.det_ratio == s.worst.front().det_ratio && u.centroid == s.worst.front().centroid,
          "pack/unpack round trip");
    mq::Summary part_a, part_b;
    part_a.elements = 1;
    part_a.worst = {s.worst.back()};
    part_b.elements = 1;
    part_b.flat = 1;
    part_b.worst = {s.worst.front()};
    mq::Summary total;
    mq::merge(total, part_a, 5);
    mq::merge(total, part_b, 5);
    check(total.elements == 2 && total.flat == 1 && total.worst.front().index == 1, "merge keeps the worst");
    mq::Summary one;
    mq::merge(one, part_a, 1);
    mq::merge(one, part_b, 1);
    check(one.worst.size() == 1 && one.worst.front().index == 1, "merge truncates to keep");
  }
  if (failures) return 1;
  std::puts("mesh check of solver-failure messages on bad meshes: passed");
  return 0;
}
