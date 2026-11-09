// Worst elements of a mesh for solver-failure messages (axionfem/mesh_quality.hpp).
#include "axionfem/mesh_quality.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <mfem.hpp>

#include "axionfem/core.hpp"

namespace axf::mesh_quality {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// a size field named in a remedy: the region's own, or the global one (mesh.size / mesh.surface_size) for air, the
// infinity shell and attributes that are no design region
bool global_region(const std::string& region) {
  return region == "air" || region == "infinity_shell" || region.rfind("attribute ", 0) == 0;
}

std::string size_field(const std::string& region, const char* field) {
  if (global_region(region)) return std::string(field) == "mesh_size" ? "mesh.size" : strcat("mesh.", field);
  return strcat("the ", field, " of region '", region, "'");
}

bool simplex(mfem::Geometry::Type g) {
  return g == mfem::Geometry::SEGMENT || g == mfem::Geometry::TRIANGLE || g == mfem::Geometry::TETRAHEDRON;
}

std::string fmt_aspect(double a) {
  if (!std::isfinite(a)) return "infinite (det J <= 0)";
  return strcat(a);
}

}  // namespace

bool worse(const ElementQuality& a, const ElementQuality& b) {
  const bool ia = a.inverted(), ib = b.inverted();
  if (ia != ib) return ia;
  if (ia && a.det_ratio != b.det_ratio) return a.det_ratio < b.det_ratio;
  if (a.aspect != b.aspect) return a.aspect > b.aspect;
  if (a.det_ratio != b.det_ratio) return a.det_ratio < b.det_ratio;
  return a.index < b.index;
}

ElementQuality measure(mfem::Mesh& m, int e) {
  ElementQuality q;
  q.index = e;
  q.attribute = m.GetAttribute(e);
  const mfem::Geometry::Type geom = m.GetElementBaseGeometry(e);
  const int dim = m.Dimension(), sdim = m.SpaceDimension();
  const mfem::FiniteElementSpace* nfes = m.GetNodalFESpace();
  q.curved = nfes && nfes->GetFE(e)->GetOrder() > 1;
  q.affine = !q.curved && simplex(geom);
  mfem::ElementTransformation* T = m.GetElementTransformation(e);
  const mfem::IntegrationPoint& centre = mfem::Geometries.GetCenter(geom);
  mfem::Vector x(sdim);
  T->Transform(centre, x);
  for (int c = 0; c < std::min(sdim, 3); ++c) q.centroid[c] = x(c);
  mfem::DenseMatrix J(sdim, dim);
  double dmin = kInf, dmax = 0, amax = 1;
  auto sample = [&](const mfem::IntegrationPoint& ip) {
    T->SetIntPoint(&ip);
    const double w = T->Weight();
    dmin = std::min(dmin, w);
    dmax = std::max(dmax, std::fabs(w));
    mfem::Geometries.JacToPerfJac(geom, T->Jacobian(), J);
    const double smax = J.CalcSingularvalue(0), smin = J.CalcSingularvalue(dim - 1);
    const double a = (w > 0 && smin > 0) ? smax / smin : kInf;
    amax = std::max(amax, a);
  };
  sample(centre);
  if (!q.affine) {
    // the reference vertices and the midpoints of every vertex pair (the edges of a simplex; for other shapes also
    // face and body diagonals), computed here rather than through a GeometryRefiner (a shared lazy cache)
    const mfem::IntegrationRule* verts = mfem::Geometries.GetVertices(geom);
    const int nv = verts->GetNPoints();
    for (int i = 0; i < nv; ++i) {
      sample(verts->IntPoint(i));
      for (int j = i + 1; j < nv; ++j) {
        const mfem::IntegrationPoint& a = verts->IntPoint(i);
        const mfem::IntegrationPoint& b = verts->IntPoint(j);
        mfem::IntegrationPoint mid;
        mid.Set3(0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z));
        sample(mid);
      }
    }
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(geom, 6);
    for (int k = 0; k < ir.GetNPoints(); ++k) sample(ir.IntPoint(k));
  }
  q.aspect = amax;
  q.det_ratio = dmax > 0 ? dmin / dmax : 0.0;   // 0: degenerate (det J = 0 at every sample point)
  return q;
}

void merge(Summary& into, const Summary& other, int keep) {
  into.elements += other.elements;
  into.inverted += other.inverted;
  into.distorted += other.distorted;
  into.flat += other.flat;
  into.worst.insert(into.worst.end(), other.worst.begin(), other.worst.end());
  std::sort(into.worst.begin(), into.worst.end(), worse);
  if (static_cast<int>(into.worst.size()) > keep) into.worst.resize(keep);
}

Summary scan(mfem::Mesh& m, int keep, const std::vector<int>* global_index) {
  Summary s;
  s.dim = m.Dimension();
  const int ne = m.GetNE();
  s.elements = ne;
  for (int e = 0; e < ne; ++e) {
    ElementQuality q = measure(m, e);
    if (global_index) q.index = (*global_index)[e];
    if (q.inverted()) {
      ++s.inverted;
    } else {
      if (!q.affine && q.det_ratio <= kDistortedDetRatio) ++s.distorted;
      if (q.aspect > kFlatAspect) ++s.flat;
    }
    const bool full = static_cast<int>(s.worst.size()) >= keep;
    if (full && !worse(q, s.worst.back())) continue;
    s.worst.push_back(q);
    std::sort(s.worst.begin(), s.worst.end(), worse);
    if (static_cast<int>(s.worst.size()) > keep) s.worst.pop_back();
  }
  return s;
}

void pack(const ElementQuality& q, double* rec) {
  rec[0] = static_cast<double>(q.index);
  rec[1] = q.attribute;
  rec[2] = q.centroid[0];
  rec[3] = q.centroid[1];
  rec[4] = q.centroid[2];
  rec[5] = q.aspect;
  rec[6] = q.det_ratio;
  rec[7] = q.curved ? 1.0 : 0.0;
  rec[8] = q.affine ? 1.0 : 0.0;
}

ElementQuality unpack(const double* rec) {
  ElementQuality q;
  q.index = static_cast<long long>(rec[0]);
  q.attribute = static_cast<int>(rec[1]);
  q.centroid = {rec[2], rec[3], rec[4]};
  q.aspect = rec[5];
  q.det_ratio = rec[6];
  q.curved = rec[7] > 0.5;
  q.affine = rec[8] > 0.5;
  return q;
}

std::string describe(const Summary& s, const RegionName& region, const DescribeOptions& o) {
  if (s.worst.empty()) return " Mesh check: the mesh has no elements.";
  const ElementQuality& w = s.worst.front();
  const std::string reg = region(w.attribute);
  std::string out = strcat(" Mesh check (run because this solve failed): the worst of the ", s.elements,
                           " elements is element ", w.index, " in region '", reg, "' (attribute ", w.attribute, ")");
  if (o.three_d)
    out += strcat(", centred at (", w.centroid[0], ", ", w.centroid[1], ", ", w.centroid[2], ") m");
  else
    out += strcat(", centred at ", o.coords_2d, " = (", w.centroid[0], ", ", w.centroid[1], ") m");
  const char* where = w.affine ? "" : " at its sampled points (vertices, edge midpoints, quadrature points)";
  out += strcat(": ", w.inverted() ? "INVERTED or degenerate (det J <= 0), " : "",
                w.curved ? "curved, " : "straight-sided, ", "min det J / max |det J| = ", w.det_ratio, where,
                ", Jacobian aspect ratio ", fmt_aspect(w.aspect), ".");
  out += strcat(" At the sampled points the mesh has ", s.inverted, " inverted, ", s.distorted,
                " distorted (min det J <= ", kDistortedDetRatio, " max |det J|) and ", s.flat,
                " flat (aspect ratio > ", kFlatAspect, ") elements");
  out += o.listed_in.empty() ? std::string(".")
                             : strcat("; mesh.info.worst_elements in ", o.listed_in, " lists the worst ",
                                      s.worst.size(), ".");
  const bool folded = w.inverted() || (!w.affine && w.det_ratio <= kDistortedDetRatio);
  const bool flat = !folded && w.aspect > kFlatAspect;
  if (!o.imported_file.empty() && (folded || flat || w.aspect > 100)) {
    out += strcat(" Fix: the mesh is imported (mesh.file '", o.imported_file, "'), so the design's mesh settings do "
                  "not change it. Repair or regenerate it in the tool that wrote it: ",
                  folded ? "an inverted or folded element"
                         : flat ? "a flat (sliver) element"
                                : "probably not needed (the worst aspect ratio is moderate), but a more uniform mesh",
                  " near that point.");
    return out;
  }
  const std::string surface = size_field(reg, "surface_size");
  std::string fix;
  if (folded && w.curved && o.three_d) {
    fix = strcat("the curved (order-2) geometry folds there. Set mesh.geometry_order to 1 (straight-sided "
                 "elements), or change ", surface, " or mesh.curvature_points so that the mesher can untangle "
                 "it; mesh.optimize_netgen: true also removes slivers.");
  } else if (folded && w.curved) {
    fix = strcat("the curved (order-2) triangles fold there. Make ", surface, " (or the surface_size of the "
                 "adjacent curve) smaller so that the curved boundary is resolved, or set mesh.order to 1.");
  } else if (folded) {
    fix = "an inverted or degenerate straight-sided element: the geometry intersects itself or has a "
          "zero-thickness fragment near that point. Repair the geometry there and remesh.";
  } else if (flat) {
    fix = o.three_d ? strcat("a flat (sliver) element. Set mesh.optimize_netgen: true, change ", surface,
                             " so that it is not close to the thickness of a nearby feature, or remove thin CAD "
                             "fragments near that point (axf mesh shows the mesh without solving).")
                    : strcat("a flat triangle, from a feature thinner than the local element size. Change ",
                             surface, " or ", size_field(reg, "mesh_size"), " near that point.");
  } else if (w.aspect > 100) {
    fix = strcat("probably none: no sampled point shows an inverted, distorted or flat element, and the worst aspect "
                 "ratio is moderate (", surface, " could still be made more uniform near that point). Follow the "
                 "solver hint.");
  } else {
    fix = "none needed: no sampled point shows an inverted, distorted or flat element, so the mesh is probably not "
          "the cause. Follow the solver hint.";
  }
  out += " Fix in the design: " + fix;
  return out;
}

nlohmann::json to_json(const Summary& s, const RegionName& region) {
  nlohmann::json rows = nlohmann::json::array();
  for (const auto& q : s.worst) {
    rows.push_back({{"element", q.index},
                    {"region", region(q.attribute)},
                    {"attribute", q.attribute},
                    {"centroid_m", s.dim == 3 ? nlohmann::json{q.centroid[0], q.centroid[1], q.centroid[2]}
                                              : nlohmann::json{q.centroid[0], q.centroid[1]}},
                    {"jacobian_aspect_ratio", std::isfinite(q.aspect) ? nlohmann::json(q.aspect) : nlohmann::json(nullptr)},
                    {"min_over_max_det_J", q.det_ratio},
                    {"inverted", q.inverted()},
                    {"curved", q.curved},
                    {"affine", q.affine}});
  }
  return {{"computed_because", "a solve broke down or did not converge"},
          {"elements", s.elements},
          {"inverted", s.inverted},
          {"distorted_curved", s.distorted},
          {"flat", s.flat},
          {"thresholds", {{"distorted_min_over_max_det_J", kDistortedDetRatio}, {"flat_aspect_ratio", kFlatAspect}}},
          {"worst", rows}};
}

}  // namespace axf::mesh_quality
