// Element quality for solver-failure messages: the worst element(s) of a mesh, where they are, in which region,
// and how to fix them in the design. Computed only when a solve breaks down or stalls (no cost to a successful
// solve). MPI-free; the 3D engine gathers the per-rank summaries (model3d_impl.hpp, worst_elements_note).
#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mfem {
class Mesh;
}

namespace axf::mesh_quality {

// Thresholds: a curved element is distorted at min det J <= 1e-3 max |det J| (the 3D engine's curved-element test,
// model3d.cpp curved_element_invalid); an element is flat (a sliver) at a Jacobian aspect ratio above 1e3 (the 3D
// engine's mesh warning).
inline constexpr double kDistortedDetRatio = 1e-3;
inline constexpr double kFlatAspect = 1e3;

struct ElementQuality {
  long long index = -1;              // element number (the serial mesh's in 3D MPI runs)
  int attribute = 0;                 // element attribute (region)
  std::array<double, 3> centroid{};  // image of the reference centre (m; z = 0 in 2D, where x = r and y = z)
  // largest Jacobian aspect ratio sigma_max / sigma_min over the sample points, measured against the perfect
  // element as Mesh::GetElementSize does (1 for an equilateral simplex; infinite where det J <= 0)
  double aspect = 1;
  double det_ratio = 1;              // min det J / max |det J| over the sample points (<= 0: inverted or degenerate)
  bool curved = false;               // high-order geometry
  // Affine elements (straight simplices) have a constant Jacobian, sampled at the centre. Others (curved, or
  // straight quadrilaterals, prisms, hexahedra) are sampled at the centre, the reference vertices and edge midpoints
  // (where a P2 element's det J usually has its minimum) and the order-6 quadrature points.
  bool affine = true;
  bool inverted() const { return !(det_ratio > 0); }
};

// strict weak order "a is worse than b": inverted first (most negative det ratio first), then by aspect ratio,
// then by distortion; ties by the lower element number (deterministic across ranks)
bool worse(const ElementQuality& a, const ElementQuality& b);

ElementQuality measure(mfem::Mesh& m, int e);

struct Summary {
  std::vector<ElementQuality> worst;  // worst first
  long long elements = 0;
  long long inverted = 0;             // det J <= 0 at some sample point
  long long distorted = 0;            // not affine, not inverted, min det J <= 1e-3 max |det J| (JSON
                                      // "distorted_curved": curved, or straight but not a simplex)
  long long flat = 0;                 // aspect ratio > 1e3 (inverted elements excluded)
  int dim = 3;
};

// The `keep` worst elements of a mesh. global_index maps a local element to its global (serial) number.
Summary scan(mfem::Mesh& m, int keep, const std::vector<int>* global_index = nullptr);
// Combine two summaries of disjoint element sets (counts add, the worst `keep` survive).
void merge(Summary& into, const Summary& other, int keep);

// fixed-size records for gathering across MPI ranks
inline constexpr int kRecordDoubles = 9;
void pack(const ElementQuality& q, double* rec);
ElementQuality unpack(const double* rec);

using RegionName = std::function<std::string(int attribute)>;

struct DescribeOptions {
  bool three_d = true;                     // 3D or 2D remedies
  std::string coords_2d = "(r, z)";        // 2D coordinate names: "(r, z)" axisymmetric, "(x, y)" planar
  std::string imported_file;               // the mesh was read from mesh.file: the remedy is about that file
  std::string listed_in = "the results";   // where mesh.info.worst_elements can be read; empty: nowhere (a thrown error)
};

// The text appended to a solver-failure message: the worst element (number, region, centroid, measures), the
// counts, and what to change in the design (for an imported mesh: in the tool that wrote it). Starts with a space.
std::string describe(const Summary& s, const RegionName& region, const DescribeOptions& o = {});
nlohmann::json to_json(const Summary& s, const RegionName& region);

}  // namespace axf::mesh_quality
