#include "axionfem/fluids/mesh_checks.hpp"

#include <mfem.hpp>

namespace axf::fluids {

InfSupMeshReport check_taylor_hood_mesh(const mfem::Mesh& mesh) {
  InfSupMeshReport rep;
  std::vector<char> on_bdr(mesh.GetNV(), 0);
  mfem::Array<int> v;
  for (int b = 0; b < mesh.GetNBE(); ++b) {
    const int face = mesh.GetBdrElementFaceIndex(b);
    if (face >= 0 && mesh.FaceIsInterior(face)) continue;  // an interface, not the exterior boundary
    mesh.GetBdrElementVertices(b, v);
    for (int i : v) on_bdr[i] = 1;
  }
  for (int e = 0; e < mesh.GetNE(); ++e) {
    const auto g = mesh.GetElementGeometry(e);
    const bool simplex = g == mfem::Geometry::TRIANGLE || g == mfem::Geometry::TETRAHEDRON;
    mesh.GetElementVertices(e, v);
    bool all = true;
    for (int i : v) all = all && on_bdr[i];
    if (simplex) {
      ++rep.checked_elements;
      if (all) rep.all_boundary_vertices.push_back(e);
    } else if (all) {
      rep.tensor_all_boundary_vertices.push_back(e);
    }
  }
  return rep;
}

}  // namespace axf::fluids
