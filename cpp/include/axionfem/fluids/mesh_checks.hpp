// Mesh conditions for inf-sup stability of Taylor–Hood elements.
//
//  * 2D (triangles): no element may have all of its vertices on the boundary. Such a corner triangle has no
//    interior velocity node to balance its pressure modes, and P2/P1 is then not uniformly inf-sup stable.
//  * 3D (tetrahedra): every tetrahedron needs at least one interior vertex (Boffi 1997); a tet whose four vertices
//    all lie on the boundary is flagged.
//  Quadrilaterals/hexahedra are reported separately (`tensor_all_boundary_vertices`) as a warning: the classical
//  Q2/Q1 stability proofs assume shape-regular meshes and we do not claim more; a corner quad with all vertices on
//  the boundary still has an interior (bubble) velocity node, so it is not refused.
//  Only exterior boundary faces count: boundary elements on interior faces (material interfaces) are ignored.
//
// Call it on the SERIAL mesh before partitioning: on a ParMesh, a vertex can lie on the global boundary while the
// boundary element that shows it lives on another rank.
#pragma once

#include <vector>

namespace mfem {
class Mesh;
}

namespace axf::fluids {

struct InfSupMeshReport {
  long checked_elements = 0;               // simplices examined
  std::vector<int> all_boundary_vertices;  // simplices that violate the condition (refuse)
  std::vector<int> tensor_all_boundary_vertices;  // quads/hexes with all vertices on the boundary (warn)
  bool ok() const { return all_boundary_vertices.empty(); }
};

InfSupMeshReport check_taylor_hood_mesh(const mfem::Mesh& mesh);

}  // namespace axf::fluids
