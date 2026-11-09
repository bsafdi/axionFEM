// 3D meshes for the fluids library (3D Stokes).
//
//  * mesh_from_arrays_3d: tetrahedra (4 vertices) or hexahedra (8 vertices, basix LEXICOGRAPHIC vertex order, x
//    fastest), boundary faces (3 or 4 vertices; a quadrilateral as a cycle) with attributes >= 1, and optionally the
//    curved isoparametric geometry: 10 nodes per tetrahedron / 27 per hexahedron in the basix reference-node order
//    (REF_P2_TET / REF_Q2_HEX of the 3D Stokes check module, repeated in this file). MFEM may permute the vertices of a
//    tetrahedron (orientation fix); every element keeps an AFFINE reference map between the input reference cell and
//    MFEM's (RefMap3D), used to place the curved nodes and, later, to evaluate at INPUT reference points.
//  * taylor_hood_counts_3d: three counts (Boffi's hypothesis (a), its weakening (b), the
//    certain failure (c)) for tetrahedral meshes, and the hexahedra with all 8 vertices on the boundary.
#pragma once

#include <mfem.hpp>

#include <array>
#include <vector>

namespace axf::fluids {

// x_in = b + M x_mfem (reference coordinates of the input cell from those of MFEM's), row-major M.
struct RefMap3D {
  std::array<double, 9> M{1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::array<double, 3> b{0, 0, 0};
  double det() const;
  // input reference point -> MFEM reference point
  void to_mfem(const double* x_in, double* x_m) const;
  void to_input(const double* x_m, double* x_in) const;
};

struct Mesh3DInput {
  std::vector<double> vertices;    // nv * 3
  std::vector<int> elements;       // ne * vpe, vpe = 4 (tet) or 8 (hex, basix lexicographic order)
  int vpe = 4;
  std::vector<int> bdr_faces;      // nb * fv, fv = 3 (triangles) or 4 (quadrilaterals, cyclic order)
  int fv = 3;
  std::vector<int> bdr_attributes; // nb, >= 1
  std::vector<int> elem_attributes;  // ne or empty (1)
  std::vector<double> nodes;       // ne * nn * 3 (nn = 10 / 27, basix reference-node order) or empty (affine)
};

// The serial mesh. With curved nodes the mesh carries order-2 H1 nodes equal to the given ones (shared nodes must
// coincide: refused otherwise) and every element map must keep a positive Jacobian at a degree-6 rule.
// `curved` false ignores the given nodes (the straight-sided mesh of the same vertices).
mfem::Mesh mesh_from_arrays_3d(const Mesh3DInput& in, bool curved = true);

// Reference map of every element of `mesh` relative to the input element whose vertex coordinates are given
// (input order). Matching by coordinates (tolerance 1e-12 x the element size), so it works on a ParMesh too.
RefMap3D element_ref_map(mfem::Mesh& mesh, int e, const double* input_vertex_xyz, int vpe);

// Basix reference coordinates of the input cell: the vertices and the order-2 nodes.
const std::vector<std::array<double, 3>>& ref_vertices(int vpe);
const std::vector<std::array<double, 3>>& ref_nodes2(int vpe);

struct TaylorHood3DCounts {
  long tets_without_interior_vertex = 0;  // (a) Boffi's hypothesis violated
  long tets_violating_weak_boffi = 0;     // (b) fewer than three non-coplanar interior edges
  long tets_all_p2_nodes_on_boundary = 0; // (c) all six edges on the boundary: a certain spurious mode
  long hexes_all_vertices_on_boundary = 0;
  long tets = 0, hexes = 0;
};
// From the element connectivity (exterior faces = faces of exactly one element).
TaylorHood3DCounts taylor_hood_counts_3d(const std::vector<int>& elements, int vpe, int nv);

}  // namespace axf::fluids
