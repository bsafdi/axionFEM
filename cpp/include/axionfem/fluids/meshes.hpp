// Small mesh builders for fluids tests and benchmarks (no gmsh needed).
#pragma once

#include <mfem.hpp>

#include <vector>

namespace axf::fluids {

// Annulus R1 <= r <= R2, nr x nt quadrilaterals, curved isoparametrically to `order` with the nodes placed exactly
// on the polar map (r, theta) of each element's reference square, so both circles are exact at the nodes.
// Boundary attributes: 1 inner circle, 2 outer circle.
mfem::Mesh annulus_mesh(int nr, int nt, double R1, double R2, int order = 2);

// Mesh from vertex and element arrays (triangles: 3 vertices per element, quadrilaterals: 4), boundary segments
// with attributes; element attributes default to 1. Clockwise elements are reoriented by MFEM; a triangle with all
// three vertices on the boundary is accepted here (check_taylor_hood_mesh reports it).
//
// p2_nodes (triangles only): an isoparametric P2 geometry, 12 doubles per element = the 6 geometry nodes
// [v0, v1, v2, m01, m12, m20] (x, y interleaved) in the element's INPUT vertex order; v0..v2 must equal the vertex
// coordinates; a shared edge must have the same node in both elements (refused otherwise). The mesh then carries P2 (H1) nodes and its elements are curved wherever an edge node is off the
// chord.
mfem::Mesh mesh_from_arrays(const std::vector<double>& xy, const std::vector<int>& elements, int verts_per_element,
                            const std::vector<int>& bdr_segments, const std::vector<int>& bdr_attributes,
                            const std::vector<int>& elem_attributes = {},
                            const std::vector<double>& p2_nodes = {});

// The geometry nodes of every element at the 6 P2 reference nodes (0,0) (1,0) (0,1) (1/2,0) (1/2,1/2) (0,1/2):
// 12 doubles per element (straight elements give the chord midpoints).
std::vector<double> element_p2_nodes(const mfem::Mesh& mesh);

// A periodic identification: every vertex on the boundary elements of `partner` is identified with the vertex of
// `attribute` at x - translation (x_partner = x_attribute + translation).
struct PeriodicPair {
  int attribute = 0, partner = 0;
  std::vector<double> translation;
};

// The periodic mesh (MFEM Mesh::MakePeriodic; discontinuous nodes keep the element geometry). Refuses when a
// vertex of a partner boundary has no image (tolerance 1e-10 x the bounding-box diameter) or the two boundaries
// have different vertex counts. The boundary elements of both attributes are kept; they now lie on interior faces,
// which is how StokesSolver recognizes an identified pair. Element numbering is unchanged.
mfem::Mesh make_periodic_mesh(const mfem::Mesh& mesh, const std::vector<PeriodicPair>& pairs);

}  // namespace axf::fluids
