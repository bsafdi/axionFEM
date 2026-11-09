// Discrete topology on tetrahedral meshes: first cohomology of element subsets.
//
// For a set S of elements (e.g. one superconducting body) we compute a basis
// of H^1(S; R): closed, non-exact 1-cochains on the edges of S (values on mesh
// edges, oriented from the lower to the higher vertex returned by
// Mesh::GetEdgeVertices).  Closed means the oriented sum around every
// triangular face of S vanishes.
//
// Algorithm: a spanning forest of the edge graph is gauged to zero; the other
// edge values follow from the face equations by propagation.  When the
// propagation stalls, a free parameter is introduced; after all edges are
// expressed in the parameters, the face equations not used for propagation
// give linear constraints whose null space (SVD) is H^1.  This is robust:
// spurious stalls only add parameters that the constraints remove.
#pragma once

#include <vector>

#include <mfem.hpp>

namespace axf::topo {

struct Cohomology {
  int betti1 = 0;                              // dimension of H^1(S)
  std::vector<std::vector<double>> cochains;   // betti1 vectors of length mesh.GetNEdges()
  int parameters = 0;                          // diagnostic: parameters introduced
};

Cohomology cohomology_h1(const mfem::Mesh& mesh, const std::vector<char>& in_elem);

// Euler characteristic of the subset (V - E + F - T) for diagnostics.
long euler_characteristic(const mfem::Mesh& mesh, const std::vector<char>& in_elem);

}  // namespace axf::topo
