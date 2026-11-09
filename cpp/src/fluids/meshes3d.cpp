// 3D meshes from arrays, reference maps and the 3D Taylor–Hood mesh counts: see meshes3d.hpp.
#include "axionfem/fluids/meshes3d.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "axionfem/core.hpp"

namespace axf::fluids {

namespace {
// MFEM hexahedron vertex k is basix (lexicographic) vertex kHexBasixOfMfem[k]
constexpr int kHexBasixOfMfem[8] = {0, 1, 3, 2, 4, 5, 7, 6};
// MFEM reference vertices
const double kTetRefMfem[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
const double kHexRefMfem[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
}  // namespace

double RefMap3D::det() const {
  const auto& m = M;
  return m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

void RefMap3D::to_input(const double* xm, double* xi) const {
  for (int i = 0; i < 3; ++i) xi[i] = b[i] + M[3 * i] * xm[0] + M[3 * i + 1] * xm[1] + M[3 * i + 2] * xm[2];
}

void RefMap3D::to_mfem(const double* xi, double* xm) const {
  // M is a signed permutation-like matrix of a cell symmetry; invert it generally
  const auto& m = M;
  const double d = det();
  const double r[3] = {xi[0] - b[0], xi[1] - b[1], xi[2] - b[2]};
  const double inv[9] = {(m[4] * m[8] - m[5] * m[7]) / d, (m[2] * m[7] - m[1] * m[8]) / d, (m[1] * m[5] - m[2] * m[4]) / d,
                         (m[5] * m[6] - m[3] * m[8]) / d, (m[0] * m[8] - m[2] * m[6]) / d, (m[2] * m[3] - m[0] * m[5]) / d,
                         (m[3] * m[7] - m[4] * m[6]) / d, (m[1] * m[6] - m[0] * m[7]) / d, (m[0] * m[4] - m[1] * m[3]) / d};
  for (int i = 0; i < 3; ++i) xm[i] = inv[3 * i] * r[0] + inv[3 * i + 1] * r[1] + inv[3 * i + 2] * r[2];
}

const std::vector<std::array<double, 3>>& ref_vertices(int vpe) {
  static const std::vector<std::array<double, 3>> tet = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  static const std::vector<std::array<double, 3>> hex = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
                                                         {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
  return vpe == 4 ? tet : hex;
}

const std::vector<std::array<double, 3>>& ref_nodes2(int vpe) {
  // basix order, identical to f1c_stokes_check.REF_P2_TET / REF_Q2_HEX
  static const std::vector<std::array<double, 3>> tet = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}, {0, 0, .5}, {0, .5, 0},
      {.5, 0, 0}};
  static const std::vector<std::array<double, 3>> hex = {
      {0, 0, 0},   {1, 0, 0},   {0, 1, 0},   {1, 1, 0},  {0, 0, 1},   {1, 0, 1},   {0, 1, 1},
      {1, 1, 1},   {.5, 0, 0},  {0, .5, 0},  {0, 0, .5}, {1, .5, 0},  {1, 0, .5},  {.5, 1, 0},
      {0, 1, .5},  {1, 1, .5},  {.5, 0, 1},  {0, .5, 1}, {1, .5, 1},  {.5, 1, 1},  {.5, .5, 0},
      {.5, 0, .5}, {0, .5, .5}, {1, .5, .5}, {.5, 1, .5}, {.5, .5, 1}, {.5, .5, .5}};
  return vpe == 4 ? tet : hex;
}

RefMap3D element_ref_map(mfem::Mesh& mesh, int e, const double* xin, int vpe) {
  mfem::Array<int> mv;
  mesh.GetElementVertices(e, mv);
  AXF_REQUIRE(mv.Size() == vpe, strcat("element_ref_map: element ", e, " has ", mv.Size(), " vertices, ", vpe,
                                       " expected"));
  double size = 0;
  for (int k = 1; k < vpe; ++k)
    size = std::max(size, std::hypot(xin[3 * k] - xin[0], std::hypot(xin[3 * k + 1] - xin[1], xin[3 * k + 2] - xin[2])));
  const double tol = 1e-9 * std::max(size, 1e-300);
  std::vector<int> perm(vpe, -1);  // MFEM vertex k = input vertex perm[k]
  for (int k = 0; k < vpe; ++k) {
    const double* x = mesh.GetVertex(mv[k]);
    for (int j = 0; j < vpe; ++j)
      if (std::fabs(x[0] - xin[3 * j]) <= tol && std::fabs(x[1] - xin[3 * j + 1]) <= tol &&
          std::fabs(x[2] - xin[3 * j + 2]) <= tol) {
        perm[k] = j;
        break;
      }
    AXF_REQUIRE(perm[k] >= 0, strcat("element_ref_map: vertex ", k, " of element ", e,
                                     " matches no vertex of the input element"));
  }
  const auto& R = ref_vertices(vpe);
  const int o = 0;
  const int ax[3] = {1, vpe == 4 ? 2 : 3, vpe == 4 ? 3 : 4};  // MFEM vertices at e_x, e_y, e_z
  RefMap3D m;
  for (int i = 0; i < 3; ++i) {
    m.b[i] = R[perm[o]][i];
    for (int d = 0; d < 3; ++d) m.M[3 * i + d] = R[perm[ax[d]]][i] - R[perm[o]][i];
  }
  // consistency on every vertex (a cell symmetry maps vertices to vertices)
  const double (*Rm)[3] = vpe == 4 ? kTetRefMfem : kHexRefMfem;
  for (int k = 0; k < vpe; ++k) {
    double xi[3];
    m.to_input(Rm[k], xi);
    for (int i = 0; i < 3; ++i)
      AXF_REQUIRE(std::fabs(xi[i] - R[perm[k]][i]) < 1e-12,
                  strcat("element_ref_map: element ", e, " is not an affine image of the input cell (vertex ", k, ")"));
  }
  AXF_REQUIRE(std::fabs(std::fabs(m.det()) - 1.0) < 1e-12, "element_ref_map: reference map is not a cell symmetry");
  return m;
}

mfem::Mesh mesh_from_arrays_3d(const Mesh3DInput& in, bool curved) {
  const int vpe = in.vpe, fv = in.fv;
  AXF_REQUIRE(vpe == 4 || vpe == 8, "mesh_from_arrays_3d: 4 (tetrahedra) or 8 (hexahedra) vertices per element");
  AXF_REQUIRE(fv == (vpe == 4 ? 3 : 4), "mesh_from_arrays_3d: boundary faces must be triangles for tetrahedra and "
                                        "quadrilaterals for hexahedra");
  AXF_REQUIRE(in.vertices.size() % 3 == 0 && in.elements.size() % vpe == 0 && in.bdr_faces.size() % fv == 0 &&
                  in.bdr_attributes.size() * fv == in.bdr_faces.size(),
              "mesh_from_arrays_3d: inconsistent array sizes");
  const int nv = static_cast<int>(in.vertices.size() / 3), ne = static_cast<int>(in.elements.size() / vpe),
            nb = static_cast<int>(in.bdr_attributes.size());
  AXF_REQUIRE(ne > 0, "mesh_from_arrays_3d: no elements");
  AXF_REQUIRE(in.elem_attributes.empty() || static_cast<int>(in.elem_attributes.size()) == ne,
              "mesh_from_arrays_3d: element attributes size");
  const int nn = vpe == 4 ? 10 : 27;
  AXF_REQUIRE(in.nodes.empty() || static_cast<long>(in.nodes.size()) == 3L * nn * ne,
              strcat("mesh_from_arrays_3d: nodes has ", in.nodes.size(), " values; ", 3 * nn, " per element expected"),
              "nodes", "Pass an (ne, 10, 3) (tetrahedra) or (ne, 27, 3) (hexahedra) array in basix node order.");
  mfem::Mesh m(3, nv, ne, nb, 3);
  for (int v = 0; v < nv; ++v) m.AddVertex(in.vertices[3 * v], in.vertices[3 * v + 1], in.vertices[3 * v + 2]);
  for (int e = 0; e < ne; ++e) {
    const int* c = &in.elements[static_cast<size_t>(vpe) * e];
    for (int k = 0; k < vpe; ++k)
      AXF_REQUIRE(c[k] >= 0 && c[k] < nv, strcat("mesh_from_arrays_3d: element ", e, " vertex ", c[k], " out of range"));
    const int at = in.elem_attributes.empty() ? 1 : in.elem_attributes[e];
    if (vpe == 4) {
      m.AddTet(c, at);
    } else {
      int h[8];
      for (int k = 0; k < 8; ++k) h[k] = c[kHexBasixOfMfem[k]];
      m.AddHex(h, at);
    }
  }
  for (int b = 0; b < nb; ++b) {
    AXF_REQUIRE(in.bdr_attributes[b] >= 1, "mesh_from_arrays_3d: boundary attributes must be >= 1");
    const int* f = &in.bdr_faces[static_cast<size_t>(fv) * b];
    if (fv == 3) m.AddBdrTriangle(f, in.bdr_attributes[b]);
    else m.AddBdrQuad(f, in.bdr_attributes[b]);
  }
  if (vpe == 4) {
    m.FinalizeTetMesh(1, 0, true);  // fix_orientation: MFEM may permute the vertices (RefMap3D keeps track)
  } else {
    m.FinalizeHexMesh(1, 0, false);
    // every hexahedron must be right-handed at its centre (MFEM does not reorder hexahedra)
    for (int e = 0; e < ne; ++e) {
      mfem::ElementTransformation* T = m.GetElementTransformation(e);
      const mfem::IntegrationPoint& c = mfem::Geometries.GetCenter(mfem::Geometry::CUBE);
      T->SetIntPoint(&c);
      AXF_REQUIRE(T->Weight() > 0, strcat("mesh_from_arrays_3d: hexahedron ", e, " is left-handed"), "elements",
                  "Order the 8 vertices lexicographically (x fastest) with a positive Jacobian.");
    }
  }
  if (in.nodes.empty() || !curved) return m;

  m.SetCurvature(2, false, 3, mfem::Ordering::byVDIM);
  mfem::GridFunction* nodes = m.GetNodes();
  const mfem::FiniteElementSpace* nfes = nodes->FESpace();
  const auto& R = ref_nodes2(vpe);
  std::vector<char> set(nodes->Size(), 0);
  double diam = 0;
  {
    mfem::Vector pmin, pmax;
    m.GetBoundingBox(pmin, pmax);
    diam = pmax.DistanceTo(pmin);
  }
  mfem::Array<int> vd;
  for (int e = 0; e < ne; ++e) {
    std::vector<double> xin(3 * vpe);
    for (int k = 0; k < vpe; ++k)
      for (int i = 0; i < 3; ++i) xin[3 * k + i] = in.vertices[3 * in.elements[static_cast<size_t>(vpe) * e + k] + i];
    const RefMap3D map = element_ref_map(m, e, xin.data(), vpe);
    nfes->GetElementVDofs(e, vd);
    const mfem::IntegrationRule& nodes_ir = nfes->GetFE(e)->GetNodes();
    const int nd = nodes_ir.GetNPoints();
    AXF_REQUIRE(nd == nn && vd.Size() == 3 * nd, "mesh_from_arrays_3d: internal error (order-2 node count)");
    for (int k = 0; k < nd; ++k) {
      const mfem::IntegrationPoint& ip = nodes_ir.IntPoint(k);
      const double xm[3] = {ip.x, ip.y, ip.z};
      double xi[3];
      map.to_input(xm, xi);
      int idx = -1;
      for (int j = 0; j < nn; ++j)
        if (std::fabs(R[j][0] - xi[0]) < 1e-10 && std::fabs(R[j][1] - xi[1]) < 1e-10 &&
            std::fabs(R[j][2] - xi[2]) < 1e-10)
          idx = j;
      AXF_REQUIRE(idx >= 0, "mesh_from_arrays_3d: internal error (an order-2 node is no basix reference node)");
      for (int c = 0; c < 3; ++c) {
        const int d = vd[c * nd + k];
        const int dd = d >= 0 ? d : -1 - d;
        const double val = in.nodes[(static_cast<size_t>(e) * nn + idx) * 3 + c];
        if (set[dd])
          AXF_REQUIRE(std::fabs((*nodes)(dd) - val) <= 1e-12 * diam,
                      strcat("mesh_from_arrays_3d: element ", e, " gives a shared geometry node (reference node ", idx,
                             ") another position than its neighbour"), "nodes",
                      "A node on a shared edge or face must be the same point in every element.");
        (*nodes)(dd) = val;
        set[dd] = 1;
      }
    }
  }
  // the map now reproduces every given node, and stays invertible
  const auto g = vpe == 4 ? mfem::Geometry::TETRAHEDRON : mfem::Geometry::CUBE;
  const mfem::IntegrationRule& ir6 = mfem::IntRules.Get(g, 6);
  for (int e = 0; e < ne; ++e) {
    mfem::ElementTransformation* T = m.GetElementTransformation(e);
    for (int q = 0; q < ir6.GetNPoints(); ++q) {
      T->SetIntPoint(&ir6.IntPoint(q));
      AXF_REQUIRE(T->Weight() > 0, strcat("mesh_from_arrays_3d: the curved geometry of element ", e,
                                          " has a non-positive Jacobian"), "nodes");
    }
  }
  return m;
}

namespace {
struct KeyHash {
  size_t operator()(const std::array<int, 4>& k) const {
    size_t h = 1469598103934665603ULL;
    for (int v : k) h = (h ^ static_cast<size_t>(v + 1)) * 1099511628211ULL;
    return h;
  }
};
}  // namespace

TaylorHood3DCounts taylor_hood_counts_3d(const std::vector<int>& el, int vpe, int nv) {
  AXF_REQUIRE(vpe == 4 || vpe == 8, "taylor_hood_counts_3d: tetrahedra or hexahedra");
  const int ne = static_cast<int>(el.size() / vpe);
  static const int tf[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
  static const int hf[6][4] = {{0, 1, 3, 2}, {0, 1, 5, 4}, {0, 2, 6, 4}, {1, 3, 7, 5}, {2, 3, 7, 6}, {4, 5, 7, 6}};
  std::unordered_map<std::array<int, 4>, int, KeyHash> faces;
  faces.reserve(static_cast<size_t>(ne) * (vpe == 4 ? 4 : 6));
  auto key_of = [&](const int* c, const int* f, int n) {
    std::array<int, 4> k{-1, -1, -1, -1};
    for (int i = 0; i < n; ++i) k[i] = c[f[i]];
    std::sort(k.begin(), k.begin() + n);
    return k;
  };
  for (int e = 0; e < ne; ++e) {
    const int* c = &el[static_cast<size_t>(vpe) * e];
    if (vpe == 4)
      for (const auto& f : tf) ++faces[key_of(c, f, 3)];
    else
      for (const auto& f : hf) ++faces[key_of(c, f, 4)];
  }
  std::vector<char> bv(nv, 0);
  std::unordered_map<long, char> bedge;
  auto ekey = [&](int a, int b) { return static_cast<long>(std::min(a, b)) * nv + std::max(a, b); };
  for (const auto& kv : faces) {
    if (kv.second != 1) continue;
    const auto& k = kv.first;
    const int n = k[3] < 0 ? 3 : 4;
    for (int i = 0; i < n; ++i) bv[k[i]] = 1;
    if (n == 3) {
      bedge[ekey(k[0], k[1])] = 1;
      bedge[ekey(k[0], k[2])] = 1;
      bedge[ekey(k[1], k[2])] = 1;
    }
  }
  TaylorHood3DCounts r;
  static const int te[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  static const int fe[4][3] = {{3, 4, 5}, {1, 2, 5}, {0, 2, 4}, {0, 1, 3}};  // edges of the face opposite vertex v
  for (int e = 0; e < ne; ++e) {
    const int* c = &el[static_cast<size_t>(vpe) * e];
    bool all = true;
    for (int i = 0; i < vpe; ++i) all = all && bv[c[i]];
    if (vpe == 8) {
      ++r.hexes;
      r.hexes_all_vertices_on_boundary += all;
      continue;
    }
    ++r.tets;
    bool inner[6];
    int n_in = 0;
    for (int i = 0; i < 6; ++i) {
      inner[i] = !bedge.count(ekey(c[te[i][0]], c[te[i][1]]));
      n_in += inner[i];
    }
    bool three_face = false;
    for (const auto& f : fe) three_face = three_face || (inner[f[0]] && inner[f[1]] && inner[f[2]]);
    const bool ok_b = n_in >= 4 || (n_in == 3 && !three_face);
    r.tets_without_interior_vertex += all;
    r.tets_violating_weak_boffi += !ok_b;
    r.tets_all_p2_nodes_on_boundary += (n_in == 0);
  }
  return r;
}

}  // namespace axf::fluids
