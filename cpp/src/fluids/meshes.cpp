#include "axionfem/fluids/meshes.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include "axionfem/core.hpp"

namespace axf::fluids {

namespace {
class PolarMap : public mfem::VectorCoefficient {
 public:
  PolarMap(int nr, int nt, double R1, double R2) : mfem::VectorCoefficient(2), nr_(nr), nt_(nt), R1_(R1), R2_(R2) {}
  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    const int e = T.ElementNo, i = e / nt_, j = e % nt_;
    const double r = R1_ + (R2_ - R1_) * (i + ip.x) / nr_;
    const double th = 2.0 * M_PI * (j + ip.y) / nt_;
    V.SetSize(2);
    V(0) = r * std::cos(th);
    V(1) = r * std::sin(th);
  }

 private:
  int nr_, nt_;
  double R1_, R2_;
};
}  // namespace

mfem::Mesh annulus_mesh(int nr, int nt, double R1, double R2, int order) {
  AXF_REQUIRE(nr >= 1 && nt >= 3 && R2 > R1 && R1 > 0, "annulus_mesh: need nr >= 1, nt >= 3, 0 < R1 < R2");
  mfem::Mesh m(2, (nr + 1) * nt, nr * nt, 2 * nt, 2);
  for (int i = 0; i <= nr; ++i)
    for (int j = 0; j < nt; ++j) {
      const double r = R1 + (R2 - R1) * i / nr, th = 2.0 * M_PI * j / nt;
      m.AddVertex(r * std::cos(th), r * std::sin(th));
    }
  auto vid = [&](int i, int j) { return i * nt + (j % nt); };
  // element e = i*nt + j; reference (xi, eta) = (radial, angular): vertices (i,j), (i+1,j), (i+1,j+1), (i,j+1)
  for (int i = 0; i < nr; ++i)
    for (int j = 0; j < nt; ++j) m.AddQuad(vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1), 1);
  for (int j = 0; j < nt; ++j) {
    m.AddBdrSegment(vid(0, j + 1), vid(0, j), 1);
    m.AddBdrSegment(vid(nr, j), vid(nr, j + 1), 2);
  }
  m.FinalizeQuadMesh(1, 0, true);
  m.SetCurvature(order);
  PolarMap map(nr, nt, R1, R2);
  m.Transform(map);
  return m;
}

mfem::Mesh mesh_from_arrays(const std::vector<double>& xy, const std::vector<int>& elements, int vpe,
                            const std::vector<int>& bdr, const std::vector<int>& battr,
                            const std::vector<int>& eattr, const std::vector<double>& p2) {
  AXF_REQUIRE(vpe == 3 || vpe == 4, "mesh_from_arrays: 3 (triangles) or 4 (quadrilaterals) vertices per element");
  AXF_REQUIRE(xy.size() % 2 == 0 && elements.size() % vpe == 0 && bdr.size() % 2 == 0 &&
                  battr.size() * 2 == bdr.size(), "mesh_from_arrays: inconsistent array sizes");
  const int nv = static_cast<int>(xy.size() / 2), ne = static_cast<int>(elements.size() / vpe),
            nb = static_cast<int>(battr.size());
  AXF_REQUIRE(eattr.empty() || static_cast<int>(eattr.size()) == ne, "mesh_from_arrays: element attributes size");
  mfem::Mesh m(2, nv, ne, nb, 2);
  for (int v = 0; v < nv; ++v) m.AddVertex(xy[2 * v], xy[2 * v + 1]);
  for (int e = 0; e < ne; ++e) {
    const int* c = &elements[vpe * e];
    for (int k = 0; k < vpe; ++k)
      AXF_REQUIRE(c[k] >= 0 && c[k] < nv, strcat("mesh_from_arrays: element ", e, " has vertex ", c[k], " out of range"));
    const int at = eattr.empty() ? 1 : eattr[e];
    if (vpe == 3) m.AddTriangle(c[0], c[1], c[2], at);
    else m.AddQuad(c[0], c[1], c[2], c[3], at);
  }
  for (int b = 0; b < nb; ++b) {
    AXF_REQUIRE(battr[b] >= 1, "mesh_from_arrays: boundary attributes must be >= 1");
    m.AddBdrSegment(bdr[2 * b], bdr[2 * b + 1], battr[b]);
  }
  if (vpe == 3) m.FinalizeTriMesh(1, 0, true);
  else m.FinalizeQuadMesh(1, 0, true);
  if (p2.empty()) return m;
  AXF_REQUIRE(vpe == 3, "mesh_from_arrays: P2 geometry nodes are implemented for triangles only", "p2_nodes");
  AXF_REQUIRE(static_cast<int>(p2.size()) == 12 * ne,
              strcat("mesh_from_arrays: p2_nodes has ", p2.size(), " values; 12 per element (", 12 * ne,
                     ") expected"), "p2_nodes", "Pass an (ne, 6, 2) array [v0, v1, v2, m01, m12, m20] per triangle.");
  double diam = 0;
  {
    mfem::Vector pmin, pmax;
    m.GetBoundingBox(pmin, pmax);
    diam = pmax.DistanceTo(pmin);
  }
  for (int e = 0; e < ne; ++e)
    for (int k = 0; k < 3; ++k) {
      const int v = elements[3 * e + k];
      AXF_REQUIRE(std::hypot(p2[12 * e + 2 * k] - xy[2 * v], p2[12 * e + 2 * k + 1] - xy[2 * v + 1]) <= 1e-12 * diam,
                  strcat("mesh_from_arrays: p2_nodes of element ", e, ", node ", k,
                         " is not the coordinate of its vertex ", v), "p2_nodes",
                  "The first three geometry nodes are the vertices in the element's input order.");
    }
  // a shared edge's node must be the same point in both elements
  {
    std::map<std::pair<int, int>, std::pair<double, double>> mid;
    for (int e = 0; e < ne; ++e)
      for (int k = 0; k < 3; ++k) {
        const int a = elements[3 * e + k], b = elements[3 * e + (k + 1) % 3];
        const auto key = std::make_pair(std::min(a, b), std::max(a, b));
        const double x = p2[12 * e + 2 * (3 + k)], y = p2[12 * e + 2 * (3 + k) + 1];
        auto it = mid.find(key);
        if (it == mid.end()) mid[key] = {x, y};
        else
          AXF_REQUIRE(std::hypot(it->second.first - x, it->second.second - y) <= 1e-12 * diam,
                      strcat("mesh_from_arrays: the edge (", a, ", ", b, ") has two different P2 nodes in its two "
                             "elements (element ", e, ")"), "p2_nodes", "Give a shared edge one geometry node.");
      }
  }
  m.SetCurvature(2, false, 2, mfem::Ordering::byVDIM);
  mfem::GridFunction* nodes = m.GetNodes();
  const mfem::FiniteElementSpace* nfes = nodes->FESpace();
  mfem::Array<int> vd, mv;
  for (int e = 0; e < ne; ++e) {
    nfes->GetElementVDofs(e, vd);
    m.GetElementVertices(e, mv);
    const int nd = vd.Size() / 2;
    AXF_REQUIRE(nd == 6, "mesh_from_arrays: internal error (P2 triangle with != 6 nodes)");
    const int* in = &elements[3 * e];
    auto input_pos = [&](int g) {
      for (int k = 0; k < 3; ++k)
        if (in[k] == g) return k;
      throw axf::Error(strcat("mesh_from_arrays: element ", e, " lost vertex ", g));
    };
    auto set = [&](int local, int node) {
      (*nodes)(vd[local]) = p2[12 * e + 2 * node];
      (*nodes)(vd[nd + local]) = p2[12 * e + 2 * node + 1];
    };
    for (int j = 0; j < 3; ++j) set(j, input_pos(mv[j]));
    // MFEM triangle edges (0,1), (1,2), (2,0) of its own vertex order; the input edges k = (k, k+1 mod 3)
    for (int j = 0; j < 3; ++j) {
      const int a = input_pos(mv[j]), b = input_pos(mv[(j + 1) % 3]);
      int k = -1;
      for (int q = 0; q < 3; ++q)
        if ((a == q && b == (q + 1) % 3) || (b == q && a == (q + 1) % 3)) k = q;
      set(3 + j, 3 + k);
    }
  }
  // every element map must stay invertible
  for (int e = 0; e < ne; ++e) {
    mfem::ElementTransformation* T = m.GetElementTransformation(e);
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 6);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      T->SetIntPoint(&ir.IntPoint(q));
      AXF_REQUIRE(T->Weight() > 0, strcat("mesh_from_arrays: the P2 geometry of element ", e,
                                          " has a non-positive Jacobian"), "p2_nodes",
                  "Edge nodes must lie close enough to the chord midpoints.");
    }
  }
  return m;
}

std::vector<double> element_p2_nodes(const mfem::Mesh& mesh0) {
  mfem::Mesh& mesh = const_cast<mfem::Mesh&>(mesh0);
  static const double ref[6][2] = {{0, 0}, {1, 0}, {0, 1}, {0.5, 0}, {0.5, 0.5}, {0, 0.5}};
  std::vector<double> out;
  out.reserve(12 * mesh.GetNE());
  for (int e = 0; e < mesh.GetNE(); ++e) {
    AXF_REQUIRE(mesh.GetElementBaseGeometry(e) == mfem::Geometry::TRIANGLE, "element_p2_nodes: triangles only");
    mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
    for (const auto& r : ref) {
      mfem::IntegrationPoint ip;
      ip.Set2(r[0], r[1]);
      double x[3];
      mfem::Vector X(x, mesh.SpaceDimension());
      T->Transform(ip, X);
      out.push_back(x[0]);
      out.push_back(x[1]);
    }
  }
  return out;
}

mfem::Mesh make_periodic_mesh(const mfem::Mesh& mesh0, const std::vector<PeriodicPair>& pairs) {
  mfem::Mesh& mesh = const_cast<mfem::Mesh&>(mesh0);
  const int sdim = mesh.SpaceDimension(), nv = mesh.GetNV();
  mfem::Vector pmin, pmax;
  mesh.GetBoundingBox(pmin, pmax);
  const double tol = 1e-10 * pmax.DistanceTo(pmin);
  std::vector<int> v2v(nv);
  for (int v = 0; v < nv; ++v) v2v[v] = v;
  auto verts_of = [&](int attr) {
    std::set<int> s;
    mfem::Array<int> vs;
    for (int be = 0; be < mesh.GetNBE(); ++be)
      if (mesh.GetBdrAttribute(be) == attr) {
        mesh.GetBdrElementVertices(be, vs);
        for (int v : vs) s.insert(v);
      }
    return std::vector<int>(s.begin(), s.end());
  };
  for (const auto& pr : pairs) {
    AXF_REQUIRE(static_cast<int>(pr.translation.size()) == sdim,
                strcat("periodic pair (", pr.attribute, ", ", pr.partner, "): the translation needs ", sdim,
                       " components"), "periodic");
    const auto va = verts_of(pr.attribute), vp = verts_of(pr.partner);
    AXF_REQUIRE(!va.empty() && va.size() == vp.size(),
                strcat("periodic pair (", pr.attribute, ", ", pr.partner, "): the boundaries have ", va.size(),
                       " and ", vp.size(), " vertices"), "periodic",
                "Both attributes must exist and carry matching vertex distributions.");
    // order both boundaries along the translation-normal direction (used only by the shift defect)
    std::vector<int> match(vp.size(), -1);
    for (size_t i = 0; i < vp.size(); ++i) {
      const double* xp = mesh.GetVertex(vp[i]);
      for (size_t j = 0; j < va.size(); ++j) {
        const double* xa = mesh.GetVertex(va[j]);
        double d = 0;
        for (int c = 0; c < sdim; ++c) d += std::pow(xp[c] - pr.translation[c] - xa[c], 2);
        if (std::sqrt(d) <= tol) {
          match[i] = va[j];
          break;
        }
      }
      AXF_REQUIRE(match[i] >= 0,
                  strcat("periodic pair (", pr.attribute, ", ", pr.partner, "): vertex ", vp[i], " of attribute ",
                         pr.partner, " has no image on attribute ", pr.attribute, " at x - translation"),
                  "periodic", "Check the translation and that the two boundaries are meshed identically.");
    }
    for (size_t i = 0; i < vp.size(); ++i) v2v[vp[i]] = match[i];
  }
  // chains (doubly periodic corners): follow to a fixed point
  for (int v = 0; v < nv; ++v) {
    int w = v, guard = 0;
    while (v2v[w] != w && guard++ < 8) w = v2v[w];
    v2v[v] = w;
  }
  return mfem::Mesh::MakePeriodic(mesh, v2v);
}

}  // namespace axf::fluids
