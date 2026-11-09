#include "axionfem/topology.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <set>

#include "axionfem/core.hpp"

namespace axf::topo {

namespace {
// Sparse linear combination of parameters.
using Combo = std::vector<std::pair<int, double>>;

void axpy(Combo& y, double a, const Combo& x) {
  for (auto& [k, v] : x) {
    // A plain copy for the lambda: capturing a structured binding is C++20, and Apple
    // clang rejects it in C++17 ("reference to local binding declared in enclosing function").
    const int key = k;
    auto it = std::find_if(y.begin(), y.end(), [&](auto& p) { return p.first == key; });
    if (it == y.end()) y.push_back({k, a * v});
    else it->second += a * v;
  }
  y.erase(std::remove_if(y.begin(), y.end(), [](auto& p) { return std::fabs(p.second) < 1e-14; }), y.end());
}
}  // namespace

long euler_characteristic(const mfem::Mesh& m, const std::vector<char>& in) {
  std::set<int> V, E, F;
  long T = 0;
  mfem::Array<int> a, o;
  for (int e = 0; e < m.GetNE(); ++e) {
    if (!in[e]) continue;
    ++T;
    m.GetElementVertices(e, a);
    V.insert(a.begin(), a.end());
    m.GetElementEdges(e, a, o);
    E.insert(a.begin(), a.end());
    m.GetElementFaces(e, a, o);
    F.insert(a.begin(), a.end());
  }
  return static_cast<long>(V.size()) - static_cast<long>(E.size()) + static_cast<long>(F.size()) - T;
}

Cohomology cohomology_h1(const mfem::Mesh& m, const std::vector<char>& in) {
  AXF_REQUIRE(m.Dimension() == 3, "cohomology_h1 expects a 3D mesh");
  const int ned = m.GetNEdges();
  mfem::Array<int> a, o;
  // edges, faces of the subset
  std::vector<char> edge_in(ned, 0);
  std::set<int> faces;
  for (int e = 0; e < m.GetNE(); ++e) {
    if (!in[e]) continue;
    m.GetElementEdges(e, a, o);
    for (int x : a) edge_in[x] = 1;
    m.GetElementFaces(e, a, o);
    faces.insert(a.begin(), a.end());
  }
  // face -> (edges, orientations)
  std::vector<std::vector<int>> fe;
  std::vector<std::vector<int>> fo;
  fe.reserve(faces.size());
  for (int f : faces) {
    const_cast<mfem::Mesh&>(m).GetFaceEdges(f, a, o);
    fe.emplace_back(a.begin(), a.end());
    fo.emplace_back(o.begin(), o.end());
  }
  // spanning forest on vertices through subset edges (BFS)
  std::map<int, std::vector<std::pair<int, int>>> adj;  // vertex -> (neighbour, edge)
  for (int ed = 0; ed < ned; ++ed) {
    if (!edge_in[ed]) continue;
    m.GetEdgeVertices(ed, a);
    adj[a[0]].push_back({a[1], ed});
    adj[a[1]].push_back({a[0], ed});
  }
  std::vector<char> tree(ned, 0);
  std::set<int> seen;
  for (auto& [v0, nb] : adj) {
    if (seen.count(v0)) continue;
    std::deque<int> q{v0};
    seen.insert(v0);
    while (!q.empty()) {
      int v = q.front();
      q.pop_front();
      for (auto [w, ed] : adj[v]) {
        if (seen.count(w)) continue;
        seen.insert(w);
        tree[ed] = 1;
        q.push_back(w);
      }
    }
  }
  // edge values as parameter combinations; tree edges are 0 (determined)
  std::vector<char> known(ned, 0);
  std::vector<Combo> val(ned);
  for (int ed = 0; ed < ned; ++ed) if (edge_in[ed] && tree[ed]) known[ed] = 1;
  // edge -> faces incidence
  std::vector<std::vector<int>> ef(ned);
  for (size_t f = 0; f < fe.size(); ++f) for (int ed : fe[f]) ef[ed].push_back(static_cast<int>(f));
  std::vector<int> unknown_count(fe.size(), 0);
  for (size_t f = 0; f < fe.size(); ++f) for (int ed : fe[f]) if (!known[ed]) ++unknown_count[f];
  std::vector<char> used(fe.size(), 0);
  std::deque<int> ready;
  for (size_t f = 0; f < fe.size(); ++f) if (unknown_count[f] == 1) ready.push_back(static_cast<int>(f));
  int nparam = 0;
  int remaining = 0;
  for (int ed = 0; ed < ned; ++ed) if (edge_in[ed] && !known[ed]) ++remaining;

  auto set_known = [&](int ed) {
    known[ed] = 1;
    --remaining;
    for (int f : ef[ed]) {
      if (--unknown_count[f] == 1 && !used[f]) ready.push_back(f);
    }
  };
  while (remaining > 0) {
    while (!ready.empty()) {
      int f = ready.front();
      ready.pop_front();
      if (used[f] || unknown_count[f] != 1) continue;
      // the single unknown edge u: sum_k o_k h_k = 0  ->  h_u = -(1/o_u) sum_{k != u} o_k h_k
      int iu = -1;
      for (size_t k = 0; k < fe[f].size(); ++k) if (!known[fe[f][k]]) iu = static_cast<int>(k);
      Combo c;
      for (size_t k = 0; k < fe[f].size(); ++k) {
        if (static_cast<int>(k) == iu) continue;
        axpy(c, -static_cast<double>(fo[f][k]) / fo[f][iu], val[fe[f][k]]);
      }
      used[f] = 1;
      int u = fe[f][iu];
      val[u] = c;
      set_known(u);
    }
    if (remaining == 0) break;
    // stalled: introduce a parameter on an unknown edge of a face with the fewest unknowns
    int best_e = -1, best_n = 1 << 30;
    for (size_t f = 0; f < fe.size(); ++f) {
      if (used[f] || unknown_count[f] < 2 || unknown_count[f] >= best_n) continue;
      for (int ed : fe[f]) if (!known[ed]) { best_e = ed; best_n = unknown_count[f]; break; }
    }
    if (best_e < 0) {  // isolated unknown edges (no faces): free parameters
      for (int ed = 0; ed < ned; ++ed) if (edge_in[ed] && !known[ed]) { best_e = ed; break; }
    }
    val[best_e] = {{nparam++, 1.0}};
    set_known(best_e);
  }
  Cohomology out;
  out.parameters = nparam;
  if (nparam == 0) return out;
  // constraints from unused faces
  std::vector<std::vector<double>> rows;
  for (size_t f = 0; f < fe.size(); ++f) {
    if (used[f]) continue;
    Combo c;
    for (size_t k = 0; k < fe[f].size(); ++k) axpy(c, fo[f][k], val[fe[f][k]]);
    if (c.empty()) continue;
    std::vector<double> r(nparam, 0.0);
    for (auto& [kk, v] : c) r[kk] = v;
    rows.push_back(r);
  }
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(std::max<size_t>(rows.size(), 1), nparam);
  for (size_t i = 0; i < rows.size(); ++i) for (int k = 0; k < nparam; ++k) C(i, k) = rows[i][k];
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullV);
  const auto& sv = svd.singularValues();
  double smax = sv.size() ? sv(0) : 0.0;
  int rank = 0;
  for (int i = 0; i < sv.size(); ++i) if (sv(i) > 1e-10 * std::max(smax, 1.0)) ++rank;
  if (rows.empty()) rank = 0;
  const Eigen::MatrixXd& V = svd.matrixV();
  out.betti1 = nparam - rank;
  for (int j = rank; j < nparam; ++j) {
    std::vector<double> h(ned, 0.0);
    for (int ed = 0; ed < ned; ++ed) {
      if (!edge_in[ed]) continue;
      double s = 0;
      for (auto& [k, v] : val[ed]) s += v * V(k, j);
      h[ed] = s;
    }
    out.cochains.push_back(std::move(h));
  }
  return out;
}

}  // namespace axf::topo
