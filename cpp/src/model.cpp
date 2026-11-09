#include "axionfem/model.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>
#include "axionfem/omp_shim.hpp"
#include "axionfem/adapt.hpp"

#include "axionfem/conductor.hpp"
#include "axionfem/core.hpp"
#include "axionfem/cost.hpp"
#include "axionfem/hts.hpp"
#include "axionfem/mesh_quality.hpp"
#include "axionfem/scan.hpp"
#include "axionfem/units.hpp"

namespace axf {

using axi::cplx;

using namespace axi;
using units::mu0;
using units::pi;

namespace {

struct ElemInfo {
  std::string region;
  const spec::Material* mat = nullptr;
  const spec::Winding* winding = nullptr;  // winding defined on this region
  bool port = false;
};

bool is_sc(const ElemInfo& x) { return x.mat->kind == spec::MaterialKind::Superconductor; }
bool is_cond(const ElemInfo& x) { return x.mat->kind == spec::MaterialKind::Conductor; }

std::vector<ElemInfo> element_info(const spec::Design& d, const Space& sp,
                                   const std::function<std::string(int)>& region_name) {
  const int ne = sp.mesh.GetNE();
  std::vector<ElemInfo> info(ne);
  std::map<std::string, const spec::Winding*> wind_of;
  for (auto& w : d.windings) wind_of[w.region] = &w;
  static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
  for (int e = 0; e < ne; ++e) {
    ElemInfo& x = info[e];
    x.region = region_name(sp.mesh.GetAttribute(e));
    if (x.region == "air") {
      x.mat = &vacuum;
    } else {
      const spec::Region* g = d.find_region(x.region);
      x.mat = &d.materials.at(g->material);
      auto it = wind_of.find(x.region);
      if (it != wind_of.end()) x.winding = it->second;
      x.port = d.pickup && d.pickup->kind == spec::Pickup::Kind::Poloidal && d.pickup->port == x.region;
    }
  }
  return info;
}

// permeability of an element for linear (RF) problems; iron: its initial mu_r unless a field-dependent law is given
double mu_static_of(const ElemInfo& x) {
  const auto* m = x.mat;
  if (m->kind == spec::MaterialKind::Linear || m->kind == spec::MaterialKind::Iron) return mu0 * m->mu_r;
  return mu0;
}

// Lattice of reference points on a triangle (includes vertices and edges).
const mfem::IntegrationRule& lattice(int k) {
  static std::map<int, mfem::IntegrationRule> cache;
  auto it = cache.find(k);
  if (it != cache.end()) return it->second;
  mfem::IntegrationRule ir;
  for (int i = 0; i <= k; ++i)
    for (int j = 0; i + j <= k; ++j) {
      mfem::IntegrationPoint ip;
      ip.Set2(double(i) / k, double(j) / k);
      ip.weight = 0;
      ir.Append(ip);
    }
  return cache[k] = ir;
}

struct Body {
  std::vector<std::string> names;  // regions/curves
  std::vector<int> elements;
  std::vector<int> curve_attrs;
  bool on_axis = false;
};

// Order a set of mesh faces (segments) into connected chains, so that a distributed surface load
// can carry an arc-length coordinate that runs along the surface.  Open chains (sheet edges) are
// started at their free end; closed loops start anywhere.  At a junction of more than two faces
// the walk continues along an arbitrary unused face, so the arc length may jump there.
std::vector<std::vector<int>> order_face_chains(const Space& sp, const std::vector<int>& faces) {
  const mfem::Mesh& m = sp.mesh;
  std::map<int, std::vector<int>> v2f;
  mfem::Array<int> fv;
  for (int f : faces) {
    m.GetFaceVertices(f, fv);
    v2f[fv[0]].push_back(f);
    v2f[fv[1]].push_back(f);
  }
  std::set<int> used;
  std::vector<std::vector<int>> chains;
  auto walk = [&](int f0, int from_vertex) {
    std::vector<int> chain;
    int f = f0, v = from_vertex;
    for (;;) {
      used.insert(f);
      chain.push_back(f);
      m.GetFaceVertices(f, fv);
      const int w = (fv[0] == v) ? fv[1] : fv[0];
      int next = -1;
      for (int g : v2f[w]) {
        if (!used.count(g)) { next = g; break; }
      }
      if (next < 0) break;
      f = next;
      v = w;
    }
    chains.push_back(std::move(chain));
  };
  for (int f : faces) {  // open chains first, from their free end
    if (used.count(f)) continue;
    m.GetFaceVertices(f, fv);
    for (int k = 0; k < 2; ++k) {
      if (v2f[fv[k]].size() == 1) { walk(f, fv[k]); break; }
    }
  }
  for (int f : faces) {  // the rest are closed loops
    if (used.count(f)) continue;
    m.GetFaceVertices(f, fv);
    walk(f, fv[0]);
  }
  return chains;
}

// One sample of a distributed surface load: a piece of the body boundary of length `dl`, its
// azimuthally integrated force (N) and its length-averaged traction (Pa).
struct LoadSample {
  double r = 0, z = 0, s = 0, dl = 0;  // midpoint, arc length at the midpoint, length
  double nr = 0, nz = 0;               // outward unit normal of the body (length-averaged)
  double tr = 0, tz = 0;               // traction, length-averaged (Pa)
  double Fr = 0, Fz = 0;               // azimuthally integrated force on this piece (N)
};

// Superconducting (and, for RF problems, perfectly conducting) bodies:
// connected components of element sets plus sheets sharing vertices.
std::vector<Body> find_bodies(const Space& sp, const MeshBundle& mb, const std::vector<ElemInfo>& info,
                              const spec::Design& d, bool include_conductors, std::vector<int>& body_of_elem,
                              std::map<int, int>& body_of_curve_attr) {
  const int ne = sp.mesh.GetNE();
  auto in = [&](int e) { return is_sc(info[e]) || (include_conductors && is_cond(info[e])); };
  // Thin magnetic layers separate what lies on their two sides (docs/physics/thin_magnetic_layer.md): parts that
  // meet only across a layer face, or at a layer vertex from different sides, are different bodies.
  // Without layers every element is in fan 0 and nothing changes.
  std::vector<char> layer_face;
  {
    std::set<int> layer_attr;
    for (auto& c : d.curves)
      if (c.thin_layer()) layer_attr.insert(mb.attr_of_curve.at(c.name));
    if (!layer_attr.empty()) {
      layer_face.assign(sp.mesh.GetNumFaces(), 0);
      for (int be = 0; be < sp.mesh.GetNBE(); ++be)
        if (layer_attr.count(sp.mesh.GetBdrAttribute(be))) layer_face[sp.mesh.GetBdrElementFaceIndex(be)] = 1;
    }
  }
  auto crosses = [&](int f) { return layer_face.empty() || !layer_face[f]; };
  int nc = 0;
  // One region on both sides of a layer face stays one body: its copies on the two sides are then tied
  // together, so the region cuts the layer as a crossing sheet does.
  auto joins = [&](int f) {
    if (crosses(f)) return true;
    int e1 = -1, e2 = -1;
    sp.mesh.GetFaceElements(f, &e1, &e2);
    return e1 >= 0 && e2 >= 0 && sp.mesh.GetAttribute(e1) == sp.mesh.GetAttribute(e2);
  };
  std::vector<int> comp = components(sp, in, joins, nc);
  // fan of (vertex, element) at the vertices of layer faces: the elements around the vertex joined across
  // faces that are not layer faces (as the DOF copies are made in PoloidalProblem::set_thin_layers)
  std::map<std::pair<int, int>, int> fan_of;
  std::set<int> end_verts;  // layer vertices with one fan (a layer's end): no vertex unions there
  mfem::Array<int> v;
  if (!layer_face.empty()) {
    std::map<int, std::vector<int>> star;
    std::set<int> lverts;
    mfem::Array<int> fv;
    for (int f = 0; f < sp.mesh.GetNumFaces(); ++f) {
      if (!layer_face[f]) continue;
      sp.mesh.GetFaceVertices(f, fv);
      for (int x : fv) lverts.insert(x);
    }
    for (int e = 0; e < ne; ++e) {
      sp.mesh.GetElementVertices(e, v);
      for (int x : v)
        if (lverts.count(x)) star[x].push_back(e);
    }
    for (auto& [x, els] : star) {
      std::set<int> in_star(els.begin(), els.end());
      int nf = 0;
      for (int e : els) {
        if (fan_of.count({x, e})) continue;
        std::vector<int> stack{e};
        fan_of[{x, e}] = nf;
        while (!stack.empty()) {
          const int y = stack.back();
          stack.pop_back();
          for (auto [nb, f] : sp.nbr[y]) {
            if (nb < 0 || !in_star.count(nb) || fan_of.count({x, nb}) || !crosses(f)) continue;
            fan_of[{x, nb}] = nf;
            stack.push_back(nb);
          }
        }
        ++nf;
      }
      if (nf == 1) end_verts.insert(x);
    }
  }
  auto fan = [&](int x, int e) {
    auto it = fan_of.find({x, e});
    return it == fan_of.end() ? 0 : it->second;
  };
  // union-find over components and curves
  std::vector<int> parent(nc);
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int a) { return parent[a] == a ? a : parent[a] = find(parent[a]); };
  auto unite = [&](int a, int b) { a = find(a); b = find(b); if (a != b) parent[std::max(a, b)] = std::min(a, b); };
  std::map<std::pair<int, int>, int> vert_comp;  // (vertex, fan) -> node
  for (int e = 0; e < ne; ++e) {
    if (comp[e] < 0) continue;
    sp.mesh.GetElementVertices(e, v);
    for (int x : v) vert_comp[{x, fan(x, e)}] = comp[e];
  }
  std::map<int, int> curve_node;  // curve attr -> node
  for (auto& c : d.curves) {
    auto k = d.materials.at(c.material).kind;
    if (!(k == spec::MaterialKind::Superconductor || (include_conductors && k == spec::MaterialKind::Conductor))) continue;
    int attr = mb.attr_of_curve.at(c.name);
    int node = static_cast<int>(parent.size());
    parent.push_back(node);
    curve_node[attr] = node;
  }
  for (int be = 0; be < sp.mesh.GetNBE(); ++be) {
    auto it = curve_node.find(sp.mesh.GetBdrAttribute(be));
    if (it == curve_node.end()) continue;
    sp.mesh.GetBdrElementVertices(be, v);
    int fe[2] = {-1, -1};
    sp.mesh.GetFaceElements(sp.mesh.GetBdrElementFaceIndex(be), &fe[0], &fe[1]);
    for (int x : v) {
      // Parts that meet only at a layer's end are not joined there: the layer's end is a contact point the
      // model cannot resolve, so two bodies meeting there are refused when their DOFs are tied.
      if (end_verts.count(x)) continue;
      for (int e : fe) {  // a sheet meets a vertex on the side(s) of its face's elements
        if (e < 0) continue;
        const std::pair<int, int> key{x, fan(x, e)};
        auto jt = vert_comp.find(key);
        if (jt != vert_comp.end()) unite(it->second, jt->second);
        else vert_comp[key] = it->second;
      }
    }
  }
  std::map<int, int> root_body;
  std::vector<Body> bodies;
  auto body_of = [&](int node) {
    int rt = find(node);
    auto it = root_body.find(rt);
    if (it != root_body.end()) return it->second;
    root_body[rt] = static_cast<int>(bodies.size());
    bodies.emplace_back();
    return static_cast<int>(bodies.size()) - 1;
  };
  body_of_elem.assign(ne, -1);
  std::vector<std::set<std::string>> names;
  for (int e = 0; e < ne; ++e) {
    if (comp[e] < 0) continue;
    int b = body_of(comp[e]);
    body_of_elem[e] = b;
    bodies[b].elements.push_back(e);
  }
  for (auto& [attr, node] : curve_node) {
    int b = body_of(node);
    body_of_curve_attr[attr] = b;
    bodies[b].curve_attrs.push_back(attr);
  }
  names.resize(bodies.size());
  for (int e = 0; e < ne; ++e) if (body_of_elem[e] >= 0) names[body_of_elem[e]].insert(info[e].region);
  for (auto& c : d.curves) {
    auto it = mb.attr_of_curve.find(c.name);
    if (it != mb.attr_of_curve.end() && body_of_curve_attr.count(it->second)) names[body_of_curve_attr[it->second]].insert(c.name);
  }
  for (size_t b = 0; b < bodies.size(); ++b) bodies[b].names.assign(names[b].begin(), names[b].end());
  // axis contact
  for (int e = 0; e < ne; ++e) {
    if (body_of_elem[e] < 0) continue;
    for (int dd : sp.edofs(e)) if (sp.r[dd] == 0.0) bodies[body_of_elem[e]].on_axis = true;
  }
  mfem::Array<int> bd;
  for (int be = 0; be < sp.mesh.GetNBE(); ++be) {
    auto it = body_of_curve_attr.find(sp.mesh.GetBdrAttribute(be));
    if (it == body_of_curve_attr.end()) continue;
    sp.fes->GetBdrElementDofs(be, bd);
    for (int dd : bd) if (sp.r[dd] == 0.0) bodies[it->second].on_axis = true;
  }
  return bodies;
}

const spec::Superconductor* sc_state(const spec::Design& d, const std::vector<std::string>& names) {
  const spec::Superconductor* found = nullptr;
  for (auto& s : d.superconductors) {
    if (std::find(names.begin(), names.end(), s.body) == names.end()) continue;
    if (found && (found->azimuthal.kind != s.azimuthal.kind || found->azimuthal.value != s.azimuthal.value ||
                  found->poloidal.kind != s.poloidal.kind || found->poloidal.value != s.poloidal.value ||
                  found->azimuthal_slit != s.azimuthal_slit)) {
      throw Error("superconducting parts '" + found->body + "' and '" + s.body +
                      "' touch (one body) but have different states",
                  "/superconductors", "Give touching parts identical states or separate them.");
    }
    found = &s;
  }
  return found;
}

// Tie current-free cavities enclosed by a single body to that body's master.
void tie_enclosed_cavities(const Space& sp, std::vector<ARole>& roles, const std::map<int, int>& sheet_master) {
  const int ne = sp.mesh.GetNE();
  auto in = [&](int e) { return !roles[e].master; };
  auto cross = [&](int f) { return !sheet_master.count(sp.face_bdr_attr[f]); };
  int nc = 0;
  std::vector<int> comp = components(sp, in, cross, nc);
  std::vector<char> open(nc, 0), active(nc, 0);
  std::vector<std::set<int>> adj(nc);
  for (int e = 0; e < ne; ++e) {
    int c = comp[e];
    if (c < 0) continue;
    const ARole& ro = roles[e];
    if (ro.J != 0.0 || ro.bh || ro.nu_at || std::fabs(ro.nu * mu0 - 1.0) > 1e-12) active[c] = 1;
    for (auto [n, f] : sp.nbr[e]) {
      if (roles[n].master) adj[c].insert(roles[n].master_id);
      auto it = sheet_master.find(sp.face_bdr_attr[f]);
      if (it != sheet_master.end()) adj[c].insert(it->second);
    }
  }
  for (int f = 0; f < sp.mesh.GetNumFaces(); ++f) {
    int a = sp.face_bdr_attr[f];
    if (a != kAxisAttr && a != kOuterAttr) continue;
    int e1, e2;
    sp.mesh.GetFaceElements(f, &e1, &e2);
    if (e1 >= 0 && comp[e1] >= 0) open[comp[e1]] = 1;
  }
  // sheets bounding a component also count as adjacency (both sides of the sheet face)
  for (int f = 0; f < sp.mesh.GetNumFaces(); ++f) {
    auto it = sheet_master.find(sp.face_bdr_attr[f]);
    if (it == sheet_master.end()) continue;
    int e1, e2;
    sp.mesh.GetFaceElements(f, &e1, &e2);
    if (e1 >= 0 && comp[e1] >= 0) adj[comp[e1]].insert(it->second);
    if (e2 >= 0 && comp[e2] >= 0) adj[comp[e2]].insert(it->second);
  }
  for (int e = 0; e < ne; ++e) {
    int c = comp[e];
    if (c < 0 || open[c] || active[c] || adj[c].size() != 1) continue;
    roles[e].master = true;
    roles[e].master_id = *adj[c].begin();
  }
}

}  // namespace

AxiModel::AxiModel(spec::Design d) : d_(std::move(d)) {
  AXF_REQUIRE(d_.symmetry != "xsec", "a cross-section (\"symmetry\": \"xsec\") design is solved by XsecModel, not "
              "by the axisymmetric engine", "/symmetry", "Use Design.solve() (Python) or `axf solve`, which dispatch on "
              "the symmetry.");
}
AxiModel::~AxiModel() = default;

std::string AxiModel::region_name(int attr) const {
  auto it = mb_.region_of_attr.find(attr);
  return it == mb_.region_of_attr.end() ? std::string("?") : it->second;
}

std::string AxiModel::mesh_note(bool thrown) {
  const mesh_quality::Summary q = mesh_quality::scan(*mb_.mesh, 5);
  auto region = [this](int a) {
    auto it = mb_.region_of_attr.find(a);
    return it == mb_.region_of_attr.end() ? strcat("attribute ", a) : it->second;
  };
  mesh_quality::DescribeOptions opt;
  opt.three_d = false;
  if (thrown) {
    opt.listed_in.clear();  // a thrown error writes no results
  } else {
    mb_.info["worst_elements"] = mesh_quality::to_json(q, region);
  }
  return mesh_quality::describe(q, region, opt);
}

namespace {

// AXF_AXI_MAX_NEWTON caps the static poloidal Newton iterations (default 60, PoloidalProblem::solve's); tests use it
// to drive the non-converged path and its mesh check. Checked before meshing (AxiModel::mesh) and read at the solve.
int axi_max_newton() {
  const char* s = std::getenv("AXF_AXI_MAX_NEWTON");
  if (!s || !*s) return 60;
  char* end = nullptr;
  errno = 0;
  const long v = std::strtol(s, &end, 10);
  AXF_REQUIRE(end != s && *end == '\0' && errno != ERANGE && v >= 1 && v <= 1000,
              strcat("AXF_AXI_MAX_NEWTON='", s, "' is not an integer from 1 to 1000"), "",
              "Unset it, or give the cap on the static poloidal Newton iterations.");
  return static_cast<int>(v);
}

}  // namespace

void AxiModel::run(const SolveOptions& o) {
  const auto& ad = d_.mesh.adapt;
  if (ad.cycles <= 0) {
    mesh(o);
    solve(o);
    return;
  }
  // adaptive refinement: re-mesh with target sizes from the error indicators until Psi, L_p and W_mag settle
  SolveOptions oc = o;
  json hist = json::array();
  bool converged = false;
  std::string why;
  for (int cyc = 0;; ++cyc) {
    mesh(oc);
    solve(oc);
    const std::vector<double> eta2 = error_indicators();
    double tot = 0;
    for (double x : eta2) tot += x;
    json row = adapt_row(cyc, sp_->mesh.GetNE(), static_cast<long>(sp_->ndof()), results_, std::sqrt(tot));
    if (!hist.empty()) {
      const double change = adapt_change(hist.back(), row, ad.watch);
      row["max_relative_change"] = change;
      if (change < ad.tol) converged = true;
    }
    hist.push_back(row);
    log(1, strcat("adaptive cycle ", cyc, ": ", row["elements"].get<int>(), " elements, estimated error ",
                  std::sqrt(tot), row.contains("max_relative_change")
                                      ? strcat(", change ", row["max_relative_change"].get<double>()) : std::string()));
    if (converged) break;
    if (cyc >= ad.cycles) {
      why = "adaptive refinement stopped after " + std::to_string(ad.cycles) + " cycles before the quantities settled";
      break;
    }
    // target sizes: equidistribute a reduced error, error per element ~ h^(p+1) in 2D
    const Space& sp = *sp_;
    const int ne = sp.mesh.GetNE(), p = sp.order;
    const double eta_t2 = ad.reduction * ad.reduction * tot / std::max(ne, 1);
    std::vector<double> hnew(ne);
    double projected = 0;
    for (int e = 0; e < ne; ++e) {
      const double h = std::sqrt(2.0 * element_area(sp, e));
      double f = eta2[e] > 0 ? std::pow(eta_t2 / eta2[e], 0.5 / (p + 1)) : 1.0;
      f = std::clamp(f, 0.25, 1.0);
      hnew[e] = h * f;
      projected += 1.0 / (f * f);
    }
    if (projected > ad.max_elements) {
      const double g = std::sqrt(projected / ad.max_elements);
      for (double& h : hnew) h *= g;
      why = "adaptive refinement limited by mesh.adapt.max_elements";
    }
    // node values: smallest target of the adjacent elements; triangles with their vertex coordinates
    std::vector<double> hv(sp.mesh.GetNV(), 1e300);
    mfem::Array<int> vv;
    for (int e = 0; e < ne; ++e) {
      sp.mesh.GetElementVertices(e, vv);
      for (int v : vv) hv[v] = std::min(hv[v], hnew[e]);
    }
    const std::string path = oc.workdir + "/adapt_size_" + std::to_string(cyc) + ".bin";
    {
      std::ofstream f(path, std::ios::binary);
      const int32_t nn = 3;
      const int64_t cnt = ne;
      f.write(reinterpret_cast<const char*>(&nn), sizeof nn);
      f.write(reinterpret_cast<const char*>(&cnt), sizeof cnt);
      for (int e = 0; e < ne; ++e) {
        sp.mesh.GetElementVertices(e, vv);
        double rec[12];
        for (int k = 0; k < 3; ++k) {
          const double* x = sp.mesh.GetVertex(vv[k]);
          rec[3 * k] = x[0];
          rec[3 * k + 1] = x[1];
          rec[3 * k + 2] = 0.0;
          rec[9 + k] = hv[vv[k]];
        }
        f.write(reinterpret_cast<const char*>(rec), sizeof rec);
      }
    }
    oc.size_field_file = path;
  }
  results_["adapt"] = {{"converged", converged}, {"tolerance", ad.tol}, {"watch", ad.watch}, {"history", hist}};
  if (!converged && !why.empty()) results_["warnings"].push_back(why);
}

std::vector<double> AxiModel::error_indicators() const {
  const Space& sp = *sp_;
  const mfem::Mesh& m = sp.mesh;
  const int ne = m.GetNE();
  // field components per point: static (Br, Bz, Bphi) and pickup per ampere (pBr, pBz, pBphi)
  auto comps = [&](int e, const PointEval& pe, double v[6]) {
    Sample s = eval_pe(e, pe);
    v[0] = s.Br;
    v[1] = s.Bz;
    v[2] = s.Bphi;
    v[3] = s.pBr;
    v[4] = s.pBz;
    v[5] = s.pBphi;
  };
  // recovered vertex values per (vertex, region attribute): averages of the element values at the vertex
  std::map<std::pair<int, int>, std::array<double, 7>> rec;
  mfem::Array<int> vv;
  PointEval pe;
  const mfem::IntegrationRule* verts = mfem::Geometries.GetVertices(mfem::Geometry::TRIANGLE);
  for (int e = 0; e < ne; ++e) {
    m.GetElementVertices(e, vv);
    const int a = m.GetAttribute(e);
    for (int k = 0; k < 3; ++k) {
      sp.eval_at(e, verts->IntPoint(k), pe);
      double v[6];
      comps(e, pe, v);
      auto& r = rec[{vv[k], a}];
      for (int i = 0; i < 6; ++i) r[i] += v[i];
      r[6] += 1;
    }
  }
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * sp.order + 2);
  std::vector<double> el(ne), ep(ne);
  double nl = 0, np = 0;
  for (int e = 0; e < ne; ++e) {
    m.GetElementVertices(e, vv);
    const int a = m.GetAttribute(e);
    std::array<double, 7> rv[3];
    for (int k = 0; k < 3; ++k) rv[k] = rec[{vv[k], a}];
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir.IntPoint(q);
      sp.eval_at(e, ip, pe);
      double v[6];
      comps(e, pe, v);
      const double lam[3] = {1 - ip.x - ip.y, ip.x, ip.y};
      const double w = ip.weight * element_area(sp, e) * 2 * pe.r;  // r dA (reference triangle area 1/2)
      for (int i = 0; i < 6; ++i) {
        double star = 0;
        for (int k = 0; k < 3; ++k) star += lam[k] * rv[k][i] / std::max(rv[k][6], 1.0);
        const double d = v[i] - star;
        (i < 3 ? el[e] : ep[e]) += d * d * w;
        (i < 3 ? nl : np) += v[i] * v[i] * w;
      }
    }
  }
  std::vector<double> eta2(ne);
  for (int e = 0; e < ne; ++e) eta2[e] = (nl > 0 ? el[e] / nl : 0.0) + (np > 0 ? ep[e] / np : 0.0);
  return eta2;
}

void AxiModel::mesh(const SolveOptions& o) {
  (void)axi_max_newton();  // a bad value is reported before meshing
  MeshOptions mo;
  mo.workdir = o.workdir;
  mo.mesher = o.mesher;
  mo.verbose = o.verbose;
  mo.size_field_file = o.size_field_file;
  mb_ = build_mesh(d_, mo);
  int order = o.order > 0 ? o.order : d_.mesh.order;
  sp_ = std::make_unique<Space>(*mb_.mesh, order);
}

void AxiModel::set_mesh(MeshBundle mb, int order) {
  mb_ = std::move(mb);
  sp_ = std::make_unique<Space>(*mb_.mesh, order > 0 ? order : d_.mesh.order);
}

void AxiModel::freeze_poloidal(std::vector<double> y_lab, std::vector<double> ic_lab, std::vector<double> y_pick,
                               std::vector<double> ic_pick) {
  frozen_ = true;
  frozen_y_lab_ = std::move(y_lab);
  frozen_ic_lab_ = std::move(ic_lab);
  frozen_y_pick_ = std::move(y_pick);
  frozen_ic_pick_ = std::move(ic_pick);
}

// A breakdown deep in an engine (SolverFailure) is rethrown with the mesh check, which names the region.
void AxiModel::solve(const SolveOptions& o) try {
  AXF_REQUIRE(sp_ != nullptr, "call mesh() before solve()");
  mb_.info.erase("worst_elements");  // a check from an earlier solve, or copied with the mesh bundle
  Timer timer;
  const Space& sp = *sp_;
  const int ne = sp.mesh.GetNE();
  I_pick_ = o.pickup_current;

  // ------------------------------------------------------------ element info
  std::vector<ElemInfo> info = element_info(d_, sp, [this](int a) { return region_name(a); });
  // region areas (for winding current densities)
  std::map<std::string, double> area;
  for (int e = 0; e < ne; ++e) area[info[e].region] += element_area(sp, e);

  bh_.clear();
  std::map<std::string, const BHLaw*> bh_of;
  for (auto& [name, m] : d_.materials) {
    if (m.kind == spec::MaterialKind::Iron) {
      bh_.push_back(make_bh(m.bh));
      bh_of[name] = bh_.back().get();
    }
  }
  auto mat_name = [&](int e) -> std::string {
    if (info[e].region == "air") return "vacuum";
    return d_.find_region(info[e].region)->material;
  };

  auto circuit_of = [&](const std::string& name) -> const spec::Circuit* {
    for (auto& c : d_.circuits) if (c.name == name) return &c;
    return nullptr;
  };
  // azimuthal current density of a winding (persistent circuits: solved, 0 here)
  auto winding_J = [&](const spec::Winding& w) -> double {
    if (w.circuit.empty()) return w.ampere_turns / area[w.region];
    const spec::Circuit* c = circuit_of(w.circuit);
    if (c->kind == spec::Circuit::Kind::Current) return w.turns * c->current / area[w.region];
    return 0.0;
  };
  // persistent circuits as solver unknowns; rf: linkage change conserved (0)
  auto circuit_coils = [&](bool rf) {
    std::vector<CircuitCoil> out;
    for (auto& c : d_.circuits) {
      if (c.kind != spec::Circuit::Kind::Persistent) continue;
      CircuitCoil cc;
      cc.name = c.name;
      cc.flux_linkage = rf ? 0.0 : c.flux_linkage;
      for (auto& w : d_.windings) {
        if (w.circuit != c.name) continue;
        CircuitCoil::Part part;
        part.turns = w.turns;
        part.area = area[w.region];
        for (int e = 0; e < ne; ++e) if (info[e].region == w.region) part.elements.push_back(e);
        cc.parts.push_back(part);
      }
      out.push_back(cc);
    }
    return out;
  };
  bg_ = BackgroundField(d_.background);
  const bool has_az = bg_.active() || std::any_of(d_.windings.begin(), d_.windings.end(), [&](auto& w) {
    return w.type == spec::WindingType::Azimuthal && winding_J(w) != 0;
  }) || std::any_of(d_.circuits.begin(), d_.circuits.end(), [](auto& c) {
    return c.kind == spec::Circuit::Kind::Persistent && c.flux_linkage != 0;
  });
  const bool has_sc_flux = std::any_of(d_.superconductors.begin(), d_.superconductors.end(), [](auto& s) {
    return s.azimuthal.value != 0.0;
  });
  const bool has_pol = std::any_of(d_.windings.begin(), d_.windings.end(),
                                   [](auto& w) { return w.type == spec::WindingType::Poloidal && w.ampere_turns != 0; });
  json warnings = json::array();
  {
    // The toroidal-field (G = r H_phi) problems do not model zero-thickness sheets yet.
    const bool g_pickup = d_.pickup && d_.pickup->kind == spec::Pickup::Kind::Poloidal;
    if (!d_.curves.empty() && (has_pol || g_pickup)) {
      const bool layer = d_.curves.front().thin_layer();
      throw Error(strcat(layer ? "thin magnetic layer '" : "zero-thickness sheet '", d_.curves.front().name,
                         "' cannot be used with ",
                         has_pol ? "a poloidal winding (static toroidal field)" : "a poloidal (port) pickup",
                         layer ? ": thin layers are not modelled in the toroidal-field problem yet"
                               : ": sheets are not modelled in the toroidal-field problem yet"),
                  "/curves", "Model the sheet or layer as a thin region (e.g. a frame or rectangle a few mm thick).");
    }
  }
  // thin magnetic layers (docs/physics/thin_magnetic_layer.md)
  std::vector<ThinLayer> layers;
  for (auto& c : d_.curves) {
    if (!c.thin_layer()) continue;
    ThinLayer L;
    L.name = c.name;
    L.attr = mb_.attr_of_curve.at(c.name);
    L.mu_r = d_.materials.at(c.material).mu_r;
    L.thickness = c.thickness;
    L.curve_index = static_cast<int>(&c - d_.curves.data());
    layers.push_back(L);
  }
  auto region_of_elem = [this, &sp](int e) { return region_name(sp.mesh.GetAttribute(e)); };
  if (!layers.empty()) {
    // A superconducting or conducting sheet drawn on the same path as a layer would tie both copies and
    // short the layer, whichever face the conductor really sits on.
    std::map<int, const spec::Curve*> curve_of_attr;
    for (auto& c : d_.curves) curve_of_attr[mb_.attr_of_curve.at(c.name)] = &c;
    std::map<int, std::vector<const spec::Curve*>> on_face;
    for (int be = 0; be < sp.mesh.GetNBE(); ++be) {
      auto it = curve_of_attr.find(sp.mesh.GetBdrAttribute(be));
      if (it != curve_of_attr.end()) on_face[sp.mesh.GetBdrElementFaceIndex(be)].push_back(it->second);
    }
    for (auto& [f, cs] : on_face) {
      const spec::Curve* layer = nullptr;
      const spec::Curve* sheet = nullptr;
      for (auto* c : cs) (c->thin_layer() ? layer : sheet) = c;
      if (!layer || !sheet) continue;
      throw Error(strcat("the sheet '", sheet->name, "' and the thin magnetic layer '", layer->name,
                         "' share mesh faces (the sheet lies on the layer's path): the sheet would short the "
                         "layer, whichever face of the layer the conductor really sits on"),
                  strcat("/curves/", &*layer - d_.curves.data(), "/path"),
                  "Offset the sheet from the layer by about an element, or model the superconductor as a region "
                  "with the layer on its surface.");
    }
  }
  // per-layer report, with a warning where the mesh next to a layer is thinner than ~4 layer thicknesses
  auto layer_report = [&](const PoloidalProblem& p, const ASolution& s, const char* what) {
    json rep = p.thin_layer_report(s);
    const int warn_ratio = thin_layer_warn_ratio(sp.order);
    for (auto& L : rep) {
      const std::string nm = L["name"].get<std::string>();
      const double hr = L["min_h_perp_over_d"].get<double>();
      if (hr > 0 && hr < warn_ratio)
        warnings.push_back(strcat(what, ": the mesh next to thin layer '", nm, "' is only ", [hr] {
                                    char b[32];
                                    std::snprintf(b, sizeof b, "%.3g", hr);
                                    return std::string(b);
                                  }(),
                                  " layer thicknesses high (< ", warn_ratio, " at mesh order ", sp.order,
                                  "); the half-strip term can make the system indefinite there, and the model "
                                  "assumes the layer is thin compared with the resolved field"));
      // saturation: the model is linear (docs/physics/thin_magnetic_layer.md, validity condition 3)
      const spec::Material& mt = d_.materials.at(d_.find_curve(nm)->material);
      const bool given = mt.bh.B_sat > 0 && mt.kind == spec::MaterialKind::Linear && mt.bh.model == "given";
      const double Bsat = given ? mt.bh.B_sat : 0.7;
      const double Bl = L["B_layer_max_T"].get<double>();
      L["B_sat_T"] = Bsat;
      L["B_sat_source"] = given ? "material B_sat" : "default 0.7 T (mu-metal); give the material a B_sat";
      if (Bl > Bsat)
        warnings.push_back(strcat(what, ": thin layer '", nm, "' carries up to ", Bl, " T in the linear model, above ",
                                  given ? "its material's B_sat of " : "the default saturation threshold of ", Bsat,
                                  given ? " T" : " T (mu-metal; give the material a B_sat)",
                                  ": a real layer saturates there, and results that depend on its shielding are "
                                  "not reliable"));
    }
    return rep;
  };

  // ------------------------------------------------------------ static poloidal field
  json static_json;
  if (has_az || has_sc_flux) {
    std::vector<int> body_of_elem;
    std::map<int, int> body_of_curve;
    auto bodies = find_bodies(sp, mb_, info, d_, false, body_of_elem, body_of_curve);
    std::vector<Master> masters;
    for (auto& b : bodies) {
      Master m;
      m.name = b.names.empty() ? "body" : b.names.front();
      const spec::Superconductor* st = sc_state(d_, b.names);
      spec::Constraint c = st ? st->azimuthal : spec::Constraint{};
      if (b.on_axis) {
        m.fixed = true;
        m.psi = 0;
        if (c.value != 0) warnings.push_back("superconductor '" + m.name + "' touches the axis; its flux is fixed to zero");
      } else if (c.kind == spec::Constraint::Kind::Flux) {
        m.fixed = true;
        m.psi = c.value / (2 * pi);
      } else {
        m.current = c.value;
      }
      masters.push_back(m);
    }
    std::vector<ARole> roles(ne);
    for (int e = 0; e < ne; ++e) {
      ARole& ro = roles[e];
      const ElemInfo& x = info[e];
      if (body_of_elem[e] >= 0) {
        ro.master = true;
        ro.master_id = body_of_elem[e];
        continue;
      }
      switch (x.mat->kind) {
        case spec::MaterialKind::Iron: ro.bh = bh_of.at(mat_name(e)); ro.nu = 1.0 / (mu0 * x.mat->mu_r); break;
        case spec::MaterialKind::Linear: ro.nu = 1.0 / (mu0 * x.mat->mu_r); break;
        default: ro.nu = 1.0 / mu0; break;
      }
      if (x.winding && x.winding->type == spec::WindingType::Azimuthal) ro.J = winding_J(*x.winding);
      if (x.winding && !x.winding->circuit.empty()) ro.nu = 1.0 / mu0;  // keep coil regions explicit (not tied)
    }
    tie_enclosed_cavities(sp, roles, body_of_curve);
    a_lab_ = std::make_unique<PoloidalProblem>(sp, roles, masters, body_of_curve);
    if (!layers.empty()) a_lab_->set_thin_layers(layers, region_of_elem);
    if (bg_.active()) {
      // magnetic regions and thin layers respond to the background (scattered-field form)
      a_lab_->set_background_field([this](double r, double z, double& br, double& bz) { bg_.B(r, z, br, bz); });
      AXF_REQUIRE(!std::any_of(roles.begin(), roles.end(), [](const ARole& r) { return r.bh != nullptr; }),
                  "background fields cannot be combined with nonlinear iron yet", "/background");
      std::vector<double> off(sp.ndof());
      bool outside = false;
      for (int i = 0; i < sp.ndof(); ++i) {
        off[i] = bg_.psi(sp.r[i], sp.z[i]);
      }
      for (int e = 0; e < ne; ++e) {
        if (!(roles[e].master || info[e].winding)) continue;
        for (int dd : sp.edofs(e)) if (!bg_.inside(sp.r[dd], sp.z[dd])) outside = true;
      }
      if (outside) warnings.push_back("parts of superconductors, cavities or windings lie outside the background field grid (taken as zero there)");
      a_lab_->set_tie_offset(off);
      if (d_.pickup || !d_.windings.empty())  // a design with neither (an eigenmode design) has no W_mag or cost
        warnings.push_back("a background field is present: W_mag and cost exclude the background magnet");
    }
    std::vector<double> bg_link;  // background linkage of each persistent circuit
    for (auto cc : circuit_coils(false)) {
      bg_link.push_back(0.0);
      if (bg_.active()) {  // linkage of the background with the circuit
        double lam = 0;
        PointEval pe2;
        const RefTable& t2 = sp.table(sp.quad_rule());
        for (auto& part : cc.parts)
          for (int e : part.elements)
            for (int q = 0; q < t2.nq; ++q) {
              sp.eval(e, t2, q, pe2);
              lam += part.turns / part.area * 2 * pi * bg_.psi(pe2.r, pe2.z) * pe2.w;
            }
        cc.flux_linkage -= lam;
        bg_link.back() = lam;
      }
      a_lab_->add_circuit_coil(cc);
    }
    a_lab_->set_open_boundary(kOuterAttr, mb_.domain_radius, 80, mb_.domain_center_z);
    if (frozen_) {
      a_lab_sol_ = a_lab_->evaluate(frozen_y_lab_, frozen_ic_lab_);
    } else {
      log(1, "solving static poloidal field");
      const int max_newton = axi_max_newton();
      a_lab_sol_ = a_lab_->solve(o.verbose, 1e-10, max_newton);
    }
    if (a_lab_sol_.diagnostics.contains("circuits")) {
      // report total linkages (the reduced problem carries the reaction field only)
      auto& cj = a_lab_sol_.diagnostics["circuits"];
      for (size_t c = 0; c < cj.size() && c < bg_link.size(); ++c) {
        if (bg_link[c] == 0.0) continue;
        cj[c]["flux_linkage_Wb"] = cj[c]["flux_linkage_Wb"].get<double>() + bg_link[c];
        cj[c]["target_Wb"] = cj[c]["target_Wb"].get<double>() + bg_link[c];
        cj[c]["background_linkage_Wb"] = bg_link[c];
      }
    }
    {
      const json& sd = a_lab_sol_.diagnostics;
      if (!sd.value("converged", true)) {
        warnings.push_back(strcat("static poloidal field: Newton did not converge in ", sd.value("newton_iterations", 0),
                                  " iterations (relative residual ", sd.value("relative_residual", 0.0),
                                  "); results are NOT reliable. Check B-H data, circuit targets and mesh.",
                                  mesh_note()));
      }
      for (auto& c : sd.value("circuits", json::array())) {
        double got = c["flux_linkage_Wb"], want = c["target_Wb"];
        // A zero-flux circuit achieves its target by cancelling large contributions, so judge the
        // residual against the scale of the linkage integral, not against the target (which is 0).
        double scale = std::max({std::fabs(want), std::fabs(got), c.value("linkage_scale_Wb", 0.0), 1e-12});
        if (std::fabs(got - want) > 1e-6 * scale)
          warnings.push_back(strcat("circuit '", c["name"].get<std::string>(), "': flux linkage ", got,
                                    " Wb differs from its target ", want, " Wb by ", got - want,
                                    " Wb, which is more than 1e-6 of the linkage scale ", scale, " Wb"));
      }
    }
    json bj = json::array();
    for (size_t b = 0; b < bodies.size(); ++b) {
      bj.push_back({{"names", bodies[b].names},
                    {"psi_Wb_per_rad", a_lab_sol_.psi[b]},
                    {"flux_Wb", 2 * pi * a_lab_sol_.psi[b]},
                    {"net_azimuthal_current_A", a_lab_sol_.current[b]},
                    {"constraint", masters[b].fixed ? "flux" : "net_current"}});
    }
    // effective current density of persistent-circuit windings (for forces and reports)
    coil_J_.assign(ne, 0.0);
    {
      auto ccs = circuit_coils(false);
      for (size_t c = 0; c < ccs.size(); ++c)
        for (auto& part : ccs[c].parts)
          for (int e : part.elements) coil_J_[e] = part.turns * a_lab_sol_.coil_current[c] / part.area;
    }
    static_json["poloidal"] = {{"energy_J", a_lab_sol_.energy},
                               {"energy_quadratic_J", a_lab_sol_.energy_quadratic},
                               {"bodies", bj},
                               {"solver", a_lab_sol_.diagnostics}};
    if (!layers.empty()) static_json["poloidal"]["thin_layers"] = layer_report(*a_lab_, a_lab_sol_, "static field");
  }

  // ------------------------------------------------------------ static toroidal field
  // secant permeability of iron at the static field (for linearized problems)
  auto mu_secant = [this](int e, const PointEval& pe) -> double {
    const ARole& ro = a_lab_->roles()[e];
    if (!ro.bh) return 1.0 / ro.nu;
    double Br, Bz;
    a_lab_->field(a_lab_sol_, e, pe, Br, Bz);
    return 1.0 / ro.bh->nu(std::sqrt(Br * Br + Bz * Bz));
  };
  auto mu_static = [&](int e) -> double { return mu_static_of(info[e]); };  // constant per element fallback
  const bool iron_nonlinear_static = a_lab_ && std::any_of(a_lab_->roles().begin(), a_lab_->roles().end(),
                                                          [](const ARole& r) { return r.bh != nullptr; });

  if (has_pol) {
    std::vector<HRole> roles(ne);
    std::vector<Profile> profiles;
    std::map<std::string, int> prof_of;
    for (auto& w : d_.windings) {
      if (w.type != spec::WindingType::Poloidal) continue;
      Profile p;
      p.name = w.name;
      p.kind = Profile::Kind::Winding;
      p.current = w.ampere_turns;
      prof_of[w.region] = static_cast<int>(profiles.size());
      profiles.push_back(p);
    }
    for (int e = 0; e < ne; ++e) {
      HRole& ro = roles[e];
      const ElemInfo& x = info[e];
      ro.mu = mu_static(e);
      if (iron_nonlinear_static && x.mat->kind == spec::MaterialKind::Iron) ro.mu_at = mu_secant;
      if (is_sc(x)) ro.kind = HKind::Excluded;
      else if (x.winding && prof_of.count(x.region)) {
        ro.kind = HKind::Profile;
        ro.profile = prof_of[x.region];
      }
    }
    auto cons = [this](const std::vector<std::string>& names) {
      const spec::Superconductor* st = sc_state(d_, names);
      return st ? st->poloidal : spec::Constraint{};
    };
    h_lab_ = std::make_unique<ToroidalProblem>(sp, mb_, roles, profiles, cons);
    log(1, "solving static toroidal field");
    h_lab_sol_ = h_lab_->solve(o.verbose);
    // Nonlinear iron in the static toroidal field. Iron is current-free, so H = G/r there with G fixed
    // by the linked currents; B = B_BH(H) pointwise. Only flux-constrained holes couple G to mu, which a
    // secant fixed point resolves (one confirming pass otherwise).
    std::vector<int> iron_el;
    for (int e = 0; e < ne; ++e)
      if (roles[e].kind == HKind::Level && info[e].mat->kind == spec::MaterialKind::Iron) iron_el.push_back(e);
    double gmax = 0;
    for (int e : iron_el) gmax = std::max(gmax, std::fabs(h_lab_sol_.G_elem[e]));
    if (gmax > 0) {
      if (iron_nonlinear_static) {
        warnings.push_back("iron carries both poloidal and toroidal static fields: its permeability follows the "
                           "poloidal field only (approximation)");
      } else {
        int it = 0;
        double change = 1;
        for (; it < 30 && change > 1e-12; ++it) {
          // One shared copy of the element G values for all iron elements. Capturing the vector by value in
          // each element's closure cost (iron elements) x (all elements) x 8 bytes: 287 GB for a toroid whose
          // field box is filled with iron (117k of 306k elements), and every copy of `roles` duplicated it again.
          const auto Gp = std::make_shared<const std::vector<double>>(h_lab_sol_.G_elem);
          const std::vector<double>& G = *Gp;
          for (int e : iron_el) {
            const BHLaw* law = bh_of.at(mat_name(e));
            roles[e].mu_at = [Gp, law](int ee, const PointEval& pe) {
              const double H = pe.r > 0 ? std::fabs((*Gp)[ee]) / pe.r : 0.0;
              return H > 0 ? law->B_of_H(H) / H : 1.0 / law->nu0();
            };
          }
          h_lab_ = std::make_unique<ToroidalProblem>(sp, mb_, roles, profiles, cons);
          h_lab_sol_ = h_lab_->solve(o.verbose);
          change = 0;
          for (int e : iron_el) change = std::max(change, std::fabs(h_lab_sol_.G_elem[e] - G[e]) / gmax);
        }
        if (change > 1e-12)
          warnings.push_back(strcat("static toroidal field: the iron secant iteration did not converge (change ", change,
                                    ").", mesh_note()));
        // stored energy int H.dB in iron (the linear solve counts 1/2 B H)
        const RefTable& tq = sp.table(sp.quad_rule());
        double dW = 0;
        PointEval pe;
        for (int e : iron_el) {
          const BHLaw* law = bh_of.at(mat_name(e));
          for (int q = 0; q < tq.nq; ++q) {
            sp.eval(e, tq, q, pe);
            if (pe.r <= 0) continue;
            const double H = std::fabs(h_lab_sol_.G_elem[e]) / pe.r, B = law->B_of_H(H);
            dW += pe.w * pe.r * (law->energy(B) - 0.5 * B * H);
          }
        }
        h_lab_sol_.energy += 2 * pi * dW;
        h_lab_sol_.diagnostics["iron_secant_iterations"] = it;
        iron_toroidal_ = true;
      }
    }
    static_json["toroidal"] = {{"energy_J", h_lab_sol_.energy}, {"solver", h_lab_sol_.diagnostics}};
  }
  // (electrostatics-only designs need no magnet: electrostatics_analysis, cpp/src/axi_electro.cpp)
  AXF_REQUIRE(a_lab_ || h_lab_ || d_.electrostatics.enabled,
              "the design has no magnet: add a winding with nonzero ampere_turns", "/windings",
              "A design with only an 'electrostatics' section needs no magnet.");
  // Secant permeability of iron at the static field, used by the RF pickup problems. Exact for RF fields
  // perpendicular to the static field (all built-in template topologies); parallel components would need the
  // differential permeability.
  std::function<double(int, const PointEval&)>& mu_rf_iron = mu_rf_iron_;
  mu_rf_iron = nullptr;
  if (iron_nonlinear_static) {
    mu_rf_iron = mu_secant;
  } else if (iron_toroidal_) {
    // Shared, not captured by value: this closure is copied into the role of every iron element of the
    // pickup problems and, through iron_mu(), of the full-wave TE/TM problems (alpha_at, beta_at), so
    // per-element vectors captured by value would cost (iron elements) x (all elements).
    const auto G = std::make_shared<const std::vector<double>>(h_lab_sol_.G_elem);
    auto law_vec = std::make_shared<std::vector<const BHLaw*>>(ne, nullptr);  // stored problems outlive locals
    for (int e = 0; e < ne; ++e)
      if (info[e].mat->kind == spec::MaterialKind::Iron) (*law_vec)[e] = bh_of.at(mat_name(e));
    const std::shared_ptr<const std::vector<const BHLaw*>> law_of = law_vec;
    mu_rf_iron = [G, law_of](int e, const PointEval& pe) {
      const BHLaw* law = (*law_of)[e];
      const double H = pe.r > 0 ? std::fabs((*G)[e]) / pe.r : 0.0;
      return H > 0 ? law->B_of_H(H) / H : 1.0 / law->nu0();
    };
  }

  // ------------------------------------------------------------ pickup mode
  json pick_json;
  if (d_.pickup) {
    const auto& pk = *d_.pickup;
    if (pk.kind == spec::Pickup::Kind::Azimuthal) {
      std::vector<int> body_of_elem;
      std::map<int, int> body_of_curve;
      auto bodies = find_bodies(sp, mb_, info, d_, true, body_of_elem, body_of_curve);
      int pb = -1;
      std::vector<Master> masters;
      for (size_t b = 0; b < bodies.size(); ++b) {
        Master m;
        m.name = bodies[b].names.empty() ? "body" : bodies[b].names.front();
        bool is_pick = std::find(bodies[b].names.begin(), bodies[b].names.end(), pk.conductor) != bodies[b].names.end();
        const spec::Superconductor* st = sc_state(d_, bodies[b].names);
        bool slit = st && st->azimuthal_slit;
        if (bodies[b].on_axis) {
          AXF_REQUIRE(!is_pick, "the pickup conductor touches the symmetry axis and cannot carry azimuthal current",
                      "/pickup/conductor");
          m.fixed = true;
          m.psi = 0;
        } else if (is_pick) {
          m.current = I_pick_;
          pb = static_cast<int>(b);
        } else if (slit) {
          m.current = 0;
        } else {
          m.fixed = true;  // RF fluxoid conservation
          m.psi = 0;
        }
        masters.push_back(m);
      }
      AXF_REQUIRE(pb >= 0, "pickup conductor '" + pk.conductor + "' not found among conductors", "/pickup/conductor");
      {
        // A conductor in several pieces would carry the readout current in each piece, and L_p from the
        // energy and from the flux would disagree (with or without thin magnetic layers).
        int n_pick = 0;
        for (auto& b : bodies)
          if (std::find(b.names.begin(), b.names.end(), pk.conductor) != b.names.end()) ++n_pick;
        AXF_REQUIRE(n_pick == 1,
                    strcat("pickup conductor '", pk.conductor, "' forms ", n_pick, " separate superconducting bodies"),
                    "/pickup/conductor",
                    "The readout current needs one connected conductor: join the pieces (a thin magnetic layer does "
                    "not separate one region), or name one piece as its own region.");
      }
      std::vector<ARole> roles(ne);
      for (int e = 0; e < ne; ++e) {
        ARole& ro = roles[e];
        if (body_of_elem[e] >= 0) {
          ro.master = true;
          ro.master_id = body_of_elem[e];
          continue;
        }
        const auto* m = info[e].mat;
        ro.nu = 1.0 / mu_static(e);
        if (m->kind == spec::MaterialKind::Iron && mu_rf_iron) {
          ro.nu_at = [mu_rf_iron](int ee, const PointEval& pp) { return 1.0 / mu_rf_iron(ee, pp); };
        }
      }
      tie_enclosed_cavities(sp, roles, body_of_curve);
      a_pick_ = std::make_unique<PoloidalProblem>(sp, roles, masters, body_of_curve);
      if (!layers.empty()) a_pick_->set_thin_layers(layers, region_of_elem);
      for (auto& cc : circuit_coils(true)) a_pick_->add_circuit_coil(cc);
      a_pick_->set_open_boundary(kOuterAttr, mb_.domain_radius, 80, mb_.domain_center_z);
      if (frozen_) {
        a_pick_sol_ = a_pick_->evaluate(frozen_y_pick_, frozen_ic_pick_);
      } else {
        log(1, "solving pickup mode (azimuthal readout current)");
        a_pick_sol_ = a_pick_->solve(o.verbose);
      }
      double L_flux = 2 * pi * a_pick_sol_.psi[pb] / I_pick_;
      double L_energy = 2 * a_pick_sol_.energy / (I_pick_ * I_pick_);
      pick_json = {{"kind", "azimuthal"},
                   {"conductor", pk.conductor},
                   {"current_A", I_pick_},
                   {"L_p_H", L_energy},
                   {"L_p_from_flux_H", L_flux},
                   {"energy_J", a_pick_sol_.energy},
                   {"solver", a_pick_sol_.diagnostics}};
      if (!layers.empty()) {
        pick_json["thin_layers"] = layer_report(*a_pick_, a_pick_sol_, "pickup mode");
        warnings.push_back("pickup mode: thin magnetic layers are treated as non-conducting (a ferrite-like layer); "
                           "a metal shield is a conductor at the readout frequency and screens instead "
                           "(docs/physics/thin_magnetic_layer.md, validity condition 3)");
      }
    } else {
      std::vector<HRole> roles(ne);
      Profile p;
      p.name = pk.port;
      p.kind = Profile::Kind::Port;
      p.current = I_pick_;
      if (pk.inside_point) {
        p.has_inside_point = true;
        p.inside_r = pk.inside_point->r;
        p.inside_z = pk.inside_point->z;
      }
      for (int e = 0; e < ne; ++e) {
        HRole& ro = roles[e];
        const ElemInfo& x = info[e];
        ro.mu = mu_static(e);
        if (mu_rf_iron && x.mat->kind == spec::MaterialKind::Iron) ro.mu_at = mu_rf_iron;
        if (is_sc(x) || is_cond(x)) ro.kind = HKind::Excluded;
        else if (x.port) {
          ro.kind = HKind::Profile;
          ro.profile = 0;
        }
      }
      auto cons = [](const std::vector<std::string>&) {
        spec::Constraint c;
        c.kind = spec::Constraint::Kind::Flux;  // RF fluxoid conservation
        c.value = 0;
        return c;
      };
      h_pick_ = std::make_unique<ToroidalProblem>(sp, mb_, roles, std::vector<Profile>{p}, cons);
      log(1, "solving pickup mode (poloidal readout current)");
      h_pick_sol_ = h_pick_->solve(o.verbose);
      pick_json = {{"kind", "poloidal"},
                   {"port", pk.port},
                   {"current_A", I_pick_},
                   {"L_p_H", 2 * h_pick_sol_.energy / (I_pick_ * I_pick_)},
                   {"energy_J", h_pick_sol_.energy},
                   {"solver", h_pick_sol_.diagnostics}};
    }
  }
  results_ = json::object();
  results_["static"] = static_json;
  results_["pickup"] = pick_json;
  results_["warnings"] = warnings;
  solved_ = true;
  post(o);
  // The sensitivities are of the magnetoquasistatic figures of merit, so a sensitivity evaluation
  // skips the analyses that each cost solves of their own.
  if (!frozen_ && !o.minimal_post) {
    mechanics_thermal(o);  // linear elasticity and steady conduction (cpp/src/model_mech.cpp)
    fullwave_analysis(o);
    hts_analysis(o);
    electrostatics_analysis(o);  // electrostatics (cpp/src/axi_electro.cpp)
  }
  results_["timing"]["solve_s"] = timer.seconds();
} catch (const SolverFailure& failure) {
  if (!mb_.mesh) throw;
  std::string note;
  try {
    note = mesh_note(true);
  } catch (const std::exception& e) {  // the original failure must survive a failing check
    note = strcat(" (The mesh check failed: ", e.what(), ")");
  }
  if (failure.hint().empty()) note.erase(0, 1);  // the leading space
  throw Error(failure.message(), failure.where(), failure.hint() + note);
}

AxiModel::Sample AxiModel::eval(int e, const mfem::IntegrationPoint& ip) const {
  PointEval pe;
  sp_->eval_at(e, ip, pe);
  return eval_pe(e, pe);
}

void AxiModel::static_poloidal(int e, const PointEval& pe, double& Br, double& Bz, double& A) const {
  a_lab_->field(a_lab_sol_, e, pe, Br, Bz, &A);
  if (bg_.active() && !a_lab_->roles()[e].master) {  // master elements already hold the total flux
    double br, bz;
    bg_.B(pe.r, pe.z, br, bz);
    Br += br;
    Bz += bz;
    if (pe.r > 0) A += bg_.psi(pe.r, pe.z) / pe.r;
  }
}

AxiModel::Sample AxiModel::eval_pe(int e, const PointEval& pe) const {
  Sample s;
  s.region = sp_->mesh.GetAttribute(e);
  if (a_lab_) {
    static_poloidal(e, pe, s.Br, s.Bz, s.A);
    s.psi = pe.r * s.A;
  }
  if (h_lab_) {
    double H, mu;
    h_lab_->field(h_lab_sol_, e, pe, H, mu);
    s.Bphi = mu * H;
  }
  if (a_pick_) {
    double A = 0;
    a_pick_->field(a_pick_sol_, e, pe, s.pBr, s.pBz, &A);
    s.pBr /= I_pick_;
    s.pBz /= I_pick_;
    s.ppsi = pe.r * A / I_pick_;
  }
  if (h_pick_) {
    double H, mu;
    h_pick_->field(h_pick_sol_, e, pe, H, mu);
    s.pBphi = mu * H / I_pick_;
  }
  return s;
}

std::vector<AxiModel::Sample> AxiModel::sample(const std::vector<double>& r, const std::vector<double>& z) const {
  AXF_REQUIRE(solved_, "solve the model before sampling");
  AXF_REQUIRE(r.size() == z.size(), "r and z must have the same length");
  const int n = static_cast<int>(r.size());
  mfem::DenseMatrix pts(2, n);
  for (int i = 0; i < n; ++i) {
    pts(0, i) = r[i];
    pts(1, i) = z[i];
  }
  mfem::Array<int> eid;
  mfem::Array<mfem::IntegrationPoint> ips;
  sp_->mesh.FindPoints(pts, eid, ips, false);
  std::vector<Sample> out(n);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (int i = 0; i < n; ++i) {
    if (eid[i] < 0) {
      out[i].Br = out[i].Bz = out[i].Bphi = out[i].psi = out[i].A = nan;
      out[i].pBr = out[i].pBz = out[i].pBphi = nan;
      continue;
    }
    out[i] = eval(eid[i], ips[i]);
  }
  return out;
}

// ---------------------------------------------------------------- post-processing
void AxiModel::post(const SolveOptions& o) {
  Timer timer;
  const Space& sp = *sp_;
  const mfem::Mesh& m = sp.mesh;
  const int ne = m.GetNE();
  json& R = results_;
  json checks = json::array();
  json& warnings = R["warnings"];

  // ------------------------------------------------------------ coupling Psi
  double Psi = 0;
  const RefTable& tab = sp.table(sp.quad_rule());
  if (a_pick_ || h_pick_) {
#pragma omp parallel reduction(+ : Psi)
    {
      PointEval pe;
#pragma omp for schedule(dynamic, 256)
      for (int e = 0; e < ne; ++e) {
        for (int q = 0; q < tab.nq; ++q) {
          sp.eval(e, tab, q, pe);
          double val = 0;
          if (a_lab_ && h_pick_) {
            double Br, Bz, A, H, mu;
            static_poloidal(e, pe, Br, Bz, A);
            h_pick_->field(h_pick_sol_, e, pe, H, mu);
            val += A * mu * H;
          }
          if (h_lab_ && a_pick_) {
            double Br, Bz, A, H, mu;
            a_pick_->field(a_pick_sol_, e, pe, Br, Bz, &A);
            h_lab_->field(h_lab_sol_, e, pe, H, mu);
            val += A * mu * H;
          }
          Psi += pe.w * pe.r * val;
        }
      }
    }
    Psi *= 2 * pi / I_pick_;
  }

  // ------------------------------------------------------------ energy, inductance, F
  double W = 0;
  if (a_lab_) W += a_lab_sol_.energy;
  if (h_lab_) W += h_lab_sol_.energy;
  // note: R["pickup"] on a non-const json would INSERT a null member for a design with no
  // pickup, which later breaks the scene summary; look it up without creating it
  const json* pickup_json = R.contains("pickup") && R["pickup"].is_object() ? &R["pickup"] : nullptr;
  double Lp = pickup_json && pickup_json->contains("L_p_H") ? (*pickup_json)["L_p_H"].get<double>() : 0.0;
  if (a_pick_) {
    double Lf = R["pickup"]["L_p_from_flux_H"];
    checks.push_back({{"name", "pickup inductance: flux vs energy"},
                      {"relative_difference", std::fabs(Lf - Lp) / std::max(Lp, 1e-300)}});
  }
  if (a_lab_) {
    const json& sd = a_lab_sol_.diagnostics;
    checks.push_back({{"name", "static poloidal solve"},
                      {"converged", sd.value("converged", true)},
                      {"newton_iterations", sd.value("newton_iterations", 0)},
                      {"relative_residual", sd.value("relative_residual", 0.0)}});
    for (auto& c : sd.value("circuits", json::array())) {
      double got = c["flux_linkage_Wb"], want = c["target_Wb"];
      const double sc = std::max({std::fabs(want), std::fabs(got), c.value("linkage_scale_Wb", 0.0), 1e-12});
      checks.push_back({{"name", "circuit flux linkage: " + c["name"].get<std::string>()},
                        {"achieved_Wb", got}, {"target_Wb", want}, {"difference_Wb", got - want},
                        {"linkage_scale_Wb", c.value("linkage_scale_Wb", 0.0)},
                        {"relative", (got - want) / sc}});
    }
  }
  // device extent and cold envelope
  double rmax = 0, zmin = 1e300, zmax = -1e300, crmax = 0, czmin = 1e300, czmax = -1e300;
  for (auto& g : d_.regions) {
    double a, b, c, e;
    g.shape.bbox(a, b, c, e);
    rmax = std::max(rmax, b);
    zmin = std::min(zmin, c);
    zmax = std::max(zmax, e);
    const auto& mat = d_.materials.at(g.material);
    bool wind = std::any_of(d_.windings.begin(), d_.windings.end(), [&](auto& w) { return w.region == g.name; });
    bool cold = !mat.warm && (wind || mat.kind != spec::MaterialKind::Vacuum);
    if (cold) {
      crmax = std::max(crmax, b);
      czmin = std::min(czmin, c);
      czmax = std::max(czmax, e);
    }
  }
  for (auto& c : d_.curves) {
    double a, b, cc, e;
    c.path.bbox(a, b, cc, e);
    rmax = std::max(rmax, b);
    zmin = std::min(zmin, cc);
    zmax = std::max(zmax, e);
    if (d_.materials.at(c.material).warm) continue;  // e.g. a warm thin mu-metal shield: not in the cold envelope
    crmax = std::max(crmax, b);
    czmin = std::min(czmin, cc);
    czmax = std::max(czmax, e);
  }
  const double A_cold = crmax > 0 ? 2 * pi * crmax * (crmax + (czmax - czmin)) : 0.0;
  double iron_kg = 0;
  for (int e = 0; e < ne; ++e) {
    const spec::Region* g = d_.find_region(region_name(m.GetAttribute(e)));
    if (!g) continue;
    const auto& mat = d_.materials.at(g->material);
    if (mat.kind != spec::MaterialKind::Iron) continue;
    PointEval pe;
    for (int q = 0; q < tab.nq; ++q) {
      sp.eval(e, tab, q, pe);
      iron_kg += mat.density * 2 * pi * pe.r * pe.w;
    }
  }

  const double F = (Lp > 0) ? std::pow(Psi, 4) / (Lp * Lp) : 0.0;
  json coupling = {{"Psi_SI_H_T_m2", Psi},
                   {"Psi_natural_T_m3", Psi / mu0},
                   {"L_p_H", Lp},
                   {"L_p_natural_m", Lp / mu0},
                   {"F_SI_T4_H2_m8", F},
                   {"F_natural_T4_m10", F / (mu0 * mu0)},
                   {"W_mag_J", W},
                   {"device_radius_m", rmax},
                   {"device_height_m", zmax - zmin},
                   {"g_factor", (Lp > 0 && W > 0) ? Psi * Psi / (Lp * W * rmax * rmax) / (mu0 * mu0) : 0.0},
                   {"note", "g = Psi^2/(L_p W R^2) in natural units, R = device outer radius"}};
  R["coupling"] = coupling;
  R["geometry"] = {{"A_cold_m2", A_cold},
                   {"cold_envelope", {{"r_max_m", crmax}, {"z_min_m", czmin}, {"z_max_m", czmax}}},
                   {"iron_mass_kg", iron_kg}};

  // ------------------------------------------------------------ peak fields, forces, losses
  std::map<std::string, double> peak_region, sc_peak, sc_peak_regular;
  double peak_all = 0;
  if (!o.minimal_post) post_fields(o, Lp, checks, peak_region, peak_all, sc_peak, sc_peak_regular);

  // ------------------------------------------------------------ requirements, scaling, cost
  json bench = json::array();
  const json conductor_data = R.contains("_conductor_data") ? R["_conductor_data"] : json::object();
  const bool has_conductors = !conductor_data.empty();
  auto conductor_margins = [&](double fscale, double cscale, double lscale = 1.0) {
    json out = json::object();
    for (auto& [name, op] : conductor_data.items()) {
      const auto& c = d_.conductors.at(op["conductor"].get<std::string>());
      auto law = make_jc(c);
      const json* worst = nullptr;
      double frac = -1, jc_w = 0;
      for (auto& p : op["points"]) {
        const double J = cscale * p["J_engineering_A_m2"].get<double>() / c.fill_factor;
        const double jc = law->Jc(fscale * p["Bperp_T"].get<double>(), fscale * p["Bpar_T"].get<double>(), c.T_op);
        const double f = jc > 0 ? J / jc : (J > 0 ? 1e30 : 0.0);
        if (f > frac || (f == frac && worst && p["B_T"].get<double>() > (*worst)["B_T"].get<double>())) {
          frac = f;
          jc_w = jc;
          worst = &p;
        }
      }
      AXF_REQUIRE(worst != nullptr, "winding '" + name + "' has no field evaluation points");
      const double J = cscale * (*worst)["J_engineering_A_m2"].get<double>() / c.fill_factor;
      const double Bp = fscale * (*worst)["Bperp_T"].get<double>();
      const double Bt = fscale * (*worst)["Bpar_T"].get<double>();
      const double ss = load_line_scale(*law, Bp, Bt, J, c.T_op);
      json j = {{"conductor", op["conductor"]},
                {"law", law->law()},
                {"T_op_K", c.T_op},
                {"fill_factor", c.fill_factor},
                {"area_m2", lscale * lscale * op["area_m2"].get<double>()},
                {"ampere_turns_A", cscale * lscale * lscale * op["ampere_turns_A"].get<double>()},
                {"J_engineering_A_m2", cscale * (*worst)["J_engineering_A_m2"].get<double>()},
                {"J_engineering_mean_A_m2", cscale * op["J_engineering_integral_A"].get<double>() /
                                                  op["area_m2"].get<double>()},
                {"J_superconductor_A_m2", J},
                {"B_peak_T", fscale * op["B_peak_T"].get<double>()},
                {"B_peak_location", {{"r_m", lscale * op["B_peak_r_m"].get<double>()},
                                       {"z_m", lscale * op["B_peak_z_m"].get<double>()}}},
                {"worst_point", {{"r_m", lscale * (*worst)["r_m"].get<double>()},
                                   {"z_m", lscale * (*worst)["z_m"].get<double>()},
                                   {"B_T", fscale * (*worst)["B_T"].get<double>()},
                                   {"B_perp_T", Bp},
                                   {"B_par_T", Bt},
                                   {"field_angle_to_tape_deg", std::atan2(Bp, Bt) * 180 / pi}}},
                {"Jc_A_m2", jc_w},
                {"load_line_fraction", frac},
                {"short_sample", {{"scale", ss}, {"J_superconductor_A_m2", ss * J},
                                   {"B_T", ss * fscale * (*worst)["B_T"].get<double>()}}},
                {"ok", frac <= 1.0}};
      if (law->temperature_dependent()) {
        const double Tcs = current_sharing_T(*law, Bp, Bt, J, c.T_op);
        j["current_sharing_T_K"] = Tcs;
        j["temperature_margin_K"] = Tcs - c.T_op;
      } else {
        j["temperature_margin"] = "not available: this conductor's Jc does not depend on temperature "
                                    "(a field-only fit or table); give the temperature dependence to get a margin";
      }
      if (jc_w <= 0)
        j["status"] = "the field at the worst point is above the critical surface of this conductor "
                       "(Jc = 0): load_line_fraction is reported as 1e30";
      if (op["poloidal"].get<bool>())
        j["current_note"] =
            "poloidal winding: the current density is J = grad(G)/r (uniform-resistivity distribution) and "
            "varies over the cross-section; J_engineering_A_m2 is its value at the worst point and "
            "J_engineering_mean_A_m2 the area average. At a sharp inner corner of the winding path J is "
            "singular and the peak grows under mesh refinement, as the superconductor surface field does; "
            "round the corner (shape fillet) or read J_engineering_mean_A_m2, which does not";
      if (law->anisotropic())
        j["note"] = "anisotropic conductor: B_perp is the component normal to the tape plane "
                     "(conductors.tape_normal); the worst point is the point of largest J/Jc on the winding";
      out[name] = std::move(j);
    }
    return out;
  };
  auto load_line_worst = [&](double fscale, double cscale) {
    double worst = 0;
    for (auto& [name, op] : conductor_data.items()) {
      const auto& c = d_.conductors.at(op["conductor"].get<std::string>());
      auto law = make_jc(c);
      for (auto& p : op["points"]) {
        const double J = cscale * p["J_engineering_A_m2"].get<double>() / c.fill_factor;
        const double jc = law->Jc(fscale * p["Bperp_T"].get<double>(), fscale * p["Bpar_T"].get<double>(), c.T_op);
        worst = std::max(worst, jc > 0 ? J / jc : (J > 0 ? 1e30 : 0.0));
      }
    }
    return worst;
  };
  const bool linear = !(a_lab_ && std::any_of(a_lab_->roles().begin(), a_lab_->roles().end(),
                                               [](const ARole& r) { return r.bh != nullptr; })) &&
                      !iron_toroidal_;
  // field limits for the cost options: superconductor wall (regular peak, i.e. away from sharp corners)
  // and winding peak
  double sc_wall = 0, wind_peak = 0;
  for (auto& [k, v] : sc_peak_regular) sc_wall = std::max(sc_wall, v);
  for (auto& w : d_.windings)
    if (peak_region.count(w.region)) wind_peak = std::max(wind_peak, peak_region[w.region]);
  const double sc_cap = d_.cost.sc_wall_cap, b_cap = d_.cost.B_peak_cap;
  const double ll_cap = d_.cost.load_line_cap;
  auto caps_json = [&](double scale, double cscale) {
    json j = json::object();
    double viol = 0;
    if (sc_cap > 0) {
      j["sc_wall_T"] = scale * sc_wall;
      j["sc_wall_cap_T"] = sc_cap;
      viol = std::max(viol, scale * sc_wall / sc_cap - 1);
    }
    if (b_cap > 0) {
      j["winding_peak_T"] = scale * wind_peak;
      j["B_peak_cap_T"] = b_cap;
      viol = std::max(viol, scale * wind_peak / b_cap - 1);
    }
    if (has_conductors && ll_cap > 0) {
      const double frac = load_line_worst(scale, cscale);
      j["load_line_fraction"] = frac;
      j["load_line_cap"] = ll_cap;
      viol = std::max(viol, frac / ll_cap - 1);
    }
    if (sc_cap > 0 || b_cap > 0 || (has_conductors && ll_cap > 0)) {
      j["violation"] = std::max(viol, 0.0);  // max(field/cap - 1, 0)
      j["ok"] = viol <= 0;
    }
    return j;
  };
  // amplitude scaling is exact only for linear problems with no fixed (non-scaling) sources
  const bool fixed_sources =
      std::any_of(d_.superconductors.begin(), d_.superconductors.end(),
                  [](auto& sc) { return sc.azimuthal.value != 0.0 || sc.poloidal.value != 0.0; }) ||
      std::any_of(d_.circuits.begin(), d_.circuits.end(), [](auto& c) { return c.flux_linkage != 0.0; }) ||
      bg_.active();
  for (auto& b : d_.benchmarks) {
    scan::Requirement req = scan::F_required(b, d_.scan);
    json bj = {{"name", b.name}, {"Q", b.Q}, {"T_K", b.T}, {"eta_A", b.eta_A},
               {"F_req_SI", req.F_SI}, {"F_req_natural_T4_m10", req.F_nat}, {"g_min_GeV", req.g_min_GeV}};
    if (F > 0) {
      double s = std::pow(req.F_SI / F, 0.25);
      double Wr = s * s * W;
      bj["amplitude_scale"] = s;
      bj["F_over_F_req"] = F / req.F_SI;
      bj["W_mag_at_req_J"] = Wr;
      bj["B_max_at_req_T"] = s * peak_all;
      json scp = json::object(), scr = json::object();
      for (auto& [k, v] : sc_peak) scp[k] = s * v;
      for (auto& [k, v] : sc_peak_regular) scr[k] = s * v;
      bj["superconductor_surface_at_req_T"] = scp;
      bj["superconductor_surface_regular_at_req_T"] = scr;
      if (sc_cap > 0 || b_cap > 0 || (has_conductors && ll_cap > 0)) bj["caps_at_req"] = caps_json(s, s);
      if (has_conductors) bj["conductor_margins"] = conductor_margins(s, s);
      auto c = cost::evaluate(d_.cost, Wr, A_cold, iron_kg);
      bj["cost"] = c.to_json();
      if (iron_kg == 0) {
        double l = cost::optimal_scale(d_.cost, Wr, A_cold);
        auto copt = cost::evaluate(d_.cost, Wr / (l * l), A_cold * l * l, 0);
        const double fs = s * std::pow(l, -2.5);  // field scale at the cost-optimal size
        json sco = json::object();
        for (auto& [k, v] : sc_peak_regular) sco[k] = fs * v;
        bj["cost_optimal_size"] = {{"length_scale", l},
                                   {"device_radius_m", rmax * l},
                                   {"B_max_T", fs * peak_all},
                                   {"superconductor_surface_regular_T", sco},
                                   {"W_mag_J", Wr / (l * l)},
                                   {"cost", copt.to_json()}};
        if (sc_cap > 0 || b_cap > 0 || (has_conductors && ll_cap > 0))
          bj["cost_optimal_size"]["caps"] = caps_json(fs, fs / l);
        if (has_conductors) bj["cost_optimal_size"]["conductor_margins"] = conductor_margins(fs, fs / l, l);
      }
      if (!linear) {
        bj["note"] = "iron is nonlinear: amplitude scaling is a first estimate; rescale the sources and re-solve";
      } else if (fixed_sources) {
        bj["note"] = "fixed fluxes, prescribed superconductor currents, persistent-circuit linkages or a background "
                     "field do not scale with the winding currents: amplitude scaling is approximate";
      }
    }
    bench.push_back(bj);
  }
  R["benchmarks"] = bench;
  R["checks"] = checks;
  R["linear"] = linear;
  R["mesh"] = {{"elements", ne}, {"order", sp.order}, {"dofs", sp.ndof()}, {"info", mb_.info}};
  R["design"] = {{"name", d_.name}, {"parameters", d_.parameters}};
  R["version"] = kVersion;
  R["timing"]["post_s"] = timer.seconds();
  R.erase("_conductor_data");
  (void)o;
  (void)warnings;
}

// Peak fields, forces, RF surface losses, the readout-port capacitance and winding flux linkages:
// everything in the results document that the figures of merit (Psi, L_p, W, F, cost) do not need.
// Sensitivity evaluations skip it (SolveOptions::minimal_post).
void AxiModel::post_fields(const SolveOptions& o, double Lp, json& checks,
                           std::map<std::string, double>& peak_region, double& peak_all,
                           std::map<std::string, double>& sc_peak, std::map<std::string, double>& sc_peak_regular) {
  const Space& sp = *sp_;
  const mfem::Mesh& m = sp.mesh;
  const int ne = m.GetNE();
  json& R = results_;
  json& warnings = R["warnings"];
  const RefTable& tab = sp.table(sp.quad_rule());
  auto is_excluded_static = [&](int e) {
    const spec::Region* g = d_.find_region(region_name(m.GetAttribute(e)));
    return g && d_.materials.at(g->material).kind == spec::MaterialKind::Superconductor;
  };
  (void)o;
  // ------------------------------------------------------------ peak fields
  const int K = 2 * sp.order + 2;
  const auto& lat = lattice(K);
  const RefTable& ltab = sp.table(lat);
  std::vector<std::string> attr_region(m.attributes.Max() + 1);
  for (int a = 0; a <= m.attributes.Max(); ++a) attr_region[a] = region_name(a);
  const int nth = omp_get_max_threads();
  std::vector<std::map<std::string, double>> peak_t(nth);
  std::vector<double> pk(nth, 0.0), pkr(nth, 0.0), pkz(nth, 0.0);
  std::vector<std::string> pkw(nth);
#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    PointEval pe;
#pragma omp for schedule(dynamic, 256)
    for (int e = 0; e < ne; ++e) {
      if (is_excluded_static(e)) continue;
      const std::string& reg = attr_region[m.GetAttribute(e)];
      double& pr = peak_t[tid][reg];
      for (int q = 0; q < ltab.nq; ++q) {
        sp.eval(e, ltab, q, pe);
        Sample s = eval_pe(e, pe);
        double B = std::sqrt(s.Br * s.Br + s.Bz * s.Bz + s.Bphi * s.Bphi);
        pr = std::max(pr, B);
        if (B > pk[tid]) {
          pk[tid] = B;
          pkr[tid] = pe.r;
          pkz[tid] = pe.z;
          pkw[tid] = reg;
        }
      }
    }
  }
  double peak_r = 0, peak_z = 0;
  std::string peak_where;
  for (int t = 0; t < nth; ++t) {
    for (auto& [k, v] : peak_t[t]) peak_region[k] = std::max(peak_region[k], v);
    if (pk[t] > peak_all) {
      peak_all = pk[t];
      peak_r = pkr[t];
      peak_z = pkz[t];
      peak_where = pkw[t];
    }
  }
  // superconductor surfaces: fields on the non-superconducting side
  std::map<std::string, std::array<double, 2>> sc_peak_at;
  std::map<std::string, double> sc_force_z;
  // Convex sharp corners of superconductors (and sheet ends): the vacuum field is singular there, so
  // the surface peak near them grows under refinement. Detected from the curved boundary tangents.
  struct Corner {
    std::string body;
    double r, z, angle_deg, delta;
  };
  std::vector<Corner> corners;
  {
    auto sc_elem = [&](int e) {
      const spec::Region* g = d_.find_region(region_name(m.GetAttribute(e)));
      return g && d_.materials.at(g->material).kind == spec::MaterialKind::Superconductor;
    };
    struct End {
      double tx, ty, len;  // outgoing unit tangent and edge length
      int sc_elem;
    };
    std::map<int, std::vector<End>> at_vertex;
    mfem::Array<int> fv;
    for (int f = 0; f < m.GetNumFaces(); ++f) {
      int e1, e2;
      m.GetFaceElements(f, &e1, &e2);
      if (e1 < 0 || e2 < 0 || sc_elem(e1) == sc_elem(e2)) continue;
      const int sce = sc_elem(e1) ? e1 : e2;
      m.GetFaceVertices(f, fv);
      mfem::ElementTransformation* T = const_cast<mfem::Mesh&>(m).GetFaceTransformation(f);
      mfem::IntegrationPoint ip;
      double len = 0;
      for (int k = 0; k < 2; ++k) {
        ip.x = k;
        T->SetIntPoint(&ip);
        const mfem::DenseMatrix& J = T->Jacobian();
        double tx = J(0, 0), ty = J(1, 0), n = std::hypot(tx, ty);
        len = std::max(len, n);
        if (k == 1) { tx = -tx; ty = -ty; }
        at_vertex[fv[k]].push_back({tx / n, ty / n, n, sce});
      }
    }
    for (auto& [v, ends] : at_vertex) {
      if (ends.size() != 2) continue;  // junctions of several bodies are not classified
      const double* p = m.GetVertex(v);
      if (p[0] <= sp.rtol) continue;
      const End &a = ends[0], &b = ends[1];
      const double cosang = std::clamp(a.tx * b.tx + a.ty * b.ty, -1.0, 1.0);
      const double alpha = std::acos(cosang);  // smaller angle between the two boundary tangents
      // SC side: is the SC element's centroid inside the smaller wedge?
      mfem::Vector c(2);
      {
        mfem::Array<int> ev;
        m.GetElementVertices(a.sc_elem, ev);
        c = 0.0;
        for (int u : ev) { c(0) += m.GetVertex(u)[0] / ev.Size(); c(1) += m.GetVertex(u)[1] / ev.Size(); }
      }
      const double wx = c(0) - p[0], wy = c(1) - p[1];
      const double cab = a.tx * b.ty - a.ty * b.tx, caw = a.tx * wy - a.ty * wx, cwb = wx * b.ty - wy * b.tx;
      const bool inside = (cab >= 0) ? (caw >= 0 && cwb >= 0) : (caw <= 0 && cwb <= 0);
      const double theta = inside ? alpha : 2 * pi - alpha;  // interior angle of the superconductor
      if (theta < 150.0 * pi / 180.0) {
        corners.push_back({region_name(m.GetAttribute(a.sc_elem)), p[0], p[1], theta * 180 / pi,
                           3 * std::max(a.len, b.len)});
      }
    }
    // ends of open superconducting sheets
    for (auto& cv : d_.curves) {
      if (d_.materials.at(cv.material).kind != spec::MaterialKind::Superconductor) continue;
      const int attr = mb_.attr_of_curve.at(cv.name);
      std::map<int, int> deg;
      std::map<int, double> elen;
      for (int f = 0; f < m.GetNumFaces(); ++f) {
        if (sp.face_bdr_attr[f] != attr) continue;
        m.GetFaceVertices(f, fv);
        const double* p0 = m.GetVertex(fv[0]);
        const double* p1 = m.GetVertex(fv[1]);
        const double L = std::hypot(p1[0] - p0[0], p1[1] - p0[1]);
        for (int k = 0; k < 2; ++k) {
          ++deg[fv[k]];
          elen[fv[k]] = std::max(elen[fv[k]], L);
        }
      }
      for (auto& [v, n] : deg) {
        const double* p = m.GetVertex(v);
        if (n == 1 && p[0] > sp.rtol) corners.push_back({cv.name, p[0], p[1], 0.0, 3 * elen[v]});
      }
    }
  }
  auto near_corner = [&](const std::string& body, double r, double z) -> const Corner* {
    for (auto& c : corners)
      if (c.body == body && std::hypot(r - c.r, z - c.z) < c.delta) return &c;
    return nullptr;
  };
  {
    const mfem::IntegrationRule& ir1 = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp.order + 6);
    mfem::IntegrationRule lat1;
    for (int i = 0; i <= K; ++i) {
      mfem::IntegrationPoint ip;
      ip.x = double(i) / K;
      lat1.Append(ip);
    }
    auto is_sc_elem = [&](int e) {
      const spec::Region* g = d_.find_region(region_name(m.GetAttribute(e)));
      return g && d_.materials.at(g->material).kind == spec::MaterialKind::Superconductor;
    };
    std::set<int> sc_sheet_attrs;
    for (auto& c : d_.curves) {
      if (d_.materials.at(c.material).kind == spec::MaterialKind::Superconductor) sc_sheet_attrs.insert(mb_.attr_of_curve.at(c.name));
    }
    for (int f = 0; f < m.GetNumFaces(); ++f) {
      int e1, e2;
      m.GetFaceElements(f, &e1, &e2);
      bool sheet = sc_sheet_attrs.count(sp.face_bdr_attr[f]) > 0;
      std::vector<std::pair<int, int>> sides;  // (vacuum-side element, which: 1 or 2), body name
      std::string body;
      if (sheet) {
        for (auto& c : d_.curves) if (mb_.attr_of_curve.at(c.name) == sp.face_bdr_attr[f]) body = c.name;
        if (e1 >= 0 && !is_sc_elem(e1)) sides.push_back({e1, 1});
        if (e2 >= 0 && !is_sc_elem(e2)) sides.push_back({e2, 2});
      } else if (e1 >= 0 && e2 >= 0 && is_sc_elem(e1) != is_sc_elem(e2)) {
        int sc = is_sc_elem(e1) ? e1 : e2;
        body = region_name(m.GetAttribute(sc));
        sides.push_back(is_sc_elem(e1) ? std::make_pair(e2, 2) : std::make_pair(e1, 1));
      }
      if (sides.empty()) continue;
      mfem::FaceElementTransformations* FT = const_cast<mfem::Mesh&>(m).GetFaceElementTransformations(f);
      for (auto [ev, which] : sides) {
        // peak
        for (int q = 0; q < lat1.GetNPoints(); ++q) {
          FT->SetAllIntPoints(&lat1.IntPoint(q));
          const mfem::IntegrationPoint& eip = which == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
          Sample s = eval(ev, eip);
          double B = std::sqrt(s.Br * s.Br + s.Bz * s.Bz + s.Bphi * s.Bphi);
          mfem::Vector xq(2);
          FT->Transform(lat1.IntPoint(q), xq);
          if (B > sc_peak[body]) {
            sc_peak[body] = B;
            sc_peak_at[body] = {xq(0), xq(1)};
          }
          if (!near_corner(body, xq(0), xq(1))) sc_peak_regular[body] = std::max(sc_peak_regular[body], B);
        }
        // Maxwell pressure force (z) on the body: F = -int p n_SC dA, n_SC points from SC into vacuum.
        for (int q = 0; q < ir1.GetNPoints(); ++q) {
          FT->SetAllIntPoints(&ir1.IntPoint(q));
          const mfem::IntegrationPoint& eip = which == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
          Sample s = eval(ev, eip);
          double B2 = s.Br * s.Br + s.Bz * s.Bz + s.Bphi * s.Bphi;
          mfem::Vector nor(2), x(2);
          mfem::CalcOrtho(FT->Jacobian(), nor);  // points from element 1 to element 2, |nor| = dl/dxi
          FT->Transform(ir1.IntPoint(q), x);
          double nz_v = (which == 1 ? -nor(1) : nor(1));  // normal pointing from SC side into vacuum element
          // pressure pushes on the body towards the SC: force on body = -p * n_into_vacuum
          sc_force_z[body] += -(B2 / (2 * mu0)) * nz_v * 2 * pi * x(0) * ir1.IntPoint(q).weight;
        }
      }
    }
  }
  json sc_loc = json::object(), jcorners = json::array();
  for (auto& [b, x] : sc_peak_at) {
    sc_loc[b] = {{"r_m", x[0]}, {"z_m", x[1]}};
    if (const Corner* c = near_corner(b, x[0], x[1])) {
      warnings.push_back(strcat("superconductor '", b, "': the peak surface field ", sc_peak[b], " T is at a sharp ",
                                c->angle_deg == 0 ? "sheet edge" : "corner", " (r = ", c->r, ", z = ", c->z,
                                " m) where the field is singular and grows under mesh refinement; away from such "
                                "corners the peak is ", sc_peak_regular[b],
                                " T (superconductor_surface_regular_T). Round the corner (fillet) for a meaningful "
                                "peak field."));
    }
  }
  for (size_t i = 0; i < corners.size() && i < 50; ++i)
    jcorners.push_back({{"body", corners[i].body}, {"r_m", corners[i].r}, {"z_m", corners[i].z},
                        {"interior_angle_deg", corners[i].angle_deg}, {"excluded_radius_m", corners[i].delta}});
  json peaks = {{"B_max_T", peak_all},
                {"B_max_location", {{"r_m", peak_r}, {"z_m", peak_z}, {"region", peak_where}}},
                {"by_region_T", peak_region},
                {"superconductor_surface_T", sc_peak},
                {"superconductor_surface_location", sc_loc},
                {"superconductor_surface_regular_T", sc_peak_regular},
                {"superconductor_sharp_corners", jcorners}};
  R["fields"] = peaks;

  // ------------------------------------------------------------ conductor critical-current margins
  // For every winding made of a named conductor: the operating point (engineering and superconductor
  // current density, peak |B| on the winding and, for a tape, the field angle), the critical current
  // density of the worst point of the winding, the load-line margin (fraction of Jc, and the scale at
  // which the load line meets the critical surface) and the temperature margin.
  // The operating point varies over the winding (an azimuthal winding has a uniform current density,
  // a poloidal one carries J = grad G / r, which falls like 1/r), so every lattice point of the
  // winding is kept and the worst point is the one with the largest J/Jc at the scale asked for.
  struct CondPoint {
    double Bperp = 0, Bpar = 0, B = 0, Je = 0, r = 0, z = 0;
  };
  struct CondOp {
    const spec::Winding* w = nullptr;
    const spec::Conductor* c = nullptr;
    std::shared_ptr<JcLaw> law;
    double area = 0, NI = 0, Jsum = 0;          // Jsum: int |J| dA over the cross-section
    double B_peak = 0, peak_r = 0, peak_z = 0;  // peak |B| on the winding
    std::vector<CondPoint> pts;
  };
  std::vector<CondOp> conds;
  for (auto& w : d_.windings) {
    if (w.conductor.empty()) continue;
    CondOp op;
    op.w = &w;
    op.c = &d_.conductors.at(w.conductor);
    op.law = make_jc(*op.c);
    const double nr = op.c->normal_r, nz = op.c->normal_z;
    const bool azim = w.type == spec::WindingType::Azimuthal;
    AXF_REQUIRE(azim || (h_lab_ && w.type == spec::WindingType::Poloidal),
                "winding '" + w.name + "' has a conductor but no solved current distribution",
                "/windings", "Conductor margins need an azimuthal or a poloidal winding that carries current.");
    PointEval pe;
    for (int e = 0; e < ne; ++e) {
      if (region_name(m.GetAttribute(e)) != w.region) continue;
      op.area += element_area(sp, e);
      const double Jaz = azim && a_lab_ ? a_lab_->roles()[e].J + (coil_J_.empty() ? 0.0 : coil_J_[e]) : 0.0;
      const int prof = !azim ? h_lab_->roles()[e].profile : -1;
      // |J| at a point: uniform for an azimuthal winding; for a poloidal one the current function
      // gives J_r = -dG/dz / r, J_z = dG/dr / r with G = La + (Lb - La) r u_s.
      auto local_J = [&](const PointEval& q) {
        if (azim) return std::fabs(Jaz);
        if (prof < 0 || q.r <= 0) return 0.0;
        const double dL = h_lab_sol_.Lb[prof] - h_lab_sol_.La[prof];
        const auto& dd = sp.edofs(e);
        double us = 0, dr = 0, dz = 0;
        for (size_t i = 0; i < dd.size(); ++i) {
          us += h_lab_sol_.unit(dd[i]) * q.N[i];
          dr += h_lab_sol_.unit(dd[i]) * q.dN[2 * i];
          dz += h_lab_sol_.unit(dd[i]) * q.dN[2 * i + 1];
        }
        return std::hypot(dL * dz, dL * (us / q.r + dr));
      };
      for (int q = 0; q < tab.nq; ++q) {
        sp.eval(e, tab, q, pe);
        if (azim) op.NI += Jaz * pe.w;
        op.Jsum += local_J(pe) * pe.w;
      }
      for (int q = 0; q < ltab.nq; ++q) {
        sp.eval(e, ltab, q, pe);
        const Sample s = eval_pe(e, pe);
        CondPoint p;
        p.r = pe.r;
        p.z = pe.z;
        const double B2 = s.Br * s.Br + s.Bz * s.Bz + s.Bphi * s.Bphi;
        const double bn = s.Br * nr + s.Bz * nz;
        p.Bperp = std::fabs(bn);
        p.Bpar = std::sqrt(std::max(B2 - bn * bn, 0.0));
        p.B = std::sqrt(B2);
        if (p.B > op.B_peak) { op.B_peak = p.B; op.peak_r = pe.r; op.peak_z = pe.z; }
        p.Je = local_J(pe);
        op.pts.push_back(p);
      }
    }
    AXF_REQUIRE(op.area > 0, "winding '" + w.name + "' has no elements in the mesh", "/windings");
    if (!azim) op.NI = w.ampere_turns;
    conds.push_back(std::move(op));
  }
  // Keep the sampled operating points for the benchmark rescaling in post().  The public
  // conductor block below is intentionally still produced here, while this compact private
  // representation lets the requirements/cost pass recompute the nonlinear Jc and load line at
  // each benchmark scale without rerunning the field solve.
  json conductor_data = json::object();
  for (auto& op : conds) {
    json points = json::array();
    for (auto& p : op.pts)
      points.push_back({{"Bperp_T", p.Bperp}, {"Bpar_T", p.Bpar}, {"B_T", p.B},
                        {"J_engineering_A_m2", p.Je}, {"r_m", p.r}, {"z_m", p.z}});
    conductor_data[op.w->name] = {{"conductor", op.w->conductor}, {"area_m2", op.area},
                                   {"ampere_turns_A", op.NI}, {"J_engineering_integral_A", op.Jsum},
                                   {"B_peak_T", op.B_peak}, {"B_peak_r_m", op.peak_r},
                                   {"B_peak_z_m", op.peak_z},
                                   {"poloidal", op.w->type != spec::WindingType::Azimuthal},
                                   {"points", std::move(points)}};
  }
  if (!conductor_data.empty()) R["_conductor_data"] = std::move(conductor_data);
  // Margins at a given field scale, current scale and length scale (the first two are equal for
  // amplitude scaling of one geometry; at the cost-optimal size the lengths scale by `lscale`, so the
  // current density scales as field_scale / length_scale).
  auto cond_json = [&](double fscale, double cscale, double lscale = 1.0) {
    json out = json::object();
    for (auto& op : conds) {
      // worst point: the largest fraction of the critical current density at this scale
      const CondPoint* worst = nullptr;
      double frac = -1, jc_w = 0;
      for (auto& p : op.pts) {
        const double J = cscale * p.Je / op.c->fill_factor;
        const double jc = op.law->Jc(fscale * p.Bperp, fscale * p.Bpar, op.c->T_op);
        // Jc = 0 means the field is above the critical surface: a large finite fraction keeps the JSON
        // valid and the cost penalty monotone.
        const double f = jc > 0 ? J / jc : (J > 0 ? 1e30 : 0.0);
        // ties (several points above the critical surface, all at the 1e30 sentinel) go to the
        // largest field, so the reported point does not depend on the element order
        if (f > frac || (f == frac && worst && p.B > worst->B)) { frac = f; jc_w = jc; worst = &p; }
      }
      AXF_REQUIRE(worst != nullptr, "winding '" + op.w->name + "' has no field evaluation points");
      const double J = cscale * worst->Je / op.c->fill_factor;
      const double Bp = fscale * worst->Bperp, Bt = fscale * worst->Bpar;
      // load line: both the current and the field grow with the magnet current
      const double ss = load_line_scale(*op.law, Bp, Bt, J, op.c->T_op);
      json j = {{"conductor", op.w->conductor},
                {"law", op.law->law()},
                {"T_op_K", op.c->T_op},
                {"fill_factor", op.c->fill_factor},
                {"area_m2", lscale * lscale * op.area},
                {"ampere_turns_A", cscale * lscale * lscale * op.NI},
                {"J_engineering_A_m2", cscale * worst->Je},
                {"J_engineering_mean_A_m2", cscale * op.Jsum / op.area},
                {"J_superconductor_A_m2", J},
                {"B_peak_T", fscale * op.B_peak},
                {"B_peak_location", {{"r_m", lscale * op.peak_r}, {"z_m", lscale * op.peak_z}}},
                {"worst_point",
                 {{"r_m", lscale * worst->r}, {"z_m", lscale * worst->z}, {"B_T", fscale * worst->B},
                  {"B_perp_T", Bp}, {"B_par_T", Bt},
                  {"field_angle_to_tape_deg", std::atan2(Bp, Bt) * 180 / pi}}},
                {"Jc_A_m2", jc_w},
                {"load_line_fraction", frac},
                {"short_sample", {{"scale", ss}, {"J_superconductor_A_m2", ss * J}, {"B_T", ss * fscale * worst->B}}},
                {"ok", frac <= 1.0}};
      if (op.law->temperature_dependent()) {
        const double Tcs = current_sharing_T(*op.law, Bp, Bt, J, op.c->T_op);
        j["current_sharing_T_K"] = Tcs;
        j["temperature_margin_K"] = Tcs - op.c->T_op;
      } else {
        j["temperature_margin"] = "not available: this conductor's Jc does not depend on temperature "
                                  "(a field-only fit or table); give the temperature dependence to get a margin";
      }
      if (jc_w <= 0)
        j["status"] = "the field at the worst point is above the critical surface of this conductor "
                      "(Jc = 0): load_line_fraction is reported as 1e30";
      if (op.w->type != spec::WindingType::Azimuthal)
        j["current_note"] =
            "poloidal winding: the current density is J = grad(G)/r (uniform-resistivity distribution) and "
            "varies over the cross-section; J_engineering_A_m2 is its value at the worst point and "
            "J_engineering_mean_A_m2 the area average. At a sharp inner corner of the winding path J is "
            "singular and the peak grows under mesh refinement, as the superconductor surface field does; "
            "round the corner (shape fillet) for a mesh-independent margin.";
      if (op.law->anisotropic())
        j["note"] = "anisotropic conductor: B_perp is the component normal to the tape plane "
                    "(conductors.tape_normal); the worst point is the point of largest J/Jc on the winding";
      out[op.w->name] = j;
    }
    return out;
  };
  if (!conds.empty()) {
    R["conductors"] = cond_json(1.0, 1.0);
    for (auto& [n, j] : R["conductors"].items()) {
      const double jw = j["J_engineering_A_m2"], jm = j["J_engineering_mean_A_m2"];
      if (jm > 0 && jw > 1.5 * jm)
        warnings.push_back(strcat("winding '", n, "': the current density at the worst point (", jw,
                                  " A/m^2) is ", jw / jm,
                                  " times the area average, i.e. current crowding at a corner of the winding "
                                  "path. There J is singular and the margin grows under mesh refinement; round "
                                  "the corner (shape fillet) or read J_engineering_mean_A_m2, which does not"));
      if (!j["ok"].get<bool>())
        warnings.push_back(strcat("winding '", n, "' runs at ",
                                  j.contains("status") ? std::string("a field above the critical surface")
                                                       : strcat(j["load_line_fraction"].get<double>(),
                                                                " times the critical current density"),
                                  " of its conductor '", j["conductor"].get<std::string>(), "' (",
                                  j["worst_point"]["B_T"].get<double>(), " T, ", j["T_op_K"].get<double>(),
                                  " K): it would not stay superconducting"));
    }
  }

  // ------------------------------------------------------------ winding forces
  json forces = json::object();
  {
    PointEval pe;
    for (int e = 0; e < ne; ++e) {
      std::string reg = region_name(m.GetAttribute(e));
      const spec::Winding* w = nullptr;
      for (auto& ww : d_.windings) if (ww.region == reg) w = &ww;
      if (!w) continue;
      for (int q = 0; q < tab.nq; ++q) {
        sp.eval(e, tab, q, pe);
        Sample s = eval_pe(e, pe);
        json& fw = forces[w->name];
        if (!fw.is_object()) fw = {{"Fz_N", 0.0}, {"radial_force_per_radian_N", 0.0}};
        if (w->type == spec::WindingType::Azimuthal && a_lab_) {
          double J = a_lab_->roles()[e].J + (coil_J_.empty() ? 0.0 : coil_J_[e]);
          fw["Fz_N"] = fw["Fz_N"].get<double>() - 2 * pi * J * s.Br * pe.r * pe.w;
          fw["radial_force_per_radian_N"] = fw["radial_force_per_radian_N"].get<double>() + J * s.Bz * pe.r * pe.w;
        } else if (w->type == spec::WindingType::Poloidal && h_lab_) {
          // G = La + (Lb - La) r u_s  ->  J_r = -dG/dz / r, J_z = dG/dr / r
          int p = h_lab_->roles()[e].profile;
          if (p < 0) continue;  // element of the winding region that carries no current profile
          double dL = h_lab_sol_.Lb[p] - h_lab_sol_.La[p];
          const auto& dd = sp.edofs(e);
          double us = 0, dr = 0, dz = 0;
          for (size_t i = 0; i < dd.size(); ++i) {
            us += h_lab_sol_.unit(dd[i]) * pe.N[i];
            dr += h_lab_sol_.unit(dd[i]) * pe.dN[2 * i];
            dz += h_lab_sol_.unit(dd[i]) * pe.dN[2 * i + 1];
          }
          double dGdr = dL * (us + pe.r * dr), dGdz = dL * pe.r * dz;
          double Jr = -dGdz / pe.r, Jz = dGdr / pe.r;
          fw["Fz_N"] = fw["Fz_N"].get<double>() + 2 * pi * Jr * s.Bphi * pe.r * pe.w;
          fw["radial_force_per_radian_N"] = fw["radial_force_per_radian_N"].get<double>() - Jz * s.Bphi * pe.r * pe.w;
        }
      }
    }
    for (auto& [b, fz] : sc_force_z) forces["superconductor:" + b] = {{"Fz_N", fz}};
  }

  // ----------------------------------------------- winding self-force floor
  // The net axial force a rigid current loop exerts on itself is exactly zero, so whatever the
  // discretization returns for int J_w x B_w(vacuum) is pure error.  For a linear problem the field
  // superposes, so solving the same mesh with winding w alone in vacuum and subtracting that force
  // removes the self-force term and leaves the force from every other source -- including the
  // superconducting screening currents that w itself induces, which are physical.  The removal is
  // algebraically exact only where both solves see the same FE space over the winding (the vacuum
  // problem has no tied superconductor DOFs); otherwise it is as good as the two discretizations
  // agree, which the closure check then measures.
  // The hoop load is NOT corrected: a loop's radial self-repulsion is real.
  std::map<std::string, double> self_Fz, self_hoop;
  if (o.winding_self_force && a_lab_) {
    std::vector<const spec::Winding*> az;
    for (auto& w : d_.windings) {
      if (w.type != spec::WindingType::Azimuthal) continue;
      double jmax = 0;
      for (int e = 0; e < ne; ++e) {
        if (region_name(m.GetAttribute(e)) != w.region) continue;
        jmax = std::max(jmax, std::fabs(a_lab_->roles()[e].J + (coil_J_.empty() ? 0.0 : coil_J_[e])));
      }
      if (jmax > 0) az.push_back(&w);
    }
    // One extra linear solve per powered azimuthal winding.  The limit covers designs with a dozen
    // windings (e.g. ten passive packs); above it the cost would dominate the run, so the estimate is
    // skipped and said so.  `self_force = false` switches the whole thing off.
    const size_t kMaxSelfSolves = 12;
    if (az.size() > kMaxSelfSolves) {
      warnings.push_back(strcat("winding self-force floor not estimated: ", az.size(),
                                " powered azimuthal windings would need one extra linear solve each "
                                "(limit ", kMaxSelfSolves, "); forces carry an unquantified self-force residual"));
    } else {
      if (az.size() > 1)
        log(1, strcat("self-force floor: ", az.size(), " extra vacuum solves (one per powered azimuthal "
                      "winding; pass self_force = false to skip them)"));
      // the vacuum operator is the same for every winding: factor it once (solve_tangent caches the
      // factorization) and solve one right-hand side per winding
      std::vector<ARole> v0(ne);
      for (int e = 0; e < ne; ++e) v0[e].nu = 1.0 / mu0;  // vacuum everywhere: no superconductors, no iron
      PoloidalProblem base(sp, v0, {});
      base.set_open_boundary(kOuterAttr, mb_.domain_radius, 80, mb_.domain_center_z);
      for (const spec::Winding* w : az) {
        std::vector<ARole> vr(v0);
        for (int e = 0; e < ne; ++e)
          if (region_name(m.GetAttribute(e)) == w->region)
            vr[e].J = a_lab_->roles()[e].J + (coil_J_.empty() ? 0.0 : coil_J_[e]);
        PoloidalProblem vac(sp, vr, {});
        vac.set_open_boundary(kOuterAttr, mb_.domain_radius, 80, mb_.domain_center_z);
        std::vector<double> g, gc, y, yc, y0;
        vac.residual(y0, {}, g, gc);  // at y = 0: g = -(source)
        for (double& v : g) v = -v;
        base.solve_tangent(g, {}, y, yc);
        ASolution vs = vac.evaluate(y);
        double fz = 0, fh = 0;
        PointEval pe;
        for (int e = 0; e < ne; ++e) {
          if (vr[e].J == 0.0) continue;
          for (int q = 0; q < tab.nq; ++q) {
            sp.eval(e, tab, q, pe);
            double Br, Bz;
            vac.field(vs, e, pe, Br, Bz);
            fz -= 2 * pi * vr[e].J * Br * pe.r * pe.w;
            fh += vr[e].J * Bz * pe.r * pe.w;
          }
        }
        self_Fz[w->name] = fz;
        self_hoop[w->name] = fh;
      }
    }
  }
  // the same predicate the solver uses to decide whether it must run Newton (axi_poloidal.cpp)
  const bool linear_static = a_lab_ && !std::any_of(a_lab_->roles().begin(), a_lab_->roles().end(),
                                                    [](const ARole& r) { return !r.master && r.bh; });
  for (auto& [name, fz] : self_Fz) {
    if (!forces.contains(name)) continue;
    json& fw = forces[name];
    fw["Fz_self_N"] = fz;                                  // numerical self-force: error bar on Fz_N
    fw["radial_force_per_radian_self_N"] = self_hoop[name];  // physical self-repulsion (not removed)
    if (linear_static) fw["Fz_excluding_self_N"] = fw["Fz_N"].get<double>() - fz;
    else fw["self_force_note"] = "nonlinear iron: Fz_self_N is the self-force residual, i.e. the error bar "
                                 "on Fz_N; it cannot be subtracted because the fields do not superpose";
  }

  // ----------------------------------------- distributed surface loads
  // Maxwell surface traction of the static field, evaluated in the vacuum just outside a body:
  //   t = (1/mu0) [ (B.n) B - |B|^2 n / 2 ],   n = outward unit normal of the body,
  // so that F = int t dA over the body boundary is the total force on everything it encloses
  // (Maxwell's stress theorem; exact where mu = mu0 on the surface).  On a superconductor B.n = 0
  // and t is the magnetic pressure -|B|^2/(2 mu0) n.  Reported per unit meridional length with the
  // azimuth integrated out: (f_r, f_z) = 2 pi r (t_r, t_z) [N/m], so sum(f_z dl) = F_z.
  json distributions = json::object();
  json closure = json::object();
  double sc_Bn_rel = 0;
  {
    auto mat_of_elem = [&](int e) -> const spec::Material* {
      const spec::Region* g = d_.find_region(region_name(m.GetAttribute(e)));
      return g ? &d_.materials.at(g->material) : nullptr;  // null: "air"
    };
    // Bodies whose surface load we report: material bodies and winding regions, plus sheet curves.
    struct Target {
      std::string name, kind;
      std::string region;  // region body (empty for a sheet)
      int curve_attr = -1;
    };
    std::vector<Target> targets;
    for (auto& g : d_.regions) {
      const auto& mt = d_.materials.at(g.material);
      const bool wind = std::any_of(d_.windings.begin(), d_.windings.end(),
                                    [&](auto& w) { return w.region == g.name; });
      std::string kind;
      if (wind) {
        kind = "winding";  // whatever the winding region is made of: it carries current
      } else {
        switch (mt.kind) {
          case spec::MaterialKind::Superconductor: kind = "superconductor"; break;
          case spec::MaterialKind::Conductor: kind = "conductor"; break;
          case spec::MaterialKind::Iron: kind = "iron"; break;
          case spec::MaterialKind::Linear:
            // a magnetizable body feels a real force, and the closure check needs it
            if (mt.mu_r != 1.0) kind = "magnetic";
            break;
          default: break;
        }
      }
      if (!kind.empty()) targets.push_back({g.name, kind, g.name, -1});
    }
    for (auto& c : d_.curves) {
      if (c.thin_layer()) continue;  // forces on thin magnetic layers are not computed
      const auto k = d_.materials.at(c.material).kind;
      targets.push_back({c.name, k == spec::MaterialKind::Superconductor ? "superconductor" : "conductor", "",
                         mb_.attr_of_curve.at(c.name)});
    }
    const mfem::IntegrationRule& irf = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp.order + 6);
    // faces carrying a thin magnetic layer: the traction there misses the layer's own stress
    std::set<int> layer_faces;
    {
      std::set<int> lattr;
      for (auto& c : d_.curves)
        if (c.thin_layer()) lattr.insert(mb_.attr_of_curve.at(c.name));
      for (int be = 0; be < m.GetNBE() && !lattr.empty(); ++be)
        if (lattr.count(m.GetBdrAttribute(be))) layer_faces.insert(m.GetBdrElementFaceIndex(be));
    }
    // Consecutive pieces are merged down to this many per body; the merge preserves the integral, so
    // the net force is the same whatever the cap.  0 means "net forces only, no arrays": the arrays
    // are ~160 bytes per sample per body and would otherwise dominate results.json.
    const size_t kMaxSamples = o.force_samples > 0 ? static_cast<size_t>(o.force_samples) : 1;
    const bool want_arrays = o.force_samples > 0;
    for (const Target& tg : targets) {
      // boundary faces of the body, with the exterior element on each side
      std::vector<int> faces;
      std::map<int, std::vector<std::pair<int, int>>> side_of;  // face -> (exterior element, which)
      for (int f = 0; f < m.GetNumFaces(); ++f) {
        int e1, e2;
        m.GetFaceElements(f, &e1, &e2);
        std::vector<std::pair<int, int>> sides;
        if (tg.curve_attr >= 0) {
          if (sp.face_bdr_attr[f] != tg.curve_attr) continue;
          if (e1 >= 0) sides.push_back({e1, 1});
          if (e2 >= 0) sides.push_back({e2, 2});
        } else {
          const bool a = e1 >= 0 && region_name(m.GetAttribute(e1)) == tg.region;
          const bool b = e2 >= 0 && region_name(m.GetAttribute(e2)) == tg.region;
          if (a == b) continue;
          if (a && e2 >= 0) sides.push_back({e2, 2});
          else if (b && e1 >= 0) sides.push_back({e1, 1});
          else continue;  // body touches the domain boundary: no exterior element
        }
        if (sides.empty()) continue;
        faces.push_back(f);
        side_of[f] = sides;
      }
      if (faces.empty()) continue;
      std::vector<LoadSample> raw;
      double Fr_tot = 0, Fz_tot = 0, len_tot = 0, len_nonvac = 0, Bn_max = 0, B_max = 0;
      double s_run = 0;
      mfem::Vector nor(2), x(2);
      for (const auto& chain : order_face_chains(sp, faces)) {
        for (int f : chain) {
          LoadSample smp;
          // tractions are summed over the sides (a sheet is loaded from both faces); the geometry of
          // the piece is taken from the first side only, so `dl` is its true length
          double tr_i = 0, tz_i = 0, nr_i = 0, nz_i = 0, glen = 0;
          bool nonvac_face = false, first_side = true;
          mfem::FaceElementTransformations* FT = const_cast<mfem::Mesh&>(m).GetFaceElementTransformations(f);
          for (auto [ev, which] : side_of[f]) {
            const spec::Material* em = mat_of_elem(ev);
            const double mu_ext = em ? (em->kind == spec::MaterialKind::Linear ||
                                        em->kind == spec::MaterialKind::Iron ? mu0 * em->mu_r : mu0)
                                     : mu0;
            for (int q = 0; q < irf.GetNPoints(); ++q) {
              FT->SetAllIntPoints(&irf.IntPoint(q));
              const mfem::IntegrationPoint& eip =
                  which == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
              Sample sv = eval(ev, eip);
              mfem::CalcOrtho(FT->Jacobian(), nor);  // element 1 -> element 2, |nor| = dl/dxi
              FT->Transform(irf.IntPoint(q), x);
              const double sgn = which == 1 ? -1.0 : 1.0;  // outward normal of the body
              const double nx = sgn * nor(0), ny = sgn * nor(1);
              const double nl = std::hypot(nx, ny);
              if (!(nl > 0)) continue;
              const double ur = nx / nl, uz = ny / nl;
              const double Bn = sv.Br * ur + sv.Bz * uz;
              const double B2 = sv.Br * sv.Br + sv.Bz * sv.Bz + sv.Bphi * sv.Bphi;
              const double tr = (Bn * sv.Br - 0.5 * B2 * ur) / mu0;
              const double tz = (Bn * sv.Bz - 0.5 * B2 * uz) / mu0;
              const double dl = nl * irf.IntPoint(q).weight;
              tr_i += tr * dl;
              tz_i += tz * dl;
              smp.Fr += 2 * pi * x(0) * tr * dl;
              smp.Fz += 2 * pi * x(0) * tz * dl;
              if (first_side) {
                glen += dl;
                smp.r += x(0) * dl;
                smp.z += x(1) * dl;
                nr_i += ur * dl;
                nz_i += uz * dl;
              }
              if (tg.kind == "superconductor") {
                Bn_max = std::max(Bn_max, std::fabs(Bn));
                B_max = std::max(B_max, std::sqrt(B2));
              }
              if (mu_ext != mu0) nonvac_face = true;
            }
            first_side = false;
          }
          if (!(glen > 0)) continue;
          smp.dl = glen;
          smp.r /= glen;
          smp.z /= glen;
          smp.tr = tr_i / glen;  // sum of the per-side tractions
          smp.tz = tz_i / glen;
          smp.nr = nr_i / glen;  // outward normal of the first side (a sheet has two)
          smp.nz = nz_i / glen;
          smp.s = s_run + 0.5 * glen;
          s_run += glen;
          if (layer_faces.count(f)) nonvac_face = true;
          if (nonvac_face) len_nonvac += glen;
          Fr_tot += smp.Fr;
          Fz_tot += smp.Fz;
          len_tot += smp.dl;
          raw.push_back(smp);
        }
      }
      if (raw.empty()) continue;
      // merge consecutive pieces so the output stays bounded; sum(f_z dl) still equals F_z
      const size_t group = raw.size() <= kMaxSamples ? 1 : (raw.size() + kMaxSamples - 1) / kMaxSamples;
      std::vector<LoadSample> out;
      for (size_t i = 0; i < raw.size(); i += group) {
        LoadSample b;
        for (size_t k = i; k < std::min(i + group, raw.size()); ++k) {
          const LoadSample& a = raw[k];
          b.dl += a.dl;
          b.r += a.r * a.dl;
          b.z += a.z * a.dl;
          b.s += a.s * a.dl;
          b.nr += a.nr * a.dl;
          b.nz += a.nz * a.dl;
          b.tr += a.tr * a.dl;
          b.tz += a.tz * a.dl;
          b.Fr += a.Fr;
          b.Fz += a.Fz;
        }
        for (double* v : {&b.r, &b.z, &b.s, &b.nr, &b.nz, &b.tr, &b.tz}) *v /= b.dl;
        out.push_back(b);
      }
      json jr = json::array(), jz = json::array(), js = json::array(), jdl = json::array();
      json jnr = json::array(), jnz = json::array(), jtr = json::array(), jtz = json::array();
      json jfr = json::array(), jfz = json::array();
      for (const LoadSample& a : out) {
        jr.push_back(a.r); jz.push_back(a.z); js.push_back(a.s); jdl.push_back(a.dl);
        jnr.push_back(a.nr); jnz.push_back(a.nz); jtr.push_back(a.tr); jtz.push_back(a.tz);
        jfr.push_back(a.Fr / a.dl); jfz.push_back(a.Fz / a.dl);
      }
      json dist = {{"kind", tg.kind}, {"geometry", "surface"}, {"Fz_N", Fz_tot}, {"Fr_N", Fr_tot},
                   {"length_m", len_tot}, {"samples", want_arrays ? out.size() : 0}, {"faces", raw.size()},
                   {"exterior_nonvacuum_fraction", len_tot > 0 ? len_nonvac / len_tot : 0.0}};
      if (want_arrays) {
        dist.update({{"r_m", jr}, {"z_m", jz}, {"arclength_m", js}, {"dl_m", jdl},
                     {"n_r", jnr}, {"n_z", jnz},
                     {"traction_r_Pa", jtr}, {"traction_z_Pa", jtz},
                     {"f_r_N_per_m", jfr}, {"f_z_N_per_m", jfz}});
      }
      std::string key = tg.name;
      if (tg.kind == "winding") {
        // The net force on a winding is a tiny residue of two huge opposing tractions (the field
        // jumps by mu0 J t across the conductor), so the surface integral is far less accurate than
        // the volume integral of J x B.  Report it, but say by how much it differs.
        key = tg.name + ":boundary";
        double fz_lor = 0;
        for (auto& w : d_.windings)
          if (w.region == tg.region && forces.contains(w.name)) fz_lor = forces[w.name]["Fz_N"].get<double>();
        dist["Fz_lorentz_N"] = fz_lor;
        dist["note"] = "surface traction on a current-carrying body: the net force is a small difference of "
                       "large opposing tractions and is much less accurate than Fz_lorentz_N (the volume "
                       "integral of J x B, reported as the winding's own 'volume' distribution)";
      }
      distributions[key] = dist;
      if (tg.kind == "superconductor" && B_max > 0) sc_Bn_rel = std::max(sc_Bn_rel, Bn_max / B_max);
      if (len_nonvac > 0.01 * len_tot)
        warnings.push_back(strcat("the distributed surface load of '", tg.name, "' is not reliable on ",
                                  100.0 * len_nonvac / len_tot,
                                  "% of its boundary, where the neighbouring material is not magnetically vacuum "
                                  "or a thin magnetic layer lies on the surface (the Maxwell stress tensor used here "
                                  "assumes mu = mu0 on the surface; forces on layers are not computed)"));
      if (tg.kind == "iron" || tg.kind == "magnetic")
        forces[tg.kind + ":" + tg.name] = {{"Fz_N", Fz_tot}};
    }
    // Volumetric Lorentz load of each winding: f = J x B per unit volume at the element centroids,
    // with the azimuth integrated out, so that sum(f_z dV) is exactly the reported Fz_N.
    for (auto& w : d_.windings) {
      std::vector<LoadSample> cells;  // r, z, dl := dV, Fr, Fz
      PointEval pe;
      double Fr_tot = 0, Fz_tot = 0;
      for (int e = 0; e < ne; ++e) {
        if (region_name(m.GetAttribute(e)) != w.region) continue;
        LoadSample c;
        for (int q = 0; q < tab.nq; ++q) {
          sp.eval(e, tab, q, pe);
          Sample s = eval_pe(e, pe);
          const double dV = 2 * pi * pe.r * pe.w;
          double fr = 0, fz = 0;  // Lorentz force density (N/m^3)
          if (w.type == spec::WindingType::Azimuthal && a_lab_) {
            const double J = a_lab_->roles()[e].J + (coil_J_.empty() ? 0.0 : coil_J_[e]);
            fr = J * s.Bz;
            fz = -J * s.Br;
          } else if (w.type == spec::WindingType::Poloidal && h_lab_) {
            const int p = h_lab_->roles()[e].profile;
            if (p < 0) continue;
            const double dL = h_lab_sol_.Lb[p] - h_lab_sol_.La[p];
            const auto& dd = sp.edofs(e);
            double us = 0, dr = 0, dz = 0;
            for (size_t i = 0; i < dd.size(); ++i) {
              us += h_lab_sol_.unit(dd[i]) * pe.N[i];
              dr += h_lab_sol_.unit(dd[i]) * pe.dN[2 * i];
              dz += h_lab_sol_.unit(dd[i]) * pe.dN[2 * i + 1];
            }
            const double Jr = -dL * pe.r * dz / pe.r, Jz = dL * (us + pe.r * dr) / pe.r;
            fr = -Jz * s.Bphi;
            fz = Jr * s.Bphi;
          }
          c.dl += dV;        // dl holds the volume here
          c.r += pe.r * dV;
          c.z += pe.z * dV;
          c.tr += fr * dV;   // tr/tz hold the volume-weighted density
          c.tz += fz * dV;
          c.Fr += fr * dV;
          c.Fz += fz * dV;
        }
        if (!(c.dl > 0)) continue;
        c.r /= c.dl;
        c.z /= c.dl;
        c.tr /= c.dl;
        c.tz /= c.dl;
        Fr_tot += c.Fr;
        Fz_tot += c.Fz;
        cells.push_back(c);
      }
      if (cells.empty()) continue;
      std::sort(cells.begin(), cells.end(),
                [](const LoadSample& a, const LoadSample& b) { return a.z != b.z ? a.z < b.z : a.r < b.r; });
      const size_t group = cells.size() <= kMaxSamples ? 1 : (cells.size() + kMaxSamples - 1) / kMaxSamples;
      json jr = json::array(), jz = json::array(), jv = json::array();
      json jdr = json::array(), jdz = json::array(), jfr = json::array(), jfz = json::array();
      for (size_t i = 0; i < cells.size(); i += group) {
        LoadSample b;
        for (size_t k = i; k < std::min(i + group, cells.size()); ++k) {
          const LoadSample& a = cells[k];
          b.dl += a.dl;
          b.r += a.r * a.dl;
          b.z += a.z * a.dl;
          b.tr += a.tr * a.dl;
          b.tz += a.tz * a.dl;
          b.Fr += a.Fr;
          b.Fz += a.Fz;
        }
        for (double* v : {&b.r, &b.z, &b.tr, &b.tz}) *v /= b.dl;
        jr.push_back(b.r); jz.push_back(b.z); jv.push_back(b.dl);
        jdr.push_back(b.tr); jdz.push_back(b.tz); jfr.push_back(b.Fr); jfz.push_back(b.Fz);
      }
      json vd = {{"kind", "winding"}, {"geometry", "volume"}, {"Fz_N", Fz_tot}, {"Fr_N", Fr_tot},
                 {"samples", want_arrays ? jr.size() : 0}, {"cells", cells.size()}};
      if (want_arrays) {
        vd.update({{"r_m", jr}, {"z_m", jz}, {"dV_m3", jv},
                   {"force_density_r_N_per_m3", jdr}, {"force_density_z_N_per_m3", jdz},
                   {"f_r_N", jfr}, {"f_z_N", jfz}});
      }
      distributions[w.name] = vd;
    }
    if (sc_Bn_rel > 0)
      checks.push_back({{"name", "superconductor surfaces: normal field B.n = 0"},
                        {"max_Bn_over_Bmax", sc_Bn_rel}});

    // ------------------- Newton's third law closure: sum of all forces = 0
    // The device is isolated and the field decays, so the axial forces on all its bodies must cancel.
    // What is left measures the accuracy of the force post-processing: on CORE it is dominated by the
    // slowly converging Maxwell pressure on the superconducting cavity.
    // The scale is the largest force in the device, including the hoop load 2 pi dF/dphi, so that an
    // up/down symmetric design (every Fz exactly 0) is not judged against its own round-off.
    double sum = 0, scale = 0;
    json parts = json::array();
    auto add = [&](const std::string& what, double fz) {
      sum += fz;
      scale = std::max(scale, std::fabs(fz));
      parts.push_back({{"body", what}, {"Fz_N", fz}});
    };
    for (auto& w : d_.windings) {
      if (!forces.contains(w.name)) continue;
      const json& fw = forces[w.name];
      // the self-corrected value when we have it: the raw one carries the self-force floor
      const bool corrected = fw.contains("Fz_excluding_self_N");
      add("winding:" + w.name, fw[corrected ? "Fz_excluding_self_N" : "Fz_N"].get<double>());
      if (corrected) parts.back()["self_force_removed"] = true;
      scale = std::max(scale, 2 * pi * std::fabs(fw.value("radial_force_per_radian_N", 0.0)));
    }
    for (auto& [b, fz] : sc_force_z) add("superconductor:" + b, fz);
    for (auto it = distributions.begin(); it != distributions.end(); ++it) {
      const std::string bk = it.value()["kind"];
      if (bk == "iron" || bk == "magnetic") add(bk + ":" + it.key(), it.value()["Fz_N"].get<double>());
    }
    std::vector<std::string> caveats;
    {  // windings whose self-force floor is still inside the Fz_N that went into the sum
      std::string raw;
      for (auto& w : d_.windings)
        if (w.type == spec::WindingType::Azimuthal && forces.contains(w.name) &&
            !forces[w.name].contains("Fz_self_N"))
          raw += (raw.empty() ? "" : ", ") + w.name;
      if (!raw.empty())
        caveats.push_back("the self-force floor of " + raw + " was not estimated (self_force off, or "
                          "too many powered azimuthal windings), so the sum still contains it; it can "
                          "exceed the physical force");
    }
    const bool external = bg_.active();
    if (external) caveats.push_back("an external background field acts on the device: its source lies outside "
                                    "the model, so the forces inside need not sum to zero");
    if (a_lab_ && !linear_static)  // a design without a poloidal static problem (a toroid) has no iron force
      caveats.push_back("nonlinear iron: the iron force is the Maxwell stress of the vacuum field just outside, "
                        "which converges slowly at corners");
    // A closed axisymmetric poloidal winding feels no net axial force from its own toroidal field:
    //   F_z = int J_r B_phi dV = -2 pi int (mu G / r^2) dG/dz r dr dz = -pi int (mu/r) [G^2] dr,
    // and G vanishes above and below the winding, so the bracket is zero at every radius whatever
    // the cross-section looks like -- provided mu is the same on the way up as on the way down.
    // So there is nothing to estimate in the usual case; it is iron inside the winding's legs, or a
    // second poloidal winding leaving G nonzero at its ends, that would break it.
    {
      const bool iron = std::any_of(d_.regions.begin(), d_.regions.end(), [&](const spec::Region& g) {
        return d_.materials.at(g.material).kind == spec::MaterialKind::Iron;
      });
      int npol = 0;
      for (auto& w : d_.windings)
        if (w.type == spec::WindingType::Poloidal) ++npol;
      if (npol > 0 && (iron || npol > 1))
        caveats.push_back("the axial force on a poloidal winding is identically zero in its own field "
                          "when the permeability is uniform along its legs; with iron or more than one "
                          "poloidal winding that cancellation can break and the residual is not estimated");
    }
    const bool with_layers = std::any_of(d_.curves.begin(), d_.curves.end(), [](auto& c) { return c.thin_layer(); });
    if (with_layers)
      caveats.push_back("the forces on thin magnetic layers are not computed, so the sum leaves them out and is not "
                        "expected to vanish");
    closure = {{"sum_Fz_N", sum}, {"force_scale_N", scale},
               {"relative", scale > 0 ? sum / scale : 0.0}, {"bodies", parts}};
    if (!caveats.empty()) closure["caveats"] = caveats;
    if (!parts.empty())
      checks.push_back({{"name", "Newton's third law: total axial force on all bodies"},
                        {"sum_Fz_N", sum}, {"force_scale_N", scale},
                        {"relative", scale > 0 ? sum / scale : 0.0},
                        {"exact_value", 0.0},
                        {"valid", !external && !with_layers}});
  }
  if (!distributions.empty()) {
    // "distributions"/"closure" are reserved keys of `forces`; a body of that name would collide
    for (const char* k : {"distributions", "closure"}) {
      if (forces.contains(k)) {
        forces[std::string("body:") + k] = forces[k];
        forces.erase(k);
        warnings.push_back(strcat("'", k, "' is a reserved key of forces; that body's net force is reported "
                                  "under 'body:", k, "'"));
      }
    }
    forces["distributions"] = distributions;
    forces["closure"] = closure;
  }
  R["forces"] = forces;

  // ------------------------------------------------------------ RF surface losses of the pickup mode
  // Gamma_b = int |H_t|^2 dA / I^2 over the surfaces of conductor body b (vacuum side);
  // R_eff = R_s(f) Gamma, Q_b = 2 pi f L_p / R_eff, and the loss-weighted T/Q = sum T_b / Q_b.
  if ((a_pick_ || h_pick_) && Lp > 0) {
    auto mat_of = [&](int e) -> const spec::Material* {
      const spec::Region* g = d_.find_region(region_name(m.GetAttribute(e)));
      return g ? &d_.materials.at(g->material) : nullptr;
    };
    auto is_rfc = [&](int e) {
      const auto* mt = mat_of(e);
      return mt && (mt->kind == spec::MaterialKind::Superconductor || mt->kind == spec::MaterialKind::Conductor);
    };
    std::map<int, std::string> sheet_name;
    for (auto& c : d_.curves)
      if (!c.thin_layer()) sheet_name[mb_.attr_of_curve.at(c.name)] = c.name;
    std::map<std::string, double> gamma, gamma_mr;  // gamma_mr: weighted by the magnetoresistance factor of R_s
    std::map<std::string, const spec::Material*> body_mat;
    const mfem::IntegrationRule& ir1 = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp.order + 6);
    PointEval pe2;
    for (int f = 0; f < m.GetNumFaces(); ++f) {
      int e1, e2;
      m.GetFaceElements(f, &e1, &e2);
      std::vector<std::pair<int, int>> sides;
      std::string body;
      auto it = sheet_name.find(sp.face_bdr_attr[f]);
      if (it != sheet_name.end()) {
        body = it->second;
        body_mat[body] = &d_.materials.at(d_.find_curve(body)->material);
        if (e1 >= 0 && !is_rfc(e1)) sides.push_back({e1, 1});
        if (e2 >= 0 && !is_rfc(e2)) sides.push_back({e2, 2});
      } else if (e1 >= 0 && e2 >= 0 && is_rfc(e1) != is_rfc(e2)) {
        int c = is_rfc(e1) ? e1 : e2;
        body = region_name(m.GetAttribute(c));
        body_mat[body] = mat_of(c);
        sides.push_back(is_rfc(e1) ? std::make_pair(e2, 2) : std::make_pair(e1, 1));
      }
      if (sides.empty()) continue;
      mfem::FaceElementTransformations* FT = const_cast<mfem::Mesh&>(m).GetFaceElementTransformations(f);
      for (auto [ev, which] : sides) {
        for (int q = 0; q < ir1.GetNPoints(); ++q) {
          FT->SetAllIntPoints(&ir1.IntPoint(q));
          const mfem::IntegrationPoint& eip = which == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
          sp.eval_at(ev, eip, pe2);
          Sample smp = eval_pe(ev, pe2);
          const auto* mt = mat_of(ev);
          double mu = (mt && (mt->kind == spec::MaterialKind::Linear || mt->kind == spec::MaterialKind::Iron))
                          ? mu0 * mt->mu_r : mu0;
          // iron: the RF (secant) permeability the pickup solve used, not the initial mu_r
          if (mt && mt->kind == spec::MaterialKind::Iron && mu_rf_iron_) mu = mu_rf_iron_(ev, pe2);
          double H2 =(smp.pBr * smp.pBr + smp.pBz * smp.pBz + smp.pBphi * smp.pBphi) / (mu * mu);
          mfem::Vector nor(2);
          mfem::CalcOrtho(FT->Jacobian(), nor);
          const double dG = H2 * 2 * pi * pe2.r * nor.Norml2() * ir1.IntPoint(q).weight;
          gamma[body] += dG;
          const spec::Material* bm = body_mat[body];
          if (bm && !bm->rf.mr_model.empty()) {
            const double Bs = std::sqrt(smp.Br * smp.Br + smp.Bz * smp.Bz + smp.Bphi * smp.Bphi);
            gamma_mr[body] += dG * bm->rf.Rs_factor(Bs);
          }
        }
      }
    }
    json lj = json::object();
    double TQ_lo = 0, TQ_hi = 0;
    bool have_rf = false;
    const double f_lo = d_.scan.f_min, f_hi = d_.scan.f_max;
    for (auto& [b, G0] : gamma) {
      json e = {{"Gamma", G0}, {"note", "R_eff = R_s(f) * Gamma; Q = 2 pi f L_p / R_eff"}};
      const spec::Material* mt = body_mat[b];
      double G = G0;
      if (mt && !mt->rf.mr_model.empty()) {
        G = gamma_mr[b];
        e["Gamma_magnetoresistance"] = G;
        e["note"] = "R_eff = R_s(f) * Gamma_magnetoresistance (R_s weighted by the local static field); Q = 2 pi f L_p / R_eff";
      }
      if (mt && !mt->rf.model.empty()) {
        have_rf = true;
        double Rl = mt->rf.Rs(f_lo) * G, Rh = mt->rf.Rs(f_hi) * G;
        double Ql = 2 * pi * f_lo * Lp / Rl, Qh = 2 * pi * f_hi * Lp / Rh;
        e["R_eff_ohm"] = {{"f_min", Rl}, {"f_max", Rh}};
        e["Q"] = {{"f_min", Ql}, {"f_max", Qh}};
        e["T_K"] = mt->rf.T_K;
        e["T_over_Q_K"] = {{"f_min", mt->rf.T_K / Ql}, {"f_max", mt->rf.T_K / Qh}};
        TQ_lo += mt->rf.T_K / Ql;
        TQ_hi += mt->rf.T_K / Qh;
      }
      lj[b] = e;
    }
    json losses = {{"surfaces", lj}};
    if (have_rf) {
      losses["T_over_Q_total_K"] = {{"f_min", TQ_lo}, {"f_max", TQ_hi}};
      json cmp = json::array();
      for (auto& bm : d_.benchmarks)
        cmp.push_back({{"benchmark", bm.name}, {"T_over_Q_K", bm.T / bm.Q},
                       {"computed_over_benchmark", {{"f_min", TQ_lo / (bm.T / bm.Q)}, {"f_max", TQ_hi / (bm.T / bm.Q)}}}});
      losses["vs_benchmarks"] = cmp;
      losses["note"] = "only surfaces with rf loss data are included; materials without an rf block are lossless here";
    }
    R["pickup"]["losses"] = losses;
    std::set<std::string> rf_ignored;
    for (auto& g : d_.regions) {
      const spec::Material& gm = d_.materials.at(g.material);
      if (!gm.rf.model.empty() && gm.kind != spec::MaterialKind::Superconductor &&
          gm.kind != spec::MaterialKind::Conductor)
        rf_ignored.insert(g.material);
    }
    for (auto& name : rf_ignored)
      warnings.push_back(strcat("material '", name, "' has an rf block but is not a conductor or superconductor: ",
                                "pickup.losses counts only conductor and superconductor surfaces, so it has no loss"));
  }

  // ------------------------------------------------------------ readout-port gap capacitance
  // Computed by electrostatics_analysis() in cpp/src/axi_electro.cpp: the readout gap is solved as a
  // two-terminal electrostatic problem (pickup.port_capacitance).

  // ------------------------------------------------------------ winding flux linkage (per turn) and currents
  json wj = json::object();
  if (a_lab_) {
    std::map<std::string, double> phi, area_w, jsum;
    PointEval pe;
    for (int e = 0; e < ne; ++e) {
      std::string reg = region_name(m.GetAttribute(e));
      const spec::Winding* w = nullptr;
      for (auto& ww : d_.windings) if (ww.region == reg && ww.type == spec::WindingType::Azimuthal) w = &ww;
      if (!w) continue;
      for (int q = 0; q < tab.nq; ++q) {
        sp.eval(e, tab, q, pe);
        double A = 0, br_, bz_;
        static_poloidal(e, pe, br_, bz_, A);
        phi[w->name] += 2 * pi * pe.r * A * pe.w;
        area_w[w->name] += pe.w;
        jsum[w->name] += (a_lab_->roles()[e].J + (coil_J_.empty() ? 0.0 : coil_J_[e])) * pe.w;
      }
    }
    for (auto& [n, ph] : phi) {
      wj[n] = {{"flux_per_turn_Wb", ph / area_w[n]}, {"ampere_turns_A", jsum[n]}, {"area_m2", area_w[n]}};
    }
  }
  R["windings"] = wj;

}

// ---------------------------------------------------------------- full-wave analysis (beyond MQS)
// Time-harmonic Maxwell at the requested frequencies with the readout driven by I = 1 A (e^{-i omega t}).
// By reciprocity the axion EMF across the readout is V = -i omega kappa Psi(omega) for the effective
// current kappa B_lab, with Psi(omega) = (1/(i omega I)) int E_I . B_lab dV = (1/I) int A_lab . B_I dV
// (up to the exterior of the computational sphere); Psi -> the MQS Psi as omega -> 0.
void AxiModel::fullwave_analysis(const SolveOptions& o) {
  const auto& fw = d_.fullwave;
  if (!fw.enabled) return;
  (void)o;
  Timer timer;
  const Space& sp = *sp_;
  const int ne = sp.mesh.GetNE(), nd = sp.ndof();
  json& R = results_;
  json& warnings = R["warnings"];
  const double eps0 = 1.0 / (mu0 * units::c * units::c);
  const cplx I1(0.0, 1.0);
  std::vector<ElemInfo> info = element_info(d_, sp, [this](int a) { return region_name(a); });
  const bool radiation = fw.boundary == "radiation";
  // k_R uses the outer radius of all regions (coupling.device_radius_m, the R of the g factor)
  const double Rb = mb_.domain_radius, Rdev = R["coupling"].value("device_radius_m", mb_.device_radius);
  std::string driven;
  if (d_.pickup) driven = d_.pickup->kind == spec::Pickup::Kind::Azimuthal ? "TE" : "TM";
  AXF_REQUIRE(fw.frequencies.empty() || !driven.empty(), "a full-wave frequency sweep needs a pickup (readout)",
              "/fullwave", "Add a pickup, or request only eigenmodes (fullwave.modes).");
  std::vector<std::string> fams = fw.families;
  if (fams.empty() && fw.modes > 0)
    fams = driven.empty() ? std::vector<std::string>{"TE", "TM"} : std::vector<std::string>{driven};
  auto need = [&](const std::string& f) {
    return (driven == f && !fw.frequencies.empty()) ||
           (fw.modes > 0 && std::find(fams.begin(), fams.end(), f) != fams.end());
  };
  std::vector<double> eps(ne);
  for (int e = 0; e < ne; ++e) eps[e] = eps0 * info[e].mat->eps_r;
  // lossy media (dielectric loss tangent, bulk conductivity): one group per material
  std::map<std::string, int> group_of;
  std::vector<const spec::Material*> group_mat;
  std::vector<int> egroup(ne, -1);
  for (int e = 0; e < ne; ++e) {
    const auto* m = info[e].mat;
    if (is_sc(info[e]) || is_cond(info[e]) || (m->tan_delta <= 0 && m->sigma_S_m <= 0)) continue;
    auto it = group_of.find(m->name);
    if (it == group_of.end()) {
      it = group_of.emplace(m->name, static_cast<int>(group_mat.size())).first;
      group_mat.push_back(m);
    }
    egroup[e] = it->second;
  }
  // relative complex permittivity factor eps_c / eps' = 1 + i tan(delta) + i sigma / (omega eps')
  auto eps_factor = [eps0](const spec::Material* m, double w) {
    return cplx(1.0, m->tan_delta + (w > 0 ? m->sigma_S_m / (w * eps0 * m->eps_r) : 0.0));
  };
  std::map<std::string, double> area;
  std::vector<double> earea(ne);
  for (int e = 0; e < ne; ++e) {
    earea[e] = element_area(sp, e);
    area[info[e].region] += earea[e];
  }
  auto iron_mu = [&](int e) -> std::function<double(int, const PointEval&)> {
    if (info[e].mat->kind == spec::MaterialKind::Iron && mu_rf_iron_) return mu_rf_iron_;
    return nullptr;
  };
  json out = {{"boundary", fw.boundary},
              {"domain_radius_m", Rb},
              {"device_radius_m", Rdev},
              {"bounding_sphere_radius_m", mb_.device_radius},
              {"readout_current_A", 1.0},
              {"convention", "e^{-i omega t}; Z = V/I = R - i omega L; axion EMF V = -i omega kappa Psi(omega)"}};

  // wavelength resolution at the highest frequency
  if (!fw.frequencies.empty()) {
    const double fmax = *std::max_element(fw.frequencies.begin(), fw.frequencies.end());
    double worst = 0;
    std::string where;
    for (int e = 0; e < ne; ++e) {
      if (is_sc(info[e]) || is_cond(info[e])) continue;
      const double h = std::sqrt(4.0 * earea[e] / std::sqrt(3.0));
      const double kh = 2 * pi * fmax * std::sqrt(mu_static_of(info[e]) * eps[e]) * h;
      if (kh > worst) { worst = kh; where = info[e].region; }
    }
    out["mesh_kh_max"] = worst;
    if (worst > 0.5 * sp.order)
      warnings.push_back(strcat("full-wave: the mesh is coarse at ", fmax, " Hz (k h = ", worst, " in '", where,
                                "'); refine mesh.size (k h <~ ", 0.5 * sp.order, " for order ", sp.order, ")"));
    // conducting media: the skin depth must be resolved
    double hd_worst = 0;
    std::string hd_where;
    for (int e = 0; e < ne; ++e) {
      const auto* m = info[e].mat;
      if (egroup[e] < 0 || m->sigma_S_m <= 0) continue;
      const double delta = std::sqrt(2.0 / (2 * pi * fmax * mu_static_of(info[e]) * m->sigma_S_m));
      const double h = std::sqrt(4.0 * earea[e] / std::sqrt(3.0));
      if (h / delta > hd_worst) { hd_worst = h / delta; hd_where = info[e].region; }
    }
    if (hd_worst > 0) {
      out["mesh_h_over_skin_depth_max"] = hd_worst;
      if (hd_worst > 1.0)
        warnings.push_back(strcat("full-wave: elements in '", hd_where, "' are ", hd_worst,
                                  " skin depths at the highest frequency; eddy currents there are under-resolved "
                                  "(refine the region, or model it as a perfect conductor with surface losses)"));
    }
  }
  const RefTable& tab = sp.table(sp.quad_rule());

  // ------------------------------------------------------------ cavity domain (beta, docs/validation/cavity_2d.md)
  // The connected non-conductor component that contains cavity.interior_point (not crossing perfectly conducting
  // sheets) is the cavity; every other non-conductor element is excluded from the full-wave problems, so the
  // vacuum around the cavity wall adds no modes. Conductors around it stay as they are (TE masters, TM PEC).
  const auto& cav = d_.cavity;
  std::vector<char> in_cav(ne, 1);
  std::set<int> pec_sheet_attr;  // boundary attributes of perfectly conducting sheets (curves)
  for (auto& c : d_.curves) {
    const auto k = d_.materials.at(c.material).kind;
    if (k == spec::MaterialKind::Superconductor || k == spec::MaterialKind::Conductor)
      pec_sheet_attr.insert(mb_.attr_of_curve.at(c.name));
  }
  auto locate = [&](const spec::Vec2& p, int& e, PointEval& pe) {
    mfem::DenseMatrix pts(2, 1);
    pts(0, 0) = p.r;
    pts(1, 0) = p.z;
    mfem::Array<int> eid;
    mfem::Array<mfem::IntegrationPoint> ips;
    const_cast<mfem::Mesh&>(sp.mesh).FindPoints(pts, eid, ips, false);
    e = eid[0];
    if (e >= 0) {
      sp.eval_at(e, ips[0], pe);
      pe.ip = nullptr;
    }
  };
  double V_cav = 0;
  json cav_json;
  if (cav.enabled) {
    int e0 = -1;
    PointEval pe0;
    locate(cav.interior_point, e0, pe0);
    AXF_REQUIRE(e0 >= 0, strcat("cavity.interior_point (", cav.interior_point.r, ", ", cav.interior_point.z,
                                ") lies outside the mesh"),
                "/cavity/interior_point", "Give a point inside the cavity volume, in metres.");
    if (is_sc(info[e0]) || is_cond(info[e0])) {
      // a point on the cavity/wall interface: FindPoints may return the conductor element; take the
      // non-conductor element next to it (nudges far below any element size)
      const std::string creg = info[e0].region;
      const double dl = 1e-9 * std::max(Rb, 1e-3);
      int e1 = -1;
      PointEval pe1;
      for (auto [dr, dz] : std::vector<std::pair<double, double>>{{1, 0}, {-1, 0}, {0, 1}, {0, -1},
                                                                   {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}) {
        spec::Vec2 q{cav.interior_point.r + dr * dl, cav.interior_point.z + dz * dl};
        if (q.r < 0) continue;
        locate(q, e1, pe1);
        if (e1 >= 0 && !is_sc(info[e1]) && !is_cond(info[e1])) break;
        e1 = -1;
      }
      AXF_REQUIRE(e1 >= 0, strcat("cavity.interior_point lies inside the conductor region '", creg, "'"),
                  "/cavity/interior_point", "Give a point in the vacuum or dielectric inside the cavity wall.");
      warnings.push_back(strcat("cavity: interior_point (", cav.interior_point.r, ", ", cav.interior_point.z,
                                ") lies on the surface of the conductor region '", creg, "'; the adjacent '",
                                info[e1].region, "' element was used (move the point into the cavity to be explicit)"));
      e0 = e1;
    }
    int nc = 0;
    std::vector<int> comp = components(
        sp, [&](int e) { return !is_sc(info[e]) && !is_cond(info[e]); },
        [&](int f) { return !pec_sheet_attr.count(sp.face_bdr_attr[f]); }, nc);
    const int k0 = comp[e0];
    std::set<std::string> regs;
    for (int e = 0; e < ne; ++e) {
      in_cav[e] = comp[e] == k0;
      if (in_cav[e]) {
        regs.insert(info[e].region);
        PointEval pq;
        for (int q = 0; q < tab.nq; ++q) {
          sp.eval(e, tab, q, pq);
          V_cav += 2 * pi * pq.w * pq.r;
        }
      }
    }
    for (int be = 0; be < sp.mesh.GetNBE(); ++be) {
      if (sp.mesh.GetBdrAttribute(be) != kOuterAttr) continue;
      int e1, e2;
      sp.mesh.GetFaceElements(sp.mesh.GetBdrElementFaceIndex(be), &e1, &e2);
      AXF_REQUIRE(!(e1 >= 0 && in_cav[e1]),
                  "the cavity selected by cavity.interior_point is not enclosed: it reaches the outer sphere",
                  "/cavity/interior_point",
                  "Close the cavity with conductor regions (e.g. a copper wall) or move the point inside the wall.");
    }
    cav_json = {{"status", "beta (docs/validation/cavity_2d.md)"},
                {"interior_point", {cav.interior_point.r, cav.interior_point.z}},
                {"regions", regs},
                {"volume_m3", V_cav},
                {"excluded_components", nc - 1}};
  }

  // DOF fans: the elements accepted by `ok` that contain DOF d, partitioned by adjacency through faces that are not
  // perfectly conducting sheets. A sheet with the cavity on both sides splits the elements around each of its DOFs
  // into one fan per side; a fan is "clean" unless two of its elements are adjacent through a sheet face, which
  // happens only at a free edge of a sheet (the elements wrap around the edge).
  struct DofFans {
    std::vector<std::vector<int>> fans;
    std::vector<char> clean;
  };
  auto is_sheet_face = [&](int f) { return pec_sheet_attr.count(sp.face_bdr_attr[f]) > 0; };
  auto dof_fans = [&](const std::set<int>& dofs, const std::function<bool(int)>& ok) {
    std::map<int, std::vector<int>> star;
    for (int e = 0; e < ne; ++e)
      if (ok(e))
        for (int dd : sp.edofs(e))
          if (dofs.count(dd)) star[dd].push_back(e);
    std::map<int, DofFans> out;
    for (auto& [dd, els] : star) {
      std::map<int, int> lab;
      for (int e : els) lab[e] = -1;
      DofFans F;
      for (int e : els) {
        if (lab[e] >= 0) continue;
        const int id = static_cast<int>(F.fans.size());
        F.fans.emplace_back();
        F.clean.push_back(1);
        std::vector<int> stack{e};
        lab[e] = id;
        while (!stack.empty()) {
          const int x = stack.back();
          stack.pop_back();
          F.fans[id].push_back(x);
          for (auto [n, f] : sp.nbr[x]) {
            auto it = lab.find(n);
            if (it == lab.end() || it->second >= 0 || is_sheet_face(f)) continue;
            it->second = id;
            stack.push_back(n);
          }
        }
      }
      for (size_t k = 0; k < F.fans.size(); ++k)
        for (int x : F.fans[k])
          for (auto [n, f] : sp.nbr[x]) {
            auto it = lab.find(n);
            if (it != lab.end() && it->second == static_cast<int>(k) && is_sheet_face(f)) F.clean[k] = 0;
          }
      out[dd] = std::move(F);
    }
    return out;
  };

  // ------------------------------------------------------------ TE0 (azimuthal E): u = A_phi
  int pb = -1;
  if (need("TE")) {
    std::vector<int> body_of_elem;
    std::map<int, int> body_of_curve;
    auto bodies = find_bodies(sp, mb_, info, d_, true, body_of_elem, body_of_curve);
    std::vector<Master> masters;
    for (size_t b = 0; b < bodies.size(); ++b) {
      Master m;
      m.name = bodies[b].names.empty() ? "body" : bodies[b].names.front();
      const bool is_pick = driven == "TE" && std::find(bodies[b].names.begin(), bodies[b].names.end(),
                                                       d_.pickup->conductor) != bodies[b].names.end();
      const spec::Superconductor* st = sc_state(d_, bodies[b].names);
      if (bodies[b].on_axis) {
        AXF_REQUIRE(!is_pick, "the pickup conductor touches the symmetry axis", "/pickup/conductor");
        m.fixed = true;
      } else if (is_pick) {
        m.current = 1.0;
        pb = static_cast<int>(b);
      } else if (st && st->azimuthal_slit) {
        m.current = 0;
      } else {
        m.fixed = true;  // closed loop: no RF fluxoid change
      }
      masters.push_back(m);
    }
    AXF_REQUIRE(driven != "TE" || pb >= 0, "pickup conductor '" + d_.pickup->conductor + "' not found",
                "/pickup/conductor");
    std::vector<WaveRole> roles(ne);
    for (int e = 0; e < ne; ++e) {
      WaveRole& ro = roles[e];
      if (body_of_elem[e] >= 0) {
        ro.master = true;
        ro.master_id = body_of_elem[e];
        continue;
      }
      if (!in_cav[e]) {  // outside the cavity: not assembled
        ro.excluded = true;
        continue;
      }
      ro.alpha = 1.0 / mu_static_of(info[e]);
      if (auto mu = iron_mu(e)) ro.alpha_at = [mu](int ee, const PointEval& pp) { return 1.0 / mu(ee, pp); };
      ro.beta = eps[e];
      ro.group = egroup[e];
    }
    if (cav.enabled) {
      // the cut must run along conductors: a cavity DOF shared with an excluded element has to be tied to a body
      // (or lie on the axis), otherwise it would carry a spurious natural (magnetic-wall) condition
      std::vector<char> tied(nd, 0), cut(nd, 0);
      for (int e = 0; e < ne; ++e) {
        if (roles[e].master) for (int dd : sp.edofs(e)) tied[dd] = 1;
        if (roles[e].excluded) for (int dd : sp.edofs(e)) cut[dd] = 1;
      }
      mfem::Array<int> bd;
      for (int be = 0; be < sp.mesh.GetNBE(); ++be)
        if (body_of_curve.count(sp.mesh.GetBdrAttribute(be))) {
          sp.fes->GetBdrElementDofs(be, bd);
          for (int dd : bd) tied[dd] = 1;
        }
      for (int e = 0; e < ne; ++e) {
        if (!in_cav[e]) continue;
        for (int dd : sp.edofs(e))
          AXF_REQUIRE(!cut[dd] || tied[dd] || sp.r[dd] == 0.0,
                      strcat("the cavity touches the region outside its wall at (r, z) = (", sp.r[dd], ", ", sp.z[dd],
                             ") without a conductor in between"),
                      "/cavity/interior_point", "Close the cavity wall (conductor regions or conducting sheets).");
      }
    }
    fw_te_ = std::make_unique<AxiWave>(sp, roles, masters, body_of_curve);
    {
      std::vector<WaveGroup> gs;
      for (const auto* m : group_mat)  // TE: the permittivity (mass) carries the loss
        gs.push_back({m->name, [](double) { return cplx(1.0); }, [m, eps_factor](double w) { return eps_factor(m, w); }});
      fw_te_->set_groups(gs);
    }
    // persistent circuits: RF flux-linkage change zero
    for (auto& c : d_.circuits) {
      if (c.kind != spec::Circuit::Kind::Persistent) continue;
      Eigen::VectorXd cv = Eigen::VectorXd::Zero(nd);
      PointEval pe;
      for (auto& w : d_.windings) {
        if (w.circuit != c.name) continue;
        const double k = w.turns / area[w.region];
        for (int e = 0; e < ne; ++e) {
          if (info[e].region != w.region) continue;
          const auto& dd = sp.edofs(e);
          for (int q = 0; q < tab.nq; ++q) {
            sp.eval(e, tab, q, pe);
            for (size_t i = 0; i < dd.size(); ++i) cv(dd[i]) += k * pe.w * pe.r * pe.N[i];
          }
        }
      }
      fw_te_->add_constraint("circuit " + c.name, cv);
    }
    fw_te_->set_outer(kOuterAttr, Rb, 1.0 / mu0, radiation, true, fw.dtn_modes, mb_.domain_center_z);
  }

  // ------------------------------------------------------------ TM0 (azimuthal H): u = H' = H - H_MQS
  std::vector<double> m_src, a_vec;  // int mu r H_MQS N_i, int mu r A_lab N_i (per radian, per ampere)
  // an imposed background's part of the domain integral (a diagnostic; a_vec has it already): int mu r A_bg N_i and
  // int mu r A_bg H_MQS / I (per radian)
  std::vector<double> a_bg_vec;
  double psi_mqs_bg = 0;
  double bg_area = 0;          // int r dA over the elements of the TM problem (per radian)
  double bg_area_outside = 0;  // the part outside the background grid, where its field is taken as zero
  int n_cavities = 0;
  if (need("TM")) {
    std::vector<WaveRole> roles(ne);
    for (int e = 0; e < ne; ++e) {
      WaveRole& ro = roles[e];
      ro.excluded = is_sc(info[e]) || is_cond(info[e]) || !in_cav[e];  // outside the cavity: not assembled
      ro.alpha = 1.0 / eps[e];
      ro.beta = mu_static_of(info[e]);
      ro.beta_at = iron_mu(e);
      ro.group = egroup[e];
    }
    fw_tm_ = std::make_unique<AxiWave>(sp, roles, std::vector<Master>{});
    if (cav.enabled && !pec_sheet_attr.empty()) {
      // Perfectly conducting sheets with the cavity on both sides (irises, baffles): H_phi jumps across them (the
      // surface current), with the natural condition (n x E = 0) on each face. Each extra side of a sheet DOF gets
      // its own copy of the DOF; at a free edge of the sheet the sides join (one DOF).
      auto ok = [&](int e) { return !roles[e].excluded; };
      std::set<int> sdofs;
      std::set<std::string> two_sided;
      for (int e = 0; e < ne; ++e) {
        if (!ok(e)) continue;
        for (auto [n, f] : sp.nbr[e]) {
          if (n < 0 || !ok(n) || !is_sheet_face(f)) continue;
          for (int a : sp.edofs(e))
            for (int b : sp.edofs(n))
              if (a == b) sdofs.insert(a);
          for (auto& c : d_.curves)
            if (mb_.attr_of_curve.at(c.name) == sp.face_bdr_attr[f]) two_sided.insert(c.name);
        }
      }
      std::map<int, std::map<int, int>> remap;
      std::vector<int> src;
      for (auto& [dd, F] : dof_fans(sdofs, ok))
        for (size_t k = 1; k < F.fans.size(); ++k) {
          const int nw = nd + static_cast<int>(src.size());
          src.push_back(dd);
          for (int e : F.fans[k]) remap[e][dd] = nw;
        }
      if (!src.empty()) fw_tm_->split_dofs(std::move(remap), src);
      if (!two_sided.empty()) {
        cav_json["two_sided_sheets"] = two_sided;
        cav_json["TM_split_dofs"] = src.size();
      }
    }
    {
      std::vector<WaveGroup> gs;
      for (const auto* m : group_mat)  // TM: 1/eps (stiffness) carries the loss
        gs.push_back({m->name, [m, eps_factor](double w) { return 1.0 / eps_factor(m, w); }, [](double) { return cplx(1.0); }});
      fw_tm_->set_groups(gs);
    }
    // cavities enclosed by conductors (touching neither the axis nor the outer sphere) keep zero RF toroidal flux
    int nc = 0;
    std::vector<int> comp = components(sp, [&](int e) { return !roles[e].excluded; }, [](int) { return true; }, nc);
    std::vector<char> open(nc, 0);
    for (int e = 0; e < ne; ++e)
      if (comp[e] >= 0) for (int dd : sp.edofs(e)) if (sp.r[dd] == 0.0) open[comp[e]] = 1;
    for (int be = 0; be < sp.mesh.GetNBE(); ++be) {
      if (sp.mesh.GetBdrAttribute(be) != kOuterAttr) continue;
      int e1, e2;
      sp.mesh.GetFaceElements(sp.mesh.GetBdrElementFaceIndex(be), &e1, &e2);
      if (e1 >= 0 && comp[e1] >= 0) open[comp[e1]] = 1;
    }
    for (int k = 0; k < nc; ++k) {
      if (open[k]) continue;
      Eigen::VectorXd cv = Eigen::VectorXd::Zero(fw_tm_->ndof());
      PointEval pe;
      std::string name;
      for (int e = 0; e < ne; ++e) {
        if (comp[e] != k) continue;
        if (name.empty()) name = info[e].region;
        const auto& dd = fw_tm_->edofs(e);
        for (int q = 0; q < tab.nq; ++q) {
          sp.eval(e, tab, q, pe);
          const double be = roles[e].beta_at ? roles[e].beta_at(e, pe) : roles[e].beta;
          for (size_t i = 0; i < dd.size(); ++i) cv(dd[i]) += be * pe.w * pe.N[i];
        }
      }
      fw_tm_->add_constraint("enclosed cavity (" + name + ")", cv);
      ++n_cavities;
    }
    fw_tm_->set_outer(kOuterAttr, Rb, 1.0 / eps0, radiation, false, fw.dtn_modes, mb_.domain_center_z);
    if (driven == "TM") {
      AXF_REQUIRE(h_pick_ != nullptr, "internal: the MQS pickup field is missing");
      m_src.assign(nd, 0.0);
      a_vec.assign(nd, 0.0);
      if (bg_.active()) a_bg_vec.assign(nd, 0.0);
      PointEval pe;
      for (int e = 0; e < ne; ++e) {
        if (roles[e].excluded) continue;
        const auto& dd = sp.edofs(e);
        for (int q = 0; q < tab.nq; ++q) {
          sp.eval(e, tab, q, pe);
          double H, mu;
          h_pick_->field(h_pick_sol_, e, pe, H, mu);
          double Br = 0, Bz = 0, A = 0;
          if (a_lab_) static_poloidal(e, pe, Br, Bz, A);
          for (size_t i = 0; i < dd.size(); ++i) {
            m_src[dd[i]] += mu * pe.r * (H / I_pick_) * pe.N[i] * pe.w;
            a_vec[dd[i]] += mu * pe.r * A * pe.N[i] * pe.w;
          }
          if (bg_.active() && pe.r > 0) {
            const double Ab = bg_.psi(pe.r, pe.z) / pe.r;
            psi_mqs_bg += mu * pe.r * Ab * (H / I_pick_) * pe.w;
            for (size_t i = 0; i < dd.size(); ++i) a_bg_vec[dd[i]] += mu * pe.r * Ab * pe.N[i] * pe.w;
          }
          if (bg_.active()) {
            bg_area += pe.r * pe.w;
            if (!bg_.inside(pe.r, pe.z)) bg_area_outside += pe.r * pe.w;
          }
        }
      }
    }
  }
  if (need("TM")) out["enclosed_cavities"] = n_cavities;

  // ------------------------------------------------------------ TM: exterior part of the reciprocity integral
  // int_{rho > Rb} A_lab mu0 H' dV with A_lab = sum a_n (Rb/rho)^{n+1} P_n^1 (static multipoles) and
  // H' = sum b_n h_n(k rho)/h_n(k Rb) P_n^1 (outgoing):  2 pi mu0 sum a_n b_n J_n / inv_norm(n)
  // (the MQS pickup field vanishes outside the device, so only Psi gets this term, not Z).
  // With a background, A_lab is the reaction field (its sources lie inside Rb), and a grid background has its own
  // exterior term, int_{rho > Rb} A_bg mu0 H' dV = -mu0 sum b_n G_n(k) / c_n with c_n = sqrt((2n + 1) / (4 pi)) and the
  // moments G_n of axi::BackgroundExterior, which the 3D sweep shares (docs/physics/fullwave_background.md). A uniform
  // background has A_phi ~ rho outside, so its own exterior integral diverges and is left out; A_lab's term (windings,
  // superconductor reactions) is kept, as in 3D.
  struct OutPt {
    int e;
    PointEval pe;
    double c, x;  // r dl / rho^2 (= sin(theta) d theta), cos(theta)
    double rho;   // distance from the domain centre (Rb up to the boundary's curvature error)
  };
  std::vector<OutPt> opts;
  std::vector<double> a_ext;
  const int Next = fw.dtn_modes;
  auto inv_norm = [](int n) { return (2.0 * n + 1.0) / (2.0 * n * (n + 1.0)); };
  const bool bg_grid = bg_.active() && d_.background.type == "grid";
  const bool ext_tail = driven == "TM" && radiation && a_lab_;
  BackgroundExterior bg_ext;      // a grid background outside Rb (its moments)
  std::vector<double> bg_sliver;  // their correction for the slivers between the outer boundary and the sphere Rb
  // the labels, warnings and background diagnostics describe Psi(omega), so only a sweep reports them
  const bool sweep = !fw.frequencies.empty();
  if (sweep && driven == "TM" && radiation && !ext_tail) {
    warnings.push_back("full wave: the exterior reciprocity term of Psi(omega) is omitted (no static field), so "
                       "Psi(omega) depends on mesh.domain_radius above the MQS limit (the MQS Psi does not)");
    out["exterior_term"] = "omitted (no static field)";
  } else if (sweep && ext_tail) {
    out["exterior_term"] = "included";
  }
  if (sweep && driven == "TM" && bg_.active()) {
    out["background"] = {{"type", d_.background.type},
                         {"form", "Psi part (1/I) int A_bg mu H dV with A_bg = psi / r over the domain (the static "
                                  "reciprocal form, as for the static field; Psi_background.volume) and, with the "
                                  "radiating boundary and a grid, outside it (Psi_background.exterior)"}};
    if (bg_grid) {
      // what the sweep sees of the grid, as in 3D: the share of the domain outside it (where the field is taken as
      // zero) and how large A_bg still is on its edge
      const auto& bgs = d_.background;
      double a_max = 0;
      double a_edge = 0;
      const size_t nr = bgs.r.size();
      const size_t nz = bgs.z.size();
      for (size_t j = 0; j < nz; ++j)
        for (size_t i = 1; i < nr; ++i) {
          const double a = std::fabs(bg_.psi(bgs.r[i], bgs.z[j]) / bgs.r[i]);
          a_max = std::max(a_max, a);
          if (i == nr - 1 || j == 0 || j == nz - 1) a_edge = std::max(a_edge, a);
        }
      const double out_frac = bg_area > 0 ? bg_area_outside / bg_area : 0.0;
      const double edge_frac = a_max > 0 ? a_edge / a_max : 0.0;
      out["background"]["physical_volume_outside_grid_fraction"] = out_frac;
      out["background"]["grid_edge_A_over_max"] = edge_frac;
      if (out_frac > 1e-6 && edge_frac > 1e-3)
        warnings.push_back(strcat("full wave: ", out_frac, " of the domain (by volume) lies outside the background "
                                  "grid, whose edge still carries A_bg = psi/r at ", edge_frac, " of its maximum: Psi "
                                  "sees the background cut off there; extend the grid over the domain sphere"));
    }
    if (!radiation) {
      out["exterior_term_background"] = "none (closed: a perfectly conducting sphere at R1, outside which E = 0)";
    } else if (!bg_grid) {
      // a uniform background has A_phi ~ rho outside, so its exterior integral diverges and is left out (A_lab's term,
      // exterior_term, is included)
      out["exterior_term_background"] = "omitted (uniform background: the exterior integral diverges)";
      warnings.push_back("full wave: the exterior reciprocity term of Psi(omega) is omitted for the uniform "
                         "background, whose field fills all space (the design's own static field, its windings and "
                         "superconductor reactions, keeps its term), so Psi(omega) depends on mesh.domain_radius "
                         "above the MQS limit: between R1 and 1.5 R1 it moved by 0.7-44 % at kR = 0.4 and by up to "
                         "159 % at kR = 0.8 in docs/validation/axi_background_exterior.md; vary mesh.domain_radius to "
                         "see how much, or give the magnet's field as a grid to include the term");
    }
  }
  if (ext_tail) {
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp.order + 14);
    std::vector<double> Pn;
    a_ext.assign(Next + 1, 0.0);
    for (int be = 0; be < sp.mesh.GetNBE(); ++be) {
      if (sp.mesh.GetBdrAttribute(be) != kOuterAttr) continue;
      const int f = sp.mesh.GetBdrElementFaceIndex(be);
      mfem::FaceElementTransformations* FT = const_cast<mfem::Mesh&>(sp.mesh).GetFaceElementTransformations(f);
      if (FT->Elem1No < 0) continue;
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        FT->SetAllIntPoints(&ir.IntPoint(q));
        OutPt p;
        p.e = FT->Elem1No;
        sp.eval_at(p.e, FT->GetElement1IntPoint(), p.pe);
        p.pe.ip = nullptr;
        mfem::Vector nor(2);
        mfem::CalcOrtho(FT->Jacobian(), nor);
        const double zc = p.pe.z - mb_.domain_center_z;  // multipoles about the domain centre (0, z0)
        const double rho = std::hypot(p.pe.r, zc);
        p.c = p.pe.r * nor.Norml2() * ir.IntPoint(q).weight / (rho * rho);
        p.x = zc / rho;
        p.rho = rho;
        double Br = 0, Bz = 0, A = 0;
        if (bg_.active()) {
          // the reaction field only (the background has its own term); master elements hold the total flux
          a_lab_->field(a_lab_sol_, p.e, p.pe, Br, Bz, &A);
          if (a_lab_->roles()[p.e].master && p.pe.r > 0) A -= bg_.psi(p.pe.r, p.pe.z) / p.pe.r;
        } else {
          static_poloidal(p.e, p.pe, Br, Bz, A);
        }
        assoc_legendre_p1(p.x, Next, Pn);
        for (int n = 1; n <= Next; ++n) a_ext[n] += inv_norm(n) * p.c * A * Pn[n];
        opts.push_back(std::move(p));
      }
    }
  }
  if (ext_tail && bg_grid && sweep) {
    // the grid up to the sphere inscribed in its box, its exterior multipoles beyond (axi::background_exterior), to
    // the order of the exterior map
    const auto& bgs = d_.background;
    const double z0 = mb_.domain_center_z;
    const double r_max = bgs.r.back();
    const double z_lo = bgs.z.front();
    const double z_hi = bgs.z.back();
    const int panels = background_exterior_panels();
    bg_ext = background_exterior(bg_, r_max, z_lo, z_hi, Rb, z0, Next, panels);
    // quadrature check: the k -> 0 moments again with twice the panels (to order 12)
    double gmax = 0;
    const double quad_change =
        background_exterior_quadrature_check(bg_, r_max, z_lo, z_hi, Rb, z0, panels, bg_ext, &gmax);
    // the slivers between the outer boundary (order-2 geometry) and the sphere Rb: neither the finite-element domain
    // nor the quadrature (from Rb outwards) covers them; first order in (Rb - rho), at rho = Rb, as in 3D
    bg_sliver.assign(Next + 1, 0.0);
    std::vector<double> dy;
    for (const auto& p : opts) {
      if (!(p.pe.r > 0)) continue;
      const double A = bg_.psi(p.pe.r, p.pe.z) / p.pe.r;
      dtheta_y0(p.x, Next, dy);
      for (int n = 1; n <= Next; ++n) bg_sliver[n] += 2 * pi * p.c * (Rb - p.rho) * Rb * Rb * A * dy[n];
    }
    double sliver = 0;
    for (int n = 1; n <= Next; ++n) sliver = std::max(sliver, std::fabs(bg_sliver[n]));
    out["background"]["exterior"] = {
        {"n_max", Next}, {"radial_nodes", bg_ext.rho.size()}, {"panels", panels},
        {"grid_max_distance_from_domain_center_m", std::max(std::hypot(r_max, z_hi - z0), std::hypot(r_max, z_lo - z0))},
        {"continuation_radius_m", bg_ext.rho_s},
        {"continuation_mismatch", bg_ext.tail_mismatch}, {"continuation_samples", bg_ext.tail_samples},
        {"quadrature_relative_change_doubled_panels", quad_change},
        {"sliver_correction_over_moments", gmax > 0 ? sliver / gmax : 0.0},
        {"scope", "int_{rho > Rb} A_bg mu0 H' dV from the outgoing multipoles of H' (n <= n_max, the exterior map's "
                  "order): the grid up to the sphere inscribed in its box (continuation_radius_m), its exterior "
                  "multipoles beyond (continuation_mismatch: relative rms against the grid where the box reaches "
                  "farther; 0 samples: none to compare), or, with no such sphere outside Rb, the grid over its box; "
                  "quadrature check on the k -> 0 moments to order 12; first-order correction for the slivers between "
                  "the outer boundary and the sphere (the same moments as the 3D sweep's, axi::BackgroundExterior)"}};
    out["exterior_term_background"] =
        bg_ext.rho.empty() ? "none (the grid ends inside the domain sphere, where its field is cut off: "
                             "grid_edge_A_over_max)"
        : bg_ext.rho_s > 0 ? "included (grid, then its exterior multipoles)"
                           : "included (grid only: cut at its box)";
    const double edge_frac = out["background"].value("grid_edge_A_over_max", 0.0);
    if (bg_ext.rho.empty() && edge_frac > 1e-3)
      warnings.push_back(strcat("full wave: the background grid ends inside the domain sphere (its farthest point lies "
                                "at rho = ", std::max(std::hypot(r_max, z_hi - z0), std::hypot(r_max, z_lo - z0)),
                                " m, R1 = ", Rb, " m) while A_bg on its edge is still ", edge_frac, " of its maximum: "
                                "the magnet's field beyond the grid is coupled neither inside the domain nor outside "
                                "it; extend the grid beyond the domain sphere, and far enough that its inscribed "
                                "sphere encloses the magnet"));
    if (!(quad_change <= 1e-4))
      warnings.push_back(strcat("full wave: the background's exterior quadrature changes by ", quad_change,
                                " with doubled panels; set AXF_FW_BG_PANELS above ", panels));
    if (bg_ext.rho_s > 0 && !(bg_ext.tail_mismatch <= 1e-2))
      warnings.push_back(strcat("full wave: beyond the sphere rho = ", bg_ext.rho_s, " m inscribed in the background "
                                "grid, its exterior multipoles differ from the grid by ", bg_ext.tail_mismatch,
                                " (rms): the magnet's sources may reach that sphere; extend the grid so that its "
                                "inscribed sphere encloses them"));
    if (!bg_ext.rho.empty() && bg_ext.rho_s == 0)
      warnings.push_back("full wave: the background grid's inscribed sphere lies inside the domain sphere, so the "
                         "exterior term uses the grid only up to its box, where the magnet's field is cut off; extend "
                         "the grid beyond the domain sphere in every direction");
  }

  // ------------------------------------------------------------ conductor surfaces (RF losses)
  // Gamma_b(omega) = int |H|^2 dA / I^2 on the vacuum side of conductor body b; R_loss = sum R_s(f) Gamma_b.
  struct SurfPt {
    int e;
    PointEval pe;
    double w, mu;
    double mr = 1;  // magnetoresistance factor of R_s at the local static field
    double nr = 0, nz = 0;  // unit face normal pointing out of element e (into the conductor)
    int f = -1;             // mesh face
    std::string body;
  };
  std::vector<SurfPt> spts;
  std::map<std::string, const spec::Material*> body_mat;
  if (!fw.frequencies.empty() || (cav.enabled && cav.losses)) {
    const mfem::Mesh& m = sp.mesh;
    auto is_rfc = [&](int e) { return is_sc(info[e]) || is_cond(info[e]); };
    std::map<int, std::string> sheet_name;
    for (auto& c : d_.curves)
      if (!c.thin_layer()) sheet_name[mb_.attr_of_curve.at(c.name)] = c.name;
    const mfem::IntegrationRule& ir1 = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp.order + 6);
    for (int f = 0; f < m.GetNumFaces(); ++f) {
      int e1, e2;
      m.GetFaceElements(f, &e1, &e2);
      std::vector<std::pair<int, int>> sides;
      std::string body;
      auto it = sheet_name.find(sp.face_bdr_attr[f]);
      if (it != sheet_name.end()) {
        body = it->second;
        body_mat[body] = &d_.materials.at(d_.find_curve(body)->material);
        if (e1 >= 0 && !is_rfc(e1)) sides.push_back({e1, 1});
        if (e2 >= 0 && !is_rfc(e2)) sides.push_back({e2, 2});
      } else if (e1 >= 0 && e2 >= 0 && is_rfc(e1) != is_rfc(e2)) {
        const int c = is_rfc(e1) ? e1 : e2;
        body = info[c].region;
        body_mat[body] = info[c].mat;
        sides.push_back(is_rfc(e1) ? std::make_pair(e2, 2) : std::make_pair(e1, 1));
      }
      if (sides.empty()) continue;
      mfem::FaceElementTransformations* FT = const_cast<mfem::Mesh&>(m).GetFaceElementTransformations(f);
      for (auto [ev, which] : sides) {
        for (int q = 0; q < ir1.GetNPoints(); ++q) {
          FT->SetAllIntPoints(&ir1.IntPoint(q));
          const mfem::IntegrationPoint& eip = which == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
          SurfPt p;
          p.e = ev;
          p.body = body;
          sp.eval_at(ev, eip, p.pe);
          p.pe.ip = nullptr;
          mfem::Vector nor(2);
          mfem::CalcOrtho(FT->Jacobian(), nor);
          p.w = 2 * pi * p.pe.r * nor.Norml2() * ir1.IntPoint(q).weight;
          p.nr = nor(0) / nor.Norml2();
          p.nz = nor(1) / nor.Norml2();
          p.f = f;
          {  // orient out of element ev: away from its centroid
            mfem::Array<int> ev_v;
            m.GetElementVertices(ev, ev_v);
            double cr = 0, cz = 0;
            for (int vv : ev_v) {
              cr += m.GetVertex(vv)[0] / ev_v.Size();
              cz += m.GetVertex(vv)[1] / ev_v.Size();
            }
            if ((cr - p.pe.r) * p.nr + (cz - p.pe.z) * p.nz > 0) {
              p.nr = -p.nr;
              p.nz = -p.nz;
            }
          }
          p.mu = mu_static_of(info[ev]);
          const spec::Material* bm = body_mat[body];
          if (bm && !bm->rf.mr_model.empty()) {
            Sample smp = eval_pe(ev, p.pe);
            const double Bs = std::sqrt(smp.Br * smp.Br + smp.Bz * smp.Bz + smp.Bphi * smp.Bphi);
            p.mr = bm->rf.Rs_factor(Bs);
          }
          spts.push_back(std::move(p));
        }
      }
    }
  }
  bool have_rf = false;
  for (auto& [b, mt] : body_mat) if (mt && !mt->rf.model.empty()) have_rf = true;

  // ------------------------------------------------------------ frequency sweep
  const double Psi_mqs = R["coupling"].value("Psi_SI_H_T_m2", 0.0), L_mqs = R["coupling"].value("L_p_H", 0.0);
  if (!fw.frequencies.empty()) {
    out["family"] = driven == "TE" ? "TE0 (azimuthal E; azimuthal readout current)"
                                   : "TM0 (azimuthal H; poloidal readout current)";
    json sweep = json::array();
    bool capacitive = false;
    for (double f : fw.frequencies) {
      const double w = 2 * pi * f;
      cplx Z = 0, Psi = 0;
      double Prad = 0;
      json diag;
      std::map<std::string, double> gamma, gamma_mr;
      std::map<std::string, double> loss_g;  // dissipated power per lossy material (W per A^2 amplitude)
      json tail_row;                          // exterior reciprocity term (TM)
      json bg_row;                            // an imposed background's parts of Psi (TM)
      if (driven == "TE") {
        auto s = fw_te_->solve(w);
        Z = -I1 * w * 2.0 * pi * s.psi[pb];
        for (auto& p : spts) {
          const auto& dd = sp.edofs(p.e);
          cplx a = 0, dr = 0, dz = 0;
          for (size_t i = 0; i < dd.size(); ++i) {
            a += s.u[dd[i]] * p.pe.N[i];
            dr += s.u[dd[i]] * p.pe.dN[2 * i];
            dz += s.u[dd[i]] * p.pe.dN[2 * i + 1];
          }
          const cplx Br = -dz, Bz = dr + (p.pe.r > 0 ? a / p.pe.r : dr);
          gamma[p.body] += p.w * (std::norm(Br) + std::norm(Bz)) / (p.mu * p.mu);
          gamma_mr[p.body] += p.mr * p.w * (std::norm(Br) + std::norm(Bz)) / (p.mu * p.mu);
        }
        if (h_lab_) {
          PointEval pe;
          for (int e = 0; e < ne; ++e) {
            for (int q = 0; q < tab.nq; ++q) {
              sp.eval(e, tab, q, pe);
              double H, mu;
              h_lab_->field(h_lab_sol_, e, pe, H, mu);
              if (H == 0.0) continue;
              Psi += pe.w * pe.r * mu * H * fw_te_->value(s, e, pe);
            }
          }
          Psi *= 2 * pi;
        }
        Prad = -pi * w * s.dtn_im;
        diag = s.diagnostics;
        for (size_t gi = 0; gi < group_mat.size(); ++gi)  // P = 1/2 omega Im(eps_c) |E|^2, E = i omega A
          loss_g[group_mat[gi]->name] = pi * w * w * w * eps_factor(group_mat[gi], w).imag() * s.group_M[gi];
      } else {
        std::vector<cplx> rhs(nd);
        for (int i = 0; i < nd; ++i) rhs[i] = w * w * m_src[i];
        auto s = fw_tm_->solve(w, rhs);
        for (auto& p : spts) {
          double H, mu;
          h_pick_->field(h_pick_sol_, p.e, p.pe, H, mu);
          const cplx Ht = H / I_pick_ + fw_tm_->value(s, p.e, p.pe);
          gamma[p.body] += p.w * std::norm(Ht);
          gamma_mr[p.body] += p.mr * p.w * std::norm(Ht);
        }
        cplx dl = 0, dp = 0;
        for (int i = 0; i < nd; ++i) {
          dl += m_src[i] * s.u[i];
          dp += a_vec[i] * s.u[i];
        }
        Z = -I1 * w * (L_mqs + 2 * pi * dl);  // Z = -i omega (1/I^2) int mu H_MQS . H dV
        Psi = Psi_mqs + 2 * pi * dp;
        if (ext_tail) {
          std::vector<cplx> b_ext(Next + 1, 0.0);
          std::vector<double> Pn;
          for (auto& p : opts) {
            const cplx h = fw_tm_->value(s, p.e, p.pe);
            assoc_legendre_p1(p.x, Next, Pn);
            for (int n = 1; n <= Next; ++n) b_ext[n] += inv_norm(n) * p.c * h * Pn[n];
          }
          const auto J = exterior_radial_integrals(w / units::c, Rb, Next);
          cplx tail = 0;
          for (int n = 1; n <= Next; ++n) tail += a_ext[n] * b_ext[n] * J[n] / inv_norm(n);
          tail *= 2 * pi * mu0;
          Psi += tail;
          tail_row = {{"re", tail.real()}, {"im", tail.imag()}};
          if (bg_ext.N > 0 && !bg_ext.rho.empty()) {
            // the grid background's exterior term: -mu0 sum b_n (G_n + sliver_n) / c_n (G_n with dY_n0/dtheta)
            const auto G = bg_ext.G(w / units::c);
            cplx bg_tail = 0;
            cplx term = 0;
            cplx previous = 0;
            for (int n = 1; n <= Next; ++n) {
              previous = term;
              term = -b_ext[n] * (G[n] + bg_sliver[n]) / std::sqrt((2.0 * n + 1.0) / (4 * pi));
              bg_tail += term;
            }
            // the last two orders: a magnet symmetric about the domain centre has no even orders
            const double last = (std::abs(term) + std::abs(previous)) / std::max(std::abs(bg_tail), 1e-300);
            bg_tail *= mu0;
            Psi += bg_tail;
            bg_row["exterior"] = {{"re", bg_tail.real()}, {"im", bg_tail.imag()}};
            bg_row["exterior_last_order_fraction"] = last;
          }
        }
        if (!a_bg_vec.empty()) {
          // the background's part of the domain integral (included in Psi above): (1/I) int mu A_bg H dV
          cplx db = 0;
          for (int i = 0; i < nd; ++i) db += a_bg_vec[i] * s.u[i];
          const cplx vol = 2 * pi * (psi_mqs_bg + db);
          bg_row["volume"] = {{"re", vol.real()}, {"im", vol.imag()}};
          if (!bg_row.contains("exterior")) bg_row["exterior"] = nullptr;   // no exterior term computed
        }
        Prad = -pi / w * s.dtn_im;
        diag = s.diagnostics;
        for (size_t gi = 0; gi < group_mat.size(); ++gi)  // P = -(1/2 omega) Im(1/eps_c) |curl H|^2 (curl H_MQS = 0 there)
          loss_g[group_mat[gi]->name] = -pi / w * (1.0 / eps_factor(group_mat[gi], w)).imag() * s.group_K[gi];
      }
      double Ploss = 0;
      for (auto& [k, v] : loss_g) Ploss += v;
      const double L = -Z.imag() / w, Rr = Z.real();
      json row = {{"f_Hz", f},
                  {"k_R", w / units::c * Rdev},
                  {"Psi_re", Psi.real()},
                  {"Psi_im", Psi.imag()},
                  {"Psi_abs", std::abs(Psi)},
                  {"Z_re_ohm", Z.real()},
                  {"Z_im_ohm", Z.imag()},
                  {"L_H", L},
                  {"R_rad_ohm", Rr},
                  {"radiated_W", Prad},
                  {"solver", diag}};
      if (Psi_mqs != 0.0) row["Psi_over_MQS"] = std::abs(Psi) / std::fabs(Psi_mqs);
      if (!tail_row.is_null()) row["Psi_exterior"] = tail_row;
      if (!bg_row.is_null()) row["Psi_background"] = bg_row;
      if (L_mqs > 0) row["L_over_MQS"] = L / L_mqs;
      if (L > 0) {
        row["F_SI_T4_H2_m8"] = std::pow(std::abs(Psi), 4) / (L * L);
        if (Rr > 0) row["Q_rad"] = w * L / Rr;
      } else {
        capacitive = true;
      }
      if (!loss_g.empty()) {
        // R = Re Z now contains the radiation and the bulk losses; split it
        json lj = json::object();
        for (auto& [k, v] : loss_g) lj[k] = {{"P_W", v}, {"R_ohm", 2 * v}, {"Q", L > 0 && v > 0 ? w * L / (2 * v) : 0.0}};
        row["bulk_losses"] = lj;
        row["R_rad_ohm"] = 2 * Prad;
        row["R_bulk_ohm"] = 2 * Ploss;
        if (L > 0 && Prad > 0) row["Q_rad"] = w * L / (2 * Prad);
        if (L > 0 && Ploss > 0) row["Q_bulk"] = w * L / (2 * Ploss);
      }
      {
        const double Pin = 0.5 * Z.real(), Pout = Prad + Ploss;
        if (radiation || !loss_g.empty())
          row["power_balance_rel"] = std::fabs(Pin - Pout) / std::max({std::fabs(Pin), std::fabs(Pout), 1e-300});
      }
      if (!gamma.empty()) {
        row["surface_Gamma"] = gamma;
        if (have_rf) {
          // surface resistance at this frequency: R_loss = sum_b R_s,b(f) Gamma_b (perturbative)
          double Rl = 0, TQ = 0;
          for (auto& [b, G] : gamma) {
            const spec::Material* mt = body_mat[b];
            if (!mt || mt->rf.model.empty()) continue;
            const double Rb = mt->rf.Rs(f) * (mt->rf.mr_model.empty() ? G : gamma_mr[b]);
            Rl += Rb;
            if (L > 0 && Rb > 0) TQ += mt->rf.T_K / (w * L / Rb);
          }
          row["R_loss_ohm"] = Rl;
          if (L > 0) {
            if (Rl > 0) row["Q_loss"] = w * L / Rl;
            if (Rl + std::max(Rr, 0.0) > 0) row["Q_total"] = w * L / (Rl + std::max(Rr, 0.0));
            row["T_over_Q_K"] = TQ;
          }
        }
      }
      sweep.push_back(row);
      log(1, strcat("full-wave ", driven, " f = ", f, " Hz: |Psi|/Psi_MQS = ", row.value("Psi_over_MQS", 0.0),
                    ", L/L_MQS = ", row.value("L_over_MQS", 0.0), ", R_rad = ", Rr, " ohm"));
    }
    if (capacitive)
      warnings.push_back("full-wave: the readout is capacitive (L(omega) <= 0) at some frequencies, i.e. above its "
                         "first self-resonance; F and Q_rad are omitted there");
    out["sweep"] = sweep;
    // RF field kept for scenes (per readout ampere)
    const double fs = fw.scene_frequency > 0 ? fw.scene_frequency : fw.frequencies.front();
    const double ws = 2 * pi * fs;
    if (driven == "TE") {
      fw_sol_ = std::make_unique<AxiWave::Solution>(fw_te_->solve(ws));
    } else {
      std::vector<cplx> rhs(nd);
      for (int i = 0; i < nd; ++i) rhs[i] = ws * ws * m_src[i];
      fw_sol_ = std::make_unique<AxiWave::Solution>(fw_tm_->solve(ws, rhs));
    }
    fw_sol_family_ = driven;
    fw_sol_f_ = fs;
    fw_eps_.assign(ne, 0.0);  // eps_c = eps' (1 + i tan delta + i sigma/(omega eps')): E = (i/omega eps_c) curl H'
    for (int e = 0; e < ne; ++e) fw_eps_[e] = eps[e] * eps_factor(info[e].mat, ws);
    out["scene_frequency_Hz"] = fs;
  }

  // ------------------------------------------------------------ eigenmodes (closed sphere)
  if (fw.modes > 0) {
    json eig = json::object();
    const double w_target = 2 * pi * fw.target_f_Hz;
    // cavity: static field B0 for the form factor (poloidal field incl. background; toroidal field)
    double B_ref = 0;
    bool B0_negligible = false, Bref_negligible = false;
    std::vector<double> B2_cav;  // int |B0|^2 dV over the cavity (index 0)
    std::vector<std::array<double, 3>> B0q;  // (B_r, B_z, B_phi) of the static field at cavity quadrature points
    std::vector<std::array<double, 3>> B0s;  // the same B0 at the wall surface points spts (flux_flow walls only)
    double B0_scale = 0;                     // max |B0| over the design
    std::map<std::string, const spec::Material*> reg_mat;
    for (int e = 0; e < ne; ++e) reg_mat[info[e].region] = info[e].mat;
    std::unique_ptr<AxiModel> aux;  // static solve with the superconducting cavity walls transparent
    if (cav.enabled) {
      // "applied" B0: superconducting regions and sheets that bound the cavity do not expel the magnet field (a
      // haloscope cavity sits far above Hc1 in the mixed state, or is a thin film); they are re-solved as
      // `conductor` (transparent to static fields) in an auxiliary static problem on the same mesh.
      std::set<std::string> sc_regs, sc_curves;
      if (cav.B0_field == "applied") {
        for (int e = 0; e < ne; ++e) {
          if (!in_cav[e]) continue;
          for (auto [n, f] : sp.nbr[e]) {
            if (n >= 0 && is_sc(info[n])) sc_regs.insert(info[n].region);
            if (is_sheet_face(f))
              for (auto& c : d_.curves)
                if (mb_.attr_of_curve.at(c.name) == sp.face_bdr_attr[f] &&
                    d_.materials.at(c.material).kind == spec::MaterialKind::Superconductor)
                  sc_curves.insert(c.name);
          }
        }
      }
      if (!sc_regs.empty() || !sc_curves.empty()) {
        spec::Design d2 = d_;
        auto transparent = [&](const std::string& mat) {
          const std::string nm = "__cavity_wall_transparent__" + mat;
          if (!d2.materials.count(nm)) {
            spec::Material m2 = d2.materials.at(mat);
            m2.name = nm;
            m2.kind = spec::MaterialKind::Conductor;
            d2.materials[nm] = m2;
          }
          return nm;
        };
        for (auto& rg : d2.regions)
          if (sc_regs.count(rg.name)) rg.material = transparent(rg.material);
        for (auto& c : d2.curves)
          if (sc_curves.count(c.name)) c.material = transparent(c.material);
        d2.superconductors.erase(std::remove_if(d2.superconductors.begin(), d2.superconductors.end(),
                                                [&](const spec::Superconductor& x) {
                                                  return sc_regs.count(x.body) || sc_curves.count(x.body);
                                                }),
                                 d2.superconductors.end());
        d2.fullwave.enabled = false;
        d2.fullwave.modes = 0;
        d2.cavity.enabled = false;
        d2.hts.enabled = false;
        d2.mechanics.enabled = false;
        d2.thermal.enabled = false;
        d2.electrostatics.enabled = false;
        d2.pickup.reset();
        MeshBundle mb2;
        mb2.mesh = std::make_unique<mfem::Mesh>(*mb_.mesh);
        mb2.region_of_attr = mb_.region_of_attr;
        mb2.attr_of_region = mb_.attr_of_region;
        mb2.attr_of_curve = mb_.attr_of_curve;
        mb2.domain_radius = mb_.domain_radius;
        mb2.domain_center_z = mb_.domain_center_z;
        mb2.device_radius = mb_.device_radius;
        mb2.info = mb_.info;
        aux = std::make_unique<AxiModel>(d2);
        aux->set_mesh(std::move(mb2), sp.order);
        SolveOptions so = o;
        so.verbose = 0;
        so.minimal_post = true;
        so.winding_self_force = false;
        const int saved = log_level();
        log_level() = 0;
        try {
          aux->solve(so);
        } catch (...) {
          log_level() = saved;
          throw;
        }
        log_level() = saved;
        std::vector<std::string> names(sc_regs.begin(), sc_regs.end());
        names.insert(names.end(), sc_curves.begin(), sc_curves.end());
        cav_json["B0_transparent_walls"] = names;
      }
      cav_json["B0_field"] = cav.B0_field;
      auto B0_at = [&](int e, const PointEval& pe) { return aux ? aux->eval_pe(e, pe) : eval_pe(e, pe); };
      auto mag = [](const Sample& x) { return std::sqrt(x.Br * x.Br + x.Bz * x.Bz + x.Bphi * x.Bphi); };
      double B2 = 0, Bmax_cav = 0, Bscale = 0;
      PointEval pe;
      B0q.assign(static_cast<size_t>(ne) * tab.nq, {0.0, 0.0, 0.0});
      for (int e = 0; e < ne; ++e) {
        for (int q = 0; q < tab.nq; ++q) {
          sp.eval(e, tab, q, pe);
          Sample sm = B0_at(e, pe);
          Bscale = std::max(Bscale, mag(sm));  // field scale of the whole design (background, windings)
          if (!in_cav[e]) continue;
          Bmax_cav = std::max(Bmax_cav, mag(sm));
          B0q[static_cast<size_t>(e) * tab.nq + q] = {sm.Br, sm.Bz, sm.Bphi};
          B2 += 2 * pi * pe.w * pe.r * (sm.Br * sm.Br + sm.Bz * sm.Bz + sm.Bphi * sm.Bphi);
        }
      }
      B2_cav.push_back(B2);
      B0_scale = Bscale;
      // B0 at the wall points of flux_flow walls: vortex density |B0| and orientation sin^2(J, B0) (wall losses)
      B0s.assign(spts.size(), {0.0, 0.0, 0.0});
      for (size_t i = 0; i < spts.size(); ++i) {
        const spec::Material* bm = body_mat[spts[i].body];
        if (!bm || !bm->rf.flux_flow) continue;
        const Sample sm = B0_at(spts[i].e, spts[i].pe);
        B0s[i] = {sm.Br, sm.Bz, sm.Bphi};
      }
      const double B0_rms = V_cav > 0 ? std::sqrt(B2 / V_cav) : 0.0;
      // Default normalisation: the volume rms of B0 over the cavity, B_ref^2 V = int |B0|^2 dV (then C = C_rms
      // and C does not depend on which point selects the cavity); explicit B_ref_point: |B0| there.
      if (cav.B_ref_point) {
        const spec::Vec2 pref = *cav.B_ref_point;
        int er = -1;
        PointEval per;
        locate(pref, er, per);
        AXF_REQUIRE(er >= 0, strcat("cavity.form_factor.B_ref_point (", pref.r, ", ", pref.z, ") lies outside the mesh"),
                    "/cavity/form_factor/B_ref_point", "Give a point inside the meshed domain.");
        B_ref = mag(B0_at(er, per));
        cav_json["B_ref"] = "point";
        cav_json["B_ref_point"] = {pref.r, pref.z};
        if (B0_rms > 0) cav_json["B_ref_over_B0_rms"] = B_ref / B0_rms;
      } else {
        B_ref = B0_rms;
        cav_json["B_ref"] = "volume rms of B0 over the cavity";
      }
      cav_json["B_ref_T"] = B_ref;
      cav_json["B0_rms_T"] = B0_rms;
      cav_json["B0_max_cavity_T"] = Bmax_cav;
      cav_json["B0_max_design_T"] = Bscale;
      // B0 negligible against the field scale of the design: C would be round-off (e.g. a closed superconducting
      // cavity expelling the field with B0 = "static")
      const double tiny = 1e-6 * Bscale;
      B0_negligible = !(B0_rms > tiny);
      Bref_negligible = !(B_ref > tiny);
      cav_json["B0_negligible"] = B0_negligible;
      if (!(Bscale > 0)) {
        warnings.push_back("cavity: the static field B0 is zero everywhere, so the form factor C is not reported "
                           "(give a background field or windings)");
      } else if (B0_negligible) {
        warnings.push_back(strcat("cavity: the static field in the cavity is negligible (rms ", B0_rms, " T against ",
                                  Bscale, " T in the design", cav.B0_field == "static" ? "; closed superconducting "
                                  "walls expel it with cavity.form_factor.B0 = \"static\"" : "",
                                  "), so C and C_rms would be round-off and are not reported"));
      } else if (Bref_negligible) {
        warnings.push_back(strcat("cavity: |B0| at cavity.form_factor.B_ref_point is ", B_ref, " T (negligible against ",
                                  Bscale, " T in the design), so C is not reported (C_rms is)"));
      }
    }
    for (auto& fam : fams) {
      AxiWave& wv = fam == "TE" ? *fw_te_ : *fw_tm_;
      json d;
      std::vector<Eigenmode> modes = wv.eigenmodes(fw.modes, d, w_target);
      json list = json::array();
      // TE walls: H_t is a derivative of u = A_phi, so the wall current K_phi = (H x n_out)_phi is recovered from
      // the weak reaction (consistent flux: M_wall k = (K - lambda M) u at the wall DOFs, M_wall = int N_i N_j r dl);
      // the pointwise derivative is kept as a diagnostic (G_ohm_pointwise). The unknowns are per DOF and per side
      // ("fan", see dof_fans): on a conducting sheet with the cavity on both sides each face carries its own current
      // (the net reaction would give K+ + K-, i.e. zero for equal and opposite face currents). The reaction of a fan
      // is summed over its elements only. At a free edge of a sheet the elements of both sides join, so the
      // reaction cannot separate the faces there: those rows use the moment of the pointwise current of each face.
      struct WallKey {
        int d, fan, e;  // e = -1 for clean fans (reaction rows), else the element (pointwise rows)
        bool operator<(const WallKey& o) const { return std::tie(d, fan, e) < std::tie(o.d, o.fan, o.e); }
      };
      std::map<WallKey, int> wkey;
      std::vector<WallKey> wkeys;
      std::map<std::pair<int, int>, int> fan_of;  // (element, DOF) -> fan
      std::map<int, DofFans> wfans;
      std::map<int, std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> emat;  // element (K, M) of clean-fan elements
      std::set<std::string> edge_sheets;  // sheets with a free edge inside the cavity
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> mwall;
      const bool te_flux = fam == "TE" && cav.enabled && cav.losses;
      auto spt_ok = [&](const SurfPt& p) { return in_cav[p.e] && !wv.roles()[p.e].master && !wv.roles()[p.e].excluded; };
      auto key_of = [&](int e, int dd) -> WallKey {
        const int k = fan_of.at({e, dd});
        return {dd, k, wfans.at(dd).clean[k] ? -1 : e};
      };
      if (te_flux) {
        std::set<int> wset;
        for (auto& p : spts)
          if (spt_ok(p)) for (int dd : wv.edofs(p.e)) wset.insert(dd);
        wfans = dof_fans(wset, [&](int e) { return in_cav[e] && !wv.roles()[e].master && !wv.roles()[e].excluded; });
        for (auto& [dd, F] : wfans)
          for (size_t k = 0; k < F.fans.size(); ++k)
            for (int e : F.fans[k]) fan_of[{e, dd}] = static_cast<int>(k);
        std::map<WallKey, double> dg;
        double dmax = 0;
        for (auto& p : spts) {
          if (!spt_ok(p)) continue;
          const auto& dd = wv.edofs(p.e);
          for (size_t i = 0; i < dd.size(); ++i)
            dmax = std::max(dmax, dg[key_of(p.e, dd[i])] += p.w * p.pe.N[i] * p.pe.N[i]);
        }
        for (auto& [k, v] : dg)
          if (v > 1e-10 * dmax) {
            wkey[k] = static_cast<int>(wkeys.size());
            wkeys.push_back(k);
            if (k.e >= 0) {
              for (auto& p : spts)
                if (p.e == k.e && spt_ok(p)) edge_sheets.insert(p.body);
            } else {
              for (int e : wfans.at(k.d).fans[k.fan])
                if (!emat.count(e)) wv.element_matrices(e, emat[e].first, emat[e].second);
            }
          }
        std::vector<Eigen::Triplet<double>> tr;
        for (auto& p : spts) {
          if (!spt_ok(p)) continue;
          const auto& dd = wv.edofs(p.e);
          for (size_t i = 0; i < dd.size(); ++i) {
            auto ii = wkey.find(key_of(p.e, dd[i]));
            if (ii == wkey.end()) continue;
            for (size_t j = 0; j < dd.size(); ++j) {
              auto jj = wkey.find(key_of(p.e, dd[j]));
              if (jj != wkey.end()) tr.emplace_back(ii->second, jj->second, p.w / (2 * pi) * p.pe.N[i] * p.pe.N[j]);
            }
          }
        }
        Eigen::SparseMatrix<double> Mw(wkeys.size(), wkeys.size());
        Mw.setFromTriplets(tr.begin(), tr.end());
        if (!wkeys.empty()) {
          mwall.compute(Mw);
          AXF_REQUIRE(mwall.info() == Eigen::Success, "internal: the cavity wall mass matrix could not be factorized");
        }
        if (!edge_sheets.empty()) {
          std::string names;
          for (auto& n : edge_sheets) names += (names.empty() ? "" : ", ") + n;
          warnings.push_back("cavity: the conducting sheet(s) " + names + " end inside the cavity (free edge). The TE "
                             "wall loss of a zero-thickness edge diverges logarithmically under mesh refinement "
                             "wherever the mode's field does not vanish at the edge, so G and Q of such modes are "
                             "mesh dependent; model the iris as a thin region with a finite edge for converged losses");
          cav_json["sheets_with_free_edge"] = edge_sheets;
        }
      }
      std::set<std::string> lossless_fam;  // walls without rf data (not cavity.allow_lossless_walls)
      bool q0_omitted = false;             // Q0 omitted: no wall rf data, dielectric loss only
      std::set<std::string> flux_flow_walls;  // walls whose R_s has a flux_flow term
      std::set<std::string> ff_nodir;         // flux_flow walls with points where B0 = 0 (orientation factor 1)
      std::map<std::string, std::string> rs_warn;  // RFLoss::warnings by material/key (text of the last mode)
      for (auto& m : modes) {
        // energy by region: electric (TE: eps |A|^2) or magnetic (TM: mu |H|^2); equal totals at resonance
        std::map<std::string, double> en;
        double tot = 0;
        // cavity integrals (per radian): overlap int E.B0, int eps_r |E|^2, electric energy per region,
        // magnetic int mu |H|^2, with E in units of the eigenvector (any common factor cancels)
        double I_EB = 0, I_EE = 0, I_HH = 0;
        std::map<std::string, double> we_reg;
        PointEval pe;
        for (int e = 0; e < ne; ++e) {
          const WaveRole& ro = wv.roles()[e];
          if (ro.master || ro.excluded) continue;
          const auto& dd = wv.edofs(e);
          const double eps_r = eps[e] / eps0;
          for (int q = 0; q < tab.nq; ++q) {
            sp.eval(e, tab, q, pe);
            const double v = wv.value(m.x, e, pe);
            const double be = ro.beta_at ? ro.beta_at(e, pe) : ro.beta;
            const double x = be * v * v * pe.r * pe.w;
            en[info[e].region] += x;
            tot += x;
            if (!cav.enabled) continue;
            double dr = 0, dz = 0;
            for (size_t i = 0; i < dd.size(); ++i) {
              dr += m.x[dd[i]] * pe.dN[2 * i];
              dz += m.x[dd[i]] * pe.dN[2 * i + 1];
            }
            const double cbr = -dz, cbz = dr + (pe.r > 0 ? v / pe.r : dr);  // curl(u phi^) = (B_r, B_z)
            const double al = ro.alpha_at ? ro.alpha_at(e, pe) : ro.alpha;
            const auto& s0 = B0q[static_cast<size_t>(e) * tab.nq + q];
            const double wr = pe.w * pe.r;
            double ee;
            if (fam == "TM") {  // u = H_phi; E = (i/omega)(1/eps) curl(u phi^): E ~ (cbr, cbz)/eps_r
              const double er_ = cbr / eps_r, ez_ = cbz / eps_r;
              I_EB += wr * (er_ * s0[0] + ez_ * s0[1]);
              ee = wr * eps_r * (er_ * er_ + ez_ * ez_);
              I_HH += x;  // mu |H|^2
            } else {  // u = A_phi; E_phi = i omega u; B = curl(u phi^), mu |H|^2 = |B|^2 / mu
              I_EB += wr * v * s0[2];
              ee = wr * eps_r * v * v;
              I_HH += wr * al * (cbr * cbr + cbz * cbz);
            }
            I_EE += ee;
            we_reg[info[e].region] += ee;
          }
        }
        std::vector<std::pair<double, std::string>> v;
        for (auto& [k, x] : en) v.push_back({x / std::max(tot, 1e-300), k});
        std::sort(v.rbegin(), v.rend());
        json fr = json::object();
        for (size_t i = 0; i < v.size() && i < 3; ++i) fr[v[i].second] = v[i].first;
        json row = {{"f_Hz", m.f_Hz}, {"k_R", m.omega / units::c * Rdev}, {"residual", m.residual},
                    {"residual_true", m.residual_true}, {"energy_fraction", fr}};
        if (cav.enabled) {
          // ---- form factor C = |int E.B0 dV|^2 / (B_ref^2 V int eps_r |E|^2 dV)
          const double OEB = 2 * pi * I_EB, OEE = 2 * pi * I_EE;
          json ff = {{"V_m3", V_cav},
                     {"overlap_EB_T_m32", std::fabs(OEB) / std::sqrt(std::max(OEE, 1e-300))},
                     {"B_ref_T", B_ref},
                     {"note", "C = |int E.B0 dV|^2 / (B_ref^2 V int eps_r |E|^2 dV), V = cavity volume (conductors "
                              "excluded), B_ref = cavity rms of B0 unless form_factor.B_ref_point is given; C_rms "
                              "uses int |B0|^2 dV in place of B_ref^2 V"}};
          if (!B0_negligible && !Bref_negligible) ff["C"] = OEB * OEB / (B_ref * B_ref * V_cav * OEE);
          if (!B0_negligible) ff["C_rms"] = OEB * OEB / (B2_cav[0] * OEE);
          row["form_factor"] = ff;
          // ---- losses: dielectric (tan delta + sigma/(omega eps')) and walls (perturbative R_s)
          if (cav.losses) {
            json lj = json::object();
            double inv_Qd = 0;
            json dl = json::object();
            for (auto& [rg, we] : we_reg) {
              const spec::Material* mt = reg_mat.at(rg);
              const double td = mt->tan_delta + (m.omega > 0 ? mt->sigma_S_m / (m.omega * eps0 * mt->eps_r) : 0.0);
              if (!(td > 0)) continue;
              const double fr_e = we / std::max(I_EE, 1e-300);
              inv_Qd += td * fr_e;
              dl[rg] = {{"tan_delta_eff", td}, {"electric_energy_fraction", fr_e}, {"Q", 1.0 / (td * fr_e)}};
            }
            // wall: Gamma_b = oint_b |H_t|^2 dA (per eigenvector units^2), G_b = omega int mu |H|^2 / Gamma_b
            std::map<std::string, double> gam, gam_mr, gam_pt;
            // flux_flow walls: oint |H_t|^2 dR_ff(B0, s) dA with the orientation factor s = sin^2(J_rf, B0), the same
            // with s = 1, and oint |H_t|^2 s dA
            std::map<std::string, double> gam_ff, gam_ff_perp, gam_ff_s;
            Eigen::VectorXd kw;
            // pointwise tangential field of a TE surface point: B = curl(u phi^) in its element; K_phi =
            // (H x n_out)_phi = H_z n_r - H_r n_z (signed, as the reaction), and |H_t|^2
            auto te_point = [&](const SurfPt& p, double& Kphi, double& ht2p) {
              const auto& dd = wv.edofs(p.e);
              double dr = 0, dz = 0;
              for (size_t i = 0; i < dd.size(); ++i) {
                dr += m.x[dd[i]] * p.pe.dN[2 * i];
                dz += m.x[dd[i]] * p.pe.dN[2 * i + 1];
              }
              const double val = wv.value(m.x, p.e, p.pe);
              const double br = -dz, bz = dr + (p.pe.r > 0 ? val / p.pe.r : dr);
              const double bn = br * p.nr + bz * p.nz, btr = br - bn * p.nr, btz = bz - bn * p.nz;
              ht2p = (btr * btr + btz * btz) / (p.mu * p.mu);
              Kphi = (bz * p.nr - br * p.nz) / p.mu;
            };
            if (te_flux && !wkeys.empty()) {
              const double lam = m.omega * m.omega;
              Eigen::VectorXd b = Eigen::VectorXd::Zero(static_cast<long>(wkeys.size()));
              std::map<int, Eigen::VectorXd> erx;  // (K_e - lambda M_e) u_e of clean-fan elements
              for (auto& [e, KM] : emat) {
                const auto& dd = wv.edofs(e);
                Eigen::VectorXd ue(static_cast<long>(dd.size()));
                for (size_t i = 0; i < dd.size(); ++i) ue(static_cast<long>(i)) = m.x[dd[i]];
                erx[e] = KM.first * ue - lam * (KM.second * ue);
              }
              for (size_t k = 0; k < wkeys.size(); ++k) {
                const WallKey& wk = wkeys[k];
                if (wk.e >= 0) continue;
                for (int e : wfans.at(wk.d).fans[wk.fan]) {
                  const auto& dd = wv.edofs(e);
                  for (size_t i = 0; i < dd.size(); ++i)
                    if (dd[i] == wk.d) b(static_cast<long>(k)) += erx.at(e)(static_cast<long>(i));
                }
              }
              for (auto& p : spts) {  // free sheet edges: moments of the pointwise current of each face
                if (!spt_ok(p)) continue;
                const auto& dd = wv.edofs(p.e);
                double Kp = 0, h2 = 0;
                bool done = false;
                for (size_t i = 0; i < dd.size(); ++i) {
                  auto it = wkey.find(key_of(p.e, dd[i]));
                  if (it == wkey.end() || it->first.e < 0) continue;
                  if (!done) {
                    te_point(p, Kp, h2);
                    done = true;
                  }
                  b(it->second) += p.w / (2 * pi) * Kp * p.pe.N[i];
                }
              }
              kw = mwall.solve(b);
            }
            for (size_t ip = 0; ip < spts.size(); ++ip) {
              const SurfPt& p = spts[ip];
              const WaveRole& ro = wv.roles()[p.e];
              if (!in_cav[p.e] || ro.master || ro.excluded) continue;
              double ht2;
              const double val = wv.value(m.x, p.e, p.pe);
              if (fam == "TM") {
                ht2 = val * val;  // H_phi is tangential to every wall
              } else {
                double Kp;
                te_point(p, Kp, ht2);
                gam_pt[p.body] += p.w * ht2;
                if (kw.size() > 0) {
                  const auto& dd = wv.edofs(p.e);
                  double K = 0;
                  for (size_t i = 0; i < dd.size(); ++i) {
                    auto it = wkey.find(key_of(p.e, dd[i]));
                    if (it != wkey.end()) K += kw(it->second) * p.pe.N[i];
                  }
                  ht2 = K * K;
                }
              }
              gam[p.body] += p.w * ht2;
              gam_mr[p.body] += p.mr * p.w * ht2;
              const spec::Material* fm = body_mat[p.body];
              if (fm && fm->rf.flux_flow) {
                // vortices along B0; RF wall current along n x H: TM (H = H_phi) in the (r, z) plane, perpendicular
                // to phi and n, i.e. (-n_z, n_r); TE azimuthal. Lorentz drive ~ |J x B0|: s = 1 - (J.B0)^2/|B0|^2
                const auto& b0 = B0s[ip];
                const double Bm = std::sqrt(b0[0] * b0[0] + b0[1] * b0[1] + b0[2] * b0[2]);
                const double Bv = fm->rf.ff_B_local() ? Bm : std::fabs(fm->rf.ff_B);
                AXF_REQUIRE(Bv < fm->rf.ff_Bc2,
                            strcat("flux_flow wall '", p.body, "': static field ", Bv, " T at (r, z) = (", p.pe.r, ", ",
                                   p.pe.z, ") is above Bc2_T = ", fm->rf.ff_Bc2, " T (the wall would be normal)"),
                            "/materials/" + fm->name + "/rf/flux_flow", "Use a normal-metal rf model for this wall.");
                double sf;
                if (fm->rf.ff_orientation >= 0) {
                  sf = fm->rf.ff_orientation;
                } else if (Bm > 1e-9 * B0_scale && Bm > 0) {
                  const double jb = fam == "TM" ? (-p.nz * b0[0] + p.nr * b0[1]) : b0[2];
                  sf = std::clamp(1 - jb * jb / (Bm * Bm), 0.0, 1.0);
                } else {
                  sf = 1;  // no static field at this point: direction undefined, upper bound
                  ff_nodir.insert(p.body);
                }
                const double wh = p.w * ht2;
                gam_ff[p.body] += wh * fm->rf.Rs_flux_flow(m.f_Hz, Bv, sf);
                gam_ff_perp[p.body] += wh * fm->rf.Rs_flux_flow(m.f_Hz, Bv, 1.0);
                gam_ff_s[p.body] += wh * sf;
              }
            }
            const double WH = 2 * pi * I_HH;  // int mu |H|^2 dV
            {
              // peak wall fields at the wall quadrature points, the definition of the 3D cavity solver
              // (model3d_fullwave.cpp "peak_wall_fields", same keys): the largest normal |E| and tangential |B| over the
              // Gauss points of the wall faces (never at mesh vertices), evaluated in the cavity element; B_rms over
              // the cavity volume with mu_r = 1. E_t_max (should vanish on a PEC wall; TM: a natural boundary
              // condition here, so only weakly) is a diagnostic.
              double pE = 0, pB = 0, pEt = 0, xE[2] = {0, 0}, xB[2] = {0, 0};
              int npts = 0;
              std::map<std::pair<int, int>, std::array<double, 3>> facc;  // (face, element) -> (A, int E_n^2, int B_t^2)
              for (const SurfPt& p : spts) {
                const WaveRole& ro = wv.roles()[p.e];
                if (!in_cav[p.e] || ro.master || ro.excluded) continue;
                const auto& dd = wv.edofs(p.e);
                double dr = 0, dz = 0;
                for (size_t i = 0; i < dd.size(); ++i) {
                  dr += m.x[dd[i]] * p.pe.dN[2 * i];
                  dz += m.x[dd[i]] * p.pe.dN[2 * i + 1];
                }
                const double val = wv.value(m.x, p.e, p.pe);
                const double cbr = -dz, cbz = dr + (p.pe.r > 0 ? val / p.pe.r : dr);  // curl(u phi^)
                double En, Et, Bt;
                if (fam == "TM") {  // u = H_phi: E = curl(u phi^)/(omega eps), B = mu0 u (tangential)
                  const double er = cbr / (m.omega * eps[p.e]), ez = cbz / (m.omega * eps[p.e]);
                  En = std::fabs(er * p.nr + ez * p.nz);
                  Et = std::fabs(-er * p.nz + ez * p.nr);
                  Bt = mu0 * std::fabs(val);
                } else {  // u = A_phi: E_phi = omega u (tangential), B = curl(u phi^)
                  En = 0;
                  Et = m.omega * std::fabs(val);
                  const double bn = cbr * p.nr + cbz * p.nz;
                  Bt = std::sqrt(std::max(0.0, cbr * cbr + cbz * cbz - bn * bn));
                }
                if (En > pE) { pE = En; xE[0] = p.pe.r; xE[1] = p.pe.z; }
                if (Bt > pB) { pB = Bt; xB[0] = p.pe.r; xB[1] = p.pe.z; }
                pEt = std::max(pEt, Et);
                ++npts;
                auto& fa = facc[{p.f, p.e}];
                fa[0] += p.w;
                fa[1] += p.w * En * En;
                fa[2] += p.w * Bt * Bt;
              }
              double pEf = 0, pBf = 0;
              for (auto& [k, fa] : facc)
                if (fa[0] > 0) {
                  pEf = std::max(pEf, std::sqrt(fa[1] / fa[0]));
                  pBf = std::max(pBf, std::sqrt(fa[2] / fa[0]));
                }
              const double Brms = V_cav > 0 ? std::sqrt(mu0 * std::max(WH, 0.0) / V_cav) : 0.0;
              json pw = {{"E_max", pE}, {"B_t_max", pB}, {"E_t_max", pEt}, {"B_rms", Brms}, {"E_max_face_rms", pEf},
                         {"B_t_max_face_rms", pBf}, {"E_max_at_rz", {xE[0], xE[1]}}, {"B_t_max_at_rz", {xB[0], xB[1]}},
                         {"n_points", npts},
                         {"note", "largest normal |E| and tangential |B| over the wall quadrature points (Gauss points of "
                                  "the wall faces, evaluated in the cavity element; the 3D cavity's definition), in the "
                                  "units of the eigenvector (ratios are what matter); B_rms over the cavity volume, mu_r = 1; "
                                  "*_face_rms: largest face rms; E_t_max: largest tangential |E| (0 on a PEC wall in the "
                                  "limit; a resolution diagnostic). cEpk_over_Brms = E_max/(c B_rms)"}};
              if (Brms > 0 && npts > 0) {  // no wall points (a cavity closed by the domain boundary): no ratios
                pw["Bpk_over_Brms"] = pB / Brms;
                pw["cEpk_over_Brms"] = pE / (units::c * Brms);
                pw["Bpk_face_rms_over_Brms"] = pBf / Brms;
                pw["cEpk_face_rms_over_Brms"] = pEf / (units::c * Brms);
                pw["cEt_max_over_Brms"] = pEt / (units::c * Brms);
              }
              row["peak_wall_fields"] = pw;
            }
            double inv_Qw = 0, gsum = 0, TQ = 0;
            bool any_rf = false, rs_missing = false;
            std::vector<std::string> lossless;
            json bj = json::object();
            for (auto& [b, G] : gam) {
              gsum += G;
              json e = {{"G_ohm", m.omega * WH / std::max(G, 1e-300)}};
              if (gam_pt.count(b)) e["G_ohm_pointwise"] = m.omega * WH / std::max(gam_pt[b], 1e-300);
              const spec::Material* mt = body_mat[b];
              double Rs0 = 0;
              bool miss = false;
              if (mt && !mt->rf.model.empty()) {
                try {
                  Rs0 = mt->rf.Rs_base(m.f_Hz);
                  for (auto& [k, t] : mt->rf.warnings(m.f_Hz)) rs_warn[mt->rf.owner + "/" + k] = t;
                } catch (const std::exception& ex) {  // e.g. outside an rf table: no Q for this mode
                  e["note"] = std::string("R_s not available: ") + ex.what();
                  rs_missing = miss = true;
                }
              }
              const bool ffw = mt && mt->rf.flux_flow && !miss;
              // additive vortex term, |H_t|^2-weighted over the wall with the local B0 and orientation
              const double Rff = ffw ? gam_ff[b] / std::max(G, 1e-300) : 0.0;
              const double Rs = Rs0 * (mt && !mt->rf.mr_model.empty() ? gam_mr[b] / std::max(G, 1e-300) : 1.0) + Rff;
              if (Rs > 0) {
                const double Qb = m.omega * WH / std::max(Rs * G, 1e-300);
                e["Rs_ohm"] = Rs;
                e["Rs_model"] = mt->rf.model;
                if (ffw) {  // order-of-magnitude model, reported separately
                  e["Rs_flux_flow_ohm"] = Rff;
                  e["Rs_flux_flow_perpendicular_ohm"] = gam_ff_perp[b] / std::max(G, 1e-300);
                  e["flux_flow_orientation_factor"] = gam_ff_s[b] / std::max(G, 1e-300);
                  flux_flow_walls.insert(b);
                }
                e["Q"] = Qb;
                e["T_K"] = mt->rf.T_K;
                inv_Qw += 1.0 / Qb;
                TQ += mt->rf.T_K / Qb;
                any_rf = true;
              } else if (!miss) {
                e["note"] = cav.allow_lossless_walls ? "no rf data: lossless (cavity.allow_lossless_walls)"
                                                     : "no rf data: lossless";
                lossless.push_back(b);
              }
              bj[b] = e;
            }
            if (!lossless.empty()) {
              // Q_wall and Q0 then cover only the walls with rf data (warned unless allowed; as in 3D and xsec)
              lj["lossless_walls"] = lossless;
              if (!cav.allow_lossless_walls) for (auto& b : lossless) lossless_fam.insert(b);
            }
            if (cav.allow_lossless_walls) lj["walls_lossless_by_request"] = !lossless.empty();
            if (any_rf)
              for (auto it = bj.begin(); it != bj.end(); ++it)
                if (it->contains("Q")) (*it)["P_frac"] = (1.0 / (*it)["Q"].get<double>()) / inv_Qw;
            lj["walls"] = bj;
            if (gsum > 0) lj["G_ohm"] = m.omega * WH / gsum;
            if (any_rf && !rs_missing) {
              lj["Q_wall"] = 1.0 / inv_Qw;
              lj["T_over_Q_K"] = TQ;
            }
            if (!dl.empty()) {
              lj["dielectric"] = dl;
              lj["Q_dielectric"] = 1.0 / inv_Qd;
            }
            if (rs_missing) {
              warnings.push_back(strcat("cavity: ", fam, " mode at ", m.f_Hz, " Hz has walls without R_s at that "
                                        "frequency (see losses.walls); Q_wall and Q0 are not reported for it"));
            } else if (any_rf || (inv_Qd > 0 && cav.allow_lossless_walls)) {
              // Q0 needs every wall's loss: with no wall rf data at all it would be Q_dielectric alone, reported
              // only when lossless walls are explicitly allowed
              lj["Q0"] = 1.0 / (inv_Qw + inv_Qd);
            } else if (inv_Qd > 0) {
              q0_omitted = true;
            }
            lj["note"] = "Q0 = omega U / P; walls: perturbative R_s(f) with H_t of the lossless mode, G_ohm = Q R_s "
                         "(geometry factor; the total G_ohm assumes one R_s on all walls)";
            row["losses"] = lj;
          }
        }
        list.push_back(row);
      }
      for (auto& [k, t] : rs_warn) warnings.push_back("cavity: " + t);
      if (!flux_flow_walls.empty()) {
        std::string names;
        for (auto& n : flux_flow_walls) names += (names.empty() ? "" : ", ") + n;
        warnings.push_back(strcat("cavity: the R_s of walls ", names, " includes a flux_flow (vortex) term "
                                  "(Rs_flux_flow_ohm): an order-of-magnitude model with large uncertainty, weighted "
                                  "per wall point by |H_t|^2 and the orientation factor sin^2(J_rf, B0) (B0 = the "
                                  "form-factor static field; Rs_flux_flow_perpendicular_ohm: without it). Current "
                                  "parallel to the vortices is assumed lossless; misalignment, roughness and vortex "
                                  "tilt make it lossy in practice (docs/physics/surface_resistance.md Sec. 4)"));
      }
      if (!ff_nodir.empty()) {
        std::string names;
        for (auto& n : ff_nodir) names += (names.empty() ? "" : ", ") + n;
        warnings.push_back("cavity: flux_flow walls " + names + " have points where the static field vanishes; the "
                           "orientation factor is set to 1 there");
      }
      if (!lossless_fam.empty()) {
        std::string names;
        for (auto& n : lossless_fam) names += (names.empty() ? "" : ", ") + n;
        warnings.push_back("cavity: " + fam + " modes: the walls " + names + " have no rf data and are lossless: "
                           "Q_wall, Q0 and P_frac leave out their loss (see losses.lossless_walls and their G_ohm). "
                           "Give them an rf model, or set cavity.allow_lossless_walls if this is intended");
      }
      if (q0_omitted)
        warnings.push_back("cavity: " + fam + " modes: no wall has rf data, so Q0 is not reported (it would be "
                           "Q_dielectric alone); Q_dielectric is. Give the walls an rf model or set "
                           "cavity.allow_lossless_walls");
      if (cav.enabled) fw_modes_[fam] = std::move(modes);  // eigenvectors are kept for the cavity analysis only
      json fj = {{"modes", list}, {"solver", d},
                 {"note", fam == "TE" ? "azimuthal E; closed sphere (A = 0); pickup and slit bodies open"
                                      : "azimuthal H; closed sphere; readout port open; enclosed cavities "
                                        "conserve their toroidal flux (static modes removed)"}};
      if (fw.target_f_Hz > 0) fj["target_f_Hz"] = fw.target_f_Hz;
      if (cav.enabled) fj["note"] = fam == "TE" ? "azimuthal E (TE0np) inside the cavity" : "azimuthal H (TM0np) inside the cavity";
      eig[fam] = fj;
    }
    out["eigenmodes"] = eig;
  }
  if (cav.enabled) out["cavity"] = cav_json;
  out["seconds"] = timer.seconds();
  R["fullwave"] = out;
}

// ---------------------------------------------------------------- critical state (HTS screening currents)
// Time-dependent A-formulation with the E-J power law in the designated regions (hts.hpp).  The static
// and pickup solves above keep the ideal-superconductor model, so this analysis reports what the
// screening currents change: their distribution, the screening-current-induced field (SCIF) at the
// field-of-view points, the trapped flux after a ramp-down, the AC loss per cycle, and Psi recomputed
// with the critical-state field.
void AxiModel::hts_analysis(const SolveOptions& o) {
  const auto& hs = d_.hts;
  if (!hs.enabled) return;
  Timer timer;
  const Space& sp = *sp_;
  const mfem::Mesh& m = sp.mesh;
  const int ne = m.GetNE();
  json& R = results_;
  json& warnings = R["warnings"];
  std::vector<ElemInfo> info = element_info(d_, sp, [this](int a) { return region_name(a); });

  AXF_REQUIRE(a_lab_ != nullptr,
              "the critical-state analysis needs a poloidal static field (azimuthal winding currents "
              "or a background field)", "/hts",
              "Screening currents in this model are azimuthal; a toroidal static field (poloidal winding) "
              "is not supported.");
  for (auto& w : d_.windings)
    AXF_REQUIRE(w.type == spec::WindingType::Azimuthal,
                "the critical-state analysis supports azimuthal windings only", "/windings",
                "The screening currents and the field they produce are poloidal (azimuthal currents).");
  for (auto& c : d_.circuits)
    AXF_REQUIRE(c.kind != spec::Circuit::Kind::Persistent,
                "persistent circuits are not supported in the critical-state analysis yet", "/circuits",
                "Remove the circuit, or drive the design with a prescribed current.");
  for (auto& cv : d_.curves)
    AXF_REQUIRE(d_.materials.at(cv.material).kind != spec::MaterialKind::Superconductor,
                "zero-thickness superconducting sheets are not supported in the critical-state analysis",
                "/curves", "Model the sheet as a thin region.");

  // ---- materials of the critical-state regions
  hts_laws_.clear();
  std::vector<HTSMaterial> mats;
  std::map<std::string, int> mat_of_region;
  json rj = json::array();
  for (auto& g : hs.regions) {
    const spec::Conductor& c = d_.conductors.at(g.conductor);
    hts_laws_.push_back(make_jc(c));
    HTSMaterial M;
    M.name = g.conductor;
    M.jc = hts_laws_.back().get();
    M.n = c.n_value;
    M.E0 = c.E0;
    M.T_op = c.T_op;
    M.fill = c.fill_factor;
    M.normal_r = c.normal_r;
    M.normal_z = c.normal_z;
    M.e_floor = hs.e_floor;
    mat_of_region[g.region] = static_cast<int>(mats.size());
    double area = 0, rmin = 1e300, rmax = 0, zmin = 1e300, zmax = -1e300;
    for (int e = 0; e < ne; ++e) {
      if (info[e].region != g.region) continue;
      area += element_area(sp, e);
      double rc, zc;
      element_center(sp, e, rc, zc);
      rmin = std::min(rmin, rc); rmax = std::max(rmax, rc);
      zmin = std::min(zmin, zc); zmax = std::max(zmax, zc);
    }
    AXF_REQUIRE(area > 0, "critical-state region '" + g.region + "' has no elements in the mesh", "/hts/regions");
    rj.push_back({{"region", g.region}, {"conductor", g.conductor}, {"law", M.jc->law()},
                  {"initial", g.initial}, {"n_value", M.n}, {"E0_V_m", M.E0}, {"T_op_K", M.T_op},
                  {"fill_factor", M.fill}, {"area_m2", area},
                  {"Jc_self_field_A_m2", M.Jc_eng(0, 0)},
                  {"r_range_m", {rmin, rmax}}, {"z_range_m", {zmin, zmax}}});
    mats.push_back(M);
  }

  for (auto& g : hs.regions)
    for (auto& sc : d_.superconductors)
      if (sc.body == g.region)
        AXF_REQUIRE(sc.azimuthal.value == 0.0 && sc.poloidal.value == 0.0 && !sc.azimuthal_slit,
                    "critical-state region '" + g.region + "' also carries an explicit superconductor state",
                    "/superconductors",
                    "The critical-state model replaces the ideal constraints of that body; remove them, or "
                    "drop the region from `hts` to keep the ideal model.");
  // ---- remaining ideal superconductors stay exact masters; the critical-state regions do not
  static const spec::Material vacuum_mat = spec::builtin_materials().at("vacuum");
  std::vector<ElemInfo> info_ideal = info;
  for (int e = 0; e < ne; ++e)
    if (mat_of_region.count(info[e].region)) info_ideal[e].mat = &vacuum_mat;
  std::vector<int> body_of_elem;
  std::map<int, int> body_of_curve;
  auto bodies = find_bodies(sp, mb_, info_ideal, d_, false, body_of_elem, body_of_curve);
  std::vector<Master> masters;
  for (auto& b : bodies) {
    Master mm;
    mm.name = b.names.empty() ? "body" : b.names.front();
    const spec::Superconductor* st = sc_state(d_, b.names);
    spec::Constraint cst = st ? st->azimuthal : spec::Constraint{};
    if (b.on_axis) { mm.fixed = true; mm.psi = 0; }
    else if (cst.kind == spec::Constraint::Kind::Flux) { mm.fixed = true; mm.psi = cst.value / (2 * pi); }
    else mm.current = cst.value;
    masters.push_back(mm);
  }
  const bool fixed_sc_source = std::any_of(masters.begin(), masters.end(), [](const Master& x) {
    return (x.fixed && x.psi != 0.0) || (!x.fixed && x.current != 0.0);
  });
  if (!bodies.empty())
    warnings.push_back(strcat("critical-state analysis: ", bodies.size(),
                              " ideal superconducting body(ies) keep their static constraint at every time "
                              "step (their state is not evolved)"));
  if (fixed_sc_source)
    warnings.push_back("critical-state analysis: an ideal superconductor carries a prescribed nonzero flux or "
                       "net current; that source does not follow the drive waveform and is present already in "
                       "the initial state, so a zero-field-cooled run does not start from a field-free state");
  {
    bool normal = false;
    for (int e = 0; e < ne; ++e) normal = normal || info[e].mat->kind == spec::MaterialKind::Conductor;
    if (normal)
      warnings.push_back("critical-state analysis: 'conductor' (normal metal) regions carry no eddy currents in "
                         "the time-dependent solve, so the reported dissipation is the HTS loss alone");
  }

  // ---- element roles
  std::map<std::string, double> area;
  for (int e = 0; e < ne; ++e) area[info[e].region] += element_area(sp, e);
  auto winding_J = [&](const spec::Winding& w) {
    if (w.circuit.empty()) return w.ampere_turns / area[w.region];
    for (auto& c : d_.circuits)
      if (c.name == w.circuit) return w.turns * c.current / area[w.region];
    return 0.0;
  };
  bool iron = false;
  std::vector<HTSRole> roles(ne);
  for (int e = 0; e < ne; ++e) {
    HTSRole& ro = roles[e];
    if (body_of_elem[e] >= 0) { ro.master = true; ro.master_id = body_of_elem[e]; ro.nu = 1.0 / mu0; continue; }
    const ElemInfo& x = info[e];
    auto it = mat_of_region.find(x.region);
    if (it != mat_of_region.end()) {
      ro.hts = it->second;
      ro.nu = 1.0 / mu0;  // a superconductor is not magnetic
    } else if (x.mat->kind == spec::MaterialKind::Iron) {
      ro.nu = 1.0 / (mu0 * x.mat->mu_r);
      iron = true;
    } else if (x.mat->kind == spec::MaterialKind::Linear) {
      ro.nu = 1.0 / (mu0 * x.mat->mu_r);
    } else {
      ro.nu = 1.0 / mu0;
    }
    if (x.winding && x.winding->type == spec::WindingType::Azimuthal) ro.J = winding_J(*x.winding);
  }
  if (iron)
    warnings.push_back("critical-state analysis: iron is linearized at its initial permeability mu_r "
                       "(the nonlinear B-H law is not used in the time-dependent solve)");
  {
    // Vacuum cavities fully enclosed by an ideal superconductor are tied to it, as in the static
    // problem.  The critical-state regions are marked "active" so that they are never tied.
    std::vector<ARole> tmp(ne);
    for (int e = 0; e < ne; ++e) {
      tmp[e].master = roles[e].master;
      tmp[e].master_id = roles[e].master_id;
      tmp[e].nu = roles[e].nu;
      tmp[e].J = roles[e].hts >= 0 ? 1.0 : roles[e].J;
    }
    tie_enclosed_cavities(sp, tmp, body_of_curve);
    for (int e = 0; e < ne; ++e) {
      roles[e].master = tmp[e].master;
      roles[e].master_id = tmp[e].master_id;
    }
  }

  // ---- the solver, and a second copy with the regions transparent (the applied field)
  auto make_solver = [&](bool with_hts) {
    std::vector<HTSRole> rr = roles;
    if (!with_hts)
      for (auto& x : rr) x.hts = -1;
    auto cs = std::make_unique<CriticalState>(sp, rr, mats, masters);
    cs->set_open_boundary(kOuterAttr, mb_.domain_radius, 80);
    if (bg_.active()) {
      std::vector<double> psib(sp.ndof());
      for (int i = 0; i < sp.ndof(); ++i) psib[i] = bg_.psi(sp.r[i], sp.z[i]);
      const BackgroundField* bg = &bg_;
      cs->set_background(std::move(psib),
                         [bg](double r, double z) { return r > 0 ? bg->psi(r, z) / r : 0.0; },
                         [bg](double r, double z, double& br, double& bz) { bg->B(r, z, br, bz); });
    }
    cs->set_newton(hs.newton_tol, hs.max_newton);
    cs->setup();
    return cs;
  };
  hts_cs_ = make_solver(true);
  std::unique_ptr<CriticalState> applied = make_solver(false);

  // ---- the drive waveform, refined into time steps
  std::vector<double> ts{hs.times.front()}, scs{hs.scales.front()};
  std::vector<int> breakpoint{0};
  for (size_t k = 1; k < hs.times.size(); ++k) {
    for (int i = 1; i <= hs.steps_per_segment; ++i) {
      const double f = double(i) / hs.steps_per_segment;
      ts.push_back(hs.times[k - 1] + f * (hs.times[k] - hs.times[k - 1]));
      scs.push_back(hs.scales[k - 1] + f * (hs.scales[k] - hs.scales[k - 1]));
    }
    breakpoint.push_back(static_cast<int>(ts.size()) - 1);
  }
  const bool field_cooled =
      std::any_of(hs.regions.begin(), hs.regions.end(), [](auto& g) { return g.initial == "field_cooled"; });
  if (field_cooled && std::any_of(hs.regions.begin(), hs.regions.end(),
                                  [](auto& g) { return g.initial != "field_cooled"; }))
    warnings.push_back("critical-state analysis: mixed cooldown states are treated as field cooled "
                       "(the initial state is one field solution for the whole design)");

  // ---- quadrature helpers for the per-step integrals
  const RefTable& tab = sp.table(sp.quad_rule());
  auto region_integrals = [&](const CriticalState::State& st) {
    struct Agg { double I = 0, Jmax = 0, frac_max = 0, moment = 0; };
    std::vector<Agg> agg(mats.size());
    PointEval pe;
    for (int e = 0; e < ne; ++e) {
      const int h = roles[e].hts;
      if (h < 0 || roles[e].master) continue;
      for (int q = 0; q < tab.nq; ++q) {
        sp.eval(e, tab, q, pe);
        double jc = 0;
        const double J = hts_cs_->J_at(st, e, pe, &jc);
        agg[h].I += J * pe.w;
        agg[h].moment += pi * pe.r * pe.r * J * pe.w;
        agg[h].Jmax = std::max(agg[h].Jmax, std::fabs(J));
        if (jc > 0) agg[h].frac_max = std::max(agg[h].frac_max, std::fabs(J) / jc);
      }
    }
    return agg;
  };
  // Psi with the critical-state field (only the poloidal lab field enters; see post())
  auto psi_of = [&](const CriticalState::State& st) {
    if (!h_pick_) return 0.0;
    double P = 0;
    PointEval pe;
    for (int e = 0; e < ne; ++e) {
      for (int q = 0; q < tab.nq; ++q) {
        sp.eval(e, tab, q, pe);
        double Br, Bz, A = 0, H, mu;
        hts_cs_->field(st, e, pe, Br, Bz, &A);
        h_pick_->field(h_pick_sol_, e, pe, H, mu);
        P += pe.w * pe.r * A * mu * H;
      }
    }
    return P * 2 * pi / I_pick_;
  };
  // field-of-view point location
  const int nfov = static_cast<int>(hs.field_of_view.size());
  mfem::Array<int> fov_e;
  mfem::Array<mfem::IntegrationPoint> fov_ip;
  if (nfov > 0) {
    mfem::DenseMatrix pts(2, nfov);
    for (int i = 0; i < nfov; ++i) { pts(0, i) = hs.field_of_view[i].r; pts(1, i) = hs.field_of_view[i].z; }
    const_cast<mfem::Mesh&>(m).FindPoints(pts, fov_e, fov_ip, false);
    for (int i = 0; i < nfov; ++i)
      if (fov_e[i] < 0)
        warnings.push_back(strcat("critical-state field-of-view point (", hs.field_of_view[i].r, ", ",
                                  hs.field_of_view[i].z, ") is outside the mesh"));
  }
  auto fov_fields = [&](const CriticalState::State& st, const CriticalState& cs, std::vector<double>& Br,
                        std::vector<double>& Bz) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Br.assign(nfov, nan);
    Bz.assign(nfov, nan);
    PointEval pe;
    for (int i = 0; i < nfov; ++i) {
      if (fov_e[i] < 0) continue;
      sp.eval_at(fov_e[i], fov_ip[i], pe);
      cs.field(st, fov_e[i], pe, Br[i], Bz[i]);
    }
  };

  // ---- time stepping
  CriticalState::State st = hts_cs_->start(ts.front(), scs.front(), field_cooled);
  json steps = json::array();
  std::vector<double> cum_loss_at_break(1, 0.0);
  int not_converged = 0;
  double max_rel_res = 0, worst_balance = 0, worst_update = 0, worst_be = 0;
  auto record = [&](int k, const CriticalState::StepInfo* si) {
    const auto agg = region_integrals(st);
    std::vector<double> tBr, tBz, aBr, aBz;
    fov_fields(st, *hts_cs_, tBr, tBz);
    CriticalState::State ast = applied->start(st.t, st.scale, true);
    fov_fields(ast, *applied, aBr, aBz);
    json jr = json::array();
    for (size_t i = 0; i < mats.size(); ++i)
      jr.push_back({{"region", hs.regions[i].region}, {"screening_current_A", agg[i].I},
                    {"max_J_A_m2", agg[i].Jmax}, {"max_J_over_Jc", agg[i].frac_max},
                    {"moment_Am2", agg[i].moment}});
    json jf = json::array();
    for (int i = 0; i < nfov; ++i)
      jf.push_back({{"r_m", hs.field_of_view[i].r}, {"z_m", hs.field_of_view[i].z},
                    {"Br_T", tBr[i]}, {"Bz_T", tBz[i]},
                    {"Br_applied_T", aBr[i]}, {"Bz_applied_T", aBz[i]},
                    {"scif_Br_T", tBr[i] - aBr[i]}, {"scif_Bz_T", tBz[i] - aBz[i]}});
    json j = {{"step", k}, {"t_s", st.t}, {"scale", st.scale}, {"W_reduced_J", st.W_J},
              {"cumulative_loss_J", st.loss_J}, {"cumulative_work_J", st.work_J}, {"regions", jr}};
    if (nfov > 0) j["field_of_view"] = jf;
    if (h_pick_) j["Psi_SI_H_T_m2"] = psi_of(st);
    if (si) {
      j["newton_iterations"] = si->newton;
      j["relative_residual"] = si->rel_residual;
      j["newton_update"] = si->update;
      j["converged"] = si->converged;
      j["converged_on"] = si->converged_on;
      j["loss_J"] = si->loss_J;
      j["work_J"] = si->work_J;
      j["dW_J"] = si->dW_J;
      j["numerical_dissipation_J"] = si->numerical_dissipation_J;
      j["energy_balance_rel"] = si->balance_rel;
      j["energy_balance_backward_euler_rel"] = si->be_balance_rel;
    }
    steps.push_back(j);
  };
  double psi_peak = 0, psi_peak_scale = 0, psi_peak_t = 0;
  auto track_psi = [&]() {
    if (!h_pick_ || std::fabs(st.scale) <= std::fabs(psi_peak_scale)) return;
    psi_peak_scale = st.scale;
    psi_peak_t = st.t;
    psi_peak = steps.back().value("Psi_SI_H_T_m2", 0.0);
  };
  record(0, nullptr);
  track_psi();
  for (size_t k = 1; k < ts.size(); ++k) {
    const CriticalState::StepInfo si = hts_cs_->advance(st, ts[k], scs[k], o.verbose);
    if (!si.converged) ++not_converged;
    max_rel_res = std::max(max_rel_res, si.rel_residual);
    worst_update = std::max(worst_update, si.update);
    worst_be = std::max(worst_be, std::fabs(si.be_balance_rel));
    worst_balance = std::max(worst_balance, std::fabs(si.balance_rel));
    record(static_cast<int>(k), &si);
    track_psi();
    if (std::find(breakpoint.begin(), breakpoint.end(), static_cast<int>(k)) != breakpoint.end())
      cum_loss_at_break.push_back(st.loss_J);
    if (o.verbose > 0 && (k % 10 == 0 || k + 1 == ts.size()))
      log(1, strcat("  critical state: step ", k, "/", ts.size() - 1, " t = ", st.t, " s, scale ", st.scale,
                    ", loss ", st.loss_J, " J"));
  }
  if (not_converged)
    warnings.push_back(strcat("critical-state analysis: ", not_converged, " of ", ts.size() - 1,
                              " time steps did not reach the Newton tolerance (worst update ", worst_update,
                              ", worst relative residual ", max_rel_res, ", worst backward-Euler energy identity ",
                              worst_be, "); raise hts.max_newton or hts.newton_tol"));

  hts_state_ = std::make_unique<CriticalState::State>(st);  // the final state (hts_sample, profiles)
  json out = {{"regions", rj},
              {"drive", {{"times_s", hs.times}, {"scales", hs.scales}, {"steps", ts.size() - 1},
                         {"steps_per_segment", hs.steps_per_segment}}},
              {"initial_state", field_cooled ? "field_cooled" : "zero_field_cooled"},
              {"steps", steps},
              {"total_loss_J", st.loss_J},
              {"total_work_J", st.work_J},
              {"newton",
               {{"max_relative_residual", max_rel_res}, {"max_update", worst_update},
                {"steps_not_converged", not_converged}, {"tolerance", hs.newton_tol},
                {"max_iterations", hs.max_newton},
                {"note", "convergence is measured on the Newton update; the residual ratio is reported for "
                         "reference and is pessimistic (the power-law Jacobian spans many orders of magnitude)"}}},
              {"energy_balance_worst_rel", worst_balance},
              {"energy_balance_backward_euler_worst_rel", worst_be}};

  // AC loss per cycle: the loss between successive returns of the drive to its starting scale
  {
    json cyc = json::array();
    double prev = 0;
    for (size_t k = 1; k < hs.scales.size(); ++k) {
      if (hs.scales[k] != hs.scales.front()) continue;
      if (k + 1 < hs.scales.size() && hs.scales[k + 1] == hs.scales.front()) continue;  // hold at the start
      const double l = cum_loss_at_break[k];
      cyc.push_back(l - prev);
      prev = l;
    }
    if (!cyc.empty()) {
      out["loss_per_cycle_J"] = cyc;
      out["ac_loss_per_cycle_J"] = cyc.back();
    }
  }
  // final state: trapped flux, field quality, screening-current profiles
  {
    const auto agg = region_integrals(st);
    std::vector<double> tBr, tBz, aBr, aBz;
    fov_fields(st, *hts_cs_, tBr, tBz);
    CriticalState::State ast = applied->start(st.t, st.scale, true);
    fov_fields(ast, *applied, aBr, aBz);
    json fj = {{"t_s", st.t}, {"scale", st.scale}, {"W_reduced_J", st.W_J},
               {"note", "W_reduced_J is the energy of the field of the screening currents and windings; "
                        "a background magnet's own energy is excluded, as in the static results"}};
    json jr = json::array();
    for (size_t i = 0; i < mats.size(); ++i)
      jr.push_back({{"region", hs.regions[i].region}, {"screening_current_A", agg[i].I},
                    {"max_J_A_m2", agg[i].Jmax}, {"max_J_over_Jc", agg[i].frac_max},
                    {"moment_Am2", agg[i].moment}});
    fj["regions"] = jr;
    if (nfov > 0) {
      auto quality = [&](const std::vector<double>& B) {
        double lo = 1e300, hi = -1e300, sum = 0;
        int n = 0;
        for (double x : B) {
          if (!std::isfinite(x)) continue;
          lo = std::min(lo, x); hi = std::max(hi, x); sum += x; ++n;
        }
        const double mean = n > 0 ? sum / n : 0.0;
        json j = {{"mean_Bz_T", mean}, {"min_Bz_T", lo}, {"max_Bz_T", hi}};
        j["peak_to_peak_over_mean"] = std::fabs(mean) > 0 ? (hi - lo) / std::fabs(mean) : 0.0;
        return j;
      };
      double scif_max = 0;
      for (int i = 0; i < nfov; ++i)
        if (std::isfinite(tBz[i])) scif_max = std::max(scif_max, std::fabs(tBz[i] - aBz[i]));
      fj["field_quality"] = {{"with_screening_currents", quality(tBz)}, {"applied", quality(aBz)},
                             {"max_scif_Bz_T", scif_max}};
    }
    if (st.scale == 0.0) {
      json tr = {{"moment_Am2", 0.0}};
      double mom = 0;
      for (auto& a : agg) mom += a.moment;
      tr["moment_Am2"] = mom;
      if (nfov > 0) {
        double mx = 0;
        json pts = json::array();
        for (int i = 0; i < nfov; ++i) {
          pts.push_back({{"r_m", hs.field_of_view[i].r}, {"z_m", hs.field_of_view[i].z},
                         {"Br_T", tBr[i]}, {"Bz_T", tBz[i]}});
          if (std::isfinite(tBz[i])) mx = std::max(mx, std::fabs(tBz[i]));
        }
        tr["max_Bz_T"] = mx;
        tr["points"] = pts;
      }
      tr["note"] = "the drive is back at zero: the remaining field and moment are trapped flux";
      fj["trapped"] = tr;
    }
    // radial profile of the screening current density at each region's mid-height
    if (hs.profile_points > 1) {
      json profs = json::array();
      for (size_t i = 0; i < hs.regions.size(); ++i) {
        const double r0 = rj[i]["r_range_m"][0], r1 = rj[i]["r_range_m"][1];
        const double z0 = rj[i]["z_range_m"][0], z1 = rj[i]["z_range_m"][1];
        const double zc = 0.5 * (z0 + z1);
        const int np = hs.profile_points;
        std::vector<double> pr(np), pz(np, zc);
        for (int k = 0; k < np; ++k) pr[k] = r0 + (r1 - r0) * k / (np - 1);
        auto smp = hts_sample(pr, pz);
        json rows = json::array();
        for (int k = 0; k < np; ++k)
          rows.push_back({{"r_m", pr[k]}, {"z_m", zc}, {"J_A_m2", smp[k].J}, {"Jc_A_m2", smp[k].Jc},
                          {"E_V_m", smp[k].E}, {"Bz_T", smp[k].Bz}, {"Br_T", smp[k].Br}});
        profs.push_back({{"region", hs.regions[i].region}, {"z_m", zc}, {"points", rows}});
      }
      fj["current_profiles"] = profs;
    }
    out["final"] = fj;
  }
  // Psi with the critical-state field compared with the ideal-superconductor model
  if (h_pick_) {
    // The baseline Psi is the static solve at full drive, so the comparison is made at the step where
    // the drive is largest; at the final state the drive may be back at zero.
    const double psi_cs = psi_peak, psi_final = psi_of(st);
    const double psi0 = R["coupling"].value("Psi_SI_H_T_m2", 0.0);
    json pj = {{"Psi_ideal_SI_H_T_m2", psi0},
               {"Psi_critical_state_SI_H_T_m2", psi_cs},
               {"at_peak_drive", {{"scale", psi_peak_scale}, {"t_s", psi_peak_t}}},
               {"final", {{"scale", st.scale}, {"t_s", st.t}, {"Psi_SI_H_T_m2", psi_final}}}};
    if (psi0 != 0 && psi_peak_scale != 0) pj["relative_change"] = psi_cs / (psi0 * psi_peak_scale) - 1.0;
    pj["note"] = "Psi recomputed with the critical-state static field and the same pickup mode, at the step "
                 "with the largest drive scale; the baseline is the ideal (Meissner) model at drive scale 1, "
                 "so relative_change divides by the peak scale. Psi(t) of every step is in `steps`";
    out["psi"] = pj;
  } else {
    out["psi"] = {{"note", "Psi is not affected by a poloidal static field for this pickup (or there is no "
                           "pickup): only a port (poloidal) pickup couples to it"}};
  }
  out["seconds"] = timer.seconds();
  R["hts"] = out;
}

std::vector<AxiModel::HTSSample> AxiModel::hts_sample(const std::vector<double>& r,
                                                      const std::vector<double>& z) const {
  AXF_REQUIRE(hts_cs_ && hts_state_, "this design has no critical-state (hts) analysis");
  AXF_REQUIRE(r.size() == z.size(), "r and z must have the same length");
  const int n = static_cast<int>(r.size());
  mfem::DenseMatrix pts(2, n);
  for (int i = 0; i < n; ++i) { pts(0, i) = r[i]; pts(1, i) = z[i]; }
  mfem::Array<int> eid;
  mfem::Array<mfem::IntegrationPoint> ips;
  sp_->mesh.FindPoints(pts, eid, ips, false);
  std::vector<HTSSample> out(n);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  PointEval pe;
  for (int i = 0; i < n; ++i) {
    if (eid[i] < 0) {
      out[i].J = out[i].Jc = out[i].E = out[i].Br = out[i].Bz = out[i].A = nan;
      continue;
    }
    sp_->eval_at(eid[i], ips[i], pe);
    out[i].region = sp_->mesh.GetAttribute(eid[i]);
    out[i].hts = hts_cs_->roles()[eid[i]].hts >= 0;
    hts_cs_->field(*hts_state_, eid[i], pe, out[i].Br, out[i].Bz, &out[i].A);
    out[i].E = hts_cs_->E_at(*hts_state_, eid[i], pe);
    out[i].J = hts_cs_->J_at(*hts_state_, eid[i], pe, &out[i].Jc);
  }
  return out;
}

bool AxiModel::fw_fields(int e, const PointEval& pe, cplx B[3], cplx E[3]) const {
  for (int k = 0; k < 3; ++k) B[k] = E[k] = 0.0;
  if (!fw_sol_) return false;
  const Space& sp = *sp_;
  const AxiWave& wv = fw_sol_family_ == "TE" ? *fw_te_ : *fw_tm_;
  const WaveRole& ro = wv.roles()[e];
  const double w = fw_sol_->omega, r = pe.r;
  cplx a = 0, dr = 0, dz = 0;
  if (!ro.excluded && !ro.master) {
    const auto& dd = sp.edofs(e);
    for (size_t i = 0; i < dd.size(); ++i) {
      a += fw_sol_->u[dd[i]] * pe.N[i];
      dr += fw_sol_->u[dd[i]] * pe.dN[2 * i];
      dz += fw_sol_->u[dd[i]] * pe.dN[2 * i + 1];
    }
  }
  const cplx curl_r = -dz, curl_z = dr + (r > 0 ? a / r : dr);  // curl of (u phi-hat)
  const cplx I1(0.0, 1.0);
  if (fw_sol_family_ == "TE") {  // u = A_phi: B = curl A, E = i omega A (conductor interiors: 0)
    if (ro.master || ro.excluded) return true;
    B[0] = curl_r;
    B[2] = curl_z;
    E[1] = I1 * w * a;
  } else {  // u = H' = H_phi - H_MQS: B_phi = mu H_phi, E = (i / omega eps) curl H'
    if (ro.excluded) return true;
    double H = 0, mu = mu0;
    h_pick_->field(h_pick_sol_, e, pe, H, mu);
    B[1] = mu * (H / I_pick_ + a);
    const cplx f = I1 / (w * fw_eps_[e]);
    E[0] = f * curl_r;
    E[2] = f * curl_z;
  }
  return true;
}

json run_design(const json& doc, const SolveOptions& o) {
  AxiModel model(spec::parse_design(doc));
  model.run(o);
  return model.results();
}

}  // namespace axf
