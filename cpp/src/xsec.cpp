// Cross-section eigensolver for z-uniform cavities ("symmetry": "xsec"; beta, docs/validation/cavity_xsec.md).
// Formulation and conventions: axionfem/xsec.hpp; validation: docs/validation/cavity_xsec.md.
#include "axionfem/xsec.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>

#include "axionfem/axi.hpp"
#include "axionfem/core.hpp"
#include "axionfem/fullwave.hpp"
#include "axionfem/units.hpp"

namespace axf {

using json = nlohmann::json;
using axi::PointEval;
using SpMat = Eigen::SparseMatrix<double>;
using Vec = Eigen::VectorXd;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kEps0 = 1.0 / (units::mu0 * units::c * units::c);

// Reference triangle (MFEM): vertices (0,0), (1,0), (0,1); local edge k runs from vertex k to vertex k+1 (mod 3).
const double kRefV[3][2] = {{0, 0}, {1, 0}, {0, 1}};

}  // namespace

struct XsecModel::Impl {
  std::unique_ptr<mfem::Mesh> mesh;
  std::unique_ptr<axi::Space> sp;
  std::map<int, std::string> region_of_attr;
  std::map<int, const spec::Curve*> curve_of_attr;
  int outer_attr = axi::kOuterAttr;
  // per element
  std::vector<int> kind;  // 0 cavity, 1 conductor, 2 non-conductor outside the cavity (not assembled)
  std::vector<const spec::Material*> mat;
  std::vector<std::string> reg;
  std::vector<std::vector<int>> eedges;  // mesh edge (face) index of local edge k
  // cavity wall: one entry per (edge, cavity element) side
  struct Wall {
    int face = -1, elem = -1, ledge = -1;
    std::string body;
    const spec::Material* m = nullptr;  // nullptr: no rf data (lossless)
    bool sheet = false;                 // two-sided sheet face
    std::vector<int> loc;               // local DOFs of elem on this edge
  };
  std::vector<Wall> walls;
  std::vector<char> sheet_face;  // per mesh face: conducting sheet with the cavity on both sides
  std::map<std::string, const spec::Material*> body_mat;
  std::vector<std::vector<int>> dof_elems;  // Space DOF -> cavity elements containing it
  double area = 0, area_full = 0;
  std::set<std::string> outer_bodies;  // bodies on the outer boundary loop (area_full)
  std::string wall_body = "wall";
  // per family
  struct Fam {
    bool tm = true;
    std::vector<std::vector<int>> edofs;  // per element (cavity elements; split copies for TE at sheets)
    int ndof = 0;
    std::vector<char> dirichlet;           // per family DOF
    std::vector<Eigen::MatrixXd> Ke, Me;   // unscaled element matrices (alpha grad.grad, beta N N)
    std::vector<axi::Eigenmode> modes;     // eigenvectors over all family DOFs
    std::vector<double> norm;              // scale to int eps_r |E|^2 dA = 1 (with sign)
  };
  std::map<std::string, Fam> fam;

  double eps_r(int e) const { return mat[e] ? mat[e]->eps_r : 1.0; }
  double mu_r(int e) const { return mat[e] ? mat[e]->mu_r : 1.0; }

  // Fans: the cavity elements around `dof` grouped by adjacency through faces that are not two-sided sheets.
  std::vector<std::vector<int>> fans(int dof) const {
    const auto& el = dof_elems[dof];
    std::vector<int> comp(el.size(), -1);
    int nc = 0;
    for (size_t s = 0; s < el.size(); ++s) {
      if (comp[s] >= 0) continue;
      comp[s] = nc;
      std::vector<size_t> stack{s};
      while (!stack.empty()) {
        const size_t a = stack.back();
        stack.pop_back();
        for (auto [nb, f] : sp->nbr[el[a]]) {
          if (sheet_face[f]) continue;
          for (size_t b = 0; b < el.size(); ++b)
            if (el[b] == nb && comp[b] < 0) {
              comp[b] = nc;
              stack.push_back(b);
            }
        }
      }
      ++nc;
    }
    std::vector<std::vector<int>> out(nc);
    for (size_t s = 0; s < el.size(); ++s) out[comp[s]].push_back(el[s]);
    return out;
  }

  // Unscaled fields of a mode at a point of cavity element e (E, H real amplitudes: E_c = i E, H_c = H).
  void eval(const Fam& F, const axi::Eigenmode& m, int e, const PointEval& pe, double E[3], double H[3]) const {
    const auto& dd = F.edofs[e];
    double u = 0, ux = 0, uy = 0;
    for (size_t i = 0; i < dd.size(); ++i) {
      const double xi = m.x[dd[i]];
      u += xi * pe.N[i];
      ux += xi * pe.dN[2 * i];
      uy += xi * pe.dN[2 * i + 1];
    }
    const double w = m.omega;
    if (F.tm) {  // E = u z^, H = curl(u z^)/(omega mu) = (u_y, -u_x)/(omega mu)
      const double s = 1.0 / (w * units::mu0 * mu_r(e));
      E[0] = E[1] = 0;
      E[2] = u;
      H[0] = s * uy;
      H[1] = -s * ux;
      H[2] = 0;
    } else {  // H = u z^, E = curl(u z^)/(omega eps) = (u_y, -u_x)/(omega eps)
      const double s = 1.0 / (w * kEps0 * eps_r(e));
      H[0] = H[1] = 0;
      H[2] = u;
      E[0] = s * uy;
      E[1] = -s * ux;
      E[2] = 0;
    }
  }

  // Points, weights (dl) and outward normals on local edge k of element e (Gauss rule on the reference edge).
  struct EdgePt {
    PointEval pe;
    double dl = 0, nx = 0, ny = 0;
  };
  std::vector<EdgePt> edge_points(int e, int k, int order) const {
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, order);
    mfem::ElementTransformation* T = mesh->GetElementTransformation(e);
    const double* a = kRefV[k];
    const double* b = kRefV[(k + 1) % 3];
    std::vector<EdgePt> out(ir.GetNPoints());
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const double t = ir.IntPoint(q).x;
      mfem::IntegrationPoint ip;
      ip.Set2(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
      T->SetIntPoint(&ip);
      const mfem::DenseMatrix& J = T->Jacobian();
      const double tx = J(0, 0) * (b[0] - a[0]) + J(0, 1) * (b[1] - a[1]);
      const double ty = J(1, 0) * (b[0] - a[0]) + J(1, 1) * (b[1] - a[1]);
      const double len = std::hypot(tx, ty);
      const double sgn = J.Det() >= 0 ? 1.0 : -1.0;  // counter-clockwise element: outward normal = (t_y, -t_x)
      EdgePt& p = out[q];
      sp->eval_at(e, ip, p.pe);
      p.pe.ip = nullptr;
      p.dl = ir.IntPoint(q).weight * len;
      p.nx = sgn * ty / len;
      p.ny = -sgn * tx / len;
    }
    return out;
  }
};

XsecModel::XsecModel(spec::Design d) : d_(std::move(d)), p_(std::make_unique<Impl>()) {
  AXF_REQUIRE(d_.symmetry == "xsec", "XsecModel needs a cross-section design (\"symmetry\": \"xsec\")", "/symmetry");
}
XsecModel::~XsecModel() = default;

int XsecModel::mode_count(const std::string& family) const {
  auto it = p_->fam.find(family);
  return it == p_->fam.end() ? 0 : static_cast<int>(it->second.modes.size());
}

void XsecModel::run(const SolveOptions& o) {
  Timer timer;
  Impl& P = *p_;
  const auto& cav = d_.cavity;
  const auto& fw = d_.fullwave;
  json warnings = json::array();
  const double pi = units::pi;

  // ------------------------------------------------------------ mesh (axf-mesher "plane" mode)
  std::error_code ec;
  std::filesystem::create_directories(o.workdir, ec);
  AXF_REQUIRE(!ec, "cannot create working directory '" + o.workdir + "': " + ec.message());
  double xmin = 1e300, xmax = -1e300, ymin = 1e300, ymax = -1e300;
  for (auto& g : d_.regions) {
    double a, b, c, e;
    g.shape.bbox(a, b, c, e);
    xmin = std::min(xmin, a); xmax = std::max(xmax, b);
    ymin = std::min(ymin, c); ymax = std::max(ymax, e);
  }
  AXF_REQUIRE(!d_.regions.empty() && xmax > xmin && ymax > ymin, "the cross-section has no extent", "/regions");
  const double ext = std::max(xmax - xmin, ymax - ymin);
  const double h = d_.mesh.size > 0 ? d_.mesh.size : ext / 30.0;
  const double hs = d_.mesh.surface_size > 0 ? d_.mesh.surface_size : h / 2.0;
  json in;
  in["plane"] = true;
  in["size"] = h;
  in["surface_size"] = hs;
  in["surface_layer"] = 4.0 * h;
  in["order"] = std::min(d_.mesh.order, 2);
  in["outer_attr"] = P.outer_attr;
  json regs = json::array(), curves = json::array(), refine = json::array();
  int attr = 2;
  for (auto& g : d_.regions) {
    json r = {{"name", g.name}, {"attr", attr}, {"shape", g.shape.resolved()}};
    if (g.mesh_size > 0) r["size"] = g.mesh_size;
    if (g.surface_size > 0) r["surface_size"] = g.surface_size;
    regs.push_back(r);
    P.region_of_attr[attr] = g.name;
    const auto k = d_.materials.at(g.material).kind;
    if (g.surface_size <= 0 && (k == spec::MaterialKind::Conductor || k == spec::MaterialKind::Superconductor ||
                                d_.materials.at(g.material).eps_r != 1.0))
      refine.push_back(g.name);
    ++attr;
  }
  int cattr = axi::kCurveAttr0;
  for (auto& c : d_.curves) {
    json cj = {{"name", c.name}, {"attr", cattr}, {"path", c.path.resolved()}};
    if (c.surface_size > 0) cj["surface_size"] = c.surface_size;
    curves.push_back(cj);
    P.curve_of_attr[cattr++] = &c;
  }
  in["regions"] = regs;
  in["curves"] = curves;
  in["refine_regions"] = refine;
  const std::string base = o.workdir + "/mesh";
  in["output"] = base + ".msh";
  in["info"] = base + "_info.json";
  in["verbose"] = o.verbose > 1 ? 1 : 0;
  std::ofstream(base + "_in.json") << in.dump(2);
  const int rc = axi::run_process({axi::find_mesher(o.mesher), base + "_in.json"}, base + "_mesher.log");
  json minfo;
  {
    std::ifstream f(base + "_info.json");
    if (f) minfo = json::parse(f, nullptr, false);
  }
  if (rc != 0 || !minfo.is_object() || !minfo.value("ok", false)) {
    const std::string why = minfo.is_object() ? minfo.value("error", std::string("unknown")) : "no info file";
    throw Error("cross-section mesh generation failed: " + why, "/regions",
                "Check the shapes (x, y in metres) and that curves lie inside the regions; see " + base + "_mesher.log.");
  }
  P.mesh = std::make_unique<mfem::Mesh>((base + ".msh").c_str(), 1, 1, true);
  for (int i = 0; i < d_.mesh.refinements; ++i) P.mesh->UniformRefinement();
  const int order = o.order > 0 ? o.order : d_.mesh.order;
  AXF_REQUIRE(order >= 1 && order <= 6, strcat("FE order ", order, " out of range"), "/mesh/order", "Use 1..6 (2 default).");
  P.sp = std::make_unique<axi::Space>(*P.mesh, order);
  const axi::Space& sp = *P.sp;
  const int ne = P.mesh->GetNE();
  log(1, strcat("xsec mesh: ", ne, " triangles, order ", order));

  // ------------------------------------------------------------ element roles, cavity component
  P.kind.assign(ne, 2);
  P.mat.assign(ne, nullptr);
  P.reg.assign(ne, "");
  P.eedges.assign(ne, {});
  std::vector<char> cond(ne, 0);
  for (int e = 0; e < ne; ++e) {
    const int a = P.mesh->GetAttribute(e);
    auto it = P.region_of_attr.find(a);
    AXF_REQUIRE(it != P.region_of_attr.end(), strcat("internal: element attribute ", a, " has no region"));
    P.reg[e] = it->second;
    const spec::Region* g = d_.find_region(it->second);
    P.mat[e] = &d_.materials.at(g->material);
    cond[e] = P.mat[e]->kind == spec::MaterialKind::Conductor || P.mat[e]->kind == spec::MaterialKind::Superconductor;
    mfem::Array<int> ed, eo;
    P.mesh->GetElementEdges(e, ed, eo);
    P.eedges[e].assign(ed.begin(), ed.end());
  }
  const int nf = P.mesh->GetNumFaces();
  std::vector<int> face_curve(nf, 0);  // curve attribute of each face (0: none)
  for (int f = 0; f < nf; ++f)
    if (P.curve_of_attr.count(sp.face_bdr_attr[f])) face_curve[f] = sp.face_bdr_attr[f];
  // components of non-conductor elements, connected through faces that are not sheets
  std::vector<int> comp(ne, -1);
  int ncomp = 0;
  for (int s = 0; s < ne; ++s) {
    if (cond[s] || comp[s] >= 0) continue;
    comp[s] = ncomp;
    std::vector<int> st{s};
    while (!st.empty()) {
      const int a = st.back();
      st.pop_back();
      for (auto [b, f] : sp.nbr[a])
        if (!cond[b] && comp[b] < 0 && !face_curve[f]) {
          comp[b] = ncomp;
          st.push_back(b);
        }
    }
    ++ncomp;
  }
  AXF_REQUIRE(ncomp > 0, "the cross-section has no vacuum or dielectric region (only conductors)", "/regions");
  int cav_comp = 0;
  if (cav.interior_given) {
    auto locate = [&](double x, double y) {
      mfem::DenseMatrix pts(2, 1);
      pts(0, 0) = x;
      pts(1, 0) = y;
      mfem::Array<int> eid;
      mfem::Array<mfem::IntegrationPoint> ips;
      P.mesh->FindPoints(pts, eid, ips, false);
      return eid[0];
    };
    int e0 = locate(cav.interior_point.r, cav.interior_point.z);
    AXF_REQUIRE(e0 >= 0, strcat("cavity.interior_point (", cav.interior_point.r, ", ", cav.interior_point.z,
                                ") lies outside the cross-section"), "/cavity/interior_point",
                "Give a point [x, y] inside the cavity, in metres.");
    if (cond[e0]) {
      const double dl = 1e-9 * ext;
      int e1 = -1;
      for (auto [dx, dy] : std::vector<std::pair<double, double>>{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
        const int c = locate(cav.interior_point.r + dx * dl, cav.interior_point.z + dy * dl);
        if (c >= 0 && !cond[c]) { e1 = c; break; }
      }
      AXF_REQUIRE(e1 >= 0, strcat("cavity.interior_point lies inside the conductor region '", P.reg[e0], "'"),
                  "/cavity/interior_point", "Give a point in the vacuum or dielectric of the cavity.");
      warnings.push_back(strcat("cavity: interior_point lies on the surface of '", P.reg[e0], "'; the adjacent '",
                                P.reg[e1], "' element selects the cavity"));
      e0 = e1;
    }
    cav_comp = comp[e0];
  } else {
    AXF_REQUIRE(ncomp == 1, strcat("the cross-section has ", ncomp, " separate vacuum/dielectric components (cells "
                                   "divided by conductors or sheets); say which one is the cavity"),
                "/cavity/interior_point", "Add cavity.interior_point [x, y] inside the wanted cell.");
  }
  std::set<std::string> cav_regions, excluded_regions;
  for (int e = 0; e < ne; ++e) {
    if (cond[e]) P.kind[e] = 1;
    else if (comp[e] == cav_comp) {
      P.kind[e] = 0;
      cav_regions.insert(P.reg[e]);
    } else {
      excluded_regions.insert(P.reg[e]);
    }
  }
  for (auto& g : d_.regions)
    if (g.name == "wall") P.wall_body = "wall (domain boundary)";
  for (auto& c : d_.curves)
    if (c.name == "wall") P.wall_body = "wall (domain boundary)";

  // ------------------------------------------------------------ walls
  P.sheet_face.assign(nf, 0);
  const spec::Material* wall_mat = cav.wall_material.empty() ? nullptr : &d_.materials.at(cav.wall_material);
  const int qe = 2 * order + 4;  // edge quadrature order
  std::set<std::string> two_sided;
  for (int e = 0; e < ne; ++e) {
    if (P.kind[e] != 0) continue;
    for (int k = 0; k < 3; ++k) {
      const int f = P.eedges[e][k];
      int other = -1;
      for (auto [b, ff] : sp.nbr[e])
        if (ff == f) other = b;
      Impl::Wall w;
      w.face = f;
      w.elem = e;
      w.ledge = k;
      if (face_curve[f]) {
        const spec::Curve* c = P.curve_of_attr.at(face_curve[f]);
        w.body = c->name;
        w.m = &d_.materials.at(c->material);
        w.sheet = other >= 0 && P.kind[other] == 0;
        if (w.sheet) {
          P.sheet_face[f] = 1;
          two_sided.insert(c->name);
        }
      } else if (other < 0) {
        w.body = P.wall_body;
        w.m = wall_mat;
      } else if (P.kind[other] == 1) {
        w.body = P.reg[other];
        w.m = P.mat[other];
      } else {
        continue;  // interior face of the cavity
      }
      // local DOFs on this edge: the shape functions that do not vanish on it
      auto pts = P.edge_points(e, k, qe);
      const int nd = static_cast<int>(sp.edofs(e).size());
      for (int i = 0; i < nd; ++i) {
        double mx = 0;
        for (auto& p : pts) mx = std::max(mx, std::fabs(p.pe.N[i]));
        if (mx > 1e-10) w.loc.push_back(i);
      }
      P.body_mat[w.body] = w.m;
      P.walls.push_back(std::move(w));
    }
  }
  AXF_REQUIRE(!P.walls.empty(), "internal: the cavity has no wall");
  // DOF -> cavity elements
  P.dof_elems.assign(sp.ndof(), {});
  for (int e = 0; e < ne; ++e)
    if (P.kind[e] == 0)
      for (int dof : sp.edofs(e)) P.dof_elems[dof].push_back(e);

  // areas: cavity A, and the area enclosed by the outer wall (signed int x n_x dl per connected set of wall edges)
  {
    const axi::RefTable& tab = sp.table(sp.quad_rule());
    PointEval pe;
    for (int e = 0; e < ne; ++e) {
      if (P.kind[e] != 0) continue;
      for (int q = 0; q < tab.nq; ++q) {
        sp.eval(e, tab, q, pe);
        P.area += pe.w;
      }
    }
    std::vector<int> parent(P.mesh->GetNV());
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int a) { return parent[a] == a ? a : parent[a] = find(parent[a]); };
    for (auto& w : P.walls) {
      mfem::Array<int> v;
      P.mesh->GetEdgeVertices(w.face, v);
      parent[find(v[0])] = find(v[1]);
    }
    std::map<int, double> loop_area;
    for (auto& w : P.walls) {
      mfem::Array<int> v;
      P.mesh->GetEdgeVertices(w.face, v);
      double s = 0;
      for (auto& p : P.edge_points(w.elem, w.ledge, qe)) s += p.pe.r * p.nx * p.dl;
      loop_area[find(v[0])] += s;
    }
    P.area_full = 0;
    int outer_root = -1;
    for (auto& [k, a] : loop_area)
      if (a > P.area_full) {
        P.area_full = a;
        outer_root = k;
      }
    // bodies (not sheets/curves) that form the outer boundary: a post touching or overlapping the wall is part of it
    // and is therefore excluded from area_full (C_full); several bodies there are reported
    for (auto& w : P.walls) {
      if (w.sheet || P.curve_of_attr.count(P.sp->face_bdr_attr[w.face])) continue;
      mfem::Array<int> v;
      P.mesh->GetEdgeVertices(w.face, v);
      if (find(v[0]) == outer_root) P.outer_bodies.insert(w.body);
    }
  }
  // free edges of sheets (a sheet vertex with one sheet face and no other wall face): field singular there
  std::set<std::string> free_edge_sheets;
  {
    std::map<int, int> nsheet, nother;
    std::map<int, std::string> vbody;
    for (auto& w : P.walls) {
      mfem::Array<int> v;
      P.mesh->GetEdgeVertices(w.face, v);
      for (int a : v) {
        if (w.sheet) { nsheet[a] += 1; vbody[a] = w.body; }  // counted twice per face (both sides)
        else nother[a] += 1;
      }
    }
    for (auto& [vtx, n] : nsheet)
      if (n == 2 && !nother.count(vtx)) free_edge_sheets.insert(vbody[vtx]);
  }

  // ------------------------------------------------------------ eigenproblems
  const double Lc = std::sqrt(P.area);
  std::vector<std::string> families = fw.families.empty() ? std::vector<std::string>{"TM"} : fw.families;
  json eig = json::object();
  double Bz = 0;
  if (d_.background.type == "uniform") Bz = std::fabs(d_.background.Bz * d_.background.scale);
  std::map<std::string, std::string> rs_warn;
  for (const std::string& fname : families) {
    Timer ft;
    Impl::Fam& F = P.fam[fname];
    F.tm = fname == "TM";
    // DOFs: Space DOFs of cavity elements; TE: one extra copy per extra fan of a DOF on a two-sided sheet
    F.edofs.assign(ne, {});
    F.ndof = sp.ndof();
    int nsplit = 0;
    for (int e = 0; e < ne; ++e)
      if (P.kind[e] == 0) F.edofs[e] = sp.edofs(e);
    if (!F.tm) {
      std::set<int> sdofs;
      for (auto& w : P.walls)
        if (w.sheet)
          for (int i : w.loc) sdofs.insert(sp.edofs(w.elem)[i]);
      for (int dof : sdofs) {
        auto fs = P.fans(dof);
        for (size_t k = 1; k < fs.size(); ++k) {
          const int nw = F.ndof++;
          ++nsplit;
          for (int e : fs[k])
            for (auto& x : F.edofs[e])
              if (x == dof) x = nw;
        }
      }
    }
    F.dirichlet.assign(F.ndof, 0);
    if (F.tm)
      for (auto& w : P.walls)
        for (int i : w.loc) F.dirichlet[F.edofs[w.elem][i]] = 1;
    // element matrices and assembly (K scaled by c^2: eigenvalue omega^2)
    std::vector<char> used(F.ndof, 0);
    for (int e = 0; e < ne; ++e)
      for (int x : F.edofs[e]) used[x] = 1;
    std::vector<int> red(F.ndof, -1);
    int nred = 0;
    for (int i = 0; i < F.ndof; ++i)
      if (used[i] && !F.dirichlet[i]) red[i] = nred++;
    AXF_REQUIRE(nred > 0, "the cavity has no free degrees of freedom (mesh too coarse?)", "/mesh/size");
    F.Ke.assign(ne, Eigen::MatrixXd());
    F.Me.assign(ne, Eigen::MatrixXd());
    std::vector<Eigen::Triplet<double>> tk, tm;
    const axi::RefTable& tab = sp.table(sp.quad_rule());
    PointEval pe;
    const double c2 = units::c * units::c;
    for (int e = 0; e < ne; ++e) {
      if (P.kind[e] != 0) continue;
      const double al = F.tm ? 1.0 / P.mu_r(e) : 1.0 / P.eps_r(e);
      const double be = F.tm ? P.eps_r(e) : P.mu_r(e);
      const int nd = static_cast<int>(F.edofs[e].size());
      Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero(nd, nd), Me = Eigen::MatrixXd::Zero(nd, nd);
      for (int q = 0; q < tab.nq; ++q) {
        sp.eval(e, tab, q, pe);
        for (int i = 0; i < nd; ++i)
          for (int j = 0; j < nd; ++j) {
            Ke(i, j) += pe.w * al * (pe.dN[2 * i] * pe.dN[2 * j] + pe.dN[2 * i + 1] * pe.dN[2 * j + 1]);
            Me(i, j) += pe.w * be * pe.N[i] * pe.N[j];
          }
      }
      for (int i = 0; i < nd; ++i) {
        const int ri = red[F.edofs[e][i]];
        if (ri < 0) continue;
        for (int j = 0; j < nd; ++j) {
          const int rj = red[F.edofs[e][j]];
          if (rj < 0) continue;
          tk.emplace_back(ri, rj, c2 * Ke(i, j));
          tm.emplace_back(ri, rj, Me(i, j));
        }
      }
      F.Ke[e] = Ke;
      F.Me[e] = Me;
    }
    SpMat K(nred, nred), M(nred, nred);
    K.setFromTriplets(tk.begin(), tk.end());
    M.setFromTriplets(tm.begin(), tm.end());
    // shift-invert (as AxiWave::eigenmodes): below the spectrum by default; about target_f_Hz on request
    const bool targeted = fw.target_f_Hz > 0;
    const double wt = 2 * pi * fw.target_f_Hz;
    const double sigma = targeted ? wt * wt : -std::pow(0.1 * units::c / Lc, 2);
    SpMat As = K - sigma * M;
    As.makeCompressed();
    Eigen::SimplicialLDLT<SpMat> ch;
    Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>> lu;
    bool use_lu = false;
    ch.compute(As);
    if (ch.info() != Eigen::Success) {
      AXF_REQUIRE(targeted, "the cross-section eigenproblem could not be factorized");
      use_lu = true;
    } else if (targeted) {
      std::mt19937 rng(20261022u);
      std::normal_distribution<double> nd;
      Vec b(nred);
      for (int i = 0; i < nred; ++i) b(i) = nd(rng);
      const Vec y = ch.solve(b);
      if (!((As * y - b).norm() / b.norm() < 1e-10)) use_lu = true;
    }
    if (use_lu) {
      lu.analyzePattern(As);
      lu.factorize(As);
      AXF_REQUIRE(lu.info() == Eigen::Success,
                  strcat("K - sigma M is singular at the target ", fw.target_f_Hz,
                         " Hz (the target coincides with an eigenfrequency to round-off)"),
                  "/fullwave/target_f_Hz", "Move target_f_Hz slightly.");
    }
    auto inv = [&](const Vec& b) -> Vec { return use_lu ? Vec(lu.solve(b)) : Vec(ch.solve(b)); };
    // TE: the constant H_z (omega = 0) is an eigenvector of the natural problem; ask for one more and drop it
    const int extra = F.tm ? 0 : 1;
    json ld;
    std::vector<axi::Eigenmode> modes = axi::lanczos(M, inv, sigma, fw.modes + extra, ld, !targeted);
    const double lam_floor = 1e-8 * std::pow(units::c / Lc, 2);
    int dropped = 0;
    std::vector<axi::Eigenmode> kept;
    for (auto& m : modes) {
      if (!F.tm && m.omega * m.omega < lam_floor) { ++dropped; continue; }
      kept.push_back(std::move(m));
    }
    if (static_cast<int>(kept.size()) > fw.modes) {
      if (targeted)  // nearest to the target
        std::stable_sort(kept.begin(), kept.end(), [&](const auto& a, const auto& b) {
          return std::fabs(a.omega * a.omega - sigma) < std::fabs(b.omega * b.omega - sigma);
        });
      kept.resize(fw.modes);
      std::sort(kept.begin(), kept.end(), [](const auto& a, const auto& b) { return a.omega < b.omega; });
    }
    for (auto& m : kept) {
      Eigen::Map<const Vec> x(m.x.data(), static_cast<long>(m.x.size()));
      const Vec Kx = K * x;
      m.residual_true = (Kx - m.omega * m.omega * (M * x)).norm() / std::max(Kx.norm(), 1e-300);
      std::vector<double> full(F.ndof, 0.0);
      for (int i = 0; i < F.ndof; ++i)
        if (red[i] >= 0) full[i] = m.x[red[i]];
      m.x = std::move(full);
    }
    F.modes = std::move(kept);
    json solver = ld;
    solver["shift_rad2_s2"] = sigma;
    solver["factorization"] = use_lu ? "LU" : "LDLT";
    solver["dofs"] = nred;
    if (targeted) solver["target_f_Hz"] = fw.target_f_Hz;
    if (!F.tm) {
      solver["split_dofs"] = nsplit;
      solver["dropped_static_modes"] = dropped;
    }

    // ---------------------------------------------------------- wall unknowns for the consistent flux (TM)
    std::map<std::pair<int, int>, int> wkey;  // (Space DOF, element) -> wall unknown
    int nw = 0;
    SpMat Mw;
    if (F.tm) {
      std::set<int> wdofs;
      for (auto& w : P.walls)
        for (int i : w.loc) wdofs.insert(sp.edofs(w.elem)[i]);
      for (int dof : wdofs)
        for (auto& fan : P.fans(dof)) {
          for (int e : fan) wkey[{dof, e}] = nw;
          ++nw;
        }
      std::vector<Eigen::Triplet<double>> tw;
      for (auto& w : P.walls) {
        const auto& dd = sp.edofs(w.elem);
        for (auto& p : P.edge_points(w.elem, w.ledge, qe))
          for (int i : w.loc)
            for (int j : w.loc)
              tw.emplace_back(wkey.at({dd[i], w.elem}), wkey.at({dd[j], w.elem}), p.dl * p.pe.N[i] * p.pe.N[j]);
      }
      Mw.resize(nw, nw);
      Mw.setFromTriplets(tw.begin(), tw.end());
    }
    Eigen::SimplicialLDLT<SpMat> wch;
    if (F.tm) {
      wch.compute(Mw);
      AXF_REQUIRE(wch.info() == Eigen::Success, "internal: the wall mass matrix could not be factorized");
    }

    // ---------------------------------------------------------- per-mode post-processing
    json list = json::array();
    F.norm.assign(F.modes.size(), 1.0);
    std::set<std::string> lossless_fam;  // walls without rf data (not cavity.allow_lossless_walls)
    bool q0_omitted = false;             // no wall rf data: Q0 omitted, Q_dielectric kept
    for (size_t mi = 0; mi < F.modes.size(); ++mi) {
      const axi::Eigenmode& m = F.modes[mi];
      json row = {{"f_Hz", m.f_Hz}, {"omega_rad_s", m.omega}, {"residual", m.residual},
                  {"residual_true", m.residual_true}};
      double Ie = 0, IH = 0, IH0 = 0, Iz = 0, best = -1, sgn_max = 1;
      std::map<std::string, double> Ireg;
      double E[3], H[3];
      for (int e = 0; e < ne; ++e) {
        if (P.kind[e] != 0) continue;
        for (int q = 0; q < tab.nq; ++q) {
          sp.eval(e, tab, q, pe);
          P.eval(F, m, e, pe, E, H);
          const double e2 = E[0] * E[0] + E[1] * E[1] + E[2] * E[2];
          const double h2 = H[0] * H[0] + H[1] * H[1] + H[2] * H[2];
          Ie += pe.w * P.eps_r(e) * e2;
          Ireg[P.reg[e]] += pe.w * P.eps_r(e) * e2;
          IH += pe.w * P.mu_r(e) * h2;
          IH0 += pe.w * h2;
          Iz += pe.w * E[2];
          if (e2 > best * (1 + 1e-9)) {
            best = e2;
            int k = 0;
            for (int j = 1; j < 3; ++j)
              if (std::fabs(E[j]) > std::fabs(E[k])) k = j;
            sgn_max = E[k] >= 0 ? 1.0 : -1.0;
          }
        }
      }
      AXF_REQUIRE(Ie > 0, strcat("internal: ", fname, " mode ", mi, " has zero electric energy"));
      // sign: int E_z dA > 0 when it is not round-off (TM), else the largest E component at max |E| positive
      const bool zsign = F.tm && std::fabs(Iz) > 1e-8 * std::sqrt(P.area * Ie);
      const double sgn = zsign ? (Iz > 0 ? 1.0 : -1.0) : sgn_max;
      F.norm[mi] = sgn / std::sqrt(Ie);
      // energies: eps0 Ie = mu0 IH at resonance (a check)
      row["energy_balance"] = units::mu0 * IH / (kEps0 * Ie) - 1.0;
      json ef = json::object();
      for (auto& [r, v] : Ireg) ef[r] = v / Ie;
      row["energy_fraction"] = ef;
      // form factor (uniform axial B0)
      json ff = {{"A_m2", P.area}, {"A_full_m2", P.area_full}};
      if (F.tm) {
        ff["C"] = Iz * Iz / (P.area * Ie);
        ff["C_full"] = Iz * Iz / (P.area_full * Ie);
        ff["overlap_Ez_m"] = std::fabs(Iz) / std::sqrt(Ie);  // |int E_z dA| / (int eps_r E_z^2 dA)^(1/2)
      } else {
        ff["C"] = 0.0;
        ff["C_full"] = 0.0;
        ff["note"] = "TE_z: E is transverse, no overlap with an axial B0";
      }
      if (cav.length > 0) ff["V_m3"] = P.area * cav.length;
      row["form_factor"] = ff;
      // ---- losses
      if (cav.losses) {
        const double w = m.omega;
        const double WH = units::mu0 * IH;  // int mu |H|^2 dA (unscaled)
        std::map<std::string, double> gam, gam_pt;
        if (F.tm) {
          // consistent flux: reaction (K_e - k^2 M_e) u_e = oint q N_i dl, q = (1/mu_r) du/dn; H_t = q / (omega mu0)
          Vec r = Vec::Zero(nw);
          const double k2 = w * w / (units::c * units::c);
          for (int e = 0; e < ne; ++e) {
            if (P.kind[e] != 0) continue;
            const auto& dd = F.edofs[e];
            Vec ue(dd.size());
            for (size_t i = 0; i < dd.size(); ++i) ue(i) = m.x[dd[i]];
            const Vec re = F.Ke[e] * ue - k2 * (F.Me[e] * ue);
            for (size_t i = 0; i < dd.size(); ++i) {
              auto it = wkey.find({dd[i], e});
              if (it != wkey.end()) r(it->second) += re(i);
            }
          }
          const Vec qv = wch.solve(r);
          for (auto& wl : P.walls) {
            const auto& dd = sp.edofs(wl.elem);
            const auto& fd = F.edofs[wl.elem];
            for (auto& p : P.edge_points(wl.elem, wl.ledge, qe)) {
              double q = 0, ux = 0, uy = 0;
              for (int i : wl.loc) q += qv(wkey.at({dd[i], wl.elem})) * p.pe.N[i];
              for (size_t i = 0; i < fd.size(); ++i) {
                ux += m.x[fd[i]] * p.pe.dN[2 * i];
                uy += m.x[fd[i]] * p.pe.dN[2 * i + 1];
              }
              const double qpt = (ux * p.nx + uy * p.ny) / P.mu_r(wl.elem);
              const double s = 1.0 / (w * units::mu0);
              gam[wl.body] += p.dl * q * q * s * s;
              gam_pt[wl.body] += p.dl * qpt * qpt * s * s;
            }
          }
        } else {  // TE: H_t = H_z = u on the wall (each side of a sheet from its own DOFs)
          for (auto& wl : P.walls) {
            const auto& fd = F.edofs[wl.elem];
            for (auto& p : P.edge_points(wl.elem, wl.ledge, qe)) {
              double u = 0;
              for (size_t i = 0; i < fd.size(); ++i) u += m.x[fd[i]] * p.pe.N[i];
              gam[wl.body] += p.dl * u * u;
            }
          }
        }
        json lj = json::object(), bj = json::object();
        double inv_Q2 = 0, gsum = 0, TQ = 0;
        bool any_rf = false, rs_missing = false;
        std::vector<std::string> lossless;
        // cap: an end cap (z = 0, L), whose RF current is in the cap plane, i.e. perpendicular to the axial vortices
        auto rs_of = [&](const spec::Material* mt, json& e, double& Rs, bool cap = false) {
          Rs = 0;
          if (!mt || mt->rf.model.empty()) return false;
          double Rs0 = 0;
          try {  // e.g. outside an rf table: no Q for this mode
            Rs0 = mt->rf.Rs_base(m.f_Hz);
            for (auto& [k, t] : mt->rf.warnings(m.f_Hz)) rs_warn[mt->rf.owner + "/" + k] = t;
          } catch (const std::exception& ex) {
            e["note"] = std::string("R_s not available: ") + ex.what();
            rs_missing = true;
            return false;
          }
          Rs = Rs0 * (mt->rf.mr_model.empty() ? 1.0 : mt->rf.Rs_factor(Bz));
          if (mt->rf.flux_flow) {
            // vortices along the axial B0; RF wall current on the barrel along z (TM_z: parallel, s = 0) or in-plane
            // (TE_z: s = 1); on the end caps (TM_z) radial/in-plane, perpendicular to the vortices (s = 1)
            const double Bv = mt->rf.ff_B_local() ? Bz : std::fabs(mt->rf.ff_B);
            AXF_REQUIRE(Bv < mt->rf.ff_Bc2, strcat("flux_flow wall material '", mt->name, "': static field ", Bv,
                                                   " T is above Bc2_T = ", mt->rf.ff_Bc2, " T"),
                        "/materials/" + mt->name + "/rf/flux_flow", "Use a normal-metal rf model for this wall.");
            const double sf = mt->rf.ff_orientation >= 0 ? mt->rf.ff_orientation
                                                         : (Bz > 0 ? ((F.tm && !cap) ? 0.0 : 1.0) : 1.0);
            const double Rff = mt->rf.Rs_flux_flow(m.f_Hz, Bv, sf);
            e["Rs_flux_flow_ohm"] = Rff;
            e["flux_flow_orientation_factor"] = sf;
            Rs += Rff;
          }
          e["Rs_ohm"] = Rs;
          e["Rs_model"] = mt->rf.model;
          e["T_K"] = mt->rf.T_K;
          return Rs > 0;
        };
        for (auto& [b, G] : gam) {
          gsum += G;
          json e = {{"G_ohm", w * WH / std::max(G, 1e-300)}};
          if (gam_pt.count(b)) e["G_ohm_pointwise"] = w * WH / std::max(gam_pt[b], 1e-300);
          double Rs = 0;
          const spec::Material* mt = P.body_mat.count(b) ? P.body_mat.at(b) : nullptr;
          if (rs_of(mt, e, Rs)) {
            const double Qb = w * WH / (Rs * G);
            e["Q"] = Qb;
            inv_Q2 += 1.0 / Qb;
            TQ += mt->rf.T_K / Qb;
            any_rf = true;
          } else if (!e.contains("note")) {
            e["note"] = cav.allow_lossless_walls ? "no rf data: lossless (cavity.allow_lossless_walls)" : "no rf data: lossless";
            lossless.push_back(b);
          }
          bj[b] = e;
        }
        // end caps at z = 0 and z = L (TM_z: H transverse, z-independent): oint over both caps 2 int |H|^2 dA
        double inv_Qc = 0;
        json cj;
        if (F.tm && cav.length > 0) {
          const double Gc = w * cav.length * WH / (2.0 * IH0);
          cj = {{"G_ohm", Gc}, {"length_m", cav.length}, {"material", cav.end_cap_material}};
          double Rs = 0;
          const spec::Material* mt = cav.end_cap_material.empty() ? nullptr : &d_.materials.at(cav.end_cap_material);
          if (rs_of(mt, cj, Rs, true)) {
            cj["Q"] = Gc / Rs;
            inv_Qc = Rs / Gc;
            TQ += mt->rf.T_K * inv_Qc;
            any_rf = true;
          } else if (!cj.contains("note")) {
            cj["note"] = cav.allow_lossless_walls ? "no rf data: lossless (cavity.allow_lossless_walls)" : "no rf data: lossless";
            lossless.push_back("end_caps");
          }
          lj["end_caps"] = cj;
        }
        // dielectric losses: 1/Q_d = sum tan_eff W_e,reg / W_e
        double inv_Qd = 0;
        json dl = json::object();
        for (auto& [rname, v] : Ireg) {
          const spec::Material& mt = d_.materials.at(d_.find_region(rname)->material);
          const double tan_eff = mt.tan_delta + mt.sigma_S_m / (w * kEps0 * mt.eps_r);
          if (!(tan_eff > 0)) continue;
          const double fr = v / Ie;
          dl[rname] = {{"tan_delta_eff", tan_eff}, {"electric_energy_fraction", fr}, {"Q", 1.0 / (tan_eff * fr)}};
          inv_Qd += tan_eff * fr;
        }
        const double inv_Qw = inv_Q2 + inv_Qc;
        if (any_rf && !rs_missing)
          for (auto it = bj.begin(); it != bj.end(); ++it)
            if (it->contains("Q")) (*it)["P_frac"] = (1.0 / (*it)["Q"].get<double>()) / inv_Qw;
        if (any_rf && !rs_missing && cj.contains("Q")) lj["end_caps"]["P_frac"] = inv_Qc / inv_Qw;
        lj["walls"] = bj;
        if (gsum > 0) {
          lj["G_ohm_2d"] = w * WH / gsum;
          lj["G_ohm"] = (F.tm && cav.length > 0) ? 1.0 / (gsum / (w * WH) + 2.0 * IH0 / (w * cav.length * WH))
                                                   : w * WH / gsum;
        }
        if (any_rf && !rs_missing) {
          if (inv_Q2 > 0) lj["Q_wall_2d"] = 1.0 / inv_Q2;
          lj["Q_wall"] = 1.0 / inv_Qw;
          lj["T_over_Q_K"] = TQ;
          lj["Q0"] = 1.0 / (inv_Qw + inv_Qd);
          if (inv_Q2 + inv_Qd > 0) lj["Q0_2d"] = 1.0 / (inv_Q2 + inv_Qd);
        } else if (!rs_missing && inv_Qd > 0) {
          // no wall rf data: Q0 would be Q_dielectric alone, reported only when lossless walls are allowed (as in 3D)
          if (cav.allow_lossless_walls) lj["Q0"] = lj["Q0_2d"] = 1.0 / inv_Qd;
          else q0_omitted = true;
        }
        if (rs_missing)
          warnings.push_back(strcat("cavity: ", fname, " mode at ", m.f_Hz, " Hz has walls without R_s at that "
                                    "frequency (see losses.walls); Q_wall and Q0 are not reported for it"));
        if (!dl.empty()) {
          lj["dielectric"] = dl;
          lj["Q_dielectric"] = 1.0 / inv_Qd;
        }
        if (!lossless.empty()) {
          lj["lossless_walls"] = lossless;
          if (!cav.allow_lossless_walls) for (auto& b : lossless) lossless_fam.insert(b);
        }
        if (cav.allow_lossless_walls) lj["walls_lossless_by_request"] = !lossless.empty();
        lj["note"] = F.tm ? (cav.length > 0 ? "Q0 = omega U / P with the barrel walls per unit length (Q_wall_2d) and "
                                              "the two end caps at distance length (end_caps); G_ohm = Q R_s for one "
                                              "R_s on all surfaces, G_ohm_2d without the caps"
                                            : "Q0 = omega U / P per unit length (infinitely long cavity, no end caps; "
                                              "set cavity.length for the cap losses)")
                          : "TE_z: wall losses per unit length only (z-independent TE modes do not satisfy end caps)";
        row["losses"] = lj;
      }
      list.push_back(row);
    }
    if (!lossless_fam.empty()) {
      std::string names;
      for (auto& n : lossless_fam) names += (names.empty() ? "" : ", ") + n;
      warnings.push_back("cavity: " + fname + " modes: the walls " + names + " have no rf data and are lossless: "
                         "Q_wall, Q0 and P_frac leave out their loss (see losses.lossless_walls). Give them an rf "
                         "model, or set cavity.allow_lossless_walls if this is intended");
    }
    if (q0_omitted)
      warnings.push_back("cavity: " + fname + " modes: no wall has rf data, so Q0 is not reported (it would be "
                         "Q_dielectric alone); Q_dielectric is. Set cavity.wall_material (with rf data) or "
                         "cavity.allow_lossless_walls");
    solver["seconds"] = ft.seconds();
    json fj = {{"modes", list}, {"solver", solver},
               {"note", F.tm ? "TM_z: E = E_z(x, y) z^, z-independent (TM_mn0; exact 3D modes with flat end caps)"
                             : "TE_z: H = H_z(x, y) z^, z-independent cutoff modes (C = 0; not modes of a capped "
                               "cavity, which need z dependence)"}};
    if (targeted) fj["target_f_Hz"] = fw.target_f_Hz;
    eig[fname] = fj;
  }
  for (auto& [k, t] : rs_warn) warnings.push_back("cavity: " + t);
  if (!free_edge_sheets.empty()) {
    std::string names;
    for (auto& n : free_edge_sheets) names += (names.empty() ? "" : ", ") + n;
    warnings.push_back("cavity: sheets " + names + " end inside the cavity; the field is singular at a free sheet edge "
                       "(|H_t|^2 ~ 1/r), so its wall loss diverges logarithmically under refinement and C, f converge "
                       "slowly; use a region of finite thickness for converged Q");
  }

  // ------------------------------------------------------------ results
  json cj = {{"status", "beta (docs/validation/cavity_xsec.md)"},
             {"model", "z-uniform cross-section: TM_z / TE_z modes without z dependence"},
             {"regions", std::vector<std::string>(cav_regions.begin(), cav_regions.end())},
             {"excluded_regions", std::vector<std::string>(excluded_regions.begin(), excluded_regions.end())},
             {"components", ncomp},
             {"area_m2", P.area},
             {"area_full_m2", P.area_full},
             {"wall_material", cav.wall_material},
             {"B0", "uniform, axial (C does not depend on its magnitude)"}};
  if (cav.length > 0) {
    cj["length_m"] = cav.length;
    cj["volume_m3"] = P.area * cav.length;
    cj["end_cap_material"] = cav.end_cap_material;
  }
  if (Bz > 0) cj["B0_T"] = Bz;
  json wl = json::object();
  for (auto& w : P.walls) {
    double len = 0;
    for (auto& p : P.edge_points(w.elem, w.ledge, qe)) len += p.dl;
    if (!wl.contains(w.body)) wl[w.body] = {{"length_m", 0.0}, {"rf", w.m && !w.m->rf.model.empty()}};
    wl[w.body]["length_m"] = wl[w.body]["length_m"].get<double>() + len;
  }
  cj["walls"] = wl;
  if (P.outer_bodies.size() > 1) {
    std::string names;
    for (auto& n : P.outer_bodies) names += (names.empty() ? "" : ", ") + n;
    cj["outer_boundary_bodies"] = std::vector<std::string>(P.outer_bodies.begin(), P.outer_bodies.end());
    warnings.push_back("cavity: the outer boundary of the cavity is formed by several bodies (" + names + "): a post "
                       "touching or overlapping the wall is part of it, so area_full (C_full) excludes that post; a "
                       "tangent contact is a singular geometry (keep a finite gap for converged results)");
  }
  if (!two_sided.empty()) cj["two_sided_sheets"] = std::vector<std::string>(two_sided.begin(), two_sided.end());
  if (!free_edge_sheets.empty())
    cj["sheets_with_free_edge"] = std::vector<std::string>(free_edge_sheets.begin(), free_edge_sheets.end());
  int ndofs = sp.ndof();
  R_ = json::object();
  R_["name"] = d_.name;
  R_["symmetry"] = "xsec";
  R_["parameters"] = d_.parameters;
  R_["mesh"] = {{"mesher", minfo}, {"elements", ne}, {"order", order}, {"dofs", ndofs}, {"size", h},
                {"surface_size", hs}, {"refinements", d_.mesh.refinements}};
  R_["fullwave"] = {{"eigenmodes", eig}, {"cavity", cj}};
  R_["warnings"] = warnings;
  R_["seconds"] = timer.seconds();
  solved_ = true;
  std::ofstream(o.workdir + "/results.json") << R_.dump(2);
}

// ---------------------------------------------------------------- fields
namespace {

void lattice(int k, std::vector<mfem::IntegrationPoint>& pts, std::vector<std::array<int, 3>>& tris) {
  std::map<std::pair<int, int>, int> id;
  for (int j = 0; j <= k; ++j)
    for (int i = 0; i + j <= k; ++i) {
      mfem::IntegrationPoint ip;
      ip.Set2(double(i) / k, double(j) / k);
      ip.weight = 0;
      id[{i, j}] = static_cast<int>(pts.size());
      pts.push_back(ip);
    }
  for (int j = 0; j < k; ++j)
    for (int i = 0; i + j < k; ++i) {
      tris.push_back({id[{i, j}], id[{i + 1, j}], id[{i, j + 1}]});
      if (i + j + 1 < k) tris.push_back({id[{i + 1, j}], id[{i + 1, j + 1}], id[{i, j + 1}]});
    }
}

}  // namespace

static const XsecModel::Impl::Fam& family_of(const XsecModel::Impl& P, bool solved, const std::string& family,
                                              int index) {
  AXF_REQUIRE(solved, "the cross-section model has not been solved");
  auto it = P.fam.find(family);
  AXF_REQUIRE(it != P.fam.end(), strcat("no cavity eigenmodes of family '", family, "'"), "/fullwave/families",
              "Use \"TM\" or \"TE\" (one of the families that were solved).");
  AXF_REQUIRE(index >= 0 && index < static_cast<int>(it->second.modes.size()),
              strcat("cavity mode index ", index, " out of range for family ", family, " (", it->second.modes.size(),
                     " modes)"), "/fullwave/modes", "Use an index from Results.cavity_modes().");
  return it->second;
}

static void push_point(const XsecModel::Impl& P, const XsecModel::Impl::Fam& F, const axi::Eigenmode& m, double s,
                       int e, const PointEval& pe, XsecModel::ModeField& out) {
  double E[3], H[3];
  P.eval(F, m, e, pe, E, H);
  out.x.push_back(pe.r);
  out.y.push_back(pe.z);
  for (int k = 0; k < 3; ++k) {
    out.E[k].push_back(s * E[k]);
    out.H[k].push_back(s * H[k]);
  }
  out.eps_r.push_back(P.eps_r(e));
  out.mu_r.push_back(P.mu_r(e));
  out.elem.push_back(e);
}

XsecModel::ModeField XsecModel::mode_mesh(const std::string& family, int index, int subdiv) const {
  const Impl& P = *p_;
  const auto& F = family_of(P, solved_, family, index);
  AXF_REQUIRE(subdiv >= 1 && subdiv <= 16, "subdiv must be between 1 and 16");
  const auto& m = F.modes[index];
  ModeField out;
  out.family = family;
  out.index = index;
  out.f_Hz = m.f_Hz;
  out.norm = F.norm[index];
  std::vector<mfem::IntegrationPoint> pts;
  std::vector<std::array<int, 3>> tris;
  lattice(subdiv, pts, tris);
  PointEval pe;
  for (int e = 0; e < P.mesh->GetNE(); ++e) {
    if (P.kind[e] != 0) continue;
    const int base = static_cast<int>(out.x.size());
    for (auto& ip : pts) {
      P.sp->eval_at(e, ip, pe);
      push_point(P, F, m, out.norm, e, pe, out);
    }
    for (auto& t : tris)
      for (int k = 0; k < 3; ++k) out.tris.push_back(base + t[k]);
  }
  return out;
}

XsecModel::ModeField XsecModel::mode_sample(const std::string& family, int index, const std::vector<double>& x,
                                            const std::vector<double>& y) const {
  const Impl& P = *p_;
  const auto& F = family_of(P, solved_, family, index);
  AXF_REQUIRE(x.size() == y.size(), "x and y must have the same length");
  const auto& m = F.modes[index];
  ModeField out;
  out.family = family;
  out.index = index;
  out.f_Hz = m.f_Hz;
  out.norm = F.norm[index];
  const int n = static_cast<int>(x.size());
  mfem::DenseMatrix pts(2, std::max(n, 1));
  for (int i = 0; i < n; ++i) {
    pts(0, i) = x[i];
    pts(1, i) = y[i];
  }
  mfem::Array<int> eid;
  mfem::Array<mfem::IntegrationPoint> ips;
  if (n > 0) P.mesh->FindPoints(pts, eid, ips, false);
  PointEval pe;
  for (int i = 0; i < n; ++i) {
    const int e = eid[i];
    if (e < 0 || P.kind[e] != 0) {
      out.x.push_back(x[i]);
      out.y.push_back(y[i]);
      for (int k = 0; k < 3; ++k) {
        out.E[k].push_back(kNaN);
        out.H[k].push_back(kNaN);
      }
      out.eps_r.push_back(kNaN);
      out.mu_r.push_back(kNaN);
      out.elem.push_back(e < 0 ? -1 : e);
      continue;
    }
    P.sp->eval_at(e, ips[i], pe);
    push_point(P, F, m, out.norm, e, pe, out);
    out.x.back() = x[i];
    out.y.back() = y[i];
  }
  return out;
}

json XsecModel::geometry(int subdiv) const {
  AXF_REQUIRE(solved_, "the cross-section model has not been solved");
  const Impl& P = *p_;
  std::vector<mfem::IntegrationPoint> pts;
  std::vector<std::array<int, 3>> tris;
  lattice(std::max(1, subdiv), pts, tris);
  std::vector<std::string> names;
  std::map<std::string, int> idx;
  for (auto& [a, n] : P.region_of_attr) {
    idx[n] = static_cast<int>(names.size());
    names.push_back(n);
  }
  std::vector<double> X, Y;
  std::vector<int> T, reg, cavity;
  PointEval pe;
  for (int e = 0; e < P.mesh->GetNE(); ++e) {
    const int base = static_cast<int>(X.size());
    for (auto& ip : pts) {
      P.sp->eval_at(e, ip, pe);
      X.push_back(pe.r);
      Y.push_back(pe.z);
    }
    for (auto& t : tris)
      for (int k = 0; k < 3; ++k) T.push_back(base + t[k]);
    reg.push_back(idx[P.reg[e]]);
    cavity.push_back(P.kind[e] == 0 ? 1 : 0);
  }
  json kinds = json::object();
  for (auto& g : d_.regions) {
    const auto k = d_.materials.at(g.material).kind;
    kinds[g.name] = (k == spec::MaterialKind::Conductor || k == spec::MaterialKind::Superconductor) ? "conductor"
                    : (d_.materials.at(g.material).eps_r != 1.0 ? "dielectric" : "vacuum");
  }
  // wall polylines: per wall side, the edge sampled at subdiv + 1 points (curved edges)
  json walls = json::array();
  std::set<std::pair<int, int>> seen;
  for (auto& w : P.walls) {
    if (!seen.insert({w.face, w.sheet ? 0 : w.elem}).second) continue;
    json seg = json::array();
    mfem::ElementTransformation* Tr = P.mesh->GetElementTransformation(w.elem);
    const double* a = kRefV[w.ledge];
    const double* b = kRefV[(w.ledge + 1) % 3];
    const int ns = std::max(2, 2 * subdiv);
    for (int i = 0; i <= ns; ++i) {
      const double t = double(i) / ns;
      mfem::IntegrationPoint ip;
      ip.Set2(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
      mfem::Vector xp;
      Tr->Transform(ip, xp);
      seg.push_back({xp(0), xp(1)});
    }
    walls.push_back({{"body", w.body}, {"sheet", w.sheet}, {"points", seg}});
  }
  return {{"x", X}, {"y", Y}, {"tris", T}, {"element_region", reg}, {"element_cavity", cavity},
          {"points_per_element", static_cast<int>(pts.size())}, {"regions", names}, {"region_kind", kinds},
          {"walls", walls}};
}

}  // namespace axf
