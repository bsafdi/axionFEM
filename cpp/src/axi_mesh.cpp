#include <sys/stat.h>
#include <filesystem>
#include <set>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>

#include "axionfem/axi.hpp"
#include "axionfem/morph.hpp"
#include "axionfem/units.hpp"
#include "axionfem/core.hpp"

extern char** environ;

namespace axf::axi {

namespace {
bool is_exe(const std::string& p) { return !p.empty() && access(p.c_str(), X_OK) == 0; }

}  // namespace

int run_process(const std::vector<std::string>& args, const std::string& log) {
  // posix_spawn (not fork + freopen): safe in a multithreaded host process (Python, OpenMP, MPI).
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, STDOUT_FILENO, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  posix_spawn_file_actions_adddup2(&fa, STDOUT_FILENO, STDERR_FILENO);
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, argv[0], &fa, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  if (rc != 0) throw Error(strcat("could not start '", args[0], "': ", std::strerror(rc)));
  int status = 0;
  for (;;) {
    const pid_t w = waitpid(pid, &status, 0);
    if (w == pid) break;
    if (w < 0 && errno == EINTR) continue;  // interrupted by a signal: keep waiting
    throw Error(strcat("waiting for '", args[0], "' failed: ", std::strerror(errno)));
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

std::string find_mesher(const std::string& hint) {
  std::vector<std::string> cand;
  if (!hint.empty()) cand.push_back(hint);
  if (const char* e = std::getenv("AXF_MESHER")) cand.push_back(e);
  // next to the running executable / library build tree
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n > 0) {
    buf[n] = 0;
    std::string dir(buf);
    dir = dir.substr(0, dir.find_last_of('/'));
    cand.push_back(dir + "/axf-mesher");
    cand.push_back(dir + "/bin/axf-mesher");
  }
#ifdef AXF_SOURCE_DIR
  cand.push_back(std::string(AXF_SOURCE_DIR) + "/build/axf-mesher");
#endif
  for (auto& c : cand) if (is_exe(c)) return c;
  throw Error("cannot find the axf-mesher executable", "",
              "axf-mesher is the separate program that runs gmsh (GPL, kept out of libaxionfem; "
              "see docs/install.md, \"Licensing of each wheel\"). Installed from a wheel: `pip install gmsh`, which provides the "
              "library the shipped mesher links. Source checkout: build the cmake target "
              "axf-mesher. Either way, $AXF_MESHER overrides the search. See docs/install.md.");
}

MeshBundle build_mesh(const spec::Design& d, const MeshOptions& opt) {
  Timer timer;
  std::error_code ec;
  std::filesystem::create_directories(opt.workdir, ec);
  AXF_REQUIRE(!ec, "cannot create working directory '" + opt.workdir + "': " + ec.message());
  // Device extent (shared with the mesh morphing, which needs the same domain radius).  The outer
  // semicircle (and the DtN expansion) is centred on the axis at z = z0, so all radii are measured
  // from (r, z) = (0, z0), not from the origin.
  const double z0 = d.mesh.domain_center_z;
  double Rdev = 0, Rb = 0, L = 0;
  morph::mesh_extent(d, Rdev, Rb, L);
  AXF_REQUIRE(Rdev > 0 && L > 0, "the design has zero extent", "/regions");
  AXF_REQUIRE(Rb > 1.02 * Rdev,
              strcat("domain radius ", Rb, " must exceed the device radius ", Rdev, " about the domain centre z = ", z0),
              "/mesh/domain_radius",
              "Use at least 1.5x the distance of the farthest part from the domain centre (0, mesh.domain_center_z); "
              "for a device far from z = 0, set mesh.domain_center_z near its middle.");
  double h = d.mesh.size > 0 ? d.mesh.size : L / 40.0;
  double hs = d.mesh.surface_size > 0 ? d.mesh.surface_size : h / 4.0;
  double hfar = std::min(Rb / 6.0, M_PI * Rb / 96.0);  // >= 96 elements on the outer arc
  if (d.fullwave.enabled && !d.fullwave.frequencies.empty()) {
    // full-wave: about 12 nodes per wavelength (order 2) everywhere outside conductors
    const double fmax = *std::max_element(d.fullwave.frequencies.begin(), d.fullwave.frequencies.end());
    double nmax = 1;
    for (auto& g : d.regions) {
      const auto& m = d.materials.at(g.material);
      if (m.kind == spec::MaterialKind::Vacuum || m.kind == spec::MaterialKind::Linear)
        nmax = std::max(nmax, std::sqrt(m.eps_r * m.mu_r));
    }
    const double hcap = units::c / (fmax * nmax) * std::min(d.mesh.order, 2) / 12.0;
    h = std::min(h, hcap);
    hs = std::min(hs, hcap);
    hfar = std::min(hfar, hcap);
  }

  json in;
  in["domain_radius"] = Rb;
  in["domain_center_z"] = z0;
  in["device_radius"] = Rdev;
  in["size"] = h;
  in["surface_size"] = hs;
  in["far_size"] = hfar;
  in["order"] = std::min(d.mesh.order, 2);
  in["axis_attr"] = kAxisAttr;
  in["outer_attr"] = kOuterAttr;
  in["air_attr"] = kAirAttr;
  in["surface_layer"] = 4.0 * h;
  json regs = json::array(), curves = json::array();
  json refine = json::array();
  MeshBundle mb;
  int attr = 2;
  for (auto& g : d.regions) {
    json r = {{"name", g.name}, {"attr", attr}, {"shape", g.shape.resolved()}};
    if (g.mesh_size > 0) r["size"] = g.mesh_size;
    if (g.surface_size > 0) r["surface_size"] = g.surface_size;
    regs.push_back(r);
    mb.attr_of_region[g.name] = attr;
    mb.region_of_attr[attr] = g.name;
    auto k = d.materials.at(g.material).kind;
    bool is_winding = std::any_of(d.windings.begin(), d.windings.end(), [&](auto& w) { return w.region == g.name; });
    bool is_port = d.pickup && d.pickup->port == g.name;
    // Regions with their own surface_size get a dedicated refinement field in the mesher and are
    // left out of the group that follows mesh.surface_size.
    if (g.surface_size <= 0 && (k == spec::MaterialKind::Superconductor || k == spec::MaterialKind::Iron ||
                                k == spec::MaterialKind::Conductor || is_winding || is_port)) {
      refine.push_back(g.name);
    }
    ++attr;
  }
  mb.region_of_attr[kAirAttr] = "air";
  mb.attr_of_region["air"] = kAirAttr;
  int cattr = kCurveAttr0;
  for (auto& c : d.curves) {
    json cj = {{"name", c.name}, {"attr", cattr}, {"path", c.path.resolved()}};
    if (c.mesh_size > 0) cj["size"] = c.mesh_size;
    if (c.surface_size > 0) cj["surface_size"] = c.surface_size;
    curves.push_back(cj);
    mb.attr_of_curve[c.name] = cattr++;
  }
  in["regions"] = regs;
  in["curves"] = curves;
  in["refine_regions"] = refine;
  if (!opt.size_field_file.empty()) in["size_field_file"] = opt.size_field_file;
  const std::string base = opt.workdir + "/mesh";
  in["output"] = base + ".msh";
  in["info"] = base + "_info.json";
  in["verbose"] = opt.verbose > 1 ? 1 : 0;
  {
    std::ofstream f(base + "_in.json");
    f << in.dump(2);
  }
  std::string mesher = find_mesher(opt.mesher);
  int rc = run_process({mesher, base + "_in.json"}, base + "_mesher.log");
  json info;
  {
    std::ifstream f(base + "_info.json");
    if (f) info = json::parse(f, nullptr, false);
  }
  if (rc != 0 || !info.value("ok", false)) {
    std::string why = info.is_object() ? info.value("error", std::string("unknown")) : "no info file";
    throw Error("mesh generation failed: " + why, "/regions",
                "Check that shapes are valid and lie within r >= 0; see " + base + "_mesher.log.");
  }
  log(1, strcat("mesh: ", info.value("elements_2d", 0), " triangles in ", info.value("seconds", 0.0), " s"));

  mb.mesh = std::make_unique<mfem::Mesh>((base + ".msh").c_str(), 1, 1, true);
  for (int i = 0; i < d.mesh.refinements; ++i) mb.mesh->UniformRefinement();
  mb.domain_radius = Rb;
  mb.domain_center_z = z0;
  mb.device_radius = Rdev;
  in.erase("regions");
  in.erase("curves");
  mb.info = {{"mesher", info}, {"settings", in}, {"seconds", timer.seconds()}, {"elements", mb.mesh->GetNE()},
             {"refinements", d.mesh.refinements}};
  return mb;
}

// ---------------------------------------------------------------- Space
Space::Space(mfem::Mesh& m, int p) : mesh(m), order(p) {
  fec = std::make_unique<mfem::H1_FECollection>(p, 2);
  fes = std::make_unique<mfem::FiniteElementSpace>(&mesh, fec.get());
  const int ne = mesh.GetNE();
  for (int e = 0; e < ne; ++e) {
    AXF_REQUIRE(mesh.GetElementBaseGeometry(e) == mfem::Geometry::TRIANGLE,
                "axisymmetric meshes must consist of triangles");
  }
  field_fe_ = fes->GetFE(0);
  // geometry nodes
  static mfem::Linear2DFiniteElement p1;
  const mfem::GridFunction* nodes = mesh.GetNodes();
  if (nodes) {
    const mfem::FiniteElementSpace* nfes = nodes->FESpace();
    geom_fe_ = nfes->GetFE(0);
    ng_ = geom_fe_->GetDof();
    gx_.resize(static_cast<size_t>(ne) * ng_ * 2);
    mfem::Array<int> vd;
    for (int e = 0; e < ne; ++e) {
      nfes->GetElementVDofs(e, vd);
      for (int k = 0; k < ng_; ++k) {
        gx_[(static_cast<size_t>(e) * ng_ + k) * 2 + 0] = (*nodes)(vd[k]);
        gx_[(static_cast<size_t>(e) * ng_ + k) * 2 + 1] = (*nodes)(vd[ng_ + k]);
      }
    }
  } else {
    geom_fe_ = &p1;
    ng_ = 3;
    gx_.resize(static_cast<size_t>(ne) * 6);
    mfem::Array<int> v;
    for (int e = 0; e < ne; ++e) {
      mesh.GetElementVertices(e, v);
      for (int k = 0; k < 3; ++k) {
        const double* x = mesh.GetVertex(v[k]);
        gx_[(static_cast<size_t>(e) * 3 + k) * 2 + 0] = x[0];
        gx_[(static_cast<size_t>(e) * 3 + k) * 2 + 1] = x[1];
      }
    }
  }
  // axis tolerance from the geometry nodes, before any point evaluation uses it
  {
    double gmax = 0;
    for (size_t k = 0; k < gx_.size(); k += 2) gmax = std::max(gmax, std::fabs(gx_[k]));
    rtol = 1e-10 * std::max(gmax, 1e-30);
  }
  const int nd = fes->GetNDofs();
  r.assign(nd, 0.0);
  z.assign(nd, 0.0);
  edofs_.resize(ne);
  double rmax = 0;
  mfem::Array<int> dofs;
  const mfem::IntegrationRule& nodes_ir = field_fe_->GetNodes();
  PointEval pe;
  for (int e = 0; e < ne; ++e) {
    fes->GetElementDofs(e, dofs);
    edofs_[e].assign(dofs.begin(), dofs.end());
    for (int i = 0; i < nodes_ir.GetNPoints(); ++i) {
      eval_at(e, nodes_ir.IntPoint(i), pe);
      r[dofs[i]] = pe.r;
      z[dofs[i]] = pe.z;
      rmax = std::max(rmax, pe.r);
    }
  }
  rtol = 1e-10 * std::max(rmax, 1e-30);
  for (int i = 0; i < nd; ++i) {
    if (std::fabs(r[i]) < rtol) r[i] = 0.0;
  }
  // topology
  nbr.assign(ne, {});
  face_bdr_attr.assign(mesh.GetNumFaces(), 0);
  for (int f = 0; f < mesh.GetNumFaces(); ++f) {
    int e1, e2;
    mesh.GetFaceElements(f, &e1, &e2);
    if (e1 >= 0 && e2 >= 0) {
      nbr[e1].push_back({e2, f});
      nbr[e2].push_back({e1, f});
    }
  }
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    face_bdr_attr[mesh.GetBdrElementFaceIndex(be)] = mesh.GetBdrAttribute(be);
  }
  table(quad_rule());
}

const mfem::IntegrationRule& Space::quad_rule() const {
  return mfem::IntRules.Get(mfem::Geometry::TRIANGLE, quad_order());
}

const RefTable& Space::table(const mfem::IntegrationRule& ir) const {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  auto it = tables_.find(&ir);
  if (it != tables_.end()) return *it->second;
  auto t = std::make_unique<RefTable>();
  t->ir = &ir;
  t->nq = ir.GetNPoints();
  t->nd = field_fe_->GetDof();
  t->ng = ng_;
  t->wq.resize(t->nq);
  t->N.resize(static_cast<size_t>(t->nq) * t->nd);
  t->dN.resize(static_cast<size_t>(t->nq) * t->nd * 2);
  t->G.resize(static_cast<size_t>(t->nq) * t->ng);
  t->dG.resize(static_cast<size_t>(t->nq) * t->ng * 2);
  mfem::Vector v;
  mfem::DenseMatrix dm;
  for (int q = 0; q < t->nq; ++q) {
    const auto& ip = ir.IntPoint(q);
    t->wq[q] = ip.weight;
    v.SetSize(t->nd);
    dm.SetSize(t->nd, 2);
    field_fe_->CalcShape(ip, v);
    field_fe_->CalcDShape(ip, dm);
    for (int i = 0; i < t->nd; ++i) {
      t->N[q * t->nd + i] = v(i);
      t->dN[(q * t->nd + i) * 2 + 0] = dm(i, 0);
      t->dN[(q * t->nd + i) * 2 + 1] = dm(i, 1);
    }
    v.SetSize(t->ng);
    dm.SetSize(t->ng, 2);
    geom_fe_->CalcShape(ip, v);
    geom_fe_->CalcDShape(ip, dm);
    for (int k = 0; k < t->ng; ++k) {
      t->G[q * t->ng + k] = v(k);
      t->dG[(q * t->ng + k) * 2 + 0] = dm(k, 0);
      t->dG[(q * t->ng + k) * 2 + 1] = dm(k, 1);
    }
  }
  auto& ref = *t;
  tables_[&ir] = std::move(t);
  return ref;
}

void Space::fill_geometry(int e, const double* G, const double* dG, const double* dNref, int nd, PointEval& pe) const {
  const double* X = gx_.data() + static_cast<size_t>(e) * ng_ * 2;
  double x = 0, y = 0, J00 = 0, J01 = 0, J10 = 0, J11 = 0;
  for (int k = 0; k < ng_; ++k) {
    const double xk = X[2 * k], yk = X[2 * k + 1];
    x += G[k] * xk;
    y += G[k] * yk;
    J00 += xk * dG[2 * k];
    J01 += xk * dG[2 * k + 1];
    J10 += yk * dG[2 * k];
    J11 += yk * dG[2 * k + 1];
  }
  const double det = J00 * J11 - J01 * J10;
  pe.r = std::fabs(x) < rtol ? 0.0 : x;  // round-off off the axis would blow up A/r, G/r
  pe.z = y;
  pe.nd = nd;
  pe.dN.resize(2 * nd);
  const double id = 1.0 / det;
  for (int i = 0; i < nd; ++i) {
    const double a = dNref[2 * i], b = dNref[2 * i + 1];
    pe.dN[2 * i] = (J11 * a - J10 * b) * id;
    pe.dN[2 * i + 1] = (-J01 * a + J00 * b) * id;
  }
  pe.w = std::fabs(det);  // multiplied by the quadrature weight by the caller
}

void Space::eval(int e, const RefTable& t, int q, PointEval& pe) const {
  pe.N.assign(t.N.begin() + static_cast<long>(q) * t.nd, t.N.begin() + static_cast<long>(q + 1) * t.nd);
  fill_geometry(e, &t.G[static_cast<size_t>(q) * t.ng], &t.dG[static_cast<size_t>(q) * t.ng * 2],
                &t.dN[static_cast<size_t>(q) * t.nd * 2], t.nd, pe);
  pe.w *= t.wq[q];
  pe.ip = &t.ir->IntPoint(q);
}

void Space::eval_at(int e, const mfem::IntegrationPoint& ip, PointEval& pe) const {
  const int nd = field_fe_->GetDof();
  mfem::Vector v(nd), g(ng_);
  mfem::DenseMatrix dm(nd, 2), dg(ng_, 2);
  field_fe_->CalcShape(ip, v);
  field_fe_->CalcDShape(ip, dm);
  geom_fe_->CalcShape(ip, g);
  geom_fe_->CalcDShape(ip, dg);
  std::vector<double> G(ng_), dG(2 * ng_), dN(2 * nd);
  for (int k = 0; k < ng_; ++k) {
    G[k] = g(k);
    dG[2 * k] = dg(k, 0);
    dG[2 * k + 1] = dg(k, 1);
  }
  for (int i = 0; i < nd; ++i) {
    dN[2 * i] = dm(i, 0);
    dN[2 * i + 1] = dm(i, 1);
  }
  pe.N.assign(v.GetData(), v.GetData() + nd);
  fill_geometry(e, G.data(), dG.data(), dN.data(), nd, pe);
  pe.w *= ip.weight;
  pe.ip = &ip;
}

std::vector<int> components(const Space& sp, const std::function<bool(int)>& in,
                            const std::function<bool(int)>& cross, int& ncomp) {
  const int ne = sp.mesh.GetNE();
  std::vector<int> comp(ne, -1);
  ncomp = 0;
  std::vector<int> stack;
  for (int s = 0; s < ne; ++s) {
    if (comp[s] >= 0 || !in(s)) continue;
    comp[s] = ncomp;
    stack.push_back(s);
    while (!stack.empty()) {
      int e = stack.back();
      stack.pop_back();
      for (auto [n, f] : sp.nbr[e]) {
        if (comp[n] < 0 && in(n) && cross(f)) {
          comp[n] = ncomp;
          stack.push_back(n);
        }
      }
    }
    ++ncomp;
  }
  return comp;
}

std::vector<std::vector<int>> boundary_loops(const Space& sp, const std::vector<char>& in_set) {
  // Boundary faces of the set: faces with exactly one adjacent element in the set.
  const mfem::Mesh& m = sp.mesh;
  std::vector<int> faces;
  for (int f = 0; f < m.GetNumFaces(); ++f) {
    int e1, e2;
    m.GetFaceElements(f, &e1, &e2);
    bool a = e1 >= 0 && in_set[e1], b = e2 >= 0 && in_set[e2];
    if (a != b) faces.push_back(f);
  }
  // vertex -> faces
  std::map<int, std::vector<int>> v2f;
  mfem::Array<int> fv;
  for (int f : faces) {
    m.GetFaceVertices(f, fv);
    v2f[fv[0]].push_back(f);
    v2f[fv[1]].push_back(f);
  }
  std::set<int> used;
  std::vector<std::vector<int>> loops;
  std::vector<double> areas;
  for (int f0 : faces) {
    if (used.count(f0)) continue;
    std::vector<int> loop;
    double area2 = 0;
    int f = f0;
    m.GetFaceVertices(f, fv);
    int start = fv[0], v = fv[1];
    used.insert(f);
    loop.push_back(f);
    auto acc = [&](int a, int b) {
      const double* pa = m.GetVertex(a);
      const double* pb = m.GetVertex(b);
      area2 += pa[0] * pb[1] - pb[0] * pa[1];
    };
    acc(start, v);
    while (v != start) {
      int next = -1;
      for (int g : v2f[v]) {
        if (!used.count(g)) {
          next = g;
          break;
        }
      }
      if (next < 0) break;  // open chain (should not happen)
      used.insert(next);
      loop.push_back(next);
      m.GetFaceVertices(next, fv);
      int w = (fv[0] == v) ? fv[1] : fv[0];
      acc(v, w);
      v = w;
    }
    loops.push_back(loop);
    areas.push_back(std::fabs(area2) / 2);
  }
  // outer loop first
  std::vector<size_t> idx(loops.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return areas[a] > areas[b]; });
  std::vector<std::vector<int>> out;
  for (size_t i : idx) out.push_back(loops[i]);
  return out;
}

double element_area(const Space& sp, int e) {
  const RefTable& t = sp.table(sp.quad_rule());
  PointEval pe;
  double a = 0;
  for (int q = 0; q < t.nq; ++q) {
    sp.eval(e, t, q, pe);
    a += pe.w;
  }
  return a;
}

void element_center(const Space& sp, int e, double& r, double& z) {
  mfem::IntegrationPoint ip;
  ip.Set2(1.0 / 3.0, 1.0 / 3.0);
  PointEval pe;
  sp.eval_at(e, ip, pe);
  r = pe.r;
  z = pe.z;
}

}  // namespace axf::axi
