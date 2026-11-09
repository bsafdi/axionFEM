// Mechanics and heat in 3D: linear elasticity from the static field, and steady heat conduction.  Same design keys
// (`mechanics`, `thermal`, material blocks `mechanical` / `thermal`) and the same result layout as the
// axisymmetric module (cpp/src/model_mech.cpp, cpp/src/axi_mech.cpp; docs/physics/mechanics_thermal.md#3d,
// docs/design_format.md and docs/validation/mech3d.md); the choices specific to 3D are explained below.
//
// Both problems use H1 elements of order mesh.order on the ParMesh of the magnetoquasistatic solve, assembled
// over the elements of matter only (a material with a `mechanical` / `thermal` block); every degree of freedom
// that no such element touches is eliminated.  Touching regions share nodes and are bonded.
//
//  * Elasticity, small strain, isotropic: sigma = lambda tr(eps) I + 2 mu eps - (3 lambda + 2 mu) eps_th I with
//    eps_th = -alpha under `mechanics.cooldown` (0 for `warm` materials).  Vector H1 space ordered by vdim.
//    Loads (all assembled by hand so that their resultants are integrated in the same pass):
//      - Lorentz body force f = J x B in windings: J is the static source current density of the MQS solve
//        (model3d.cpp keeps it in Impl::J_static), B = curl A_lab + background (+ mechanics.external_field,
//        a prescribed field of sources that are not in the model, e.g. the other coils of a toroid sector);
//      - Maxwell pressure t = -(B^2/2 mu0) n on the faces of superconductors (n out of the superconductor, B on
//        the field side, from the gathered serial field as in the forces post-processing of model3d.cpp);
//      - prescribed pressures (mechanics.pressures) on the faces between two regions or a region and the
//        non-structural surroundings;
//      - the thermal prestress of a cooldown.
//    Constraints, all exact (by elimination): supports fix "all", "x", "y" or "z" on every node of a region;
//    symmetry planes (mechanics.symmetry_planes, any orientation) fix u.n = 0 on the structure's boundary faces
//    that lie in the plane (end faces in plane strain, the cut faces of a sector model).  A node with constraint normals that are
//    not Cartesian is rotated: u = Q u~ with the first columns of Q spanning the constraint normals, so the
//    system becomes Q^T K Q (hypre RAP) with the constrained components of u~ eliminated.  Every rigid motion
//    of a connected assembly that the constraints leave free is found from the Gram matrix of the six rigid
//    modes on the constrained nodes and pinned at as many unconstrained nodes (greedy pivoting), and is
//    projected out afterwards (volume-weighted L2 projection), as the axisymmetric solver does for its axial
//    translation.  Reactions are K u - F at the constrained degrees of freedom; force and moment equilibrium
//    (sum of applied loads + reactions = 0) are identities of the discrete system and are reported as
//    closure checks.
//
//  * Conduction: div(k grad T) + q = 0 with k constant or a piecewise-linear table k(T) (Picard iteration on
//    k(T_prev)); fixed temperatures on whole regions, adiabatic elsewhere; heat balance from the reactions at
//    the prescribed temperatures, split by sign as in 2D.
#include <mfem.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "model3d_impl.hpp"

#include "axionfem/core.hpp"
#include "axionfem/mech.hpp"
#include "axionfem/model3d.hpp"
#include "axionfem/prescribed_source.hpp"
#include "axionfem/spec.hpp"
#include "axionfem/units.hpp"

namespace axf {

namespace {

using units::mu0;
using units::pi;
using Vec3 = std::array<double, 3>;

double gsum(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return v;
}
void gsum(double* v, int n) {
  if (n > 0) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}
double gmax(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return v;
}
long long gsuml(long long v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  return v;
}
double gdot(const mfem::Vector& a, const mfem::Vector& b) { return mfem::InnerProduct(MPI_COMM_WORLD, a, b); }

std::string join(const std::set<std::string>& names) {
  std::string s;
  for (auto& n : names) s += (s.empty() ? "" : ", ") + n;
  return s;
}

json vec_json(const double* v) { return json::array({v[0], v[1], v[2]}); }

// Connected components of the serial elements selected by `in`, joined through shared vertices: exactly the
// connectivity of a bonded H1 assembly of any order (elements that share an edge or a face share its vertices).
std::vector<int> serial_components(const mfem::Mesh& m, const std::function<bool(int)>& in, int& nc) {
  const int ne = m.GetNE();
  std::vector<int> par(ne);
  std::iota(par.begin(), par.end(), 0);
  std::function<int(int)> find = [&](int a) { return par[a] == a ? a : par[a] = find(par[a]); };
  std::vector<int> first(m.GetNV(), -1);
  mfem::Array<int> vv;
  for (int e = 0; e < ne; ++e) {
    if (!in(e)) continue;
    m.GetElementVertices(e, vv);
    for (int v : vv) {
      if (first[v] < 0) {
        first[v] = e;
        continue;
      }
      int a = find(first[v]), b = find(e);
      if (a != b) par[std::max(a, b)] = std::min(a, b);
    }
  }
  std::vector<int> comp(ne, -1);
  std::map<int, int> id;
  nc = 0;
  for (int e = 0; e < ne; ++e) {
    if (!in(e)) continue;
    auto it = id.find(find(e));
    if (it == id.end()) it = id.emplace(find(e), nc++).first;
    comp[e] = it->second;
  }
  return comp;
}

// The same, joined through shared faces only.  Parts of one vertex-connected assembly that touch only along an edge or at
// a point form a mechanism (a hinge), which the six rigid modes of the assembly do not cover.
int face_component_count(const mfem::Mesh& m, const std::function<bool(int)>& in) {
  const int ne = m.GetNE();
  std::vector<int> par(ne);
  std::iota(par.begin(), par.end(), 0);
  std::function<int(int)> find = [&](int a) { return par[a] == a ? a : par[a] = find(par[a]); };
  for (int f = 0; f < m.GetNumFaces(); ++f) {
    int e1 = -1, e2 = -1;
    m.GetFaceElements(f, &e1, &e2);
    if (e1 < 0 || e2 < 0 || !in(e1) || !in(e2)) continue;
    int a = find(e1), b = find(e2);
    if (a != b) par[std::max(a, b)] = std::min(a, b);
  }
  std::set<int> roots;
  for (int e = 0; e < ne; ++e)
    if (in(e)) roots.insert(find(e));
  return static_cast<int>(roots.size());
}

// Lattice of reference points on a tetrahedron (vertices, edges, faces, interior), where the extrema of a
// polynomial field are looked for (the 3D analogue of the triangle lattice of model_mech.cpp).
const mfem::IntegrationRule& tet_lattice(int k) {
  static std::map<int, mfem::IntegrationRule> cache;
  auto it = cache.find(k);
  if (it != cache.end()) return it->second;
  mfem::IntegrationRule ir;
  for (int i = 0; i <= k; ++i)
    for (int j = 0; i + j <= k; ++j)
      for (int l = 0; i + j + l <= k; ++l) {
        mfem::IntegrationPoint ip;
        ip.Set3(double(i) / k, double(j) / k, double(l) / k);
        ip.weight = 0;
        ir.Append(ip);
      }
  return cache[k] = ir;
}

// Rigid mode k (0..2 translations, 3..5 rotations about the axes through c, scaled by 1/Lc) at x.
void rigid_mode(int k, const double* x, const double* c, double Lc, double* m) {
  m[0] = m[1] = m[2] = 0;
  if (k < 3) {
    m[k] = 1;
    return;
  }
  const double r[3] = {(x[0] - c[0]) / Lc, (x[1] - c[1]) / Lc, (x[2] - c[2]) / Lc};
  const int a = k - 3;  // omega = e_a: m = e_a x r
  const int b = (a + 1) % 3, d = (a + 2) % 3;
  m[d] = r[b];
  m[b] = -r[d];
}

// Running extremes of a scalar (min / max), reduced over the ranks.
struct Range {
  double lo = std::numeric_limits<double>::infinity(), hi = -std::numeric_limits<double>::infinity();
  void add(double v) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  json to_json() const {
    if (lo > hi) return {{"min", 0.0}, {"max", 0.0}};
    return {{"min", lo}, {"max", hi}};
  }
};
void reduce_ranges(std::vector<Range>& r) {
  std::vector<double> lo(r.size()), hi(r.size());
  for (size_t i = 0; i < r.size(); ++i) {
    lo[i] = r[i].lo;
    hi[i] = r[i].hi;
  }
  if (!r.empty()) {
    MPI_Allreduce(MPI_IN_PLACE, lo.data(), static_cast<int>(lo.size()), MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, hi.data(), static_cast<int>(hi.size()), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  }
  for (size_t i = 0; i < r.size(); ++i) {
    r[i].lo = lo[i];
    r[i].hi = hi[i];
  }
}

// Running maximum with its location, reduced over the ranks (the location of the winning rank).
struct Peak {
  double value = -1;
  double x[3] = {0, 0, 0};
  void add(double v, const double* p) {
    if (v > value) {
      value = v;
      for (int k = 0; k < 3; ++k) x[k] = p[k];
    }
  }
  json loc() const { return {{"x_m", x[0]}, {"y_m", x[1]}, {"z_m", x[2]}}; }
};
void reduce_peaks(std::vector<Peak>& p, int rank) {
  const int n = static_cast<int>(p.size());
  if (n == 0) return;
  struct VI {
    double v;
    int r;
  };
  std::vector<VI> vi(n);
  for (int i = 0; i < n; ++i) vi[i] = {p[i].value, rank};
  MPI_Allreduce(MPI_IN_PLACE, vi.data(), n, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  std::vector<double> loc(3 * n, 0.0);
  for (int i = 0; i < n; ++i)
    if (vi[i].r == rank)
      for (int k = 0; k < 3; ++k) loc[3 * i + k] = p[i].x[k];
  gsum(loc.data(), 3 * n);
  for (int i = 0; i < n; ++i) {
    p[i].value = vi[i].v;
    for (int k = 0; k < 3; ++k) p[i].x[k] = loc[3 * i + k];
  }
}

// A plane on which the normal displacement of the structure's boundary vanishes.
struct Plane {
  std::string name, origin;
  double p[3] = {0, 0, 0}, n[3] = {0, 0, 1};
};

std::vector<Vec3> parse_points(const json& doc, const char* section) {
  std::vector<Vec3> pts;
  if (!doc.contains(section) || !doc.at(section).contains("probes")) return pts;
  const json& pj = doc.at(section).at("probes");
  const std::string path = std::string("/") + section + "/probes";
  AXF_REQUIRE(pj.is_array(), section + std::string(".probes must be a list of [x, y, z] points"), path);
  for (size_t i = 0; i < pj.size(); ++i) {
    const json& q = pj[i];
    AXF_REQUIRE(q.is_array() && q.size() == 3 && q[0].is_number() && q[1].is_number() && q[2].is_number(),
                "a probe is a point [x, y, z] in metres", strcat(path, "/", i));
    pts.push_back({q[0].get<double>(), q[1].get<double>(), q[2].get<double>()});
  }
  return pts;
}

// Points located in the local ParMesh; the lowest rank that holds a point evaluates it.
struct Located {
  std::vector<int> elem;  // local element, or -1 on the ranks that do not evaluate the point
  mfem::Array<mfem::IntegrationPoint> ips;
};
Located locate(mfem::ParMesh& pm, const std::vector<Vec3>& pts, int rank, int nranks) {
  Located L;
  const int n = static_cast<int>(pts.size());
  L.elem.assign(n, -1);
  if (n == 0) return L;
  mfem::DenseMatrix pm3(3, n);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) pm3(k, i) = pts[i][k];
  mfem::Array<int> eid;
  pm.FindPoints(pm3, eid, L.ips, false);
  std::vector<int> owner(n);
  for (int i = 0; i < n; ++i) owner[i] = eid[i] >= 0 ? rank : nranks;
  MPI_Allreduce(MPI_IN_PLACE, owner.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  for (int i = 0; i < n; ++i)
    if (owner[i] == rank && eid[i] >= 0) L.elem[i] = eid[i];
  return L;
}

// Elastic role of one local element.
struct MRole {
  bool solid = false;
  double lambda = 0, mu = 0, eps_th = 0, s0 = 0;  // s0 = (3 lambda + 2 mu) eps_th
  bool winding = false;
  int comp = -1;
};

// Thermal role of one local element.
struct TRole {
  bool domain = false;
  double k = 0, q = 0;
  const axi::KTable* kt = nullptr;
  int fixed = -1;  // index of the prescribed-temperature entry
};

class KCoefficient3 : public mfem::Coefficient {
 public:
  KCoefficient3(const std::vector<TRole>& roles, const mfem::ParGridFunction& T) : roles_(roles), T_(T) {}
  double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint& ip) override {
    const TRole& r = roles_[Tr.ElementNo];
    if (!r.domain) return 0.0;
    if (r.kt && !r.kt->empty()) return r.kt->at(T_.GetValue(Tr, ip));
    return r.k;
  }

 private:
  const std::vector<TRole>& roles_;
  const mfem::ParGridFunction& T_;
};

class QCoefficient3 : public mfem::Coefficient {
 public:
  explicit QCoefficient3(const std::vector<TRole>& roles) : roles_(roles) {}
  double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint&) override {
    const TRole& r = roles_[Tr.ElementNo];
    return r.domain ? r.q : 0.0;
  }

 private:
  const std::vector<TRole>& roles_;
};

class PWCoefficient3 : public mfem::Coefficient {  // per local element
 public:
  explicit PWCoefficient3(std::vector<double> v) : v_(std::move(v)) {}
  double Eval(mfem::ElementTransformation& Tr, const mfem::IntegrationPoint&) override { return v_[Tr.ElementNo]; }

 private:
  std::vector<double> v_;
};

// PCG + BoomerAMG on an eliminated system, with a few defect-correction passes so that the residual on the free
// rows (not hypre's criterion relative to |B|, which the eliminated rows can dominate) reaches `tol`.
json solve_spd(mfem::HypreParMatrix& Kel, const mfem::Array<int>& ess, const mfem::Vector& B, mfem::Vector& X,
               mfem::HypreBoomerAMG& amg, double tol, int max_it) {
  mfem::HyprePCG pcg(Kel);
  pcg.SetTol(tol);
  pcg.SetMaxIter(max_it);
  pcg.SetPrintLevel(0);
  pcg.SetPreconditioner(amg);
  auto free_res = [&](const mfem::Vector& x, mfem::Vector& r) {
    Kel.Mult(x, r);
    r -= B;
    for (int i : ess) r(i) = 0.0;
    return std::sqrt(std::max(0.0, gdot(r, r)));
  };
  mfem::Vector Bf(B);
  for (int i : ess) Bf(i) = 0.0;
  const double bn = std::sqrt(std::max(0.0, gdot(Bf, Bf)));
  int its = 0, total = 0;
  pcg.Mult(B, X);
  pcg.GetNumIterations(its);
  total += its;
  mfem::Vector Rv(X.Size()), dX(X.Size());
  double rn = free_res(X, Rv);
  int passes = 0;
  for (; passes < 4 && bn > 0 && rn > tol * bn; ++passes) {
    Rv.Neg();
    dX = 0.0;
    pcg.Mult(Rv, dX);
    pcg.GetNumIterations(its);
    total += its;
    X += dX;
    rn = free_res(X, Rv);
  }
  return {{"pcg_iterations", total}, {"correction_passes", passes},
          {"relative_residual", bn > 0 ? rn / bn : 0.0}};
}

}  // namespace

void Model3D::mechanics_thermal_analysis(const SolveOptions& o) {
  if (!d_.mechanics.enabled && !d_.thermal.enabled) return;
  Impl& P = *p_;
  AXF_REQUIRE(P.pmesh && P.serial, "internal: the 3D mechanics needs a meshed model");
  Timer timer;
  json& R = results_;
  if (!R.contains("warnings") || !R["warnings"].is_array()) R["warnings"] = json::array();
  json& warnings = R["warnings"];
  mfem::ParMesh& pm = *P.pmesh;
  mfem::Mesh& sm = *P.serial;
  const int ne = pm.GetNE(), nser = sm.GetNE();
  const int order = std::max(1, P.order);
  const json& doc = d_.source;

  // ------------------------------------------------------------ serial element -> region / material
  static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
  std::vector<std::string> sreg(nser);
  std::vector<const spec::Material*> smat(nser);
  for (int s = 0; s < nser; ++s) {
    const int a = sm.GetAttribute(s);
    const bool shell = P.sh_attr > 0 && a == P.sh_attr;
    sreg[s] = (!shell && P.region_of_attr.count(a)) ? P.region_of_attr.at(a) : "air";
    const spec::Region* g = d_.find_region(sreg[s]);
    smat[s] = g ? &d_.materials.at(g->material) : &vacuum;
  }
  std::vector<int> ser2loc(nser, -1);
  for (int e = 0; e < ne; ++e) ser2loc[P.loc2ser[e]] = e;
  double Lbox = 0;
  {
    mfem::Vector mn, mx;
    sm.GetBoundingBox(mn, mx);
    for (int k = 0; k < 3; ++k) Lbox = std::max(Lbox, mx(k) - mn(k));
  }
  auto is_sc = [&](int s) { return smat[s]->kind == spec::MaterialKind::Superconductor; };
  auto missing_data = [&](const char* what, const std::set<std::string>& names) {
    if (names.empty()) return;
    warnings.push_back(strcat("the ", what, " analysis skips the region(s) ", join(names),
                              ": their material has no \"", what, "\" block. Add one to the material "
                              "(see docs/design_format.md) to include them."));
  };
  // background field at a point (zero in the shell and without a background)
  auto add_bg = [&](const mfem::Vector& x, int attr, mfem::Vector& B) {
    if (!P.bg || (P.sh_attr > 0 && attr == P.sh_attr)) return;
    double b[3];
    P.bg_at(x, b);
    for (int k = 0; k < 3; ++k) B(k) += b[k];
  };
  // local element faces with their serial face and serial neighbour (-1: boundary of the mesh)
  struct FaceRef {
    int lf, sf, s_other;
  };
  auto element_faces = [&](int e, std::vector<FaceRef>& out) {
    out.clear();
    mfem::Array<int> lfs, sfs, o1, o2;
    pm.GetElementFaces(e, lfs, o1);
    const int s = P.loc2ser[e];
    sm.GetElementFaces(s, sfs, o2);
    for (int k = 0; k < lfs.Size(); ++k) {
      int e1 = -1, e2 = -1;
      sm.GetFaceElements(sfs[k], &e1, &e2);
      out.push_back({lfs[k], sfs[k], e1 == s ? e2 : e1});
    }
  };

  const std::vector<Vec3> mprobes = parse_points(doc, "mechanics"), tprobes = parse_points(doc, "thermal");

  // ============================================================ elasticity
  if (d_.mechanics.enabled) {
    const auto& ms = d_.mechanics;
    const json mdoc = doc.contains("mechanics") ? doc.at("mechanics") : json::object();
    std::vector<char> ssolid(nser, 0);
    std::set<std::string> skipped, solid_regions;
    for (int s = 0; s < nser; ++s) {
      if (smat[s]->mech.given) {
        ssolid[s] = 1;
        solid_regions.insert(sreg[s]);
      } else if (sreg[s] != "air" && smat[s]->kind != spec::MaterialKind::Vacuum) {
        skipped.insert(sreg[s]);
      }
    }
    missing_data("mechanical", skipped);
    AXF_REQUIRE(!solid_regions.empty(), "the mechanics analysis has no structure", "/mechanics",
                "Every region is vacuum-like or lacks a \"mechanical\" material block.");
    {
      std::set<std::string> soft, noalpha;
      for (int s = 0; s < nser; ++s) {
        if (!ssolid[s]) continue;
        if (smat[s]->mech.nu > 0.45) soft.insert(smat[s]->name);
        if (ms.cooldown && !smat[s]->warm && smat[s]->mech.alpha == 0.0) noalpha.insert(smat[s]->name);
      }
      if (!soft.empty())
        warnings.push_back("mechanics: material(s) " + join(soft) +
                           " have nu > 0.45. The pure-displacement formulation is not locking-free: on "
                           "a fixed mesh the stresses are over-stiff by roughly 1/(1 - 2 nu) (7 % at "
                           "nu = 0.49, 70 % at nu = 0.499 on the Lame benchmark). Refine strongly, or "
                           "reduce nu if the material is not really incompressible.");
      if (!noalpha.empty())
        warnings.push_back("mechanics: mechanics.cooldown is on but material(s) " + join(noalpha) +
                           " have no (or zero) \"alpha\" in their \"mechanical\" block, so they "
                           "contract by nothing while the parts around them do. Give them an "
                           "integrated contraction alpha = (L_300K - L_cold)/L_300K, or mark them "
                           "\"warm\": true if they really stay at room temperature.");
    }
    int ncomp = 0;
    const std::vector<int> scomp = serial_components(sm, [&](int s) { return ssolid[s] != 0; }, ncomp);
    std::vector<std::set<std::string>> comp_names(ncomp);
    for (int s = 0; s < nser; ++s)
      if (scomp[s] >= 0) comp_names[scomp[s]].insert(sreg[s]);
    if (face_component_count(sm, [&](int s) { return ssolid[s] != 0; }) != ncomp) {
      std::set<std::string> names;
      for (auto& c : comp_names)
        if (c.size() > 1) names.insert(c.begin(), c.end());
      if (names.empty()) names = solid_regions;
      throw Error("parts of the structure touch only along an edge or at a point (regions " + join(names) + ")",
                  "/mechanics",
                  "A bonded edge or point contact is a hinge, a mechanism the solver cannot pin. Join the parts "
                  "over a face, separate them, or leave one of them out of the structure.");
    }

    // ---- roles
    std::set<std::string> wind_regions;
    for (auto& w : d_.windings) wind_regions.insert(w.region);
    std::vector<MRole> roles(ne);
    std::vector<double> lam_e(ne, 0.0), mu_e(ne, 0.0);
    for (int e = 0; e < ne; ++e) {
      const int s = P.loc2ser[e];
      if (!ssolid[s]) continue;
      MRole& ro = roles[e];
      const spec::Material& mt = *smat[s];
      ro.solid = true;
      const double E = mt.mech.E_Pa, nu = mt.mech.nu;
      ro.lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      ro.mu = E / (2 * (1 + nu));
      ro.eps_th = (ms.cooldown && !mt.warm) ? -mt.mech.alpha : 0.0;
      ro.s0 = (3 * ro.lambda + 2 * ro.mu) * ro.eps_th;
      ro.winding = wind_regions.count(sreg[s]) > 0 && P.J_static != nullptr;
      ro.comp = scomp[s];
      lam_e[e] = ro.lambda;
      mu_e[e] = ro.mu;
    }
    std::set<std::string> loaded_windings, dropped_windings;
    for (int s = 0; s < nser; ++s) {
      if (!wind_regions.count(sreg[s])) continue;
      if (ssolid[s]) loaded_windings.insert(sreg[s]);
      else dropped_windings.insert(sreg[s]);
    }
    if (!dropped_windings.empty())
      warnings.push_back("the Lorentz body force of winding region(s) " + join(dropped_windings) +
                         " is not applied: their material has no \"mechanical\" block");
    if (!loaded_windings.empty() && !P.J_static)
      warnings.push_back("mechanics: no static source current is available (no static solve), so the windings "
                         "carry no Lorentz load");

    // ---- external field (sources outside the model): Lorentz loads of the windings only
    std::unique_ptr<PrescribedMagneticSource> ext;
    json ext_json = nullptr;
    if (mdoc.contains("external_field")) {
      try {
        ext = std::make_unique<PrescribedMagneticSource>(PrescribedMagneticSource::from_json(mdoc.at("external_field")));
      } catch (const std::exception& ex) {
        throw Error(std::string("mechanics.external_field: ") + ex.what(), "/mechanics/external_field",
                    "Same format as fullwave.source: {\"file\": path} or an inline {\"kind\": ...} object.");
      }
      ext_json = ext->summary();
      bool any_sc = false;
      for (int s = 0; s < nser; ++s) any_sc = any_sc || (ssolid[s] && is_sc(s));
      if (any_sc)
        warnings.push_back("mechanics.external_field is added to the Lorentz load of the windings only: it is not "
                           "part of the static solve, so superconductors do not screen it and their magnetic "
                           "pressure does not include it");
    }

    // ---- space, node coordinates and degree-of-freedom bookkeeping
    mfem::H1_FECollection fec(order, 3);
    mfem::ParFiniteElementSpace fes(&pm, &fec, 3, mfem::Ordering::byVDIM);
    const int nnodes = fes.GetNDofs(), vsize = fes.GetVSize(), tsize = fes.GetTrueVSize();
    const mfem::Operator* Pr = fes.GetProlongationMatrix();
    auto reduce_to_true = [&](const mfem::Vector& l, mfem::Vector& t) {
      t.SetSize(tsize);
      Pr->MultTranspose(l, t);
    };
    mfem::ParGridFunction xg(&fes);
    {
      mfem::VectorFunctionCoefficient xc(3, [](const mfem::Vector& x, mfem::Vector& v) { v = x; });
      xg.ProjectCoefficient(xc);
    }
    std::vector<std::array<int, 3>> td(nnodes);
    std::vector<int> owned_nodes;
    for (int i = 0; i < nnodes; ++i) {
      for (int c = 0; c < 3; ++c) td[i][c] = fes.GetLocalTDofNumber(fes.DofToVDof(i, c));
      if (td[i][0] >= 0) owned_nodes.push_back(i);
    }
    auto node_x = [&](int i, double* x) {
      for (int c = 0; c < 3; ++c) x[c] = xg(fes.DofToVDof(i, c));
    };
    // used nodes and their component (reduced to the owners)
    std::vector<char> used(nnodes, 0);
    std::vector<int> ncomp_of(nnodes, -1);
    {
      mfem::Vector ul(vsize), cl(vsize), ut, ct;
      ul = 0.0;
      cl = 0.0;
      mfem::Array<int> dofs;
      for (int e = 0; e < ne; ++e) {
        if (!roles[e].solid) continue;
        fes.GetElementDofs(e, dofs);
        for (int i : dofs) {
          ul(fes.DofToVDof(i, 0)) += 1.0;
          cl(fes.DofToVDof(i, 0)) += roles[e].comp + 1.0;
        }
      }
      reduce_to_true(ul, ut);
      reduce_to_true(cl, ct);
      for (int i : owned_nodes) {
        const double cnt = ut(td[i][0]);
        if (cnt > 0) {
          used[i] = 1;
          ncomp_of[i] = static_cast<int>(std::lround(ct(td[i][0]) / cnt)) - 1;
        }
      }
    }

    // ---- constraint sources: supports (one per region), then symmetry planes
    struct Source {
      std::string kind, name;
      std::vector<Vec3> normals;
      int plane = -1;
    };
    std::vector<Source> sources;
    for (size_t i = 0; i < ms.supports.size(); ++i) {
      const auto& sp = ms.supports[i];
      if (!solid_regions.count(sp.region))
        throw Error("support region '" + sp.region + "' is not part of the structure",
                    strcat("/mechanics/supports/", i, "/region"),
                    "Only regions whose material has a \"mechanical\" block carry displacements.");
      for (size_t j = 0; j < i; ++j)
        if (ms.supports[j].region == sp.region)
          throw Error("region '" + sp.region + "' already has a support (entry " + std::to_string(j) + ")",
                      strcat("/mechanics/supports/", i, "/region"),
                      "A region carries one support: use \"fix\": \"all\" to fix every component at once.");
      Source src{"support", sp.region, {}, -1};
      if (sp.fix == "all" || sp.fix == "x") src.normals.push_back({1, 0, 0});
      if (sp.fix == "all" || sp.fix == "y") src.normals.push_back({0, 1, 0});
      if (sp.fix == "all" || sp.fix == "z") src.normals.push_back({0, 0, 1});
      sources.push_back(src);
    }
    const int nsup = static_cast<int>(sources.size());
    // (mesh.symmetry cuts need fullwave.rf_only, which has no static solve and runs no mechanics, so the planes
    // are always explicit)
    std::vector<Plane> planes;
    if (mdoc.contains("symmetry_planes")) {
      const json& sp = mdoc.at("symmetry_planes");
      AXF_REQUIRE(sp.is_array(), "mechanics.symmetry_planes must be a list", "/mechanics/symmetry_planes");
      for (size_t i = 0; i < sp.size(); ++i) {
        const json& q = sp[i];
        const std::string path = strcat("/mechanics/symmetry_planes/", i);
        AXF_REQUIRE(q.is_object(), "a symmetry plane is {\"point\": [x, y, z], \"normal\": [nx, ny, nz]}", path);
        for (auto it = q.begin(); it != q.end(); ++it)
          AXF_REQUIRE(it.key() == "point" || it.key() == "normal" || it.key() == "name" || it.key() == "notes",
                      "unknown key '" + it.key() + "' in a symmetry plane", path,
                      "Allowed keys: point, normal, name, notes.");
        Plane pl;
        pl.name = q.value("name", strcat("plane_", i));
        pl.origin = "mechanics.symmetry_planes";
        for (const char* key : {"point", "normal"}) {
          AXF_REQUIRE(q.contains(key) && q.at(key).is_array() && q.at(key).size() == 3,
                      std::string("a symmetry plane needs '") + key + "' = [x, y, z]", path + "/" + key);
          for (int k = 0; k < 3; ++k) (std::string(key) == "point" ? pl.p : pl.n)[k] = q.at(key)[k].get<double>();
        }
        const double nn = std::sqrt(pl.n[0] * pl.n[0] + pl.n[1] * pl.n[1] + pl.n[2] * pl.n[2]);
        AXF_REQUIRE(nn > 0, "the normal of a symmetry plane must not vanish", path + "/normal");
        for (double& v : pl.n) v /= nn;
        planes.push_back(pl);
      }
    }
    for (size_t j = 0; j < planes.size(); ++j)
      sources.push_back({"plane", planes[j].name, {{planes[j].n[0], planes[j].n[1], planes[j].n[2]}}, int(j)});
    const int nsrc = static_cast<int>(sources.size());
    // node membership per source (reduced to the owners); planes carry the sign of the structure's outward
    // normal along the plane normal, for the compressive / tensile split of the reactions
    std::vector<std::vector<int>> node_src(nnodes);
    std::vector<std::vector<double>> node_sgn(nnodes);
    std::vector<long long> plane_faces(planes.size(), 0);
    {
      std::vector<mfem::Vector> mark(nsrc);
      for (auto& v : mark) {
        v.SetSize(vsize);
        v = 0.0;
      }
      mfem::Array<int> dofs, fv;
      for (int j = 0; j < nsup; ++j)
        for (int e = 0; e < ne; ++e) {
          if (!roles[e].solid || sreg[P.loc2ser[e]] != sources[j].name) continue;
          fes.GetElementDofs(e, dofs);
          for (int i : dofs) mark[j](fes.DofToVDof(i, 0)) = 1.0;
        }
      if (!planes.empty()) {
        std::vector<FaceRef> fr;
        mfem::Vector nor(3);
        for (int e = 0; e < ne; ++e) {
          if (!roles[e].solid) continue;
          element_faces(e, fr);
          for (const auto& f : fr) {
            if (f.s_other >= 0 && ssolid[f.s_other]) continue;  // interior of the structure
            pm.GetFaceVertices(f.lf, fv);
            for (size_t j = 0; j < planes.size(); ++j) {
              const Plane& pl = planes[j];
              bool on = true;
              for (int v : fv) {
                const double* X = pm.GetVertex(v);
                const double dist = (X[0] - pl.p[0]) * pl.n[0] + (X[1] - pl.p[1]) * pl.n[1] + (X[2] - pl.p[2]) * pl.n[2];
                if (std::fabs(dist) > 1e-7 * Lbox) { on = false; break; }
              }
              if (!on) continue;
              ++plane_faces[j];
              // outward normal of the element at the face centre (serial face transformation)
              mfem::FaceElementTransformations* FT = sm.GetFaceElementTransformations(f.sf);
              const mfem::IntegrationRule& irc = mfem::IntRules.Get(FT->GetGeometryType(), 1);
              FT->SetAllIntPoints(&irc.IntPoint(0));
              mfem::CalcOrtho(FT->Jacobian(), nor);
              const double sgn1 = FT->Elem1No == P.loc2ser[e] ? 1.0 : -1.0;
              const double dn = sgn1 * (nor(0) * pl.n[0] + nor(1) * pl.n[1] + nor(2) * pl.n[2]);
              fes.GetFaceDofs(f.lf, dofs);
              for (int i : dofs) mark[nsup + j](fes.DofToVDof(i, 0)) += dn > 0 ? 1.0 : -1.0;
            }
          }
        }
      }
      for (int j = 0; j < nsrc; ++j) {
        mfem::Vector t;
        mfem::Vector a(mark[j]);
        if (j >= nsup)  // count membership separately from the sign
          for (int i = 0; i < a.Size(); ++i) a(i) = mark[j](i) != 0.0 ? 1.0 : 0.0;
        reduce_to_true(a, t);
        mfem::Vector ts;
        if (j >= nsup) reduce_to_true(mark[j], ts);
        for (int i : owned_nodes) {
          if (!used[i] || !(t(td[i][0]) > 0)) continue;
          node_src[i].push_back(j);
          node_sgn[i].push_back(j >= nsup ? (ts(td[i][0]) > 0 ? 1.0 : (ts(td[i][0]) < 0 ? -1.0 : 0.0)) : 0.0);
        }
      }
    }
    {
      std::vector<double> pf(plane_faces.begin(), plane_faces.end());
      gsum(pf.data(), static_cast<int>(pf.size()));
      for (size_t j = 0; j < planes.size(); ++j) {
        plane_faces[j] = static_cast<long long>(pf[j]);
        if (plane_faces[j] == 0)
          warnings.push_back("mechanics: symmetry plane '" + planes[j].name + "' (" + planes[j].origin +
                             ") meets no boundary face of the structure, so it constrains nothing");
      }
    }
    // per owned node: orthonormal basis whose first k columns span the constraint normals
    struct NodeBasis {
      int k = 0;
      bool rotated = false;
      double Q[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};  // columns q_a: Q[c][a]
      int cart[3] = {0, 0, 0};                           // unrotated: constrained Cartesian components
    };
    std::vector<NodeBasis> nb(nnodes);
    long long n_rot = 0, n_con = 0;
    for (int i : owned_nodes) {
      if (node_src[i].empty()) continue;
      ++n_con;
      std::vector<Vec3> ns;
      for (int j : node_src[i])
        for (auto& v : sources[j].normals) ns.push_back(v);
      std::vector<Vec3> q;
      for (auto v : ns) {
        for (auto& b : q) {
          const double d = v[0] * b[0] + v[1] * b[1] + v[2] * b[2];
          for (int c = 0; c < 3; ++c) v[c] -= d * b[c];
        }
        const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (l < 1e-8) continue;
        for (double& c : v) c /= l;
        q.push_back(v);
      }
      NodeBasis& B = nb[i];
      B.k = static_cast<int>(q.size());
      bool cart = true;
      for (auto& v : q) {
        int big = 0;
        for (int c = 0; c < 3; ++c)
          if (std::fabs(v[c]) > 1 - 1e-12) ++big;
        if (big != 1) cart = false;
      }
      if (cart) {
        for (auto& v : q)
          for (int c = 0; c < 3; ++c)
            if (std::fabs(v[c]) > 0.5) B.cart[c] = 1;
        continue;
      }
      // complete the basis
      while (q.size() < 3) {
        Vec3 best{0, 0, 0};
        double bl = -1;
        for (int c = 0; c < 3; ++c) {
          Vec3 v{0, 0, 0};
          v[c] = 1;
          for (auto& b : q) {
            const double d = v[0] * b[0] + v[1] * b[1] + v[2] * b[2];
            for (int k = 0; k < 3; ++k) v[k] -= d * b[k];
          }
          const double l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
          if (l > bl) {
            bl = l;
            best = {v[0] / l, v[1] / l, v[2] / l};
          }
        }
        q.push_back(best);
      }
      B.rotated = true;
      ++n_rot;
      for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 3; ++c) B.Q[c][a] = q[a][c];
    }
    const bool any_rot = gsuml(n_rot) > 0;

    // ---- rigid modes left free by the constraints, per connected assembly
    std::vector<std::array<double, 3>> cctr(ncomp, {0, 0, 0});
    std::vector<double> clen(ncomp, 1.0);
    {
      std::vector<std::array<double, 6>> bb(ncomp, {1e300, 1e300, 1e300, -1e300, -1e300, -1e300});
      mfem::Array<int> vv;
      for (int s = 0; s < nser; ++s) {
        if (scomp[s] < 0) continue;
        sm.GetElementVertices(s, vv);
        for (int v : vv) {
          const double* X = sm.GetVertex(v);
          for (int k = 0; k < 3; ++k) {
            bb[scomp[s]][k] = std::min(bb[scomp[s]][k], X[k]);
            bb[scomp[s]][3 + k] = std::max(bb[scomp[s]][3 + k], X[k]);
          }
        }
      }
      for (int c = 0; c < ncomp; ++c) {
        double l = 0;
        for (int k = 0; k < 3; ++k) {
          cctr[c][k] = 0.5 * (bb[c][k] + bb[c][3 + k]);
          l = std::max(l, bb[c][3 + k] - bb[c][k]);
        }
        clen[c] = std::max(l, 1e-300);
      }
    }
    std::vector<double> gram(36 * ncomp, 0.0);
    for (int i : owned_nodes) {
      if (node_src[i].empty() || ncomp_of[i] < 0) continue;
      const int c = ncomp_of[i];
      double x[3], m[6][3];
      node_x(i, x);
      for (int k = 0; k < 6; ++k) rigid_mode(k, x, cctr[c].data(), clen[c], m[k]);
      const NodeBasis& B = nb[i];
      std::vector<Vec3> qs;
      if (B.rotated) {
        for (int a = 0; a < B.k; ++a) qs.push_back({B.Q[0][a], B.Q[1][a], B.Q[2][a]});
      } else {
        for (int cc = 0; cc < 3; ++cc)
          if (B.cart[cc]) {
            Vec3 v{0, 0, 0};
            v[cc] = 1;
            qs.push_back(v);
          }
      }
      for (auto& q : qs)
        for (int k = 0; k < 6; ++k)
          for (int l = 0; l < 6; ++l)
            gram[36 * c + 6 * k + l] += (m[k][0] * q[0] + m[k][1] * q[1] + m[k][2] * q[2]) *
                                        (m[l][0] * q[0] + m[l][1] * q[1] + m[l][2] * q[2]);
    }
    gsum(gram.data(), static_cast<int>(gram.size()));
    std::vector<Eigen::MatrixXd> free_modes(ncomp);  // 6 x k coefficient vectors
    for (int c = 0; c < ncomp; ++c) {
      Eigen::MatrixXd G(6, 6);
      for (int k = 0; k < 6; ++k)
        for (int l = 0; l < 6; ++l) G(k, l) = gram[36 * c + 6 * k + l];
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
      const double lmax = es.eigenvalues().maxCoeff();
      std::vector<int> fr;
      for (int k = 0; k < 6; ++k)
        // free: numerically zero against the largest eigenvalue (a clamped patch of size eps Lc gives ~eps^2;
        // 1e-12 keeps patches down to ~1e-6 Lc constrained)
        if (!(lmax > 0) || es.eigenvalues()(k) < 1e-12 * lmax) fr.push_back(k);
      free_modes[c].resize(6, static_cast<int>(fr.size()));
      for (size_t j = 0; j < fr.size(); ++j) free_modes[c].col(static_cast<int>(j)) = es.eigenvectors().col(fr[j]);
    }
    // pins: greedy pivoting over the unconstrained directions of the owned nodes of each assembly (a Cartesian
    // component of an unrotated node, or a free rotated direction q_a, a >= k, of a rotated one), so that partially
    // supported assemblies (e.g. fix "z" on one region) still have candidates
    std::vector<std::pair<int, int>> pins;  // (node, slot: Cartesian component or rotated column) owned here
    std::vector<int> npins(ncomp, 0);
    for (int c = 0; c < ncomp; ++c) {
      const int kf = static_cast<int>(free_modes[c].cols());
      if (kf == 0) continue;
      Eigen::MatrixXd W = free_modes[c].transpose();  // kf x 6
      std::vector<std::pair<int, int>> cand;          // (node, slot)
      std::vector<std::array<double, 6>> cb;          // rigid mode k along the candidate's direction
      for (int i : owned_nodes) {
        if (!used[i] || ncomp_of[i] != c) continue;
        const NodeBasis& B = nb[i];
        double x[3], m[3];
        node_x(i, x);
        for (int a = 0; a < 3; ++a) {
          double dir[3] = {0, 0, 0};
          if (B.rotated) {
            if (a < B.k) continue;
            for (int cc = 0; cc < 3; ++cc) dir[cc] = B.Q[cc][a];
          } else {
            if (B.cart[a]) continue;
            dir[a] = 1;
          }
          std::array<double, 6> v;
          for (int k = 0; k < 6; ++k) {
            rigid_mode(k, x, cctr[c].data(), clen[c], m);
            v[k] = m[0] * dir[0] + m[1] * dir[1] + m[2] * dir[2];
          }
          cand.push_back({i, a});
          cb.push_back(v);
        }
      }
      for (int step = 0; step < kf; ++step) {
        double best = -1;
        int bj = -1, bn = -1;
        for (size_t n = 0; n < cand.size(); ++n)
          for (int j = step; j < kf; ++j) {
            double v = 0;
            for (int k = 0; k < 6; ++k) v += W(j, k) * cb[n][k];
            if (std::fabs(v) > best) {
              best = std::fabs(v);
              bj = j;
              bn = static_cast<int>(n);
            }
          }
        struct {
          double v;
          int r;
        } vr{best, P.rank};
        MPI_Allreduce(MPI_IN_PLACE, &vr, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        AXF_REQUIRE(vr.v > 1e-6, strcat("mechanics: the free rigid motions of the assembly (", join(comp_names[c]),
                                        ") could not be pinned"), "/mechanics",
                    "Add a support (mechanics.supports) to that assembly.");
        double brow[7] = {0, 0, 0, 0, 0, 0, 0};
        if (vr.r == P.rank) {
          for (int k = 0; k < 6; ++k) brow[k] = cb[bn][k];
          brow[6] = bj;
          pins.push_back(cand[bn]);
          cb[bn].fill(0.0);  // a direction is pinned once
        }
        MPI_Bcast(brow, 7, MPI_DOUBLE, vr.r, MPI_COMM_WORLD);
        const int jj = static_cast<int>(brow[6]);
        W.row(step).swap(W.row(jj));
        Eigen::Map<Eigen::VectorXd> b(brow, 6);
        const double piv = W.row(step).dot(b);
        for (int j = 0; j < kf; ++j)
          if (j != step) W.row(j) -= (W.row(j).dot(b) / piv) * W.row(step);
        ++npins[c];
      }
    }
    // essential true dofs (rotated basis): unused, constrained, pins
    mfem::Array<int> ess;
    {
      std::vector<char> is_ess(tsize, 0);
      for (int i : owned_nodes) {
        if (!used[i]) {
          for (int c = 0; c < 3; ++c) is_ess[td[i][c]] = 1;
          continue;
        }
        const NodeBasis& B = nb[i];
        if (B.rotated) {
          for (int a = 0; a < B.k; ++a) is_ess[td[i][a]] = 1;
        } else {
          for (int c = 0; c < 3; ++c)
            if (B.cart[c]) is_ess[td[i][c]] = 1;
        }
      }
      for (auto& [i, c] : pins) is_ess[td[i][c]] = 1;
      for (int t = 0; t < tsize; ++t)
        if (is_ess[t]) ess.Append(t);
    }

    // ---- load vector (local dofs), with the resultants integrated in the same pass
    mfem::Vector Fl(vsize);
    Fl = 0.0;
    double applied_F[3] = {0, 0, 0}, applied_M[3] = {0, 0, 0}, load_scale = 0;
    struct WForce {
      double F[3] = {0, 0, 0}, T[3] = {0, 0, 0}, Fext[3] = {0, 0, 0};
    };
    std::map<std::string, WForce> wforce;
    for (auto& w : loaded_windings) wforce[w];
    std::set<std::string> sc_loaded, sc_dropped;
    {
      mfem::Array<int> vd;
      mfem::Vector shape, Jv(3), Bv(3), x(3), elv;
      mfem::DenseMatrix dsh;
      for (int e = 0; e < ne; ++e) {
        const MRole& ro = roles[e];
        if (!ro.solid || (!ro.winding && ro.s0 == 0.0)) continue;
        const mfem::FiniteElement* fe = fes.GetFE(e);
        const int nd = fe->GetDof();
        fes.GetElementVDofs(e, vd);
        elv.SetSize(3 * nd);
        elv = 0.0;
        shape.SetSize(nd);
        dsh.SetSize(nd, 3);
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * order + 2);
        WForce* wf = ro.winding ? &wforce[sreg[P.loc2ser[e]]] : nullptr;
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(q);
          T->SetIntPoint(&ip);
          const double wq = ip.weight * T->Weight();
          if (ro.s0 != 0.0) {
            fe->CalcPhysDShape(*T, dsh);
            for (int k = 0; k < nd; ++k)
              for (int c = 0; c < 3; ++c) elv(k + c * nd) += wq * ro.s0 * dsh(k, c);
          }
          if (ro.winding) {
            T->Transform(ip, x);
            P.J_static->Eval(Jv, *T, ip);
            Bv = 0.0;
            if (P.A_lab) P.A_lab->GetCurl(*T, Bv);
            add_bg(x, T->Attribute, Bv);
            double Be[3] = {0, 0, 0};
            if (ext) {
              const auto b = ext->field_T({x(0), x(1), x(2)});
              for (int k = 0; k < 3; ++k) Be[k] = b[k];
            }
            const double Bt[3] = {Bv(0) + Be[0], Bv(1) + Be[1], Bv(2) + Be[2]};
            const double f[3] = {Jv(1) * Bt[2] - Jv(2) * Bt[1], Jv(2) * Bt[0] - Jv(0) * Bt[2],
                                 Jv(0) * Bt[1] - Jv(1) * Bt[0]};
            const double fe_[3] = {Jv(1) * Be[2] - Jv(2) * Be[1], Jv(2) * Be[0] - Jv(0) * Be[2],
                                   Jv(0) * Be[1] - Jv(1) * Be[0]};
            fe->CalcShape(ip, shape);
            for (int k = 0; k < nd; ++k)
              for (int c = 0; c < 3; ++c) elv(k + c * nd) += wq * shape(k) * f[c];
            for (int c = 0; c < 3; ++c) {
              wf->F[c] += wq * f[c];
              wf->Fext[c] += wq * fe_[c];
              applied_F[c] += wq * f[c];
              load_scale += wq * std::fabs(f[c]);
            }
            const double tq[3] = {x(1) * f[2] - x(2) * f[1], x(2) * f[0] - x(0) * f[2], x(0) * f[1] - x(1) * f[0]};
            for (int c = 0; c < 3; ++c) {
              wf->T[c] += wq * tq[c];
              applied_M[c] += wq * tq[c];
            }
          }
        }
        Fl.AddElementVector(vd, elv);
      }
      // ---- surface loads: Maxwell pressure on superconductors (field from the gathered serial view) and
      // prescribed pressures
      std::vector<long long> pcount(ms.pressures.size(), 0);
      for (size_t i = 0; i < ms.pressures.size(); ++i) {
        const auto& pl = ms.pressures[i];
        AXF_REQUIRE(solid_regions.count(pl.on) > 0, "pressure load region '" + pl.on + "' is not part of the structure",
                    strcat("/mechanics/pressures/", i, "/on"));
      }
      mfem::ParMesh& vm = P.vmesh();
      const mfem::ParGridFunction* vA = P.vA_lab();
      std::vector<FaceRef> fr;
      mfem::Vector nor(3), xf(3), Bq(3);
      double bn_rel = 0;
      for (int e = 0; e < ne; ++e) {
        if (!roles[e].solid) continue;
        const int s = P.loc2ser[e];
        element_faces(e, fr);
        for (const auto& f : fr) {
          if (f.s_other < 0) continue;  // the boundary of the mesh carries no load
          const int so = f.s_other;
          // the loads on this face of s add up: the Maxwell pressure of a superconductor and every prescribed
          // pressure whose (on, from) pair matches (they do not exclude each other)
          const bool maxwell = is_sc(s) && !is_sc(so);
          double pconst = 0;
          bool prescribed = false;
          for (size_t i = 0; i < ms.pressures.size(); ++i) {
            const auto& pl = ms.pressures[i];
            if (sreg[s] != pl.on) continue;
            const bool from = pl.from == "vacuum" ? !ssolid[so] : sreg[so] == pl.from;
            if (!from) continue;
            pconst += pl.p_Pa;
            prescribed = true;
            ++pcount[i];
          }
          if (!maxwell && !prescribed) continue;
          if (maxwell) sc_loaded.insert(sreg[s]);
          mfem::FaceElementTransformations* FT = sm.GetFaceElementTransformations(f.sf);
          const bool s_is_1 = FT->Elem1No == s;
          const double sgn = s_is_1 ? 1.0 : -1.0;  // CalcOrtho points from element 1 to element 2
          const mfem::FiniteElement* fe = fes.GetFE(e);
          const int nd = fe->GetDof();
          fes.GetElementVDofs(e, vd);
          elv.SetSize(3 * nd);
          elv = 0.0;
          shape.SetSize(nd);
          const mfem::IntegrationRule& irf = mfem::IntRules.Get(FT->GetGeometryType(), 2 * order + 2);
          for (int q = 0; q < irf.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& ip = irf.IntPoint(q);
            FT->SetAllIntPoints(&ip);
            mfem::CalcOrtho(FT->Jacobian(), nor);
            FT->Transform(ip, xf);
            const mfem::IntegrationPoint eip = s_is_1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
            const mfem::IntegrationPoint oip = s_is_1 ? FT->GetElement2IntPoint() : FT->GetElement1IntPoint();
            const double n3[3] = {sgn * nor(0), sgn * nor(1), sgn * nor(2)};  // outward, |n3| = area element
            double p = pconst;
            if (maxwell) {
              mfem::ElementTransformation* Te = vm.GetElementTransformation(so);
              Te->SetIntPoint(&oip);
              Bq = 0.0;
              if (vA) vA->GetCurl(*Te, Bq);
              add_bg(xf, sm.GetAttribute(so), Bq);
              p += (Bq * Bq) / (2 * mu0);
              const double nl = std::sqrt(n3[0] * n3[0] + n3[1] * n3[1] + n3[2] * n3[2]), Bm = Bq.Norml2();
              if (nl > 0 && Bm > 0)
                bn_rel = std::max(bn_rel, std::fabs(Bq(0) * n3[0] + Bq(1) * n3[1] + Bq(2) * n3[2]) / (nl * Bm));
            }
            fe->CalcShape(eip, shape);
            const double t3[3] = {-p * n3[0] * ip.weight, -p * n3[1] * ip.weight, -p * n3[2] * ip.weight};
            for (int k = 0; k < nd; ++k)
              for (int c = 0; c < 3; ++c) elv(k + c * nd) += shape(k) * t3[c];
            for (int c = 0; c < 3; ++c) {
              applied_F[c] += t3[c];
              load_scale += std::fabs(t3[c]);
            }
            applied_M[0] += xf(1) * t3[2] - xf(2) * t3[1];
            applied_M[1] += xf(2) * t3[0] - xf(0) * t3[2];
            applied_M[2] += xf(0) * t3[1] - xf(1) * t3[0];
          }
          Fl.AddElementVector(vd, elv);
        }
      }
      for (int s = 0; s < nser; ++s)
        if (is_sc(s) && !ssolid[s]) sc_dropped.insert(sreg[s]);
      if (!sc_dropped.empty())
        warnings.push_back("the magnetic surface pressure on superconductor region(s) " + join(sc_dropped) +
                           " is not applied: their material has no \"mechanical\" block");
      {
        std::vector<double> pc(pcount.begin(), pcount.end());
        gsum(pc.data(), static_cast<int>(pc.size()));
        for (size_t i = 0; i < ms.pressures.size(); ++i)
          if (pc[i] == 0)
            throw Error("pressure load on '" + ms.pressures[i].on + "' from '" + ms.pressures[i].from +
                            "' touches no mesh face",
                        strcat("/mechanics/pressures/", i),
                        "The two sides must share a boundary. Check the region names.");
      }
      bn_rel = gmax(bn_rel);
      R["mechanics"]["checks"]["maxwell_traction_Bn_over_B_max"] = bn_rel;
    }
    gsum(applied_F, 3);
    gsum(applied_M, 3);
    load_scale = gsum(load_scale);
    {  // superconductor names loaded on any rank
      std::vector<double> flag(nser, 0.0);
      for (int s = 0; s < nser; ++s) flag[s] = sc_loaded.count(sreg[s]) ? 1.0 : 0.0;
      gsum(flag.data(), nser);
      for (int s = 0; s < nser; ++s)
        if (flag[s] > 0) sc_loaded.insert(sreg[s]);
    }
    json lorentz = json::object();
    for (auto& [name, wf] : wforce) {
      gsum(wf.F, 3);
      gsum(wf.T, 3);
      gsum(wf.Fext, 3);
      lorentz[name] = {{"F_N", vec_json(wf.F)}, {"torque_Nm", vec_json(wf.T)}, {"F_external_field_N", vec_json(wf.Fext)}};
    }
    mfem::Vector F(tsize);
    reduce_to_true(Fl, F);
    double assembly_scale_pre = 0;  // L1 norm of the nodal loads (round-off floor of the rigid-mode load test)
    for (int i : owned_nodes)
      if (used[i])
        for (int c = 0; c < 3; ++c) assembly_scale_pre += std::fabs(F(td[i][c]));
    assembly_scale_pre = gsum(assembly_scale_pre);
    // ---- balance the load on the free rigid modes before the solve.  A discrete load is never exactly balanced
    // along a motion no constraint fixes (e.g. the transverse net Lorentz force of a winding on an asymmetric mesh),
    // and the pins would then carry that residual as point loads, which makes the stresses near the pins depend on
    // where they are (and on the MPI partition, which picks them).  The residual is removed as nodal loads
    // proportional to the free modes themselves (F <- F - sum_j c_j phi_j with phi_k . F = 0 afterwards), so the
    // pins carry nothing; the removed generalized load is reported (`rigid_modes`) and warned about when it is not
    // small against the assembly's own loads.
    std::vector<Eigen::VectorXd> free_load(ncomp);
    std::vector<double> cscale(ncomp, 0.0);
    {
      int tot = 0;
      for (auto& fm : free_modes) tot += static_cast<int>(fm.cols());
      if (tot > 0) {
        std::vector<double> gm(36 * ncomp, 0.0), hm(6 * ncomp, 0.0);
        for (int i : owned_nodes) {
          if (!used[i] || ncomp_of[i] < 0) continue;
          const int c = ncomp_of[i];
          if (free_modes[c].cols() == 0) continue;
          double x3[3], m[6][3];
          node_x(i, x3);
          for (int k = 0; k < 6; ++k) rigid_mode(k, x3, cctr[c].data(), clen[c], m[k]);
          for (int k = 0; k < 6; ++k) {
            hm[6 * c + k] += m[k][0] * F(td[i][0]) + m[k][1] * F(td[i][1]) + m[k][2] * F(td[i][2]);
            for (int l = 0; l < 6; ++l)
              gm[36 * c + 6 * k + l] += m[k][0] * m[l][0] + m[k][1] * m[l][1] + m[k][2] * m[l][2];
          }
          for (int cc = 0; cc < 3; ++cc) cscale[c] += std::fabs(F(td[i][cc]));
        }
        gsum(gm.data(), static_cast<int>(gm.size()));
        gsum(hm.data(), static_cast<int>(hm.size()));
        gsum(cscale.data(), ncomp);
        std::vector<Eigen::VectorXd> corr(ncomp);
        for (int c = 0; c < ncomp; ++c) {
          const Eigen::MatrixXd& V = free_modes[c];
          if (V.cols() == 0) continue;
          Eigen::MatrixXd G6(6, 6);
          Eigen::VectorXd h6(6);
          for (int k = 0; k < 6; ++k) {
            h6(k) = hm[6 * c + k];
            for (int l = 0; l < 6; ++l) G6(k, l) = gm[36 * c + 6 * k + l];
          }
          free_load[c] = V.transpose() * h6;                                   // net load on each free mode
          corr[c] = V * (V.transpose() * G6 * V).ldlt().solve(free_load[c]);  // in the 6-mode basis
        }
        for (int i : owned_nodes) {
          if (!used[i] || ncomp_of[i] < 0) continue;
          const int c = ncomp_of[i];
          if (free_modes[c].cols() == 0) continue;
          double x3[3], m[3];
          node_x(i, x3);
          for (int k = 0; k < 6; ++k) {
            rigid_mode(k, x3, cctr[c].data(), clen[c], m);
            for (int cc = 0; cc < 3; ++cc) F(td[i][cc]) -= corr[c](k) * m[cc];
          }
        }
      }
    }

    // ---- stiffness, rotation, solve
    PWCoefficient3 lamc(lam_e), muc(mu_e);
    mfem::ParBilinearForm a(&fes);
    {
      auto* ei = new mfem::ElasticityIntegrator(lamc, muc);
      ei->SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * order + 2));
      a.AddDomainIntegrator(ei);
    }
    a.Assemble();
    a.Finalize();
    mfem::SparseMatrix tdiag(tsize, tsize);  // outlives Tm, which does not own it
    std::unique_ptr<mfem::HypreParMatrix> K(a.ParallelAssemble()), Kt, Kel, Ke, Tm;
    mfem::Vector Ft(F);
    if (any_rot) {
      std::vector<char> done(tsize, 0);
      for (int i : owned_nodes) {
        if (!nb[i].rotated) continue;
        for (int c = 0; c < 3; ++c) {
          for (int aa = 0; aa < 3; ++aa) tdiag.Set(td[i][c], td[i][aa], nb[i].Q[c][aa]);
          done[td[i][c]] = 1;
        }
      }
      for (int t = 0; t < tsize; ++t)
        if (!done[t]) tdiag.Set(t, t, 1.0);
      tdiag.Finalize();
      Tm = std::make_unique<mfem::HypreParMatrix>(MPI_COMM_WORLD, fes.GlobalTrueVSize(), fes.GetTrueDofOffsets(),
                                                  &tdiag);
      Kt.reset(mfem::RAP(K.get(), Tm.get()));
      Kel.reset(mfem::RAP(K.get(), Tm.get()));
      Tm->MultTranspose(F, Ft);
    } else {
      Kel.reset(a.ParallelAssemble());
    }
    ensure_diagonal(Kel, ess);
    Ke.reset(Kel->EliminateRowsCols(ess));
    mfem::Vector X(tsize), Bv(Ft);
    X = 0.0;
    Kel->EliminateBC(*Ke, ess, X, Bv);
    mfem::HypreBoomerAMG amg(*Kel);
    amg.SetPrintLevel(0);
    if (any_rot) amg.SetSystemsOptions(3, false);
    // hypre's elasticity interpolation (rigid-body modes) has no GPU path: MFEM aborts in SetElasticityOptions when
    // hypre runs on a GPU, so the systems AMG alone there (docs/validation/gpu.md)
    else if (mfem::HypreUsingGPU()) amg.SetSystemsOptions(3, fes.GetOrdering() == mfem::Ordering::byNODES);
    else amg.SetElasticityOptions(&fes);
    log(1, "solving 3D linear elasticity (Lorentz, surface and cooldown loads)");
    json sdiag = solve_spd(*Kel, ess, Bv, X, amg, 1e-12, 20000);
    // displacement and reactions in the Cartesian basis
    mfem::Vector U(tsize), Rt(tsize), Rr(tsize);
    (Kt ? *Kt : *K).Mult(X, Rt);
    Rt -= Ft;
    if (Tm) {
      Tm->Mult(X, U);
      Tm->Mult(Rt, Rr);
    } else {
      U = X;
      Rr = Rt;
    }
    // project out the pinned rigid motions (volume-weighted), per assembly
    mfem::ParGridFunction u(&fes);
    u.SetFromTrueDofs(U);
    json rigid_json = json::array();
    {
      int tot = 0;
      for (auto& m : free_modes) tot += static_cast<int>(m.cols());
      if (tot > 0) {
        std::vector<int> off(ncomp + 1, 0);
        for (int c = 0; c < ncomp; ++c) off[c + 1] = off[c] + static_cast<int>(free_modes[c].cols());
        std::vector<double> gram6(36 * ncomp, 0.0), hu(6 * ncomp, 0.0);
        mfem::Vector uv(3), x(3);
        for (int e = 0; e < ne; ++e) {
          if (!roles[e].solid) continue;
          const int c = roles[e].comp;
          if (free_modes[c].cols() == 0) continue;
          mfem::ElementTransformation* T = pm.GetElementTransformation(e);
          const mfem::IntegrationRule& ir = mfem::IntRules.Get(T->GetGeometryType(), 2 * order + 2);
          for (int q = 0; q < ir.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            T->Transform(ip, x);
            u.GetVectorValue(*T, ip, uv);
            const double wq = ip.weight * T->Weight();
            double m[6][3];
            for (int k = 0; k < 6; ++k) rigid_mode(k, x.GetData(), cctr[c].data(), clen[c], m[k]);
            for (int k = 0; k < 6; ++k) {
              hu[6 * c + k] += wq * (m[k][0] * uv(0) + m[k][1] * uv(1) + m[k][2] * uv(2));
              for (int l = 0; l < 6; ++l)
                gram6[36 * c + 6 * k + l] += wq * (m[k][0] * m[l][0] + m[k][1] * m[l][1] + m[k][2] * m[l][2]);
            }
          }
        }
        gsum(gram6.data(), static_cast<int>(gram6.size()));
        gsum(hu.data(), static_cast<int>(hu.size()));
        std::vector<Eigen::VectorXd> coef(ncomp);
        double shift_max = 0, shift_con = 0;
        for (int c = 0; c < ncomp; ++c) {
          const Eigen::MatrixXd& V = free_modes[c];
          if (V.cols() == 0) continue;
          Eigen::MatrixXd G6(6, 6);
          Eigen::VectorXd h6(6);
          for (int k = 0; k < 6; ++k) {
            h6(k) = hu[6 * c + k];
            for (int l = 0; l < 6; ++l) G6(k, l) = gram6[36 * c + 6 * k + l];
          }
          const Eigen::MatrixXd Gk = V.transpose() * G6 * V;
          coef[c] = V * Gk.ldlt().solve(V.transpose() * h6);  // rigid motion in the 6-mode basis
          const Eigen::VectorXd& gl = free_load[c];            // net load on each free mode [N or N m / Lc]
          rigid_json.push_back({{"regions", std::vector<std::string>(comp_names[c].begin(), comp_names[c].end())},
                                {"free_rigid_modes", static_cast<int>(V.cols())},
                                {"pinned_dofs", npins[c]},
                                {"net_load_on_free_modes", std::vector<double>(gl.data(), gl.data() + gl.size())},
                                {"note", "this net load was removed before the solve (balanced by nodal loads along "
                                         "the free modes), so the pins carry nothing"}});
          const double gmag = gl.cwiseAbs().maxCoeff();
          // scaled by this assembly's own nodal loads (a global scale would let a heavily loaded,
          // balanced assembly hide an unbalanced small one), with a round-off floor
          if (gmag > 1e-3 * cscale[c] && gmag > 1e-12 * std::max({assembly_scale_pre, load_scale, 1e-30}))
            warnings.push_back(strcat("mechanics: the unsupported assembly (", join(comp_names[c]),
                                      ") carries a net load of ", gmag, " N (or N m per ", clen[c],
                                      " m) along a rigid motion that no constraint fixes, so it cannot be in "
                                      "equilibrium; the solver removed that net load before solving (it is not carried "
                                      "anywhere). Add a support in mechanics.supports or a symmetry plane."));
        }
        for (int i : owned_nodes) {
          if (!used[i] || ncomp_of[i] < 0) continue;
          const int c = ncomp_of[i];
          if (free_modes[c].cols() == 0) continue;
          double x3[3], m[3], sh[3] = {0, 0, 0};
          node_x(i, x3);
          for (int k = 0; k < 6; ++k) {
            rigid_mode(k, x3, cctr[c].data(), clen[c], m);
            for (int cc = 0; cc < 3; ++cc) sh[cc] += coef[c](k) * m[cc];
          }
          for (int cc = 0; cc < 3; ++cc) U(td[i][cc]) -= sh[cc];
          // the projected motion must not move a constrained direction (it is a free mode by construction)
          const double shn = std::sqrt(sh[0] * sh[0] + sh[1] * sh[1] + sh[2] * sh[2]);
          shift_max = std::max(shift_max, shn);
          if (!node_src[i].empty()) {
            const NodeBasis& B = nb[i];
            for (int a = 0; a < 3; ++a) {
              double d = 0;
              if (B.rotated) {
                if (a >= B.k) continue;
                for (int cc = 0; cc < 3; ++cc) d += sh[cc] * B.Q[cc][a];
              } else {
                if (!B.cart[a]) continue;
                d = sh[a];
              }
              shift_con = std::max(shift_con, std::fabs(d));
            }
          }
        }
        shift_max = gmax(shift_max);
        shift_con = gmax(shift_con);
        if (shift_con > 1e-8 * std::max(shift_max, 1e-300))
          warnings.push_back(strcat("mechanics: removing the free rigid motions moved a constrained direction by ",
                                    shift_con, " m (of ", shift_max, " m): a weakly constrained motion was taken as "
                                    "free; check the supports"));
        u.SetFromTrueDofs(U);
      }
    }
    // ---- reactions per source, pins, closure
    std::vector<std::array<double, 3>> src_R(nsrc, {0, 0, 0});
    std::vector<double> plane_comp(planes.size(), 0.0), plane_tens(planes.size(), 0.0);
    double pin_F[3] = {0, 0, 0}, total_R[3] = {0, 0, 0}, total_RM[3] = {0, 0, 0}, assembly_scale = 0;
    std::map<int, std::vector<int>> pin_slots;
    for (auto& [i, c] : pins) pin_slots[i].push_back(c);
    for (int i : owned_nodes) {
      if (!used[i]) continue;
      for (int c = 0; c < 3; ++c) assembly_scale += std::fabs(F(td[i][c]));
      auto ps = pin_slots.find(i);
      if (node_src[i].empty() && ps == pin_slots.end()) continue;
      double r[3] = {Rr(td[i][0]), Rr(td[i][1]), Rr(td[i][2])}, x[3];
      node_x(i, x);
      for (int c = 0; c < 3; ++c) total_R[c] += r[c];
      total_RM[0] += x[1] * r[2] - x[2] * r[1];
      total_RM[1] += x[2] * r[0] - x[0] * r[2];
      total_RM[2] += x[0] * r[1] - x[1] * r[0];
      if (ps != pin_slots.end()) {  // the pinned (otherwise free) directions carry the pin reaction
        for (int slot : ps->second) {
          double dir[3] = {0, 0, 0};
          if (nb[i].rotated)
            for (int c = 0; c < 3; ++c) dir[c] = nb[i].Q[c][slot];
          else
            dir[slot] = 1;
          const double d = r[0] * dir[0] + r[1] * dir[1] + r[2] * dir[2];
          for (int c = 0; c < 3; ++c) {
            pin_F[c] += d * dir[c];
            r[c] -= d * dir[c];
          }
        }
      }
      if (node_src[i].empty()) continue;
      // the rest is a combination of the constraint normals of all sources at this node: the minimum-norm
      // coefficients (order independent; unique when the normals are independent), each source taking its own
      // columns (sequential projections would lose force for non-orthogonal normals)
      std::vector<int> col_src;
      std::vector<Vec3> cols;
      for (size_t kk = 0; kk < node_src[i].size(); ++kk)
        for (auto& v : sources[node_src[i][kk]].normals) {
          cols.push_back(v);
          col_src.push_back(static_cast<int>(kk));
        }
      Eigen::MatrixXd Nm(3, cols.size());
      for (size_t k = 0; k < cols.size(); ++k)
        for (int c = 0; c < 3; ++c) Nm(c, static_cast<int>(k)) = cols[k][c];
      const Eigen::Vector3d rv(r[0], r[1], r[2]);
      const Eigen::VectorXd coef = Nm.completeOrthogonalDecomposition().solve(rv);
      for (size_t k = 0; k < cols.size(); ++k) {
        const int kk = col_src[k], j = node_src[i][kk];
        for (int c = 0; c < 3; ++c) src_R[j][c] += coef(static_cast<int>(k)) * cols[k][c];
        if (sources[j].plane >= 0) {
          const double rn = coef(static_cast<int>(k)) * node_sgn[i][kk];  // along the outward normal
          if (rn < 0) plane_comp[sources[j].plane] += -rn;                 // pushes the structure inward
          else plane_tens[sources[j].plane] += rn;
        }
      }
    }
    for (auto& v : src_R) gsum(v.data(), 3);
    gsum(plane_comp.data(), static_cast<int>(plane_comp.size()));
    gsum(plane_tens.data(), static_cast<int>(plane_tens.size()));
    gsum(pin_F, 3);
    gsum(total_R, 3);
    gsum(total_RM, 3);
    assembly_scale = gsum(assembly_scale);
    // the discrete applied resultants (sum of nodal loads; equals the integrated resultant up to round-off)
    double nodal_F[3] = {0, 0, 0}, nodal_M[3] = {0, 0, 0};
    for (int i : owned_nodes) {
      if (!used[i]) continue;
      double x[3];
      node_x(i, x);
      const double f[3] = {F(td[i][0]), F(td[i][1]), F(td[i][2])};
      for (int c = 0; c < 3; ++c) nodal_F[c] += f[c];
      nodal_M[0] += x[1] * f[2] - x[2] * f[1];
      nodal_M[1] += x[2] * f[0] - x[0] * f[2];
      nodal_M[2] += x[0] * f[1] - x[1] * f[0];
    }
    gsum(nodal_F, 3);
    gsum(nodal_M, 3);
    double closure_F = 0, closure_M = 0;
    for (int c = 0; c < 3; ++c) {
      closure_F = std::max(closure_F, std::fabs(nodal_F[c] + total_R[c]));
      closure_M = std::max(closure_M, std::fabs(nodal_M[c] + total_RM[c]));
    }

    // ---- per-region extremes on a lattice of points in every element
    std::vector<std::string> rnames(solid_regions.begin(), solid_regions.end());
    std::map<std::string, int> rid;
    for (size_t i = 0; i < rnames.size(); ++i) rid[rnames[i]] = static_cast<int>(i);
    const int nr = static_cast<int>(rnames.size());
    std::vector<Peak> umax(nr), vmmax(nr);
    std::vector<Range> hoop(nr), srr(nr), szz(nr), sprin(nr);
    auto stress_at = [&](int e, mfem::ElementTransformation& T, double* s6) {
      mfem::DenseMatrix G(3);
      u.GetVectorGradient(T, G);
      const MRole& ro = roles[e];
      double eps[3][3];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) eps[i][j] = 0.5 * (G(i, j) + G(j, i));
      const double tr = eps[0][0] + eps[1][1] + eps[2][2];
      s6[0] = ro.lambda * tr + 2 * ro.mu * eps[0][0] - ro.s0;
      s6[1] = ro.lambda * tr + 2 * ro.mu * eps[1][1] - ro.s0;
      s6[2] = ro.lambda * tr + 2 * ro.mu * eps[2][2] - ro.s0;
      s6[3] = 2 * ro.mu * eps[1][2];  // yz
      s6[4] = 2 * ro.mu * eps[0][2];  // xz
      s6[5] = 2 * ro.mu * eps[0][1];  // xy
    };
    auto von_mises = [](const double* s) {
      const double a = s[0] - s[1], b = s[1] - s[2], c = s[2] - s[0];
      return std::sqrt(0.5 * (a * a + b * b + c * c) + 3 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
    };
    auto principal = [](const double* s, double& lo, double& hi) {
      Eigen::Matrix3d S;
      S << s[0], s[5], s[4], s[5], s[1], s[3], s[4], s[3], s[2];
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(S, Eigen::EigenvaluesOnly);
      lo = es.eigenvalues()(0);
      hi = es.eigenvalues()(2);
    };
    {
      const mfem::IntegrationRule& lat = tet_lattice(2 * order);
      mfem::Vector uv(3), x(3);
      for (int e = 0; e < ne; ++e) {
        if (!roles[e].solid) continue;
        AXF_REQUIRE(pm.GetElementBaseGeometry(e) == mfem::Geometry::TETRAHEDRON,
                    "the 3D mechanics expects a tetrahedral mesh");
        const int r = rid.at(sreg[P.loc2ser[e]]);
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        for (int q = 0; q < lat.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = lat.IntPoint(q);
          T->SetIntPoint(&ip);
          T->Transform(ip, x);
          u.GetVectorValue(*T, ip, uv);
          double s6[6];
          stress_at(e, *T, s6);
          umax[r].add(uv.Norml2(), x.GetData());
          vmmax[r].add(von_mises(s6), x.GetData());
          double lo, hi;
          principal(s6, lo, hi);
          sprin[r].add(lo);
          sprin[r].add(hi);
          szz[r].add(s6[2]);
          const double rho = std::hypot(x(0), x(1));
          if (rho > 1e-9 * Lbox) {  // cylindrical components about the z axis
            const double cs = x(0) / rho, sn = x(1) / rho;
            srr[r].add(cs * cs * s6[0] + sn * sn * s6[1] + 2 * cs * sn * s6[5]);
            hoop[r].add(sn * sn * s6[0] + cs * cs * s6[1] - 2 * cs * sn * s6[5]);
          }
        }
      }
    }
    reduce_peaks(umax, P.rank);
    reduce_peaks(vmmax, P.rank);
    reduce_ranges(hoop);
    reduce_ranges(srr);
    reduce_ranges(szz);
    reduce_ranges(sprin);
    json rj = json::object();
    for (int r = 0; r < nr; ++r) {
      const spec::Region* g = d_.find_region(rnames[r]);
      if (!g) continue;
      const spec::Material& mt = d_.materials.at(g->material);
      rj[rnames[r]] = {{"u_max_m", umax[r].value},
                       {"u_max_location", umax[r].loc()},
                       {"von_mises_max_Pa", vmmax[r].value},
                       {"von_mises_max_location", vmmax[r].loc()},
                       {"sigma_hoop_Pa", hoop[r].to_json()},
                       {"sigma_rr_Pa", srr[r].to_json()},
                       {"sigma_zz_Pa", szz[r].to_json()},
                       {"sigma_principal_Pa", sprin[r].to_json()},
                       {"E_Pa", mt.mech.E_Pa},
                       {"nu", mt.mech.nu},
                       {"thermal_strain", (ms.cooldown && !mt.warm) ? -mt.mech.alpha : 0.0}};
    }
    // ---- probes
    json pj = json::array();
    if (!mprobes.empty()) {
      const int np = static_cast<int>(mprobes.size());
      Located Lc = locate(pm, mprobes, P.rank, P.nranks);
      const int W = 16;  // found, solid, region attr, u[3], sigma[6], B[3], B_ext? (kept below)
      std::vector<double> buf(static_cast<size_t>(W) * np, 0.0);
      mfem::Vector uv(3), Bq(3), x(3);
      for (int i = 0; i < np; ++i) {
        const int e = Lc.elem[i];
        if (e < 0) continue;
        double* b = &buf[static_cast<size_t>(W) * i];
        b[0] = 1;
        b[2] = pm.GetAttribute(e);
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        T->SetIntPoint(&Lc.ips[i]);
        T->Transform(Lc.ips[i], x);
        Bq = 0.0;
        if (P.A_lab) P.A_lab->GetCurl(*T, Bq);
        add_bg(x, T->Attribute, Bq);
        for (int k = 0; k < 3; ++k) b[12 + k] = Bq(k);
        if (roles[e].solid) {
          b[1] = 1;
          u.GetVectorValue(*T, Lc.ips[i], uv);
          for (int k = 0; k < 3; ++k) b[3 + k] = uv(k);
          stress_at(e, *T, b + 6);
        }
      }
      gsum(buf.data(), static_cast<int>(buf.size()));
      const double nan = std::numeric_limits<double>::quiet_NaN();
      for (int i = 0; i < np; ++i) {
        const double* b = &buf[static_cast<size_t>(W) * i];
        json q = {{"x_m", json::array({mprobes[i][0], mprobes[i][1], mprobes[i][2]})}};
        const bool found = b[0] > 0, solid = b[1] > 0;
        const int attr = static_cast<int>(std::lround(b[2]));
        q["region"] = found ? (P.region_of_attr.count(attr) ? P.region_of_attr.at(attr) : std::string("air")) : "";
        q["in_mesh"] = found;
        q["in_structure"] = solid;
        auto val = [&](double v) { return solid ? v : nan; };
        q["u_m"] = json::array({val(b[3]), val(b[4]), val(b[5])});
        q["sigma_Pa"] = json::array({val(b[6]), val(b[7]), val(b[8]), val(b[9]), val(b[10]), val(b[11])});
        q["von_mises_Pa"] = solid ? von_mises(b + 6) : nan;
        q["B_T"] = found ? json::array({b[12], b[13], b[14]}) : json::array({nan, nan, nan});
        if (ext) {
          const auto be = ext->field_T(mprobes[i]);
          q["B_external_T"] = json::array({be[0], be[1], be[2]});
        }
        pj.push_back(q);
      }
    }
    // ---- results
    json sj = json::array();
    for (int j = 0; j < nsup; ++j)
      sj.push_back({{"region", ms.supports[j].region}, {"fix", ms.supports[j].fix},
                    {"reaction_F_N", vec_json(src_R[j].data())}});
    json plj = json::array();
    for (size_t j = 0; j < planes.size(); ++j)
      plj.push_back({{"name", planes[j].name}, {"origin", planes[j].origin},
                     {"point_m", vec_json(planes[j].p)}, {"normal", vec_json(planes[j].n)},
                     {"boundary_faces", plane_faces[j]},
                     {"reaction_F_N", vec_json(src_R[nsup + j].data())},
                     {"reaction_normal_compressive_N", plane_comp[j]},
                     {"reaction_normal_tensile_N", plane_tens[j]}});
    json checks = R["mechanics"].value("checks", json::object());
    checks.update({{"equilibrium_F_residual_N", closure_F},
                   {"equilibrium_F_round_off", closure_F / std::max(assembly_scale, 1e-30)},
                   {"equilibrium_M_residual_Nm", closure_M},
                   {"applied_F_N", vec_json(applied_F)},
                   {"applied_M_Nm", vec_json(applied_M)},
                   {"applied_F_nodal_N", vec_json(nodal_F)},
                   {"reaction_total_F_N", vec_json(total_R)},
                   {"load_scale_N", load_scale},
                   {"assembly_scale_N", assembly_scale},
                   {"rigid_pin_F_N", vec_json(pin_F)},
                   {"maxwell_traction_note",
                    "the magnetic surface load is the pressure part of the Maxwell traction, t = -B^2 n/(2 mu0); the "
                    "full traction has an extra (B.n) B/mu0, which vanishes on an ideal superconductor. "
                    "maxwell_traction_Bn_over_B_max is the largest |B.n|/|B| of the discrete field on those faces"},
                   {"note", "force and moment equilibrium: the sum of the nodal loads plus the reactions (supports, "
                            "symmetry planes, rigid-mode pins) vanishes identically for the discrete system, so the "
                            "residuals measure the linear solve and round-off. equilibrium_F_round_off divides by "
                            "assembly_scale_N (the L1 norm of the nodal load vector, meaningful for a self-balanced "
                            "cooldown prestress). applied_F_N / applied_M_Nm are the integrated load resultants "
                            "(moments about the origin)"}});
    sdiag.update({{"dofs", fes.GlobalTrueVSize()},
                  {"essential_dofs", gsuml(ess.Size())},
                  {"constrained_nodes", gsuml(n_con)},
                  {"rotated_nodes", gsuml(n_rot)},
                  {"components", ncomp},
                  {"rigid_mode_pins", static_cast<long long>(std::accumulate(npins.begin(), npins.end(), 0))},
                  {"solid_elements", gsuml(std::count_if(roles.begin(), roles.end(), [](const MRole& r) { return r.solid; }))},
                  {"order", order},
                  {"amg", any_rot                 ? "systems (rotated constraint basis)"
                          : mfem::HypreUsingGPU() ? "systems (hypre on a GPU: no rigid-body-mode interpolation)"
                                                  : "elasticity (rigid-body modes)"},
                  {"mpi_ranks", P.nranks},
                  {"seconds", timer.seconds()}});
    R["mechanics"] = {{"cooldown", ms.cooldown},
                      {"regions", rj},
                      {"supports", sj},
                      {"symmetry_planes", plj},
                      {"lorentz_windings", std::vector<std::string>(loaded_windings.begin(), loaded_windings.end())},
                      {"lorentz_forces", lorentz},
                      {"external_field", ext_json},
                      {"magnetic_pressure_bodies", std::vector<std::string>(sc_loaded.begin(), sc_loaded.end())},
                      {"prescribed_pressures", static_cast<int>(ms.pressures.size())},
                      {"rigid_modes", rigid_json},
                      {"checks", checks},
                      {"probes", pj},
                      {"contact", "touching regions share mesh nodes and are bonded (no contact or sliding); "
                                  "symmetry planes are frictionless (u.n = 0 only)"},
                      {"solver", sdiag}};
    if (!(sdiag.value("relative_residual", 1.0) < 1e-8))
      warnings.push_back(strcat("mechanics: the elasticity solve stopped at a relative residual of ",
                                sdiag.value("relative_residual", 1.0), " on the free degrees of freedom; the stresses "
                                "are not reliable"));
    if (closure_F > 1e-6 * std::max(load_scale, 1e-30) && closure_F > 1e-9 * std::max(assembly_scale, 1e-30))
      warnings.push_back(strcat("mechanics: force equilibrium does not close (residual ", closure_F,
                                " N); the stress results are suspect"));
    if (ms.supports.empty() && planes.empty())
      warnings.push_back("mechanics: no supports or symmetry planes were given; each assembly is free and its rigid "
                         "motion is reported relative to the volume-weighted rigid-body projection (no reactions)");
  }

  // ============================================================ conduction
  if (d_.thermal.enabled) {
    const auto& ts = d_.thermal;
    std::vector<std::unique_ptr<axi::KTable>> ktables;
    std::map<std::string, const axi::KTable*> table_of;
    for (auto& [name, mt] : d_.materials) {
      if (!mt.therm.given || mt.therm.kT_T.empty()) continue;
      auto t = std::make_unique<axi::KTable>();
      t->T = mt.therm.kT_T;
      t->k = mt.therm.kT_k;
      table_of[name] = t.get();
      ktables.push_back(std::move(t));
    }
    std::vector<char> sdom(nser, 0);
    std::set<std::string> skipped, domain;
    for (int s = 0; s < nser; ++s) {
      if (smat[s]->therm.given) {
        sdom[s] = 1;
        domain.insert(sreg[s]);
      } else if (sreg[s] != "air" && smat[s]->kind != spec::MaterialKind::Vacuum) {
        skipped.insert(sreg[s]);
      }
    }
    missing_data("thermal", skipped);
    AXF_REQUIRE(!domain.empty(), "the thermal analysis has no conducting region", "/thermal",
                "Every region is vacuum-like or lacks a \"thermal\" material block.");
    std::vector<int> sfixed(nser, -1);
    json fj = json::array();
    for (size_t i = 0; i < ts.fixed.size(); ++i) {
      const auto& f = ts.fixed[i];
      AXF_REQUIRE(f.curve.empty(), "3D designs have no curves: prescribe temperatures on regions",
                  strcat("/thermal/fixed/", i, "/curve"));
      if (!domain.count(f.region))
        throw Error("fixed-temperature region '" + f.region + "' is not part of the thermal domain",
                    strcat("/thermal/fixed/", i, "/region"),
                    "Only regions whose material has a \"thermal\" block conduct heat.");
      for (int s = 0; s < nser; ++s)
        if (sdom[s] && sreg[s] == f.region) sfixed[s] = static_cast<int>(i);
      for (size_t j = 0; j < i; ++j)  // a second entry on the same region must not silently overwrite the first
        if (ts.fixed[j].region == f.region)
          throw Error("region '" + f.region + "' already has a prescribed temperature (entry " + std::to_string(j) + ")",
                      strcat("/thermal/fixed/", i, "/region"), "Give each region one prescribed temperature.");
      fj.push_back({{"region", f.region}, {"T_K", f.T_K}});
    }
    // every connected part needs a prescribed temperature
    {
      int nc = 0;
      std::vector<int> comp = serial_components(sm, [&](int s) { return sdom[s] != 0; }, nc);
      std::vector<char> has(nc, 0);
      for (int s = 0; s < nser; ++s)
        if (comp[s] >= 0 && sfixed[s] >= 0) has[comp[s]] = 1;
      for (int c = 0; c < nc; ++c) {
        if (has[c]) continue;
        std::set<std::string> names;
        for (int s = 0; s < nser; ++s)
          if (comp[s] == c) names.insert(sreg[s]);
        throw Error("the conducting part made of region(s) " + join(names) + " has no prescribed temperature",
                    "/thermal/fixed",
                    "It is thermally isolated from every heat sink, so its temperature level is undetermined. Add a "
                    "{\"region\", \"T_K\"} entry that touches it, connect it to one, or give its material no "
                    "\"thermal\" block to keep it out of the analysis.");
      }
    }
    std::vector<TRole> roles(ne);
    std::map<std::string, double> region_vol;
    for (int e = 0; e < ne; ++e) {
      const int s = P.loc2ser[e];
      if (!sdom[s]) continue;
      TRole& ro = roles[e];
      ro.domain = true;
      ro.k = smat[s]->therm.k_const;
      auto it = table_of.find(smat[s]->name);
      if (it != table_of.end()) ro.kt = it->second;
      ro.fixed = sfixed[s];
      mfem::ElementTransformation* T = pm.GetElementTransformation(e);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(T->GetGeometryType(), 2 * order);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        T->SetIntPoint(&ir.IntPoint(q));
        region_vol[sreg[s]] += ir.IntPoint(q).weight * T->Weight();
      }
    }
    {
      std::vector<std::string> names(domain.begin(), domain.end());
      std::vector<double> v(names.size());
      for (size_t i = 0; i < names.size(); ++i) v[i] = region_vol[names[i]];
      gsum(v.data(), static_cast<int>(v.size()));
      for (size_t i = 0; i < names.size(); ++i) region_vol[names[i]] = v[i];
    }
    json lj = json::array();
    std::map<std::string, double> q_of_region;
    for (size_t i = 0; i < ts.loads.size(); ++i) {
      const auto& l = ts.loads[i];
      if (!domain.count(l.region))
        throw Error("heat load region '" + l.region + "' is not part of the thermal domain",
                    strcat("/thermal/loads/", i, "/region"),
                    "Only regions whose material has a \"thermal\" block conduct heat.");
      const double V = region_vol[l.region];
      AXF_REQUIRE(V > 0, "heat load region '" + l.region + "' has zero volume", strcat("/thermal/loads/", i));
      const double q = l.total ? l.W / V : l.W_m3;
      q_of_region[l.region] += q;
      lj.push_back({{"region", l.region}, {"W_m3", q}, {"W", q * V}, {"volume_m3", V}});
      bool all_fixed = true;
      for (int s = 0; s < nser; ++s)
        if (sdom[s] && sreg[s] == l.region && sfixed[s] < 0) all_fixed = false;
      if (all_fixed)
        warnings.push_back("thermal: the heat load on '" + l.region +
                           "' sits in a region whose temperature is prescribed everywhere, so it is absorbed by the "
                           "boundary condition and changes nothing");
    }
    for (int e = 0; e < ne; ++e)
      if (roles[e].domain && q_of_region.count(sreg[P.loc2ser[e]])) roles[e].q = q_of_region[sreg[P.loc2ser[e]]];

    mfem::H1_FECollection fec(order, 3);
    mfem::ParFiniteElementSpace fes(&pm, &fec);
    const int nl = fes.GetVSize(), tsize = fes.GetTrueVSize();
    const mfem::Operator* Pr = fes.GetProlongationMatrix();
    // used and fixed true dofs (per fixed entry, to detect two different temperatures on one node)
    const int nfix = static_cast<int>(ts.fixed.size());
    std::vector<mfem::Vector> fmark(nfix + 1);
    for (auto& v : fmark) {
      v.SetSize(nl);
      v = 0.0;
    }
    {
      mfem::Array<int> dofs;
      for (int e = 0; e < ne; ++e) {
        if (!roles[e].domain) continue;
        fes.GetElementDofs(e, dofs);
        for (int i : dofs) {
          fmark[nfix](i) = 1.0;
          if (roles[e].fixed >= 0) fmark[roles[e].fixed](i) = 1.0;
        }
      }
    }
    std::vector<mfem::Vector> ftrue(nfix + 1);
    for (int j = 0; j <= nfix; ++j) {
      ftrue[j].SetSize(tsize);
      Pr->MultTranspose(fmark[j], ftrue[j]);
    }
    mfem::Array<int> ess;
    mfem::Vector Tfix(tsize);
    Tfix = 0.0;
    std::vector<char> is_fixed(tsize, 0);
    std::vector<int> fix_of(tsize, -1);  // the prescribed-temperature entry of a fixed true dof
    double conflict = 0;
    int ca = -1, cb = -1;
    for (int t = 0; t < tsize; ++t) {
      int owner = -1;
      for (int j = 0; j < nfix; ++j) {
        if (!(ftrue[j](t) > 0)) continue;
        if (owner >= 0 && std::fabs(ts.fixed[owner].T_K - ts.fixed[j].T_K) >
                              1e-12 * std::max({1.0, std::fabs(ts.fixed[owner].T_K), std::fabs(ts.fixed[j].T_K)})) {
          conflict = 1;
          ca = owner;
          cb = j;
        }
        owner = j;
      }
      if (!(ftrue[nfix](t) > 0)) {
        ess.Append(t);  // outside the thermal domain
      } else if (owner >= 0) {
        ess.Append(t);
        is_fixed[t] = 1;
        fix_of[t] = owner;
        Tfix(t) = ts.fixed[owner].T_K;
      }
    }
    {
      double c3[3] = {conflict, conflict > 0 ? double(ca) : 0.0, conflict > 0 ? double(cb) : 0.0};
      MPI_Allreduce(MPI_IN_PLACE, c3, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (c3[0] > 0) {
        // the pair is reported from the lowest rank that saw one
        int r = conflict > 0 ? P.rank : P.nranks;
        MPI_Allreduce(MPI_IN_PLACE, &r, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        int pair[2] = {ca, cb};
        MPI_Bcast(pair, 2, MPI_INT, r, MPI_COMM_WORLD);
        throw Error(strcat("conflicting prescribed temperatures: '", ts.fixed[pair[0]].region, "' asks for ",
                           ts.fixed[pair[0]].T_K, " K and '", ts.fixed[pair[1]].region, "' for ",
                           ts.fixed[pair[1]].T_K, " K at the same mesh node"),
                    "/thermal/fixed",
                    "Two prescribed-temperature regions touch, so the temperature there is ambiguous. Separate them "
                    "with a conducting region, or give them the same temperature.");
      }
    }
    // right-hand side and generated power
    QCoefficient3 qc(roles);
    mfem::ParLinearForm b(&fes);
    {
      auto* di = new mfem::DomainLFIntegrator(qc);
      di->SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * order + 2));
      b.AddDomainIntegrator(di);
    }
    b.Assemble();
    mfem::Vector Fq(tsize);
    b.ParallelAssemble(Fq);
    double generated = 0;
    for (int t = 0; t < tsize; ++t) generated += Fq(t);
    generated = gsum(generated);
    const bool nonlinear = std::any_of(roles.begin(), roles.end(),
                                       [](const TRole& r) { return r.domain && r.kt && !r.kt->empty(); });
    const bool nonlinear_g = gmax(nonlinear ? 1.0 : 0.0) > 0;
    double Tm = 0;
    for (auto& f : ts.fixed) Tm += f.T_K / ts.fixed.size();
    mfem::ParGridFunction Tg(&fes);
    mfem::Vector X(tsize);
    for (int t = 0; t < tsize; ++t) X(t) = is_fixed[t] ? Tfix(t) : Tm;
    Tg.SetFromTrueDofs(X);
    const double tol = 1e-10;
    const int max_picard = 100;
    int it = 0;
    double change = 0;
    bool converged = !nonlinear_g;
    json sdiag;
    std::unique_ptr<mfem::HypreParMatrix> K;
    log(1, "solving 3D steady heat conduction");
    for (; it < (nonlinear_g ? max_picard : 1); ++it) {
      KCoefficient3 kc(roles, Tg);
      mfem::ParBilinearForm a(&fes);
      auto* di = new mfem::DiffusionIntegrator(kc);
      di->SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * order + 2));
      a.AddDomainIntegrator(di);
      a.Assemble();
      a.Finalize();
      K.reset(a.ParallelAssemble());
      std::unique_ptr<mfem::HypreParMatrix> Kel(a.ParallelAssemble());
      ensure_diagonal(Kel, ess);  // the nodes outside the thermal domain, where k vanishes
      std::unique_ptr<mfem::HypreParMatrix> Ke(Kel->EliminateRowsCols(ess));
      mfem::Vector Xn(tsize), Bv(Fq);
      for (int t = 0; t < tsize; ++t) Xn(t) = is_fixed[t] ? Tfix(t) : X(t);
      for (int t : ess)
        if (!is_fixed[t]) Xn(t) = 0.0;
      Kel->EliminateBC(*Ke, ess, Xn, Bv);
      mfem::HypreBoomerAMG amg(*Kel);
      amg.SetPrintLevel(0);
      sdiag = solve_spd(*Kel, ess, Bv, Xn, amg, 1e-12, 5000);
      double span = 0, dmax = 0;
      for (int t = 0; t < tsize; ++t) {
        if (!(ftrue[nfix](t) > 0)) continue;
        span = std::max(span, std::fabs(Xn(t)));
        dmax = std::max(dmax, std::fabs(Xn(t) - X(t)));
      }
      change = gmax(dmax) / std::max(gmax(span), 1e-300);
      X = Xn;
      Tg.SetFromTrueDofs(X);
      if (!nonlinear_g) break;
      if (change < tol) {
        converged = true;
        ++it;
        break;
      }
    }
    if (!nonlinear_g) it = 1;
    // heat balance at the converged temperature (reassembled with k(T_final))
    double removed = 0, added = 0, net = 0, final_res = 0;
    {
      KCoefficient3 kc(roles, Tg);
      mfem::ParBilinearForm a(&fes);
      auto* di = new mfem::DiffusionIntegrator(kc);
      di->SetIntRule(&mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * order + 2));
      a.AddDomainIntegrator(di);
      a.Assemble();
      a.Finalize();
      K.reset(a.ParallelAssemble());
      mfem::Vector Rv(tsize);
      K->Mult(X, Rv);
      Rv -= Fq;
      // heat through each prescribed-temperature entry = minus the sum of its reactions. Individual nodal reactions
      // of P2 elements are not sign definite (vertex shape integrals are negative), so the sign split is done per
      // entry, not per node.
      // Entries that share nodes (touching sinks at one temperature) form one sink group: the heat through a shared
      // node cannot be attributed to either entry, so heat is accounted per group.
      std::vector<double> adj(static_cast<size_t>(nfix) * nfix, 0.0);
      for (int t = 0; t < tsize; ++t) {
        if (!is_fixed[t]) continue;
        for (int a = 0; a < nfix; ++a) {
          if (!(ftrue[a](t) > 0)) continue;
          for (int b = a + 1; b < nfix; ++b)
            if (ftrue[b](t) > 0) adj[static_cast<size_t>(a) * nfix + b] = 1.0;
        }
      }
      gsum(adj.data(), static_cast<int>(adj.size()));
      std::vector<int> grp(nfix);
      std::iota(grp.begin(), grp.end(), 0);
      std::function<int(int)> gfind = [&](int a) { return grp[a] == a ? a : grp[a] = gfind(grp[a]); };
      for (int a = 0; a < nfix; ++a)
        for (int b = a + 1; b < nfix; ++b)
          if (adj[static_cast<size_t>(a) * nfix + b] > 0) grp[std::max(gfind(a), gfind(b))] = std::min(gfind(a), gfind(b));
      std::vector<double> qg(nfix, 0.0), dscale(1, 0.0);
      double fr2 = 0, fq2 = 0;
      const mfem::Vector Kd = stored_diagonal(*K);   // not GetDiag: rows outside the domain may be empty on a GPU
      for (int t = 0; t < tsize; ++t) {
        if (is_fixed[t]) {
          net += Rv(t);
          qg[gfind(fix_of[t])] += -Rv(t);
        } else if (ftrue[nfix](t) > 0) {
          fr2 += Rv(t) * Rv(t);
          fq2 += Fq(t) * Fq(t);
        }
        if (ftrue[nfix](t) > 0) dscale[0] += (Kd(t) * X(t)) * (Kd(t) * X(t));
      }
      net = gsum(net);
      gsum(qg.data(), nfix);
      fr2 = gsum(fr2);
      fq2 = gsum(fq2);
      dscale[0] = gsum(dscale[0]);
      for (int j = 0; j < nfix; ++j) {
        if (gfind(j) == j) {
          if (qg[j] > 0) removed += qg[j];
          else added += -qg[j];
        }
        fj[j]["sink_group"] = gfind(j);
        fj[j]["heat_absorbed_W"] = qg[gfind(j)];  // of the whole group
      }
      // free-row residual of the final temperature with k(T_final), relative to the load or the sink scale
      // scaled also by the size of the individual terms (|diag(K) T|), so an unloaded isothermal problem does not
      // divide round-off by round-off
      final_res = std::sqrt(fr2) / std::max({std::sqrt(fq2), removed + added, 1e-12 * std::sqrt(dscale[0]), 1e-300});
    }
    const double outflow = -net;
    const double balance = std::fabs(generated - outflow) / std::max({std::fabs(generated), removed, added, 1e-300});
    // per-region extremes
    std::vector<std::string> rnames(domain.begin(), domain.end());
    std::map<std::string, int> rid;
    for (size_t i = 0; i < rnames.size(); ++i) rid[rnames[i]] = static_cast<int>(i);
    std::vector<Range> tr(rnames.size());
    Range tall;
    {
      const mfem::IntegrationRule& lat = tet_lattice(2 * order);
      for (int e = 0; e < ne; ++e) {
        if (!roles[e].domain) continue;
        const int r = rid.at(sreg[P.loc2ser[e]]);
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        for (int q = 0; q < lat.GetNPoints(); ++q) {
          T->SetIntPoint(&lat.IntPoint(q));
          const double t = Tg.GetValue(*T, lat.IntPoint(q));
          tr[r].add(t);
          tall.add(t);
        }
      }
    }
    reduce_ranges(tr);
    {
      std::vector<Range> v{tall};
      reduce_ranges(v);
      tall = v[0];
    }
    json rj = json::object();
    for (size_t r = 0; r < rnames.size(); ++r) {
      const spec::Region* g = d_.find_region(rnames[r]);
      if (!g) continue;
      const spec::Material& mt = d_.materials.at(g->material);
      json e = {{"T_min_K", tr[r].lo}, {"T_max_K", tr[r].hi}, {"volume_m3", region_vol[rnames[r]]}};
      if (mt.therm.kT_T.empty()) {
        e["k_W_mK"] = mt.therm.k_const;
      } else {
        axi::KTable kt;
        kt.T = mt.therm.kT_T;
        kt.k = mt.therm.kT_k;
        e["k_W_mK"] = {{"at_T_min", kt.at(tr[r].lo)}, {"at_T_max", kt.at(tr[r].hi)}};
        e["k_table_T_range_K"] = {mt.therm.kT_T.front(), mt.therm.kT_T.back()};
        if (tr[r].lo < mt.therm.kT_T.front() - 1e-12 || tr[r].hi > mt.therm.kT_T.back() + 1e-12)
          warnings.push_back(strcat("thermal: the temperature in '", rnames[r], "' reaches [", tr[r].lo, ", ", tr[r].hi,
                                    "] K, outside the k(T) table [", mt.therm.kT_T.front(), ", ",
                                    mt.therm.kT_T.back(), "] K; k is clamped to the end values there"));
      }
      if (q_of_region.count(rnames[r])) e["q_W_m3"] = q_of_region[rnames[r]];
      rj[rnames[r]] = e;
    }
    json pj = json::array();
    if (!tprobes.empty()) {
      const int np = static_cast<int>(tprobes.size());
      Located Lc = locate(pm, tprobes, P.rank, P.nranks);
      std::vector<double> buf(3 * np, 0.0);
      for (int i = 0; i < np; ++i) {
        const int e = Lc.elem[i];
        if (e < 0) continue;
        buf[3 * i] = 1;
        if (!roles[e].domain) continue;
        buf[3 * i + 1] = 1;
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        T->SetIntPoint(&Lc.ips[i]);
        buf[3 * i + 2] = Tg.GetValue(*T, Lc.ips[i]);
      }
      gsum(buf.data(), 3 * np);
      for (int i = 0; i < np; ++i)
        pj.push_back({{"x_m", json::array({tprobes[i][0], tprobes[i][1], tprobes[i][2]})},
                      {"in_mesh", buf[3 * i] > 0},
                      {"in_thermal_domain", buf[3 * i + 1] > 0},
                      {"T_K", buf[3 * i + 1] > 0 ? buf[3 * i + 2] : std::numeric_limits<double>::quiet_NaN()}});
    }
    // converged means: Picard settled AND the last linear solve met its tolerance AND the final temperature
    // satisfies the system with k(T_final)
    converged = converged && sdiag.value("relative_residual", 1.0) < 1e-8 && final_res < 1e-6;
    sdiag.update({{"dofs", fes.GlobalTrueVSize()},
                  {"fixed_dofs", gsuml(std::count(is_fixed.begin(), is_fixed.end(), 1))},
                  {"nonlinear", nonlinear_g},
                  {"picard_iterations", it},
                  {"picard_change", change},
                  {"picard_tolerance", tol},
                  {"final_free_row_residual", final_res},
                  {"converged", converged},
                  {"order", order},
                  {"mpi_ranks", P.nranks}});
    R["thermal"] = {{"regions", rj},
                    {"fixed", fj},
                    {"loads", lj},
                    {"T_min_K", tall.lo},
                    {"T_max_K", tall.hi},
                    {"checks", {{"generated_W", generated},
                                {"outflow_W", outflow},
                                {"heat_removed_W", removed},
                                {"heat_added_W", added},
                                {"balance_relative", balance},
                                {"note", "outflow = -sum over the prescribed-temperature DOFs of (K T - F); it equals the "
                                         "generated power exactly for the discrete system, so balance_relative measures "
                                         "the linear solve and round-off. outflow is the NET flow. Each fixed[] entry "
                                         "reports heat_absorbed_W (the reactions of its sink group summed: entries that share "
                                         "nodes form one group, sink_group; nodal P2 reactions are not sign definite, "
                                         "groups are); heat_removed_W sums the groups that absorb heat "
                                         "(the sink loads) and heat_added_W those that supply it; balance_relative is "
                                         "scaled by the largest of the three"}}},
                    {"probes", pj},
                    {"solver", sdiag}};
    if (!converged)
      warnings.push_back(strcat("thermal: the conduction solve did not converge (Picard iterations ", it,
                                ", relative change ", change, ", linear residual ", sdiag.value("relative_residual", 1.0),
                                ", final free-row residual ", final_res, "); results are NOT reliable"));
    if (balance > 1e-8 && (generated != 0 || removed > 0))
      warnings.push_back(strcat("thermal: the heat balance does not close (relative residual ", balance, ")"));
  }
  (void)o;
  R["timing"]["mechanics_thermal_s"] = timer.seconds();
}

}  // namespace axf
