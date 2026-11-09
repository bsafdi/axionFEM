// axf-mesher: build and mesh an axisymmetric (r, z) geometry with gmsh/OpenCASCADE.
//
// This program links the gmsh SDK (GPL) and runs as a separate process; the
// axionFEM library only exchanges files with it, so the library itself links no GPL code.
//
// usage: axf-mesher input.json
// "plane": true meshes the (x, y) cross-section of a z-uniform cavity instead (mesh_plane()).
// input:  {"domain_radius": Rb, "domain_center_z": z0,
//          "regions": [{"name","attr","shape","size","surface_size"}...],
//          "curves": [{"name","attr","path","size","surface_size"}...],
//          "refine_regions": [names], "surface_size": h_s, "size": h_dev,
//          "far_size": h_far, "order": 2, "output": "mesh.msh", "info": "mesh.json",
//          "axis_attr": 900, "outer_attr": 901}
// The half-disk domain is centred on the axis at z = domain_center_z (default 0).
// Axisymmetric path only: a region or curve with its own "surface_size" gets a dedicated refinement
// field at its own boundary, and the rest follow the global "surface_size" within "refine_regions".
// The 3D routine uses the global "surface_size" for every refined region.
// Regions are listed in priority order: where shapes overlap, the later wins.
// Unclaimed space inside the half-disk becomes attribute "air_attr".
#include <gmsh.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <dirent.h>
#include <fcntl.h>
#include <link.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/personality.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;
namespace occ = gmsh::model::occ;
using DimTags = std::vector<std::pair<int, int>>;

#ifdef __linux__
// Whether this process is the curved-mesh process (run_curved_process): AXF_MESHER_STAGE is "curved <pid>", the pid
// of the axf-mesher that started it. Decided once, possibly before main() (sysconf below); the meshing child forked
// from that process inherits the answer.
static bool curved_stage() {
  static int stage = -1;
  if (stage < 0) {
    const char* v = std::getenv("AXF_MESHER_STAGE");
    stage = v && std::strncmp(v, "curved ", 7) == 0 && std::atol(v + 7) == static_cast<long>(getppid()) ? 1 : 0;
  }
  return stage == 1;
}
#endif

#if defined(__linux__) && defined(__GLIBC__)
// The CPU count that the curved-mesh process sees. libgomp (at start), OpenBLAS and
// OpenCASCADE (both inside libgmsh) size arrays by these counts, and on cluster nodes with 128 and with 256 CPUs that
// gave the curved meshes of STEP and surface-dump inputs different heaps. These definitions take precedence over
// glibc's for every library. In the curved-mesh process they answer a fixed 1024: at least the possible CPUs of common
// compute nodes (typically at most 512), so that indexing by a CPU number stays in range, libgomp's affinity mask fits
// at the first try, and code that branches on more than one CPU takes the branch it takes on a large node. Elsewhere
// they answer as glibc does. Limits: on a machine with more than 1024 possible CPUs libgomp's
// first affinity query fails and its arrays depend on the node again; glibc's internal callers (malloc's arena limit,
// unused on one thread) are not covered; OpenCASCADE's default thread pool now has 1023 slots, which would start that
// many threads if its parallel booleans were enabled (gmsh's Geometry.OCCParallel; no axionFEM input sets it). glibc
// only: __sysconf is glibc's.
constexpr long kCurvedCpus = 1024;
extern "C" long int __sysconf(int name) __THROW;

extern "C" long int sysconf(int name) __THROW {
  if ((name == _SC_NPROCESSORS_CONF || name == _SC_NPROCESSORS_ONLN) && curved_stage()) return kCurvedCpus;
  return __sysconf(name);
}

extern "C" int get_nprocs_conf() __THROW {
  return static_cast<int>(curved_stage() ? kCurvedCpus : __sysconf(_SC_NPROCESSORS_CONF));
}

extern "C" int get_nprocs() __THROW {
  return static_cast<int>(curved_stage() ? kCurvedCpus : __sysconf(_SC_NPROCESSORS_ONLN));
}
#endif

#ifdef __linux__
// Whether this process runs from the copy of axf-mesher in memory (run_curved_process): /proc/self/exe, which gmsh
// copies into its context, then reads "/memfd:axf-mesher (deleted)" wherever axf-mesher is installed. Allocates
// nothing.
static bool running_from_memory() {
  char path[64];
  const ssize_t n = readlink("/proc/self/exe", path, sizeof path - 1);
  return n > 0 && std::strncmp(path, "/memfd:", 7) == 0;
}
#endif

namespace {

struct Fail : std::runtime_error {
  using std::runtime_error::runtime_error;
};

double nrm(double x, double y) { return std::sqrt(x * x + y * y); }

// Names written with their double quotes replaced, per dimension: the written name -> the real one.
std::map<std::pair<int, std::string>, std::string> g_replaced_names;

std::string name_collision(const std::string& a, const std::string& b, int dim) {
  return "the names '" + a + "' and '" + b + "' (physical groups of dimension " + std::to_string(dim) +
         ") would be the same in the mesh file, where a double quote is written as a single quote (MFEM's gmsh "
         "reader cannot read one); rename one of them";
}

// A physical group named so that MFEM's gmsh reader can read the mesh back. That reader takes a name up to the next
// double quote, so a region name with one would make every load of the mesh abort (`axf mesh` too, which reads the
// mesh back to count invalid curved elements). Such quotes are written as single quotes. The mesh info keeps the real names, and
// axionFEM reads none back from the mesh file. gmsh keeps one name per dimension, so a written name that another group
// of that dimension already has, or gets later, is refused. Names without a double quote are passed on as they are,
// with no extra work unless a name with one came before.
void add_physical_group(int dim, const std::vector<int>& tags, int attr, const std::string& name) {
  if (name.find('"') == std::string::npos) {
    if (!g_replaced_names.empty()) {
      const auto it = g_replaced_names.find({dim, name});
      if (it != g_replaced_names.end()) throw Fail(name_collision(it->second, name, dim));
    }
    gmsh::model::addPhysicalGroup(dim, tags, attr, name);
    return;
  }
  std::string readable = name;
  std::replace(readable.begin(), readable.end(), '"', '\'');
  const auto it = g_replaced_names.find({dim, readable});
  if (it != g_replaced_names.end()) throw Fail(name_collision(it->second, name, dim));
  DimTags groups;
  gmsh::model::getPhysicalGroups(groups, dim);
  for (const auto& g : groups) {
    std::string other;
    gmsh::model::getPhysicalName(g.first, g.second, other);
    if (other == readable) throw Fail(name_collision(other, name, dim));
  }
  g_replaced_names[{dim, readable}] = name;
  gmsh::model::addPhysicalGroup(dim, tags, attr, readable);
}

// Wire through points with a per-vertex fillet radius (closed).  `fillet` holds one radius per
// point; a zero radius leaves that corner sharp.
int polygon_wire(const std::vector<std::pair<double, double>>& P, const std::vector<double>& fillet) {
  const size_t n = P.size();
  std::vector<int> curves;
  const bool any = std::any_of(fillet.begin(), fillet.end(), [](double f) { return f > 0; });
  if (!any) {
    std::vector<int> pts;
    for (auto& p : P) pts.push_back(occ::addPoint(p.first, p.second, 0));
    for (size_t i = 0; i < n; ++i) curves.push_back(occ::addLine(pts[i], pts[(i + 1) % n]));
    return occ::addWire(curves, -1, true);
  }
  // Tangent points and arc centres at each vertex.
  struct Corner { double ax, ay, bx, by, cx, cy; bool round; };
  std::vector<Corner> C(n);
  std::vector<double> d(n, 0.0), elen(n, 0.0);  // tangent offset at each vertex, length of edge i -> i+1
  for (size_t i = 0; i < n; ++i) {
    auto [px, py] = P[(i + n - 1) % n];
    auto [vx, vy] = P[i];
    auto [qx, qy] = P[(i + 1) % n];
    double ux = px - vx, uy = py - vy, wx = qx - vx, wy = qy - vy;
    double lu = nrm(ux, uy), lw = nrm(wx, wy);
    if (!(lu > 0) || !(lw > 0)) throw Fail("polygon has coincident points");
    elen[i] = lw;
    ux /= lu; uy /= lu; wx /= lw; wy /= lw;
    double cosang = std::clamp(ux * wx + uy * wy, -1.0, 1.0);
    double ang = std::acos(cosang);  // interior angle between edges
    const double f = i < fillet.size() ? fillet[i] : 0.0;
    if (f <= 0 || ang < 1e-6 || std::fabs(M_PI - ang) < 1e-6) {
      C[i] = {vx, vy, vx, vy, 0, 0, false};
      continue;
    }
    d[i] = f / std::tan(ang / 2);  // distance from vertex to tangent points
    double bxv = ux + wx, byv = uy + wy, lb = nrm(bxv, byv);
    double dc = f / std::sin(ang / 2);
    C[i] = {vx + d[i] * ux, vy + d[i] * uy, vx + d[i] * wx, vy + d[i] * wy,
            vx + dc * bxv / lb, vy + dc * byv / lb, true};
  }
  // Neighbouring fillets must not overrun the edge they share.
  for (size_t i = 0; i < n; ++i) {
    const size_t j = (i + 1) % n;
    if (d[i] + d[j] > elen[i] * (1 + 1e-9))
      throw Fail("polygon fillets at vertices " + std::to_string(i) + " and " + std::to_string(j) +
                 " overlap on the edge between them (needs " + std::to_string(d[i] + d[j]) + " m of " +
                 std::to_string(elen[i]) + " m)");
  }
  // Points are deduplicated by coordinate, so that a fillet ending exactly where the next corner
  // begins (or on the next vertex) yields one point instead of two coincident ones.  The tolerance
  // is OpenCASCADE's own Precision::Confusion (1e-7 m): below it OCC would merge the points itself
  // and the zero-length edge between them would fail.
  double scale = 0;
  for (size_t i = 0; i < n; ++i) scale = std::max(scale, elen[i]);
  const double tol = std::max(1e-7, 1e-9 * scale);
  std::map<std::pair<long long, long long>, int> seen;
  auto point = [&](double x, double y) {
    const auto key = std::make_pair(std::llround(x / tol), std::llround(y / tol));
    auto it = seen.find(key);
    if (it != seen.end()) return it->second;
    return seen[key] = occ::addPoint(x, y, 0);
  };
  std::vector<int> a(n), b(n);
  for (size_t i = 0; i < n; ++i) {
    a[i] = point(C[i].ax, C[i].ay);
    b[i] = C[i].round ? point(C[i].bx, C[i].by) : a[i];
    // a fillet so small that its two tangent points coincide is no fillet at all
    if (C[i].round && b[i] == a[i]) C[i].round = false;
  }
  for (size_t i = 0; i < n; ++i) {
    if (C[i].round) {
      int c = occ::addPoint(C[i].cx, C[i].cy, 0);
      curves.push_back(occ::addCircleArc(a[i], c, b[i]));
    }
    const size_t j = (i + 1) % n;
    if (b[i] != a[j]) curves.push_back(occ::addLine(b[i], a[j]));
  }
  if (curves.size() < 3) throw Fail("polygon fillets collapsed the outline to fewer than three edges");
  return occ::addWire(curves, -1, true);
}

// Closed wire of a resolved "contour" shape (spec.cpp): lines and axis-aligned elliptical arcs, point(t) =
// center + (ar cos t, az sin t).  Each arc is one exact rational quadratic B-spline (spans of at most pi/2: control
// points E(t_a), the intersection of the end tangents c + (ar cos t_m, az sin t_m)/cos(D/2) and E(t_b), middle
// weight cos(D/2); the affine image of the exact circle representation), so the wall is smooth and the curved
// (second-order) mesh nodes are placed on the true surface.  Consecutive segments share their end points; the loop
// is closed by a line back to the start unless the last segment already ends there.
int contour_wire(const json& b) {
  const double r0 = b.at("start")[0].get<double>(), z0 = b.at("start")[1].get<double>();
  double scale = std::max(std::fabs(r0), std::fabs(z0));
  for (auto& sg : b.at("segments")) {
    if (sg.contains("line")) scale = std::max({scale, std::fabs(sg["line"][0].get<double>()), std::fabs(sg["line"][1].get<double>())});
    else scale = std::max({scale, std::fabs(sg["ellipse"]["center"][0].get<double>()) + sg["ellipse"]["axes"][0].get<double>(),
                           std::fabs(sg["ellipse"]["center"][1].get<double>()) + sg["ellipse"]["axes"][1].get<double>()});
  }
  const double tol = std::max(1e-7, 1e-9 * scale);  // OpenCASCADE's confusion tolerance (as spec.cpp)
  const int p_start = occ::addPoint(r0, z0, 0);
  int cur = p_start;
  double cr = r0, cz = z0;
  std::vector<int> curves;
  const size_t nseg = b.at("segments").size();
  auto end_point = [&](double r, double z, bool last) {
    if (last && std::hypot(r - r0, z - z0) <= tol) return p_start;
    return occ::addPoint(r, z, 0);
  };
  for (size_t i = 0; i < nseg; ++i) {
    const json& sg = b.at("segments")[i];
    const bool last = i + 1 == nseg;
    if (sg.contains("line")) {
      const double r = sg["line"][0].get<double>(), z = sg["line"][1].get<double>();
      const int p = end_point(r, z, last);
      curves.push_back(occ::addLine(cur, p));
      cur = p;
      cr = r;
      cz = z;
      continue;
    }
    const json& e = sg.at("ellipse");
    const double xc = e["center"][0].get<double>(), yc = e["center"][1].get<double>();
    const double ar = e["axes"][0].get<double>(), az = e["axes"][1].get<double>();
    const double t0 = e["t"][0].get<double>(), t1 = e["t"][1].get<double>();
    const int n = std::max(1, static_cast<int>(std::ceil(std::fabs(t1 - t0) / (M_PI / 2) - 1e-12)));
    const double D = (t1 - t0) / n, w = std::cos(0.5 * D);
    std::vector<int> pts{cur};
    std::vector<double> wts{1.0};
    std::vector<double> knots;
    std::vector<int> mult;
    for (int k = 0; k < n; ++k) {
      const double ta = t0 + k * D, tm = ta + 0.5 * D, tb = ta + D;
      pts.push_back(occ::addPoint(xc + ar * std::cos(tm) / w, yc + az * std::sin(tm) / w, 0));
      wts.push_back(w);
      const double re = xc + ar * std::cos(tb), ze = yc + az * std::sin(tb);
      pts.push_back(k + 1 == n ? end_point(re, ze, last) : occ::addPoint(re, ze, 0));
      wts.push_back(1.0);
      if (k + 1 == n) { cr = re; cz = ze; }
    }
    for (int k = 0; k <= n; ++k) {
      knots.push_back(k);
      mult.push_back(k == 0 || k == n ? 3 : 2);
    }
    curves.push_back(occ::addBSpline(pts, -1, 2, wts, knots, mult));
    cur = pts.back();
  }
  if (cur != p_start) {
    if (std::hypot(cr - r0, cz - z0) <= tol) throw Fail("contour: internal: end point not merged with the start");
    curves.push_back(occ::addLine(cur, p_start));
  }
  return occ::addWire(curves, -1, true);
}

// Build an OCC surface for a resolved shape; returns its dim-tags.
DimTags build_shape(const json& s) {
  if (!s.is_object() || s.size() != 1) throw Fail("bad shape: " + s.dump());
  const std::string k = s.begin().key();
  const json& b = s.begin().value();
  if (k == "rect") {
    double r0 = b["r"][0], r1 = b["r"][1], z0 = b["z"][0], z1 = b["z"][1];
    double f = b.value("fillet", 0.0);
    return {{2, occ::addRectangle(r0, z0, 0, r1 - r0, z1 - z0, -1, f)}};
  }
  if (k == "frame") {
    double r0 = b["r"][0], r1 = b["r"][1], z0 = b["z"][0], z1 = b["z"][1];
    double f = b.value("fillet", 0.0), t = b["thickness"];
    int outer = occ::addRectangle(r0, z0, 0, r1 - r0, z1 - z0, -1, f);
    int inner = occ::addRectangle(r0 + t, z0 + t, 0, r1 - r0 - 2 * t, z1 - z0 - 2 * t, -1, std::max(f - t, 0.0));
    DimTags out;
    std::vector<DimTags> m;
    occ::cut({{2, outer}}, {{2, inner}}, out, m);
    return out;
  }
  if (k == "polygon") {
    std::vector<std::pair<double, double>> P;
    std::vector<double> F;
    const double f0 = b.value("fillet", 0.0);
    for (auto& p : b["points"]) {
      P.push_back({p[0].get<double>(), p[1].get<double>()});
      F.push_back(p.size() > 2 ? p[2].get<double>() : f0);  // [r, z] or [r, z, fillet]
    }
    if (b.contains("fillets")) {  // parallel list of per-vertex radii wins
      const json& fl = b.at("fillets");
      if (fl.size() != P.size()) throw Fail("'fillets' must have one radius per polygon point");
      for (size_t i = 0; i < P.size(); ++i) F[i] = fl[i].get<double>();
    }
    int w = polygon_wire(P, F);
    return {{2, occ::addPlaneSurface({w})}};
  }
  if (k == "disk") {
    double cr = b["center"][0], cz = b["center"][1], R = b["radius"];
    return {{2, occ::addDisk(cr, cz, 0, R, R)}};
  }
  if (k == "contour") return {{2, occ::addPlaneSurface({contour_wire(b)})}};
  if (k == "union" || k == "difference") {
    DimTags first = build_shape(b[0]);
    DimTags rest;
    for (size_t i = 1; i < b.size(); ++i) {
      DimTags c = build_shape(b[i]);
      rest.insert(rest.end(), c.begin(), c.end());
    }
    DimTags out;
    std::vector<DimTags> m;
    if (k == "union") occ::fuse(first, rest, out, m);
    else occ::cut(first, rest, out, m);
    return out;
  }
  throw Fail("unknown shape type " + k);
}

DimTags build_path(const json& s) {
  const std::string k = s.begin().key();
  const json& b = s.begin().value();
  DimTags out;
  auto line_through = [&](const std::vector<std::pair<double, double>>& P, bool closed) {
    std::vector<int> pts;
    for (auto& p : P) pts.push_back(occ::addPoint(p.first, p.second, 0));
    size_t n = P.size();
    for (size_t i = 0; i + 1 < n; ++i) out.push_back({1, occ::addLine(pts[i], pts[i + 1])});
    if (closed) out.push_back({1, occ::addLine(pts[n - 1], pts[0])});
  };
  if (k == "segment") {
    line_through({{b[0][0], b[0][1]}, {b[1][0], b[1][1]}}, false);
  } else if (k == "polyline") {
    std::vector<std::pair<double, double>> P;
    for (auto& p : b["points"]) P.push_back({p[0].get<double>(), p[1].get<double>()});
    line_through(P, b.value("closed", false));
  } else if (k == "rect") {
    double r0 = b["r"][0], r1 = b["r"][1], z0 = b["z"][0], z1 = b["z"][1];
    double f = b.value("fillet", 0.0);
    int sfc = occ::addRectangle(r0, z0, 0, r1 - r0, z1 - z0, -1, f);
    DimTags bd;
    gmsh::model::occ::synchronize();
    gmsh::model::getBoundary({{2, sfc}}, bd, false, false, false);
    for (auto& e : bd) out.push_back({1, std::abs(e.second)});
    // keep the curves, drop the surface
    occ::remove({{2, sfc}}, false);
  } else if (k == "arc") {
    double cr = b["center"][0], cz = b["center"][1], R = b["radius"];
    double a0 = b["angles"][0], a1 = b["angles"][1];
    out.push_back({1, occ::addCircle(cr, cz, 0, R, -1, a0, a1)});
  } else {
    throw Fail("unknown path type " + k);
  }
  return out;
}

// Rotation taking +z to the unit vector (ax, ay, az): returns (axis, angle) for occ::rotate.
void z_to_axis(double ax, double ay, double az, double& rx, double& ry, double& rz, double& ang) {
  double n = std::sqrt(ax * ax + ay * ay + az * az);
  ax /= n; ay /= n; az /= n;
  // axis = z x a, angle = acos(z . a)
  rx = -ay; ry = ax; rz = 0;
  double s = std::sqrt(rx * rx + ry * ry);
  ang = std::acos(std::clamp(az, -1.0, 1.0));
  if (s < 1e-12) { rx = 1; ry = 0; rz = 0; ang = az > 0 ? 0.0 : M_PI; }
}

DimTags build_solid(const json& s) {
  if (!s.is_object()) throw Fail("bad solid: " + s.dump());
  std::string k;
  for (auto it = s.begin(); it != s.end(); ++it) if (it.key() != "rotate" && it.key() != "translate") k = it.key();
  const json& b = s.at(k);
  auto v = [](const json& a, int i) { return a[i].get<double>(); };
  DimTags out;
  if (k == "box") {
    const json &mn = b["min"], &mx = b["max"];
    out = {{3, occ::addBox(v(mn, 0), v(mn, 1), v(mn, 2), v(mx, 0) - v(mn, 0), v(mx, 1) - v(mn, 1), v(mx, 2) - v(mn, 2))}};
  } else if (k == "cylinder") {
    const json &p = b["base"], &a = b["axis"];
    out = {{3, occ::addCylinder(v(p, 0), v(p, 1), v(p, 2), v(a, 0), v(a, 1), v(a, 2), b["radius"].get<double>())}};
  } else if (k == "cone") {
    const json &p = b["base"], &a = b["axis"];
    out = {{3, occ::addCone(v(p, 0), v(p, 1), v(p, 2), v(a, 0), v(a, 1), v(a, 2), b["r1"].get<double>(), b["r2"].get<double>())}};
  } else if (k == "sphere") {
    const json& c = b["center"];
    out = {{3, occ::addSphere(v(c, 0), v(c, 1), v(c, 2), b["radius"].get<double>())}};
  } else if (k == "torus") {
    const json &c = b["center"], &a = b["axis"];
    out = {{3, occ::addTorus(v(c, 0), v(c, 1), v(c, 2), b["major"].get<double>(), b["minor"].get<double>(), -1,
                             2 * M_PI, {v(a, 0), v(a, 1), v(a, 2)})}};
  } else if (k == "revolve") {
    // profile in (r, z) built in the x-y plane, rotated to the x-z plane, revolved about +z,
    // then mapped to the requested axis and centre.
    DimTags prof = build_shape(b["profile"]);
    occ::rotate(prof, 0, 0, 0, 1, 0, 0, M_PI / 2);  // (x, y, 0) -> (x, 0, y)
    double a0 = b["angles"][0].get<double>() * M_PI / 180, a1 = b["angles"][1].get<double>() * M_PI / 180;
    if (a0 != 0) occ::rotate(prof, 0, 0, 0, 0, 0, 1, a0);
    DimTags rv;
    occ::revolve(prof, 0, 0, 0, 0, 0, 1, a1 - a0, rv);
    for (auto& e : rv) if (e.first == 3) out.push_back(e);
    occ::remove(prof, true);
    const json &c = b["center"], &a = b["axis"];
    double rx, ry, rz, ang;
    z_to_axis(v(a, 0), v(a, 1), v(a, 2), rx, ry, rz, ang);
    if (ang != 0) occ::rotate(out, 0, 0, 0, rx, ry, rz, ang);
    if (v(c, 0) != 0 || v(c, 1) != 0 || v(c, 2) != 0) occ::translate(out, v(c, 0), v(c, 1), v(c, 2));
  } else if (k == "extrude") {
    std::vector<int> pts, lines;
    double z0 = b["z"][0], z1 = b["z"][1];
    for (auto& q : b["polygon"]) pts.push_back(occ::addPoint(q[0].get<double>(), q[1].get<double>(), z0));
    for (size_t i = 0; i < pts.size(); ++i) lines.push_back(occ::addLine(pts[i], pts[(i + 1) % pts.size()]));
    int w = occ::addCurveLoop(lines);
    int f = occ::addPlaneSurface({w});
    DimTags ex;
    occ::extrude({{2, f}}, 0, 0, z1 - z0, ex);
    for (auto& e : ex) if (e.first == 3) out.push_back(e);
  } else if (k == "wedge_frame") {
    // Boolean-free polyhedral frame (for example the coil cases of a toroidal magnet): the region
    // {(x, z) inside "polygon" and outside the optional "hole", -(tan_a x - c_neg) <= y <= tan_a x - c_pos} in local coordinates
    // (x, y, z); both side faces are planes and every band face is a planar trapezoid.  Built directly from
    // points, lines and plane surfaces; both band edges are parallel to y, so every band face is planar (an extrusion clipped by two half-space boxes costs minutes of OCC
    // booleans per solid at ~130 polygon vertices).  Use the generic "rotate"/"translate" keys to place it.
    const double ta = b.at("tan_a").get<double>(), cp = b.at("c_pos").get<double>(), cn = b.at("c_neg").get<double>();
    auto ring = [&](const json& poly, std::vector<int>& top, std::vector<int>& bot, std::vector<int>& ltop,
                    std::vector<int>& lbot, std::vector<int>& lv) {
      for (auto& q : poly) {
        const double x = q[0].get<double>(), z = q[1].get<double>(), wp = ta * x - cp, wn = ta * x - cn;
        if (!(wp + wn > 0)) throw Fail("wedge_frame: non-positive width at x = " + std::to_string(x));
        top.push_back(occ::addPoint(x, wp, z));
        bot.push_back(occ::addPoint(x, -wn, z));
      }
      const size_t n = top.size();
      for (size_t i = 0; i < n; ++i) {
        ltop.push_back(occ::addLine(top[i], top[(i + 1) % n]));
        lbot.push_back(occ::addLine(bot[i], bot[(i + 1) % n]));
        lv.push_back(occ::addLine(top[i], bot[i]));
      }
    };
    std::vector<int> ot, ob, olt, olb, olv, ht, hb, hlt, hlb, hlv;
    ring(b.at("polygon"), ot, ob, olt, olb, olv);
    const bool hole = b.contains("hole");
    if (hole) ring(b.at("hole"), ht, hb, hlt, hlb, hlv);
    std::vector<int> faces;
    {
      std::vector<int> w1 = {occ::addCurveLoop(olt)}, w2 = {occ::addCurveLoop(olb)};
      if (hole) { w1.push_back(occ::addCurveLoop(hlt)); w2.push_back(occ::addCurveLoop(hlb)); }
      faces.push_back(occ::addPlaneSurface(w1));
      faces.push_back(occ::addPlaneSurface(w2));
    }
    auto band = [&](const std::vector<int>& lt, const std::vector<int>& lb, const std::vector<int>& lv) {
      const size_t n = lt.size();
      for (size_t i = 0; i < n; ++i)
        faces.push_back(occ::addPlaneSurface({occ::addCurveLoop({lt[i], lv[(i + 1) % n], lb[i], lv[i]})}));
    };
    band(olt, olb, olv);
    if (hole) band(hlt, hlb, hlv);
    out = {{3, occ::addVolume({occ::addSurfaceLoop(faces)})}};
  } else if (k == "step") {
    DimTags imp;
    occ::importShapes(b["file"].get<std::string>(), imp, true);
    for (auto& e : imp) if (e.first == 3) out.push_back(e);
    double sc = b.value("scale", 1.0);
    if (sc != 1.0) occ::dilate(out, 0, 0, 0, sc, sc, sc);
    if (out.empty()) throw Fail("STEP file contains no solids: " + b["file"].get<std::string>());
  } else if (k == "union" || k == "difference" || k == "intersection") {
    DimTags first = build_solid(b[0]), rest;
    for (size_t i = 1; i < b.size(); ++i) {
      DimTags c = build_solid(b[i]);
      rest.insert(rest.end(), c.begin(), c.end());
    }
    std::vector<DimTags> m;
    if (k == "union") occ::fuse(first, rest, out, m);
    else if (k == "difference") occ::cut(first, rest, out, m);
    else occ::intersect(first, rest, out, m);
  } else {
    throw Fail("unknown solid " + k);
  }
  if (s.contains("rotate")) {
    const json& r = s["rotate"];
    const json& ax = r["axis"];
    const json o = r.value("origin", json::array({0, 0, 0}));
    occ::rotate(out, v(o, 0), v(o, 1), v(o, 2), v(ax, 0), v(ax, 1), v(ax, 2), r["angle_deg"].get<double>() * M_PI / 180);
  }
  if (s.contains("translate")) {
    const json& t = s["translate"];
    occ::translate(out, v(t, 0), v(t, 1), v(t, 2));
  }
  return out;
}

// Adaptive refinement: a target element size on the elements of a previous mesh, read from the binary
// file in["size_field_file"] (int32 nodes per element (3: triangles, 4: tetrahedra), int64 count, then per
// element the node coordinates (x, y, z per node) and the target size at each node, float64). It becomes a
// list-based post-processing view used as a PostView size field (combined with the others by Min).
// Returns the field tag (-1 without a file) and lowers *hmin to the smallest requested size.
int size_field_from_file(const json& in, double* hmin) {
  const std::string path = in.value("size_field_file", std::string());
  if (path.empty()) return -1;
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read size field file " + path);
  int32_t nn = 0;
  int64_t ne = 0;
  f.read(reinterpret_cast<char*>(&nn), sizeof nn);
  f.read(reinterpret_cast<char*>(&ne), sizeof ne);
  if (nn != 3 && nn != 4) throw std::runtime_error("size field: 3 or 4 nodes per element expected");
  std::vector<double> rec(4 * nn), data;
  data.reserve(static_cast<size_t>(ne) * 4 * nn);
  for (int64_t e = 0; e < ne; ++e) {
    f.read(reinterpret_cast<char*>(rec.data()), sizeof(double) * rec.size());
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < nn; ++k) data.push_back(rec[3 * k + c]);  // x1..xn, y1..yn, z1..zn
    for (int k = 0; k < nn; ++k) {
      data.push_back(rec[3 * nn + k]);
      *hmin = std::min(*hmin, rec[3 * nn + k]);
    }
  }
  if (!f) throw std::runtime_error("size field file " + path + " is truncated");
  const int view = gmsh::view::add("adaptive size");
  gmsh::view::addListData(view, nn == 3 ? "ST" : "SS", static_cast<int>(ne), data);
  const int fld = gmsh::model::mesh::field::add("PostView");
  gmsh::model::mesh::field::setNumber(fld, "ViewTag", view);
  return fld;
}

// Curved tetrahedra with min det J <= 1e-3 max |det J| (gmsh's Jacobian bounds; the solve's own check samples det J
// at quadrature points with the same threshold, see curved_element_invalid in model3d.cpp).
std::size_t invalid_curved_tets() {
  std::vector<int> types;
  std::vector<std::vector<std::size_t>> tags, nodes;
  gmsh::model::mesh::getElements(types, tags, nodes, 3);
  std::size_t bad = 0;
  for (const auto& t : tags) {
    if (t.empty()) continue;
    std::vector<double> lo, hi;
    gmsh::model::mesh::getElementQualities(t, lo, "minDetJac");
    gmsh::model::mesh::getElementQualities(t, hi, "maxDetJac");
    for (std::size_t i = 0; i < lo.size(); ++i)
      if (!(lo[i] > 1e-3 * std::fabs(hi[i]))) ++bad;
  }
  return bad;
}

// Largest curvature (1/m) over the model's curves, sampled at 9 points per curve; 0 if every curve is straight.
double max_curve_curvature() {
  DimTags curves;
  gmsh::model::getEntities(curves, 1);
  double kmax = 0;
  for (const auto& cv : curves) {
    std::vector<double> pmin, pmax;
    gmsh::model::getParametrizationBounds(1, cv.second, pmin, pmax);
    std::vector<double> ts;
    for (int i = 0; i <= 8; ++i) ts.push_back(pmin[0] + (pmax[0] - pmin[0]) * i / 8.0);
    std::vector<double> k;
    gmsh::model::getCurvature(1, cv.second, ts, k);
    for (double v : k)
      if (std::isfinite(v)) kmax = std::max(kmax, std::fabs(v));
  }
  return kmax;
}

bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v && *v && std::string(v) != "0";
}

// Threads of this process (Linux: the entries of /proc/self/task); 0 where that cannot be counted.
int thread_count() {
  int n = 0;
#ifdef __linux__
  if (DIR* d = opendir("/proc/self/task")) {
    while (const dirent* e = readdir(d))
      if (e->d_name[0] != '.') ++n;
    closedir(d);
  }
#endif
  return n;
}

// How the curved steps' process was started (main, run_curved_process): "on" in the curved-mesh process, which has a
// fixed command line, environment and file names; "off: <why>" where this process does them itself; "n/a" otherwise.
std::string g_canonical = "n/a";

// Says on stderr that a curved mesh is not reproducible, and why (mesh info "high_order_isolation" and
// "high_order_canonical" record the same).
void warn_not_reproducible(const std::string& why) {
  std::cerr << "axf-mesher: WARNING: this curved (order-2) mesh is not reproducible (" << why
            << "): the same input can give a different mesh in another run or another output directory\n";
}

// "off" when this process runs with address-space randomization disabled (Linux personality ADDR_NO_RANDOMIZE).
std::string address_randomization() {
#ifdef __linux__
  const int p = personality(0xffffffff);
  if (p != -1) return (p & ADDR_NO_RANDOMIZE) ? "off" : "on";
#endif
  return "unknown";
}

template <class T>
void put_vector(std::ofstream& f, const std::vector<T>& v) {
  const std::uint64_t n = v.size();
  f.write(reinterpret_cast<const char*>(&n), sizeof n);
  f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
}

template <class T>
std::vector<T> get_vector(std::ifstream& f) {
  std::uint64_t n = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof n);
  std::vector<T> v(f ? n : 0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
  return v;
}

// Curved-mesh handoff from the meshing child to the parent that runs gmsh's HighOrder patch optimizer (mesh3d): every
// entity's nodes (tags, coordinates, parametric coordinates) and elements, in gmsh's entity order. Returns false,
// without a usable file, if a node on a curve or surface has no parametric coordinates: re-created without them, it
// could no longer slide along the CAD in the optimizer.
bool save_curved_mesh(const std::string& path) {
  DimTags ents;
  gmsh::model::getEntities(ents);
  std::ofstream f(path, std::ios::binary);
  const std::uint64_t count = ents.size();
  f.write(reinterpret_cast<const char*>(&count), sizeof count);
  for (const auto& e : ents) {
    std::vector<std::size_t> nodes;
    std::vector<double> xyz, uvw;
    gmsh::model::mesh::getNodes(nodes, xyz, uvw, e.first, e.second, false, true);
    const std::size_t per_node = (e.first == 1 || e.first == 2) ? static_cast<std::size_t>(e.first) : 0;
    if (uvw.size() != per_node * nodes.size()) return false;
    std::vector<int> types;
    std::vector<std::vector<std::size_t>> elements, element_nodes;
    gmsh::model::mesh::getElements(types, elements, element_nodes, e.first, e.second);
    put_vector(f, std::vector<int>{e.first, e.second});
    put_vector(f, nodes);
    put_vector(f, xyz);
    put_vector(f, uvw);
    put_vector(f, types);
    for (std::size_t i = 0; i < types.size(); ++i) {
      put_vector(f, elements[i]);
      put_vector(f, element_nodes[i]);
    }
  }
  f.close();
  if (!f) throw Fail("cannot write the curved-mesh handoff file '" + path + "'; is the output directory writable "
                     "and not full?");
  return true;
}

// Re-creates the mesh written by save_curved_mesh on the (unmeshed) geometry of this process: all nodes first, as
// MEdgeVertex / MFaceVertex with their parametric coordinates on curves and surfaces, then the elements.
void load_curved_mesh(const std::string& path) {
  struct EntityMesh {
    std::vector<int> dim_tag;
    std::vector<std::size_t> nodes;
    std::vector<double> xyz;
    std::vector<double> uvw;
    std::vector<int> types;
    std::vector<std::vector<std::size_t>> elements;
    std::vector<std::vector<std::size_t>> element_nodes;
  };
  std::ifstream f(path, std::ios::binary);
  std::uint64_t count = 0;
  f.read(reinterpret_cast<char*>(&count), sizeof count);
  std::vector<EntityMesh> all(f ? count : 0);
  for (auto& m : all) {
    m.dim_tag = get_vector<int>(f);
    m.nodes = get_vector<std::size_t>(f);
    m.xyz = get_vector<double>(f);
    m.uvw = get_vector<double>(f);
    m.types = get_vector<int>(f);
    for (std::size_t i = 0; i < m.types.size(); ++i) {
      m.elements.push_back(get_vector<std::size_t>(f));
      m.element_nodes.push_back(get_vector<std::size_t>(f));
    }
    if (m.dim_tag.size() != 2) f.setstate(std::ios::failbit);
  }
  if (!f || all.empty())
    throw Fail("cannot read the curved-mesh handoff file '" + path + "' written by the meshing process");
  for (const auto& m : all) gmsh::model::mesh::addNodes(m.dim_tag[0], m.dim_tag[1], m.nodes, m.xyz, m.uvw);
  // The volume elements are allocated in a fixed pseudo-random order, then put back in their order in the volume
  // (reorderElements), so the patch optimizer sees them in a mixed address order, as in one process, yet the same in
  // every run. In file order (fresh memory in load order) gmsh's first pass failed on both designs where it can:
  // curved_probe_shell.json left 4 invalid tets and core30 (curved_elements_study.py) all 2418, where shuffled orders
  // (two seeds each) and 6 of 7 in-process runs cleared them in that pass.
  const std::uint64_t seed = 1;
  for (const auto& m : all)
    for (std::size_t i = 0; i < m.types.size(); ++i) {
      if (m.dim_tag[0] != 3 || m.elements[i].empty()) {
        gmsh::model::mesh::addElementsByType(m.dim_tag[1], m.types[i], m.elements[i], m.element_nodes[i]);
        continue;
      }
      const std::size_t n = m.elements[i].size();
      const std::size_t per = m.element_nodes[i].size() / n;
      std::vector<std::size_t> perm(n);
      for (std::size_t k = 0; k < n; ++k) perm[k] = k;
      std::uint64_t state = seed;
      for (std::size_t k = n; k > 1; --k) {  // Fisher-Yates with splitmix64 (the same on every platform)
        state += 0x9e3779b97f4a7c15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        z = z ^ (z >> 31);
        std::swap(perm[k - 1], perm[z % k]);
      }
      std::vector<std::size_t> tags(n), nodes(n * per), ordering(n);
      for (std::size_t k = 0; k < n; ++k) {
        tags[k] = m.elements[i][perm[k]];
        std::copy(m.element_nodes[i].begin() + per * perm[k], m.element_nodes[i].begin() + per * (perm[k] + 1),
                  nodes.begin() + per * k);
        ordering[perm[k]] = k;
      }
      gmsh::model::mesh::addElementsByType(m.dim_tag[1], m.types[i], tags, nodes);
      gmsh::model::mesh::reorderElements(m.types[i], m.dim_tag[1], ordering);
    }
}

// Node coordinates and parametric coordinates of the whole mesh, entity by entity, to undo an optimizer pass.
struct EntityNodes {
  std::size_t per_node = 0;
  std::vector<std::size_t> tags;
  std::vector<double> xyz;
  std::vector<double> uvw;
};

std::vector<EntityNodes> save_node_positions() {
  DimTags ents;
  gmsh::model::getEntities(ents);
  std::vector<EntityNodes> all(ents.size());
  for (std::size_t k = 0; k < ents.size(); ++k) {
    const int dim = ents[k].first;
    all[k].per_node = (dim == 1 || dim == 2) ? static_cast<std::size_t>(dim) : 0;
    gmsh::model::mesh::getNodes(all[k].tags, all[k].xyz, all[k].uvw, dim, ents[k].second, false, true);
    if (all[k].uvw.size() == all[k].per_node * all[k].tags.size()) continue;
    // getNodes leaves out the nodes without parametric coordinates: take each node's own (setNode ignores the zeros
    // put in for the others)
    all[k].uvw.assign(all[k].per_node * all[k].tags.size(), 0.0);
    for (std::size_t i = 0; i < all[k].tags.size(); ++i) {
      std::vector<double> xyz, uvw;
      int node_dim = 0;
      int node_tag = 0;
      gmsh::model::mesh::getNode(all[k].tags[i], xyz, uvw, node_dim, node_tag);
      for (std::size_t j = 0; j < uvw.size() && j < all[k].per_node; ++j) all[k].uvw[all[k].per_node * i + j] = uvw[j];
    }
  }
  return all;
}

void restore_node_positions(const std::vector<EntityNodes>& all) {
  for (const auto& e : all)
    for (std::size_t i = 0; i < e.tags.size(); ++i) {
      const std::vector<double> xyz(e.xyz.begin() + 3 * i, e.xyz.begin() + 3 * i + 3);
      const std::vector<double> uvw(e.uvw.begin() + e.per_node * i, e.uvw.begin() + e.per_node * (i + 1));
      gmsh::model::mesh::setNode(e.tags[i], xyz, uvw);
    }
}

int mesh3d(json in, json& info) {
  auto t0 = std::chrono::steady_clock::now();
  const json c = in.value("domain_center", json::array({0, 0, 0}));
  const int outer_attr = in.value("outer_attr", 901), air_attr = in.value("air_attr", 1);
  DimTags objects;
  std::vector<int> owner;
  const json& regions = in.value("regions", json::array());
  for (size_t i = 0; i < regions.size(); ++i) {
    DimTags s = build_solid(regions[i].at("solid"));
    for (auto& e : s) {
      objects.push_back(e);
      owner.push_back(static_cast<int>(i));
    }
  }
  // automatic extents: device bounding box -> domain radius and element sizes
  double bx0 = 1e300, by0 = 1e300, bz0 = 1e300, bx1 = -1e300, by1 = -1e300, bz1 = -1e300;
  for (auto& e : objects) {
    double a0, a1, a2, a3, a4, a5;
    occ::getBoundingBox(e.first, e.second, a0, a1, a2, a3, a4, a5);
    bx0 = std::min(bx0, a0); by0 = std::min(by0, a1); bz0 = std::min(bz0, a2);
    bx1 = std::max(bx1, a3); by1 = std::max(by1, a4); bz1 = std::max(bz1, a5);
  }
  double Rdev = 0;
  for (double xx : {bx0, bx1})
    for (double yy : {by0, by1})
      for (double zz : {bz0, bz1})
        Rdev = std::max(Rdev, std::sqrt((xx - c[0].get<double>()) * (xx - c[0].get<double>()) +
                                        (yy - c[1].get<double>()) * (yy - c[1].get<double>()) +
                                        (zz - c[2].get<double>()) * (zz - c[2].get<double>())));
  const double Ldev = std::max({bx1 - bx0, by1 - by0, bz1 - bz0});
  // with the infinity shell the open boundary is exact, so the air region can be small
  if (in.value("domain_radius", 0.0) <= 0) in["domain_radius"] = (in.value("infinity_shell", 0.0) > 1.0 ? 1.6 : 4.0) * Rdev;
  if (in.value("size", 0.0) <= 0) in["size"] = Ldev / 20.0;
  if (in.value("surface_size", 0.0) <= 0) in["surface_size"] = in["size"].get<double>() / 3.0;
  if (in.value("far_size", 0.0) <= 0) in["far_size"] = in["domain_radius"].get<double>() / 6.0;
  in["device_radius"] = Rdev;
  const double Rb = in.at("domain_radius");
  // optional "infinity shell" [Rb, Rs]: the solver maps it onto [Rb, infinity) (exact open boundary)
  const json symmetry=in.value("symmetry",json::array());
  const bool custom_domain=in.contains("domain_solid");
  const double shell_factor = (custom_domain || !symmetry.empty()) ? 0 : in.value("infinity_shell", 0.0);
  const double Rs = shell_factor > 1.0 ? shell_factor * Rb : Rb;
  const int shell_attr = in.value("shell_attr", 898);
  int shell_tag = -1;
  if (Rs > Rb) {
    int outer_ball = occ::addSphere(c[0], c[1], c[2], Rs);
    int inner_ball = occ::addSphere(c[0], c[1], c[2], Rb);
    DimTags cut;
    std::vector<DimTags> cmap;
    occ::cut({{3, outer_ball}}, {{3, inner_ball}}, cut, cmap);
    if (cut.size() != 1) throw Fail("could not build the infinity shell");
    shell_tag = cut[0].second;
  }
  DimTags domain=custom_domain ? build_solid(in.at("domain_solid")) :
                                DimTags{{3,occ::addSphere(c[0],c[1],c[2],Rb)}};
  auto clip = [&](const DimTags& input) {
    if (symmetry.empty() || input.empty()) return input;
    double lo[3]={-2*Rs,-2*Rs,-2*Rs},hi[3]={2*Rs,2*Rs,2*Rs};
    for (const auto& p:symmetry) {
      const std::string axis=p.at("axis"); const int a=axis=="x"?0:(axis=="y"?1:2);
      if (p.value("side",1.0)>0) lo[a]=0; else hi[a]=0;
    }
    int box=occ::addBox(lo[0],lo[1],lo[2],hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]);
    DimTags out; std::vector<DimTags> map;
    occ::intersect(input,{{3,box}},out,map);
    return out;
  };
  std::set<int> active_regions;
  if (!symmetry.empty()) {
    DimTags kept; std::vector<int> kept_owner;
    for (size_t i=0;i<objects.size();++i) for (const auto& e:clip({objects[i]})) {
      if (e.first==3) {kept.push_back(e);kept_owner.push_back(owner[i]);active_regions.insert(owner[i]);}
    }
    objects=kept;owner=kept_owner;
    domain=clip(domain);
  } else for (int i:owner) active_regions.insert(i);
  for (const auto& e:domain) {objects.insert(objects.begin(),e);owner.insert(owner.begin(),-1);}
  if (domain.empty()) throw Fail("computational domain is empty after symmetry clipping");
  if (shell_tag >= 0) {
    objects.push_back({3, shell_tag});
    owner.push_back(-2);  // the shell
  }
  DimTags frag;
  std::vector<DimTags> fmap;
  occ::fragment(objects, {}, frag, fmap);
  occ::synchronize();
  std::map<int, int> vol_owner;
  std::set<int> in_domain;
  std::set<int> shell_vols;
  for (size_t k = 0; k < fmap.size(); ++k) {
    for (auto& e : fmap[k]) {
      if (e.first != 3) continue;
      if (owner[k] == -2) {
        shell_vols.insert(e.second);
        continue;
      }
      if (owner[k] < 0) in_domain.insert(e.second);
      auto it = vol_owner.find(e.second);
      if (owner[k] >= 0 && (it == vol_owner.end() || it->second < owner[k])) vol_owner[e.second] = owner[k];
      else if (owner[k] < 0 && it == vol_owner.end()) vol_owner[e.second] = -1;
    }
  }
  DimTags all3;
  gmsh::model::getEntities(all3, 3);
  for (auto& e : all3) {
    if (shell_vols.count(e.second) && !in_domain.count(e.second)) continue;
    if (!in_domain.count(e.second)) throw Fail("a region extends outside the computational domain; increase mesh.domain_radius");
    if (!vol_owner.count(e.second)) vol_owner[e.second] = -1;
  }
  // COMSOL export (docs/comsol_export.md): the fragmented (conformal) OCC solids as STEP, before any PEC
  // volume is removed, with the region that owns each solid; "geometry_only" stops before meshing.
  if (in.contains("step_output")) {
    const std::string step = in.at("step_output").get<std::string>();
    gmsh::write(step);
    json solids = json::array();
    for (auto& e : all3) {
      const bool shell = shell_vols.count(e.second) && !in_domain.count(e.second);
      const int ow = vol_owner.count(e.second) ? vol_owner.at(e.second) : -1;
      double lo[3], hi[3], cm[3];
      occ::getBoundingBox(3, e.second, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]);
      occ::getCenterOfMass(3, e.second, cm[0], cm[1], cm[2]);
      double vol = 0;
      occ::getMass(3, e.second, vol);
      solids.push_back({{"volume_tag", e.second},
                        {"region", shell ? std::string("infinity_shell") : (ow < 0 ? std::string("air") : regions[ow].at("name").get<std::string>())},
                        {"attr", shell ? shell_attr : (ow < 0 ? air_attr : regions[ow].at("attr").get<int>())},
                        {"volume_m3", vol}, {"center_of_mass", {cm[0], cm[1], cm[2]}},
                        {"bbox", {lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]}}});
    }
    info["step"] = {{"file", step}, {"solids", solids}};
    if (in.value("geometry_only", false)) {
      info["ok"] = true;
      info["geometry_only"] = true;
      info["domain_radius"] = Rb;
      info["shell_outer_radius"] = Rs;
      info["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      return 0;
    }
  }
  std::map<int, std::vector<int>> by_attr;
  std::map<int, std::string> attr_name;
  std::set<int> removed_volumes;
  for (auto& [tag, ow] : vol_owner) {
    if (shell_vols.count(tag) && !in_domain.count(tag)) continue;
    int attr = ow < 0 ? air_attr : regions[ow].at("attr").get<int>();
    by_attr[attr].push_back(tag);
    attr_name[attr] = ow < 0 ? "air" : regions[ow].at("name").get<std::string>();
    if (ow>=0 && regions[ow].value("remove_pec",false)) removed_volumes.insert(tag);
  }
  std::vector<int> shell_list;
  for (int t : shell_vols) {
    if (in_domain.count(t)) continue;
    auto it = vol_owner.find(t);
    if (it != vol_owner.end() && it->second >= 0)
      throw Fail("region '" + regions[it->second].at("name").get<std::string>() +
                 "' extends into the infinity shell; increase mesh.domain_radius");
    shell_list.push_back(t);
  }
  if (!shell_list.empty()) {
    by_attr[shell_attr] = shell_list;
    attr_name[shell_attr] = "infinity_shell";
  }
  // Preserve connected PEC-body identity before deleting its volume. This is
  // required for terminals on holes, including distinct disconnected bodies
  // belonging to the same CAD region. Face contacts merge conductor bodies.
  std::map<int,std::vector<int>> surface_volumes, region_surfaces;
  std::map<int,int> parent;
  for (const auto& e:all3) {
    DimTags bd; gmsh::model::getBoundary({e},bd,false,false,false);
    for (const auto& s:bd) surface_volumes[std::abs(s.second)].push_back(e.second);
    if (removed_volumes.count(e.second)) parent[e.second]=e.second;
  }
  auto root=[&](int x) {while(parent.at(x)!=x) x=parent.at(x);return x;};
  for (const auto& [s,vs]:surface_volumes) {
    int first=-1;
    for (int v:vs) if (removed_volumes.count(v)) {
      if (first<0) first=v; else parent[root(v)]=root(first);
    }
  }
  std::map<int,int> body_ids;
  for (int v:removed_volumes) if (!body_ids.count(root(v))) body_ids[root(v)]=static_cast<int>(body_ids.size());
  for (const auto& [attr,vs]:by_attr) {
    std::set<int> surfaces;
    for (int v:vs) {DimTags bd;gmsh::model::getBoundary({{3,v}},bd,false,false,false);
      for (const auto& s:bd) surfaces.insert(std::abs(s.second));}
    region_surfaces[attr]=std::vector<int>(surfaces.begin(),surfaces.end());
  }
  json groups = json::array(), boundary_groups=json::array(), conductors=json::array();
  for (auto& [attr, tags] : by_attr) {
    std::vector<int> kept;
    for (int t:tags) if (!removed_volumes.count(t)) kept.push_back(t);
    if (!kept.empty()) {
      add_physical_group(3, kept, attr, attr_name[attr]);
      groups.push_back({{"dim", 3}, {"attr", attr}, {"name", attr_name[attr]}, {"entities", kept.size()}});
    }
  }
  for (size_t i = 0; i < regions.size(); ++i) {
    if (active_regions.count(static_cast<int>(i)) && !by_attr.count(regions[i].at("attr").get<int>()))
      throw Fail("region '" + regions[i].at("name").get<std::string>() + "' is completely covered by later regions");
  }
  std::map<int,std::vector<int>> boundary_tags;
  std::map<int,json> boundary_meta;
  boundary_meta[outer_attr]={{"attr",outer_attr},{"name","outer"},{"type","pec"}};
  for (size_t i=0;i<symmetry.size();++i) {
    const int attr=910+static_cast<int>(i);
    boundary_meta[attr]={{"attr",attr},{"name",symmetry[i]["name"]},{"type",symmetry[i]["type"]},
                         {"axis",symmetry[i]["axis"]},{"side",symmetry[i].value("side",1.0)}};
  }
  for (const auto& [s,vs]:surface_volumes) {
    int alive=0,dead=-1;
    for (int v:vs) {if (removed_volumes.count(v)) dead=v;else ++alive;}
    if (alive!=1) continue;
    int attr=outer_attr;
    if (dead>=0) {
      const int bid=body_ids.at(root(dead));attr=1000+bid;
      const int ow=vol_owner.at(dead);
      boundary_meta[attr]={{"attr",attr},{"name","pec_body_"+std::to_string(bid)},
                           {"type","pec"},{"body_id",bid},{"conductor",regions[ow]["name"]}};
    } else {
      double lo[3],hi[3];occ::getBoundingBox(2,s,lo[0],lo[1],lo[2],hi[0],hi[1],hi[2]);
      for (size_t i=0;i<symmetry.size();++i) {
        const std::string ax=symmetry[i]["axis"];const int a=ax=="x"?0:(ax=="y"?1:2);
        if (std::max(std::fabs(lo[a]),std::fabs(hi[a]))<std::max(2e-6,Rs*1e-8)) attr=910+static_cast<int>(i);
      }
    }
    boundary_tags[attr].push_back(s);
  }
  if (!removed_volumes.empty()) {
    DimTags remove;for(int v:removed_volumes) remove.push_back({3,v});
    occ::remove(remove,true);occ::synchronize();
  }
  for (const auto& [attr,tags]:boundary_tags) {
    const json& meta=boundary_meta.at(attr);
    add_physical_group(2,tags,attr,meta["name"].get<std::string>());
    boundary_groups.push_back(meta);
    groups.push_back({{"dim",2},{"attr",attr},{"name",meta["name"]},{"entities",tags.size()}});
  }
  for (const auto& [r,bid]:body_ids) {
    std::set<std::string> names;int count=0;
    for (int v:removed_volumes) if(root(v)==r) {++count;names.insert(regions[vol_owner.at(v)]["name"].get<std::string>());}
    conductors.push_back({{"body_id",bid},{"regions",names},{"removed_volumes",count}});
  }

  // sizes
  // max_size: an upper bound on every element size (full wave: a fraction of the shortest wavelength)
  const double h_cap = in.value("max_size", 0.0) > 0 ? in.value("max_size", 0.0) : 1e300;
  const double h_dev = std::min(in.at("size").get<double>(), h_cap), h_s = std::min(in.value("surface_size", h_dev), h_cap),
               h_far = std::min(in.value("far_size", Rb / 6), h_cap);
  const double r_dev = in.value("device_radius", Rb / 3);
  // extend_from_boundary = 1 grades the volume from the boundary sizes (e.g. curvature-refined curved walls; without
  // it a finely meshed convex wall meets a coarse interior in one layer of flat elements, e.g. at an SRF iris)
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", static_cast<int>(in.value("extend_from_boundary", 0.0)));
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", in.value("curvature_points", 16));
  gmsh::option::setNumber("Mesh.MeshSizeMax", h_far);
  double mesh_size_min = 0, min_curve_radius = 0;
  {  // the floor must not clip the smallest requested size (per-region sizes included)
    double hmin = std::min(h_s, h_dev);
    for (auto& g : regions)
      for (const char* key : {"size", "surface_size"}) {
        const double v = g.value(key, 0.0);
        if (v > 0) hmin = std::min(hmin, v);
      }
    mesh_size_min = hmin * 0.25;
    const double kmax = max_curve_curvature();
    min_curve_radius = kmax > 0 ? 1 / kmax : 0;
    // experimental option: nor may it clip curvature sizing on the smallest curve radius
    if (env_flag("AXF_MESHER_CURVATURE_FLOOR") && kmax > 0)
      mesh_size_min = std::min(mesh_size_min, 2 * M_PI / (kmax * in.value("curvature_points", 16.0)));
    gmsh::option::setNumber("Mesh.MeshSizeMin", mesh_size_min);
  }
  std::vector<double> fields;
  int ff = gmsh::model::mesh::field::add("MathEval");
  char expr[320];
  std::snprintf(expr, sizeof expr, "max(%.17g, %.17g + 0.3*(sqrt((x-%.17g)^2+(y-%.17g)^2+(z-%.17g)^2) - %.17g))", h_dev,
                h_dev, c[0].get<double>(), c[1].get<double>(), c[2].get<double>(), r_dev);
  gmsh::model::mesh::field::setString(ff, "F", expr);
  fields.push_back(ff);
  if (!shell_list.empty()) {  // element layers across the shell (the mapped exterior field is smooth there)
    std::vector<double> tags(shell_list.begin(), shell_list.end());
    int fc = gmsh::model::mesh::field::add("MathEval");
    const double layers = std::max(1.0, in.value("shell_layers", 2.0));
    std::snprintf(expr, sizeof expr, "%.17g", std::min(h_far, (Rs - Rb) / layers));
    gmsh::model::mesh::field::setString(fc, "F", expr);
    int fr = gmsh::model::mesh::field::add("Restrict");
    gmsh::model::mesh::field::setNumber(fr, "InField", fc);
    gmsh::model::mesh::field::setNumbers(fr, "VolumesList", tags);
    fields.push_back(fr);
  }
  for (size_t i = 0; i < regions.size(); ++i) {
    double h = regions[i].value("size", 0.0);
    if (h <= 0) continue;
    std::vector<double> tags;
    for (int t : by_attr[regions[i].at("attr").get<int>()]) if(!removed_volumes.count(t)) tags.push_back(t);
    if (tags.empty()) continue;
    int fc = gmsh::model::mesh::field::add("MathEval");
    std::snprintf(expr, sizeof expr, "%.17g", h);
    gmsh::model::mesh::field::setString(fc, "F", expr);
    int fr = gmsh::model::mesh::field::add("Restrict");
    gmsh::model::mesh::field::setNumber(fr, "InField", fc);
    gmsh::model::mesh::field::setNumbers(fr, "VolumesList", tags);
    fields.push_back(fr);
  }
  // Distance + Threshold refinement towards a set of surfaces with a given target size.  SizeMax
  // must not undercut the device size: a Threshold field returns SizeMax at every point beyond
  // DistMax and this one is not Restricted to the region.
  const double layer3 = in.value("surface_layer", 4.0 * h_dev);
  auto refine_towards3 = [&](const std::vector<double>& surfs, double hmin) {
    if (surfs.empty() || hmin <= 0) return;
    int fd = gmsh::model::mesh::field::add("Distance");
    gmsh::model::mesh::field::setNumbers(fd, "SurfacesList", surfs);
    int ft = gmsh::model::mesh::field::add("Threshold");
    gmsh::model::mesh::field::setNumber(ft, "InField", fd);
    gmsh::model::mesh::field::setNumber(ft, "SizeMin", hmin);
    gmsh::model::mesh::field::setNumber(ft, "SizeMax", std::max(h_dev, hmin));
    gmsh::model::mesh::field::setNumber(ft, "DistMin", 0.5 * hmin);
    gmsh::model::mesh::field::setNumber(ft, "DistMax", std::max(layer3, 4.0 * hmin));
    fields.push_back(ft);
  };
  auto boundary_surfaces = [&](const std::vector<int>& vols) {
    std::vector<double> out;
    std::set<int> valid;DimTags faces;gmsh::model::getEntities(faces,2);
    for(const auto& f:faces) valid.insert(f.second);
    for(const auto& [s,vs]:surface_volumes) if(valid.count(s))
      for(int v:vs) if(std::find(vols.begin(),vols.end(),v)!=vols.end()) {out.push_back(s);break;}
    return out;
  };
  // per-region surface_size: its own boundary refined to its own size
  std::set<std::string> own_surface;
  for (size_t i = 0; i < regions.size(); ++i) {
    const double hr = regions[i].value("surface_size", 0.0);
    if (hr <= 0) continue;
    own_surface.insert(regions[i].at("name").get<std::string>());
    refine_towards3(boundary_surfaces(by_attr[regions[i].at("attr").get<int>()]), hr);
  }
  const std::vector<std::string> refine_list = in.value("refine_regions", std::vector<std::string>{});
  std::set<std::string> refine(refine_list.begin(), refine_list.end());
  std::vector<double> rsurf;
  for (auto& [attr, tags] : by_attr) {
    if (!refine.count(attr_name[attr]) || own_surface.count(attr_name[attr])) continue;
    for (double t : boundary_surfaces(tags)) rsurf.push_back(t);
  }
  refine_towards3(rsurf, h_s);
  {
    // adaptive cycles: the floor of cycle 0 (per-region sizes included), lowered to the smallest adapted size
    double hmin = mesh_size_min;
    const int fa = size_field_from_file(in, &hmin);
    if (fa >= 0) {
      fields.push_back(fa);
      gmsh::option::setNumber("Mesh.MeshSizeMin", hmin);
    }
  }
  int fmin = gmsh::model::mesh::field::add("Min");
  gmsh::model::mesh::field::setNumbers(fmin, "FieldsList", fields);
  gmsh::model::mesh::field::setAsBackgroundMesh(fmin);
  gmsh::option::setNumber("Mesh.Algorithm", 6);
  gmsh::option::setNumber("Mesh.Algorithm3D", in.value("algorithm3d", 10));
  const int threads = in.value("threads", 4);  // General.NumThreads, set where meshing starts (see below)
  gmsh::option::setNumber("Mesh.Optimize", 1);
  if (in.value("optimize_netgen", false)) gmsh::option::setNumber("Mesh.OptimizeNetgen", 1);
  const char* dump = std::getenv("AXF_MESHER_SURFACE_DUMP");
  if (dump && *dump) {  // debugging aid: surface tags -> bbox
    // Written through nlohmann::json so that region names are escaped and bounding boxes keep full precision.
    auto write_dump = [&]() {
      json list = json::array();
      DimTags faces;
      gmsh::model::getEntities(faces, 2);
      for (const auto& fc : faces) {
        double a0, a1, a2, a3, a4, a5;
        gmsh::model::getBoundingBox(2, fc.second, a0, a1, a2, a3, a4, a5);
        std::vector<int> up, down;
        gmsh::model::getAdjacencies(2, fc.second, up, down);
        std::string names;
        for (int v : up) {
          auto it = vol_owner.find(v);
          names += (it == vol_owner.end() ? std::string("?") : it->second < 0 ? std::string("air")
                    : regions[it->second].at("name").get<std::string>()) + ";";
        }
        list.push_back({{"tag", fc.second}, {"bbox", {a0, a1, a2, a3, a4, a5}}, {"volumes", names}});
      }
      std::ofstream f(dump);
      if (f) f << list.dump();
      if (!f) std::cerr << "axf-mesher: warning: AXF_MESHER_SURFACE_DUMP: cannot write '" << dump << "'; no surface dump\n";
    };
    // In a forked child where that is safe (one thread): the dump's queries allocate in this process, and done here
    // they changed the curved mesh of the same input.
    pid_t pid = -1;
    if (g_canonical == "on" || thread_count() <= 1) {
      std::cout.flush();
      std::fflush(nullptr);
      pid = fork();
    }
    if (pid == 0) {
      try {
        write_dump();
      } catch (const std::exception& e) {
        std::cerr << "axf-mesher: warning: AXF_MESHER_SURFACE_DUMP: " << e.what() << "; no surface dump\n";
      }
      std::_Exit(0);
    }
    if (pid > 0) {
      int wstatus = 0;
      while (waitpid(pid, &wstatus, 0) < 0 && errno == EINTR) {
      }
    } else {
      write_dump();
    }
  }
  // AXF_MESHER_OPTIMIZE=netgen: gmsh's Netgen optimizer (face/edge swaps and smoothing) on the straight-sided tets,
  // before the order is raised. The default optimizer leaves slivers between near-coplanar nodes of curved boundaries
  // (the 5 mm slit of a CORE-type slotted pickup: width 30 nm, aspect 1e5-1e6), on which the port conduction PCG and
  // AMS break down.
  const char* optimize_env = std::getenv("AXF_MESHER_OPTIMIZE");
  const std::string optimize = optimize_env && *optimize_env ? optimize_env : "default";
  if (optimize != "default" && optimize != "netgen")
    throw Fail("AXF_MESHER_OPTIMIZE must be netgen or empty, not '" + optimize + "'");
  int order = in.value("order", 2);
  const char* ho_env = std::getenv("AXF_MESHER_HO");
  const std::string ho = ho_env && *ho_env ? ho_env : "auto";
  if (ho != "elastic" && ho != "full" && ho != "none" && ho != "auto")
    throw Fail("AXF_MESHER_HO must be elastic, full, auto or none, not '" + ho + "'");
  const std::string output = in.at("output").get<std::string>();
  json ho_info = json::object();
  json ho_passes = json::array();
  std::string patch_optimizer = "not_run";
  // straight-sided tets, raised to order 2 (nodes of curved boundaries on the CAD)
  auto mesh_order2 = [&]() {
    gmsh::model::mesh::generate(3);
    if (optimize == "netgen") gmsh::model::mesh::optimize("Netgen");
    if (order > 1) {
      gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
      gmsh::model::mesh::setOrder(std::min(order, 2));
    }
  };
  // untangle curved elements near thin curved features: gmsh's elastic pass
  auto curve = [&]() {
    if (order > 1) {
      gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
      gmsh::option::setNumber("Mesh.HighOrderOptimize", 2);
      if (ho != "none") gmsh::model::mesh::optimize("HighOrderElastic");
    }
  };
  // auto: the Jacobian-targeted patch optimizer only where the elastic pass leaves invalid elements (it removes every
  // one on the probe shell and core30 of benchmarks/curved_elements_study.py, at 2-5x the elastic pass's meshing time)
  auto count_after_elastic = [&]() {
    const std::size_t after_elastic = ho == "auto" ? invalid_curved_tets() : 0;
    if (ho == "auto") ho_info["invalid_after_elastic"] = after_elastic;
    return ho == "full" || (ho == "auto" && after_elastic > 0);
  };
  auto run_patch_optimizer = [&]() {
    // gmsh reports an optimizer that cannot bring every patch to its target quality ("Failed to reach critical
    // value") as an error, and in the API's error mode that throws from inside the optimizer, where nothing can
    // catch it: the mesher aborted (SIGABRT) on benchmarks/designs/curved_probe_shell.json. Log it instead, keep
    // what the optimizer achieved, and count what is left; the solve straightens any remaining invalid element.
    double abort_mode = 0;
    gmsh::option::getNumber("General.AbortOnError", abort_mode);
    gmsh::option::setNumber("General.AbortOnError", 0);
    // Where invalid elements remain after gmsh's pass, up to two more passes from the improved mesh with patches of 2x
    // and 4x the layers: gmsh's own adaptive strategy grows a failed patch the same way (adaptBlobLayerFact = 2,
    // maxAdaptBlob = 3 in HighOrderMeshOptimizer.h), but its API runs only the fixed-patch strategy. A pass that does
    // not lower the count is undone. Each pass re-optimizes every element outside gmsh's quality band and costs up to
    // about as much as the first, with no time limit (a time limit would make the mesh depend on the machine). With the
    // volume elements loaded in file order they rescued both designs on which gmsh's pass failed:
    // curved_probe_shell.json 6 -> 4 -> 0 (12 layers), core30 2418 -> 2418 -> 2418 (12, undone) -> 0 (24), at 3x the
    // first pass's time. The shuffled load (load_curved_mesh) is
    // a fixed draw that behaves like in-process ones, not a tuned one: over seeds, core30 cleared in gmsh's pass 5 of 5
    // times and the full-size shell 5 of 7 (the other two needed the 12-layer pass), so these
    // passes remain needed.
    double layers = 6;
    gmsh::option::getNumber("Mesh.HighOrderNumLayers", layers);
    std::size_t left = 0;
    for (int pass = 0; pass < 3; ++pass) {
      if (pass > 0 && left == 0) break;
      std::vector<EntityNodes> before;
      if (pass > 0) before = save_node_positions();
      const double pass_layers = layers * (1 << pass);
      gmsh::option::setNumber("Mesh.HighOrderNumLayers", pass_layers);
      const auto start = std::chrono::steady_clock::now();
      gmsh::model::mesh::optimize("HighOrder");
      gmsh::option::setNumber("Mesh.HighOrderNumLayers", layers);
      const std::size_t now = invalid_curved_tets();
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      const bool kept = pass == 0 || now < left;
      if (kept)
        left = now;
      else
        restore_node_positions(before);
      ho_passes.push_back({{"layers", pass_layers}, {"invalid_after", now}, {"kept", kept}, {"seconds", seconds}});
      if (pass > 0)
        std::cerr << "axf-mesher: note: extra HighOrder pass with " << pass_layers << " layers (" << seconds
                  << " s): " << now << " invalid curved tets, " << (kept ? "kept" : "undone") << "\n";
    }
    gmsh::option::setNumber("General.AbortOnError", abort_mode);
    ho_info["invalid_after_high_order"] = left;
  };
  auto write_mesh = [&]() {
    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::option::setNumber("Mesh.Binary", 1);
    gmsh::write(output);
  };
  // element quality: gamma (inscribed/circumscribed radius ratio of the straight element, 1 =
  // regular) and the scaled Jacobian of the curved element; poorly shaped wall elements spoil pointwise wall fields
  auto mesh_quality = [&]() {
    json q = json::object();
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 3);
    std::vector<std::size_t> all;
    for (auto& vv : etags) all.insert(all.end(), vv.begin(), vv.end());
    if (all.empty()) return q;
    std::vector<double> g, sj;
    gmsh::model::mesh::getElementQualities(all, g, "gamma");
    gmsh::model::mesh::getElementQualities(all, sj, "minSJ");
    std::size_t g01 = 0;
    for (double x : g) g01 += x < 0.1;
    q = {{"gamma_min", *std::min_element(g.begin(), g.end())}, {"gamma_below_0.1", g01},
         {"minSJ_min", *std::min_element(sj.begin(), sj.end())}};
    return q;
  };
  auto count_3d = [&]() {
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 3);
    std::size_t n = 0;
    for (auto& vv : etags) n += vv.size();
    return n;
  };
  // Reproducible curved meshes. gmsh orders two order-2 steps by the addresses of the mesh
  // objects (gmsh 4.15.2): the HighOrder patch optimizer keeps each patch in a std::set<MElement *> and numbers its
  // unknowns in that order (contrib/MeshOptimizer/MeshOptimizer.cpp:46, Patch.cpp:81), and the elastic pass sorts the
  // surface triangles it smooths by address (HighOrderMeshElasticAnalogy.cpp:185). Their floating-point results, and
  // the curved mesh, therefore depend on where generate() and setOrder() left the objects, which differs from run to
  // run even on one thread with ASLR off; the straight mesh and setOrder's nodes do not. So a forked child meshes and
  // raises the order, and hands the mesh over (save_curved_mesh); this process, which has not meshed, re-creates it,
  // all nodes and then the elements, the volume elements in a fixed shuffled order (load_curved_mesh), and runs the
  // elastic pass and the patch optimizer. Most of the elements then take fresh memory in a fixed order. What remained
  // with ASLR on: 1 run in 146 differed on the test design and 2 in 81 on a coarse shell, none in
  // 88 with ASLR off, and this process's memory after the geometry differed between runs only with ASLR on; main()
  // therefore re-executes the mesher once with address randomization off (Linux; AXF_MESHER_ASLR=on keeps it).
  // Whatever this process allocated before the load still decides which free chunks the elements reuse, so that must
  // not depend on the caller either: a one-character longer output path changed the mesh (run_curved_process), and on
  // Linux this process is the curved-mesh process, with a fixed command line, environment and file names.
  // Reproducible for one build of axf-mesher and gmsh, "threads": 1 (multi-threaded HXT is not reproducible,
  // model3d.cpp), on Linux/glibc. AXF_MESHER_HO_ISOLATE=0 runs everything in one process (not reproducible).
  const char* isolate_env = std::getenv("AXF_MESHER_HO_ISOLATE");
  const std::string isolate_value = isolate_env ? isolate_env : "";
  if (isolate_value != "" && isolate_value != "0" && isolate_value != "1")
    throw Fail("AXF_MESHER_HO_ISOLATE must be 1 (the default: reproducible curved meshes), 0 (one process) or empty, "
               "not '" + isolate_value + "'");
  std::string isolation = "n/a";
  int process_threads = 0;
  if (order > 1 && ho != "none") {
    isolation = "isolated";
    process_threads = thread_count();
    if (isolate_value == "0")
      isolation = "off: AXF_MESHER_HO_ISOLATE=0";
    else if (process_threads > 1)  // forking a multi-threaded process is unsafe; this one should have one thread here
      isolation = "off: " + std::to_string(process_threads) + " threads before the fork";
    if (isolation != "isolated")
      warn_not_reproducible("isolation " + isolation);
    else if (g_canonical != "on")
      warn_not_reproducible("curved-mesh process " + g_canonical);
    else if (address_randomization() == "on")
      warn_not_reproducible("address randomization on");
#ifdef __linux__
    if (isolation == "isolated" && g_canonical == "on" && !running_from_memory())
      std::cerr << "axf-mesher: warning: the curved-mesh process runs from axf-mesher's file, not from a copy in memory "
                   "(mesh info \"high_order_executable\"), so curved meshes of STEP solids and of the surface dump can "
                   "depend on where axf-mesher is installed\n";
#endif
  }
  std::size_t nel = 0;
  json quality = json::object();
  if (isolation != "isolated") {
    gmsh::option::setNumber("General.NumThreads", threads);
    mesh_order2();
    curve();
    if (order > 1 && count_after_elastic()) {
      run_patch_optimizer();
      patch_optimizer = "in_process";
    }
    write_mesh();
    nel = count_3d();
    quality = mesh_quality();
  } else {
    const std::string handoff = output + ".handoff";
    const std::string report = output + ".handoff.json";
    std::remove(handoff.c_str());  // from a run that was killed
    std::remove(report.c_str());
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    const pid_t parent = getpid();
    const pid_t pid = fork();
    if (pid < 0)
      throw Fail(std::string("cannot fork the meshing process (") + std::strerror(errno) + "); set "
                 "AXF_MESHER_HO_ISOLATE=0 to mesh in one process (curved meshes then differ from run to run)");
    if (pid == 0) {
#ifdef __linux__
      // the child dies with this process (a kill of axf-mesher's pid alone would otherwise leave it meshing)
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      if (getppid() != parent) std::_Exit(1);
#endif
      json status = json::object();
      status["pid"] = static_cast<long long>(getpid());
      int rc = 0;
      try {
        gmsh::option::setNumber("General.NumThreads", threads);
        mesh_order2();
        if (save_curved_mesh(handoff)) {
          status["next"] = "curve";
        } else {
          // nodes without parametric coordinates cannot be handed over: curve here (not reproducible)
          std::remove(handoff.c_str());
          curve();
          if (count_after_elastic()) {
            run_patch_optimizer();
            status["patch_optimizer"] = "in_process";
          }
          write_mesh();
          status["quality"] = mesh_quality();  // this process holds the mesh; the parent has none
          status["next"] = "done";
          status["isolation"] = "off: nodes without parametric coordinates";
          status["high_order_counts"] = ho_info;
          status["high_order_passes"] = ho_passes;
        }
        status["elements_3d"] = count_3d();
      } catch (const std::exception& e) {
        status["error"] = e.what();
        rc = 1;
      } catch (...) {
        status["error"] = "unknown exception";
        rc = 1;
      }
      std::ofstream(report) << status.dump(-1, ' ', false, json::error_handler_t::replace);
      std::cout.flush();
      std::fflush(nullptr);  // gmsh's terminal output ("verbose")
      std::_Exit(rc);  // no exit handlers: gmsh's state belongs to the parent
    }
    int wstatus = 0;
    while (waitpid(pid, &wstatus, 0) < 0 && errno == EINTR) {
    }
    json status;
    try {
      std::ifstream f(report);
      status = json::parse(f);
    } catch (const std::exception&) {
      status = json();
    }
    std::remove(report.c_str());
    const std::string how_far = "to see how far gmsh got, add \"verbose\": 1 to the axf-mesher input file (axf "
                                "writes it as mesh3d_in.json in the mesh directory) and run axf-mesher on it";
    if (!status.is_object()) {
      std::remove(handoff.c_str());
      if (WIFSIGNALED(wstatus))
        throw Fail("the meshing process (forked by axf-mesher) was killed by signal " +
                   std::to_string(WTERMSIG(wstatus)) + " (9: out of memory or a job limit; 11: a crash in gmsh); " +
                   how_far);
      throw Fail("the meshing process (forked by axf-mesher) exited with status " +
                 std::to_string(WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -1) + " without a report; " + how_far);
    }
    if (status.value("pid", 0LL) != static_cast<long long>(pid)) {
      std::remove(handoff.c_str());
      throw Fail("the meshing report '" + report + "' comes from another process (pid " +
                 std::to_string(status.value("pid", 0LL)) + ", expected " + std::to_string(pid) +
                 "); is another axf-mesher writing the same output? Use one output directory per run");
    }
    if (status.contains("error")) {
      std::remove(handoff.c_str());
      throw Fail(status["error"].get<std::string>());
    }
    nel = status.value("elements_3d", std::size_t(0));
    if (status.value("next", std::string()) == "curve") {
      try {
        load_curved_mesh(handoff);
      } catch (...) {
        std::remove(handoff.c_str());
        throw;
      }
      std::remove(handoff.c_str());
      gmsh::option::setNumber("General.NumThreads", threads);
      curve();
      if (count_after_elastic()) {
        run_patch_optimizer();
        patch_optimizer = "isolated";
      }
      write_mesh();
      quality = mesh_quality();
    } else {
      quality = status.value("quality", json::object());
      ho_info = status.value("high_order_counts", json::object());
      ho_passes = status.value("high_order_passes", json::array());
      patch_optimizer = status.value("patch_optimizer", std::string("not_run"));
      isolation = status.value("isolation", isolation);
      if (isolation != "isolated") warn_not_reproducible("isolation " + isolation);
    }
  }
  std::string ver;
  gmsh::option::getString("General.Version", ver);
  const json step_info = info.is_object() ? info.value("step", json()) : json();
  info = {{"ok", true}, {"gmsh_version", ver}, {"elements_3d", nel}, {"groups", groups},
          {"boundary_groups",boundary_groups},{"pec_conductors",conductors},
          {"removed_pec_volumes",removed_volumes.size()},{"symmetry",symmetry},
          {"domain_radius", Rb}, {"device_radius", Rdev}, {"size", h_dev}, {"surface_size", h_s},
          {"shell_inner_radius", Rb}, {"shell_outer_radius", Rs}, {"shell_attr", shell_list.empty() ? -1 : shell_attr},
          {"domain_center", c}, {"mesh_size_min", mesh_size_min}, {"min_curve_radius", min_curve_radius},
          {"high_order_optimize", order > 1 ? ho : std::string("n/a")}, {"high_order_counts", ho_info},
          {"high_order_patch_optimizer", patch_optimizer}, {"high_order_passes", ho_passes},
          {"high_order_isolation", isolation}, {"address_randomization", address_randomization()},
          {"high_order_canonical", isolation == "n/a" ? std::string("n/a") : g_canonical},
          {"high_order_threads_before_fork", process_threads},
          {"tet_optimize", optimize}, {"quality", quality},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()}};
  if (!step_info.is_null()) info["step"] = step_info;
  return 0;
}

// Cross-section ("plane": true) meshing for z-uniform cavities ("symmetry": "xsec"): the domain is the union of
// the region shapes in the (x, y) plane (no half-disk, no outer sphere, no clipping at x < 0). Regions are painted
// in priority order as in the axisymmetric path; curves (sheets) are embedded. The boundary of the union gets
// physical group outer_attr ("outer": the cavity wall). Sizes: "size" everywhere (no far-field growth),
// per-region size/surface_size, and "surface_size" at the outer wall, at the boundaries of "refine_regions" and
// at curves without their own surface_size.
void mesh_plane(const json& in, json& info) {
  auto t0 = std::chrono::steady_clock::now();
  const int outer_attr = in.value("outer_attr", 901);
  const json& regions = in.value("regions", json::array());
  const json& curves = in.value("curves", json::array());
  if (regions.empty()) throw Fail("a cross-section design needs at least one region (the cavity)");
  DimTags objects;
  std::vector<int> owner;
  for (size_t i = 0; i < regions.size(); ++i) {
    DimTags s = build_shape(regions[i].at("shape"));
    for (auto& e : s) {
      objects.push_back(e);
      owner.push_back(static_cast<int>(i));
    }
  }
  const int nreg = static_cast<int>(regions.size());
  for (size_t i = 0; i < curves.size(); ++i) {
    DimTags s = build_path(curves[i].at("path"));
    for (auto& e : s) {
      objects.push_back(e);
      owner.push_back(nreg + static_cast<int>(i));
    }
  }
  DimTags frag;
  std::vector<DimTags> fmap;
  if (objects.size() > 1) {
    occ::fragment(objects, {}, frag, fmap);
  } else {  // a single shape: nothing to fragment (OCC returns no map then)
    fmap = {objects};
  }
  occ::synchronize();
  std::map<int, int> surf_owner;
  std::map<int, std::set<int>> curve_items;
  for (size_t k = 0; k < fmap.size(); ++k) {
    for (auto& e : fmap[k]) {
      if (e.first == 2 && owner[k] < nreg) {
        auto it = surf_owner.find(e.second);
        if (it == surf_owner.end() || it->second < owner[k]) surf_owner[e.second] = owner[k];
      } else if (e.first == 1 && owner[k] >= nreg) {
        curve_items[e.second].insert(owner[k] - nreg);
      }
    }
  }
  DimTags all2;
  gmsh::model::getEntities(all2, 2);
  for (auto& e : all2)
    if (!surf_owner.count(e.second)) throw Fail("internal: a surface of the cross-section has no region");
  // curves must lie in the domain: the midpoint of every curve piece is inside (or on) some surface
  for (auto& [tag, items] : curve_items) {
    std::vector<double> pmin, pmax, xyz;
    gmsh::model::getParametrizationBounds(1, tag, pmin, pmax);
    gmsh::model::getValue(1, tag, {0.5 * (pmin[0] + pmax[0])}, xyz);
    bool inside = false;
    for (auto& e : all2)
      if (gmsh::model::isInside(2, e.second, xyz, false) > 0) { inside = true; break; }
    if (!inside)
      throw Fail("curve '" + curves[*items.begin()].at("name").get<std::string>() +
                 "' (or part of it) lies outside every region; a cross-section sheet must lie inside the cavity or "
                 "on its boundary");
  }
  std::map<int, std::vector<int>> by_attr;
  std::map<int, std::string> attr_name;
  for (auto& [tag, ow] : surf_owner) {
    const int attr = regions[ow].at("attr").get<int>();
    by_attr[attr].push_back(tag);
    attr_name[attr] = regions[ow].at("name").get<std::string>();
  }
  json groups = json::array();
  for (auto& [attr, tags] : by_attr) {
    add_physical_group(2, tags, attr, attr_name[attr]);
    groups.push_back({{"dim", 2}, {"attr", attr}, {"name", attr_name[attr]}, {"entities", tags.size()}});
  }
  for (size_t i = 0; i < regions.size(); ++i) {
    if (!by_attr.count(regions[i].at("attr").get<int>()))
      throw Fail("region '" + regions[i].at("name").get<std::string>() +
                 "' is completely covered by later regions; reorder or remove it");
  }
  DimTags bnd;
  gmsh::model::getBoundary(all2, bnd, true, false, false);
  std::vector<int> outer;
  std::vector<double> outer_d;
  for (auto& e : bnd) {
    outer.push_back(std::abs(e.second));
    outer_d.push_back(std::abs(e.second));
  }
  gmsh::model::addPhysicalGroup(1, outer, outer_attr, "outer");
  groups.push_back({{"dim", 1}, {"attr", outer_attr}, {"name", "outer"}});
  for (size_t i = 0; i < curves.size(); ++i) {
    std::vector<int> tags;
    for (auto& [tag, items] : curve_items)
      if (items.count(static_cast<int>(i))) tags.push_back(tag);
    const int attr = curves[i].at("attr");
    add_physical_group(1, tags, attr, curves[i].at("name").get<std::string>());
    groups.push_back({{"dim", 1}, {"attr", attr}, {"name", curves[i].at("name")}, {"entities", tags.size()}});
  }
  // ------------------------------------------------------------ mesh size
  const double h_dev = in.at("size");
  const double h_s = in.value("surface_size", h_dev);
  double hmin = std::min(h_s, h_dev);
  for (const json* list : {&regions, &curves})
    for (auto& g : *list)
      for (const char* key : {"size", "surface_size"}) {
        const double v = g.value(key, 0.0);
        if (v > 0) hmin = std::min(hmin, v);
      }
  gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
  gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", in.value("curvature_points", 24));
  gmsh::option::setNumber("Mesh.MeshSizeMax", h_dev);
  gmsh::option::setNumber("Mesh.MeshSizeMin", hmin * 0.25);
  std::vector<double> fields;
  char expr[64];
  {
    const int fc = gmsh::model::mesh::field::add("MathEval");
    std::snprintf(expr, sizeof expr, "%.17g", h_dev);
    gmsh::model::mesh::field::setString(fc, "F", expr);
    fields.push_back(fc);
  }
  for (size_t i = 0; i < regions.size(); ++i) {
    const double h = regions[i].value("size", 0.0);
    if (h <= 0) continue;
    std::vector<double> tags;
    for (int t : by_attr[regions[i].at("attr").get<int>()]) tags.push_back(t);
    const int fc = gmsh::model::mesh::field::add("MathEval");
    std::snprintf(expr, sizeof expr, "%.17g", h);
    gmsh::model::mesh::field::setString(fc, "F", expr);
    const int fr = gmsh::model::mesh::field::add("Restrict");
    gmsh::model::mesh::field::setNumber(fr, "InField", fc);
    gmsh::model::mesh::field::setNumbers(fr, "SurfacesList", tags);
    fields.push_back(fr);
  }
  const double layer = in.value("surface_layer", 4.0 * h_dev);
  auto refine_towards = [&](const std::vector<double>& cs, double hm, double hx) {
    if (cs.empty() || hm <= 0 || hm >= hx) return;
    const int fd = gmsh::model::mesh::field::add("Distance");
    gmsh::model::mesh::field::setNumbers(fd, "CurvesList", cs);
    gmsh::model::mesh::field::setNumber(fd, "Sampling", 200);
    const int ft = gmsh::model::mesh::field::add("Threshold");
    gmsh::model::mesh::field::setNumber(ft, "InField", fd);
    gmsh::model::mesh::field::setNumber(ft, "SizeMin", hm);
    gmsh::model::mesh::field::setNumber(ft, "SizeMax", hx);
    gmsh::model::mesh::field::setNumber(ft, "DistMin", 0.5 * hm);
    gmsh::model::mesh::field::setNumber(ft, "DistMax", std::max(layer, 4.0 * hm));
    fields.push_back(ft);
  };
  for (size_t i = 0; i < regions.size(); ++i) {
    const double hr = regions[i].value("surface_size", 0.0);
    if (hr <= 0) continue;
    DimTags sd, cb;
    for (int t : by_attr[regions[i].at("attr").get<int>()]) sd.push_back({2, t});
    gmsh::model::getBoundary(sd, cb, false, false, false);
    std::vector<double> cs;
    for (auto& e : cb) cs.push_back(std::abs(e.second));
    refine_towards(cs, hr, std::max(h_dev, hr));
  }
  std::vector<double> refine_curves = outer_d;
  const std::vector<std::string> refine_list = in.value("refine_regions", std::vector<std::string>{});
  const std::set<std::string> refine(refine_list.begin(), refine_list.end());
  for (auto& [attr, tags] : by_attr) {
    if (!refine.count(attr_name[attr])) continue;
    DimTags sd, cb;
    for (int t : tags) sd.push_back({2, t});
    gmsh::model::getBoundary(sd, cb, false, false, false);
    for (auto& e : cb) refine_curves.push_back(std::abs(e.second));
  }
  std::map<int, std::vector<double>> own_curves;
  for (auto& [tag, items] : curve_items) {
    int own = -1;
    for (int it : items)
      if (curves[it].value("surface_size", 0.0) > 0) { own = it; break; }
    if (own >= 0) own_curves[own].push_back(tag);
    else refine_curves.push_back(tag);
  }
  for (auto& [item, cs] : own_curves) refine_towards(cs, curves[item].value("surface_size", 0.0), h_dev);
  refine_towards(refine_curves, h_s, h_dev);
  const int fmin = gmsh::model::mesh::field::add("Min");
  gmsh::model::mesh::field::setNumbers(fmin, "FieldsList", fields);
  gmsh::model::mesh::field::setAsBackgroundMesh(fmin);
  gmsh::option::setNumber("Mesh.Algorithm", 6);
  gmsh::model::mesh::generate(2);
  const int order = in.value("order", 2);
  if (order > 1) {
    gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
    gmsh::model::mesh::setOrder(std::min(order, 2));
  }
  for (auto& e : all2) {  // counter-clockwise in (x, y)
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 2, e.second);
    if (et.empty() || enodes[0].size() < 3) continue;
    std::vector<double> c[3], pc;
    int dimn, tagn;
    for (int q = 0; q < 3; ++q) gmsh::model::mesh::getNode(enodes[0][q], c[q], pc, dimn, tagn);
    const double area = (c[1][0] - c[0][0]) * (c[2][1] - c[0][1]) - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1]);
    if (area < 0) gmsh::model::mesh::reverse({{2, e.second}});
  }
  gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
  gmsh::option::setNumber("Mesh.Binary", 0);
  gmsh::write(in.at("output").get<std::string>());
  std::vector<std::size_t> ntags;
  std::vector<double> ncoord, npar;
  gmsh::model::mesh::getNodes(ntags, ncoord, npar);
  std::vector<int> et;
  std::vector<std::vector<std::size_t>> etags, enodes;
  gmsh::model::mesh::getElements(et, etags, enodes, 2);
  size_t nel = 0;
  for (auto& v : etags) nel += v.size();
  std::string ver;
  gmsh::option::getString("General.Version", ver);
  info = {{"ok", true}, {"plane", true}, {"gmsh_version", ver}, {"nodes", ntags.size()}, {"elements_2d", nel},
          {"groups", groups},
          {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()}};
}

// Whether mesh3d will raise the order and run the curved steps in an isolated parent (its own test, before meshing).
bool curved_isolated(const json& in) {
  if (!in.is_object() || in.value("plane", false) || in.value("dim", 2) != 3 || in.value("geometry_only", false))
    return false;
  const char* ho = std::getenv("AXF_MESHER_HO");
  const char* isolate = std::getenv("AXF_MESHER_HO_ISOLATE");
  return in.value("order", 2) > 1 && !(ho && std::string(ho) == "none") && !(isolate && std::string(isolate) == "0");
}

#ifdef __linux__
// The curved-mesh process. In the isolated parent (mesh3d), the free chunks that its allocations
// before the load leave behind decide where some elements land, and with them the curved mesh. Some of those
// allocations came from outside the input:
// - the output path: with one input and one build, the coarse shell of tests/test_mesh3d_reproducible.py gave one
//   curved mesh in mesh directories up to 63 characters long and another from 64, where the report path
//   output + ".handoff.json" moves from glibc's 96-byte chunks to the 112-byte ones of gmsh's order-2 tetrahedra;
// - for inputs with STEP solids, "step_output" or the surface dump: the executable's path,
//   which gmsh copies at gmsh::initialize and adds to PATH and PYTHONPATH with setenv(), whose array grows with the
//   environment's size; whether stdin, stdout and stderr are regular files (a constructor in libgmsh, probably its
//   static libgfortran, fstat()s fds 0-2 before main and buffers the regular files); and
//   arrays that libraries size by the CPU count.
// So the curved steps run in a fresh process:
// - started as `axf-mesher input.json` from a copy of this executable in memory (/proc/self/exe then reads
//   "/memfd:axf-mesher (deleted)"); an executable that finds its libraries through $ORIGIN gets its run path,
//   expanded, in LD_LIBRARY_PATH; if the copy cannot be executed, the process runs from the file and warns;
// - in a private work directory next to the output (curved_names), where input.json is the input with every path
//   replaced by a fixed name, each a symbolic link to the user's file (mesh.msh -> the output, ...), and where the
//   process writes its mesh info (info.json), which run_curved_process copies out before it removes the directory;
// - with a fixed environment of ten variables (curved_env), and with gmsh's -noenv (main);
// - with stdin /dev/null and stdout and stderr pipes, which this process relays (relay_output), and a fixed CPU count
//   (sysconf above). (main also makes stdout's buffer static, which a piped stdout does not need.)
// The dump is written by a short-lived child (mesh3d). Still in the process's history, and so able to change the
// curved mesh: the input itself, including "verbose" and "step_output", and the build.
// A lock file next to the output, locked while the process runs, makes a second axf-mesher writing the same output
// stop with a message. Each run writes into and removes only what it created.
const char* const kCurvedStage = "AXF_MESHER_STAGE";
const char* const kCurvedInput = "input.json";
const char* const kCurvedInfo = "info.json";
const char* const kCurvedNames = "names.json";  // fixed name -> the user's path, read only to word an error

// A signal (SIGTERM, SIGINT, SIGHUP) that came while the curved-mesh process ran: passed on to it, and raised again in
// this process once it has cleaned up (main).
volatile sig_atomic_t g_stop_signal = 0;
pid_t g_curved_pid = 0;

extern "C" void forward_stop(int sig) {
  g_stop_signal = sig;
  if (g_curved_pid > 0) kill(g_curved_pid, sig);
}

// JSON as text, with any byte that is not UTF-8 replaced (paths need not be UTF-8; dump() would throw).
std::string json_text(const json& j, int indent) {
  return j.dump(indent, ' ', false, json::error_handler_t::replace);
}

// The input as the curved-mesh process reads it: each path replaced by a fixed name (keeping its extension, by which
// gmsh picks a format); `links` gets each name and the absolute path it stands for. The files of STEP solids, in the
// regions and in "domain_solid" alike, and the size field file must exist (Fail names the missing one).
json curved_input(const json& in, std::vector<std::pair<std::string, std::string>>& links) {
  json c = in;
  auto fixed = [&](const std::string& name, const std::string& path) {
    const std::string local = name + std::filesystem::path(path).extension().string();
    links.push_back({local, std::filesystem::absolute(path).string()});
    return local;
  };
  auto existing = [&](const std::string& name, const std::string& path, const std::string& what) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
      throw Fail("the " + what + " '" + path + "' does not exist (absolute path '" +
                 std::filesystem::absolute(path).string() + "')");
    return fixed(name, path);
  };
  c["output"] = fixed("mesh", in.at("output").get<std::string>());
  c["info"] = kCurvedInfo;
  if (in.contains("step_output")) c["step_output"] = fixed("geometry", in.at("step_output").get<std::string>());
  if (!in.value("size_field_file", std::string()).empty())
    c["size_field_file"] = existing("size_field", in.at("size_field_file").get<std::string>(), "size field file");
  int k = 0;
  std::function<void(json&)> walk = [&](json& j) {  // the files of "step" solids, anywhere in the input
    if (j.is_object() && j.contains("step") && j["step"].is_object() && j["step"].contains("file") &&
        j["step"]["file"].is_string())
      j["step"]["file"] = existing("input_" + std::to_string(k++), j["step"]["file"].get<std::string>(), "STEP file");
    if (j.is_structured())
      for (auto& v : j) walk(v);
  };
  walk(c);
  return c;
}

// The executable's run path (DT_RPATH, DT_RUNPATH) with $ORIGIN expanded to its directory, if it has $ORIGIN: the
// curved-mesh process gets it in LD_LIBRARY_PATH, since from memory $ORIGIN cannot be resolved (the wheel's mesher,
// CMakeLists.txt). Empty otherwise.
std::string expanded_origin_path() {
  const char* strings = nullptr;
  for (const ElfW(Dyn)* d = _DYNAMIC; d->d_tag != DT_NULL; ++d)
    if (d->d_tag == DT_STRTAB) strings = reinterpret_cast<const char*>(d->d_un.d_ptr);
  if (!strings) return "";
  char exe[4096];
  const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
  if (n <= 0) return "";
  exe[n] = '\0';
  const std::string dir = std::filesystem::path(exe).parent_path().string();
  std::string out;
  for (const ElfW(Dyn)* d = _DYNAMIC; d->d_tag != DT_NULL; ++d) {
    if (d->d_tag != DT_RPATH && d->d_tag != DT_RUNPATH) continue;
    std::string path = strings + d->d_un.d_val;
    if (path.find("$ORIGIN") == std::string::npos && path.find("${ORIGIN}") == std::string::npos) continue;
    for (const std::string token : {"${ORIGIN}", "$ORIGIN"})
      for (std::size_t at = path.find(token); at != std::string::npos; at = path.find(token, at + dir.size()))
        path.replace(at, token.size(), dir);
    out += (out.empty() ? "" : ":") + path;
  }
  return out;
}

// The curved-mesh process's environment, sorted, always the same ten variables: the six AXF_MESHER_* switches (empty
// where the caller set none; AXF_MESHER_SURFACE_DUMP made absolute), the stage marker, LD_LIBRARY_PATH (the expanded
// $ORIGIN run path, then the caller's; empty if neither), and OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. The last
// two make it single-threaded at the fork: libgmsh links OpenBLAS statically, which starts a thread per CPU when it is
// loaded unless told otherwise (without them, 4 threads in a 4-CPU job, and no isolation; the caller's environment
// may set neither). LD_PRELOAD, locale, OpenMP placement, HOME, malloc tunables, SLURM's variables and the rest are left
// out.
std::vector<std::string> curved_env() {
  std::vector<std::string> env;
  for (const char* name : {"AXF_MESHER_ASLR", "AXF_MESHER_CURVATURE_FLOOR", "AXF_MESHER_HO", "AXF_MESHER_HO_ISOLATE",
                           "AXF_MESHER_OPTIMIZE", "AXF_MESHER_SURFACE_DUMP", "LD_LIBRARY_PATH"}) {
    const char* v = std::getenv(name);
    std::string value = v ? v : "";
    if (std::string(name) == "AXF_MESHER_SURFACE_DUMP" && !value.empty())
      value = std::filesystem::absolute(value).string();
    if (std::string(name) == "LD_LIBRARY_PATH") {
      const std::string origin = expanded_origin_path();
      if (!origin.empty()) value = value.empty() ? origin : origin + ":" + value;
    }
    env.push_back(std::string(name) + "=" + value);
  }
  env.push_back(std::string(kCurvedStage) + "=curved " + std::to_string(static_cast<long>(getpid())));
  env.push_back("OPENBLAS_NUM_THREADS=1");
  env.push_back("OMP_NUM_THREADS=1");
  std::sort(env.begin(), env.end());
  return env;
}

// A copy of this executable in memory, to start the curved-mesh process from: its /proc/self/exe then reads
// "/memfd:axf-mesher (deleted)" wherever axf-mesher is installed. -1, with the reason, if that is not possible.
int executable_in_memory(std::string& why) {
  const int src = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
  const int fd = src < 0 ? -1 : memfd_create("axf-mesher", MFD_CLOEXEC);
  if (fd < 0) {
    why = std::string("memfd_create or /proc/self/exe: ") + std::strerror(errno);
    if (src >= 0) close(src);
    return -1;
  }
  char buf[1 << 16];
  ssize_t n = 0;
  while ((n = read(src, buf, sizeof buf)) > 0)
    if (write(fd, buf, static_cast<size_t>(n)) != n) n = -1;
  const int err = errno;
  close(src);
  if (n < 0) {
    why = std::string("copying the executable: ") + std::strerror(err);
    close(fd);
    return -1;
  }
  return fd;
}

// The names next to `output` of its lock file and work directory: <output>.axf-mesher.lock and
// <output>.axf-mesher.XXXXXX, or for an output file name longer than 200 bytes its first 200 and a hash of all of it
// in place of the name, so that both stay within the file system's 255 bytes whatever output name gmsh accepts.
std::string curved_stem(const std::string& output) {
  const std::filesystem::path p(output);
  const std::string name = p.filename().string();
  if (name.size() <= 200) return output;
  std::uint64_t h = 1469598103934665603ull;  // FNV-1a
  for (unsigned char ch : name) {
    h ^= ch;
    h *= 1099511628211ull;
  }
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
  return (p.parent_path() / (name.substr(0, 200) + "-" + hex)).string();
}

// Takes the lock file `lock_path` (flock), so that two axf-meshers writing one output do not run at once. Returns its
// descriptor; -1 with `error` set if another axf-mesher holds it or the file cannot be used; -2 if the file system
// cannot lock (then nothing is checked). `created`: this run created the file, wrote "host <name> pid <n>" in it, and
// removes it at the end. A file it did not create is only locked: never written, followed (a symbolic link is
// refused) or removed, and it must be a regular file of this user.
int lock_output(const std::string& lock_path, const std::string& output, bool& created, std::string& error) {
  for (int attempt = 0; attempt < 8; ++attempt) {
    created = false;
    int fd = open(lock_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd >= 0) {
      created = true;
    } else if (errno == EEXIST) {
      fd = open(lock_path.c_str(), O_RDWR | O_NOFOLLOW | O_CLOEXEC);
      if (fd < 0 && errno == ENOENT) continue;  // removed since: try again
    }
    if (fd < 0) {
      const int err = errno;
      if (err == ELOOP)
        error = "'" + lock_path + "' is a symbolic link; axf-mesher locks the output '" + output + "' with a regular "
                "file of that name, and does not follow links there. Remove it";
      else
        error = "cannot create the lock file '" + lock_path + "' (" + std::strerror(err) + "); is the output "
                "directory writable?";
      return -1;
    }
    struct stat held;
    if (fstat(fd, &held) != 0 || !S_ISREG(held.st_mode) || held.st_uid != geteuid()) {
      close(fd);
      error = "'" + lock_path + "' is not a regular file of yours; axf-mesher locks the output '" + output +
              "' with a file of that name. Remove it, or use another output name";
      return -1;
    }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
      const int err = errno;
      if (err == EWOULDBLOCK) {
        char holder[160] = {0};
        const ssize_t got = pread(fd, holder, sizeof holder - 1, 0);
        if (got > 0) holder[got] = '\0';
        for (char& ch : holder)
          if (ch == '\n') ch = '\0';
        close(fd);
        error = "another axf-mesher (" + std::string(got > 0 ? holder : "host and pid unknown") + ") is writing '" +
                output + "' now; use one output directory per run, or wait for it to finish";
        return -1;
      }
      if (created) unlink(lock_path.c_str());
      close(fd);
      return -2;
    }
    struct stat now;
    if (lstat(lock_path.c_str(), &now) == 0 && S_ISREG(now.st_mode) && held.st_ino == now.st_ino &&
        held.st_dev == now.st_dev) {
      if (created) {
        char host[128] = {0};
        gethostname(host, sizeof host - 1);
        const std::string owner = "host " + std::string(host) + " pid " + std::to_string(static_cast<long>(getpid())) +
                                  "\n";
        if (pwrite(fd, owner.data(), owner.size(), 0) < 0) {
          // the lock holds without the note
        }
      }
      return fd;
    }
    close(fd);  // the file was removed and made again meanwhile: lock the new one
  }
  error = "cannot take the lock '" + lock_path + "' (it keeps being replaced)";
  return -1;
}

// What one run made next to the output, removed when this goes out of scope (also on an exception): its work
// directory, and its lock (the file only if the run created it).
struct CurvedRunFiles {
  std::string lock_path;
  int lock_fd = -1;
  bool lock_created = false;
  std::string work;
  ~CurvedRunFiles() {
    std::error_code ec;
    if (!work.empty()) std::filesystem::remove_all(work, ec);
    if (lock_fd >= 0) {
      if (lock_created) unlink(lock_path.c_str());
      close(lock_fd);
    }
  }
};

// Passes the curved-mesh process's output on to this process's stdout and stderr until both pipes close. That process
// writes to pipes and reads /dev/null, whatever this one has: with stdin, stdout or stderr a regular file, the curved
// mesh of a STEP input differed even with nothing written.
void relay_output(int out_fd, int err_fd) {
  std::signal(SIGPIPE, SIG_IGN);  // a closed reader must not end this process before the mesher does
  struct pollfd fds[2] = {{out_fd, POLLIN, 0}, {err_fd, POLLIN, 0}};
  const int to[2] = {STDOUT_FILENO, STDERR_FILENO};
  bool writable[2] = {true, true};
  int open_count = 2;
  char buf[1 << 16];
  while (open_count > 0) {
    if (poll(fds, 2, -1) < 0) {
      if (errno == EINTR) continue;
      break;
    }
    for (int i = 0; i < 2; ++i) {
      if (fds[i].fd < 0 || !(fds[i].revents & (POLLIN | POLLHUP | POLLERR))) continue;
      const ssize_t n = read(fds[i].fd, buf, sizeof buf);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) {
        close(fds[i].fd);
        fds[i].fd = -1;
        --open_count;
        continue;
      }
      for (ssize_t done = 0; writable[i] && done < n;) {
        const ssize_t w = write(to[i], buf + done, static_cast<size_t>(n - done));
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) writable[i] = false;  // keep reading, so that the mesher does not block
        else done += w;
      }
    }
  }
}

// In the curved-mesh process, an error message with the fixed names replaced by the user's paths (names.json).
std::string user_paths(std::string message) {
  json names;
  try {
    std::ifstream f(kCurvedNames);
    names = json::parse(f);
  } catch (const std::exception&) {
    return message;
  }
  std::vector<std::pair<std::string, std::string>> pairs;
  for (auto it = names.begin(); it != names.end(); ++it) pairs.push_back({it.key(), it.value().get<std::string>()});
  std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  std::string out;  // one pass, longest name first, so that no replacement is replaced again
  for (std::size_t i = 0; i < message.size();) {
    bool hit = false;
    for (const auto& [local, path] : pairs)
      if (message.compare(i, local.size(), local) == 0) {
        out += path;
        i += local.size();
        hit = true;
        break;
      }
    if (!hit) out += message[i++];
  }
  return out;
}

// Makes the curved mesh of `in` in the curved-mesh process and waits for it. Returns false, with the reason, if that
// process cannot be set up or started; main() then stops with that reason (`optout_helps`: whether meshing in one
// process could do better, which a full file system, for one, rules out). Otherwise returns true and sets `rc` to the
// exit status for main(); the mesh is at the output path and its info, also on failure, at the input's "info" path.
bool run_curved_process(const json& in, int& rc, std::string& why, bool& optout_helps) {
  namespace fs = std::filesystem;
  optout_helps = true;
  const std::string info_path = in.value("info", std::string("mesh_info.json"));
  auto fail = [&](const std::string& error) {  // an error in the input or in its use: reported, rc 1
    std::cerr << "axf-mesher: " << error << "\n";
    std::ofstream(info_path) << json_text(json({{"ok", false}, {"error", error}}), 2);
    rc = 1;
    return true;
  };
  if (!in.contains("output") || !in.at("output").is_string()) return fail("the input has no \"output\" path");
  const std::string output = in.at("output").get<std::string>();
  std::error_code ec;
  const fs::path output_dir = fs::absolute(output).parent_path();
  if (!fs::is_directory(output_dir, ec))
    return fail("the output directory '" + output_dir.string() + "' does not exist; create it first");
  std::vector<std::pair<std::string, std::string>> links;
  json canonical;
  try {
    canonical = curved_input(in, links);
  } catch (const std::exception& e) {
    return fail(e.what());
  }
  const std::string stem = curved_stem(output);
  CurvedRunFiles files;
  files.lock_path = stem + ".axf-mesher.lock";
  std::string lock_error;
  files.lock_fd = lock_output(files.lock_path, output, files.lock_created, lock_error);
  if (files.lock_fd == -1) return fail(lock_error);
  auto full = [&](int err) {
    if (err == ENOSPC || err == EDQUOT) optout_helps = false;
  };
  std::string work_template = stem + ".axf-mesher.XXXXXX";
  if (!mkdtemp(work_template.data())) {
    full(errno);
    why = "cannot create a work directory next to the output '" + output + "' (" + std::strerror(errno) + ")";
    return false;
  }
  files.work = work_template;
  const std::string& work = files.work;
  json names = json::object();
  for (const auto& [name, target] : links) {
    fs::create_symlink(target, fs::path(work) / name, ec);
    if (ec) {
      full(ec.value());
      why = "cannot make the link '" + name + "' to '" + target + "' in its work directory (" + ec.message() + ")";
      return false;
    }
    names[name] = target;
  }
  // the handoff files are in the work directory, which is gone when an error is read
  const std::string handoff = links.front().first + ".handoff";
  names[handoff + ".json"] = handoff + ".json (in the work directory next to '" + output + "')";
  names[handoff] = handoff + " (in the work directory next to '" + output + "')";
  {
    std::ofstream f(fs::path(work) / kCurvedInput);
    f << canonical.dump(1);
    std::ofstream g(fs::path(work) / kCurvedNames);
    g << json_text(names, 1);
    if (!f || !g) {
      full(errno);
      why = "cannot write in its work directory next to the output '" + output + "'";
      return false;
    }
  }
  std::string exe_why;
  const int exe_fd = executable_in_memory(exe_why);
  const std::vector<std::string> env = curved_env();
  std::vector<char*> envp;
  for (const auto& s : env) envp.push_back(const_cast<char*>(s.c_str()));
  envp.push_back(nullptr);
  std::string arg0 = "axf-mesher";
  std::string arg1 = kCurvedInput;
  char* args[] = {arg0.data(), arg1.data(), nullptr};
  // status_pipe: the child writes {1, errno} there if the copy in memory cannot be executed (it then tries the file),
  // and {2, errno} if the file cannot either (closed on a successful exec); out_pipe and err_pipe: the process's stdout
  // and stderr (relay_output), and /dev/null its stdin
  int status_pipe[2];
  int out_pipe[2];
  int err_pipe[2];
  const int null_fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
  if (null_fd < 0 || pipe2(status_pipe, O_CLOEXEC) != 0 || pipe2(out_pipe, O_CLOEXEC) != 0 ||
      pipe2(err_pipe, O_CLOEXEC) != 0) {
    if (exe_fd >= 0) close(exe_fd);
    why = std::string("cannot open /dev/null or make pipes (") + std::strerror(errno) + ")";
    return false;
  }
  std::cout.flush();
  std::cerr.flush();
  std::fflush(nullptr);
  const pid_t parent = getpid();
  const pid_t pid = fork();
  if (pid < 0) {
    for (int fd : {status_pipe[0], status_pipe[1], out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1], null_fd})
      close(fd);
    if (exe_fd >= 0) close(exe_fd);
    why = std::string("cannot fork (") + std::strerror(errno) + ")";
    return false;
  }
  if (pid == 0) {
    close(status_pipe[0]);
    prctl(PR_SET_PDEATHSIG, SIGKILL);  // it (and its meshing child) die with this process
    if (getppid() != parent) std::_Exit(1);
    int report[2] = {0, 0};
    if (dup2(null_fd, STDIN_FILENO) < 0 || dup2(out_pipe[1], STDOUT_FILENO) < 0 ||
        dup2(err_pipe[1], STDERR_FILENO) < 0 || chdir(work.c_str()) != 0) {
      report[0] = 2;
      report[1] = errno;
    } else {
      if (exe_fd >= 0) {
        fexecve(exe_fd, args, envp.data());
        report[0] = 1;  // the copy in memory cannot be executed (vm.memfd_noexec, SELinux, ...): the file
        report[1] = errno;
        if (write(status_pipe[1], report, sizeof report) != static_cast<ssize_t>(sizeof report)) std::_Exit(126);
      }
      execve("/proc/self/exe", args, envp.data());
      report[0] = 2;
      report[1] = errno;
    }
    if (write(status_pipe[1], report, sizeof report) != static_cast<ssize_t>(sizeof report)) std::_Exit(126);
    std::_Exit(127);
  }
  g_curved_pid = pid;
  for (int sig : {SIGTERM, SIGINT, SIGHUP}) {  // passed on (forward_stop); not if the caller ignores them
    struct sigaction old = {};
    sigaction(sig, nullptr, &old);
    if (old.sa_handler == SIG_IGN) continue;
    struct sigaction act = {};
    act.sa_handler = forward_stop;
    sigemptyset(&act.sa_mask);
    sigaction(sig, &act, nullptr);
  }
  close(status_pipe[1]);
  close(out_pipe[1]);
  close(err_pipe[1]);
  close(null_fd);
  if (exe_fd >= 0) close(exe_fd);
  int reports[4] = {0, 0, 0, 0};
  std::size_t got = 0;
  while (got < sizeof reports) {
    const ssize_t n = read(status_pipe[0], reinterpret_cast<char*>(reports) + got, sizeof reports - got);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    got += static_cast<std::size_t>(n);
  }
  close(status_pipe[0]);
  relay_output(out_pipe[0], err_pipe[0]);
  int wstatus = 0;
  while (waitpid(pid, &wstatus, 0) < 0 && errno == EINTR) {
  }
  g_curved_pid = 0;
  std::string executable = exe_fd >= 0 ? "in memory" : "file (" + exe_why + ")";
  for (std::size_t r = 0; r + 1 < got / sizeof(int); r += 2) {
    if (reports[r] == 1) executable = std::string("file (the copy in memory cannot be executed: ") +
                                      std::strerror(reports[r + 1]) + ")";
    if (reports[r] == 2) {
      why = "cannot start it for the output '" + output + "' (" + std::strerror(reports[r + 1]) + ")";
      return false;
    }
  }
  json info;
  try {
    std::ifstream f(fs::path(work) / kCurvedInfo);
    info = json::parse(f);
  } catch (const std::exception&) {
    info = json();
  }
  if (!info.is_object()) {
    std::string what;
    if (g_stop_signal)
      what = "was stopped by signal " + std::to_string(g_stop_signal) + ", sent to axf-mesher";
    else if (WIFSIGNALED(wstatus))
      what = "was killed by signal " + std::to_string(WTERMSIG(wstatus)) + " (9: out of memory or a job limit; 11: a "
             "crash in gmsh)";
    else
      what = "exited with status " + std::to_string(WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : -1) +
             " without its mesh info";
    info = {{"ok", false}, {"error", "the curved-mesh process for the output '" + output + "' " + what}};
    std::cerr << "axf-mesher: " << info["error"].get<std::string>() << "\n";
  }
  if (info.contains("step") && info["step"].is_object() && in.contains("step_output"))
    info["step"]["file"] = in.at("step_output");
  if (info.contains("high_order_canonical")) info["high_order_executable"] = executable;
  std::ofstream(info_path) << json_text(info, 2);
  rc = WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : 1;
  if (rc == 0 && !info.value("ok", false)) rc = 1;
  return true;
}
#endif

}  // namespace

int main(int argc, char** argv) {
#ifdef __linux__
  // stdin, stdout and stderr open (/dev/null where a caller closed one), so that no file of the mesher's lands on
  // fds 0-2
  for (int fd = 0; fd <= 2; ++fd) {
    if (fcntl(fd, F_GETFD) != -1 || errno != EBADF) continue;
    const int null_fd = open("/dev/null", O_RDWR);
    if (null_fd >= 0 && null_fd != fd) {
      dup2(null_fd, fd);
      close(null_fd);
    }
  }
#endif
  // stdout's buffer is static, so that it is not allocated at the first write (not needed for the curved-mesh
  // process, whose stdout is a pipe)
  static char stdout_buffer[8192];
  setvbuf(stdout, stdout_buffer, isatty(STDOUT_FILENO) ? _IOLBF : _IOFBF, sizeof stdout_buffer);
  if (argc != 2) {
    std::cerr << "usage: axf-mesher input.json\n";
    return 2;
  }
  const char* aslr_env = std::getenv("AXF_MESHER_ASLR");
  const std::string aslr = aslr_env ? aslr_env : "";
  if (aslr != "" && aslr != "off" && aslr != "on") {
    std::cerr << "axf-mesher: AXF_MESHER_ASLR must be off (the default), on or empty, not '" << aslr << "'\n";
    return 2;
  }
#ifdef __linux__
  // Re-run once with address-space randomization off (what `setarch -R` does). The curved-element steps order by
  // address (mesh3d), and with randomization on, the memory the geometry leaves behind, and with it a curved mesh,
  // still differed in about 3 runs in 140 (none in 88 with it off). AXF_MESHER_ASLR=on keeps randomization.
  const int persona = personality(0xffffffff);
  if (aslr != "on" && persona != -1 && !(persona & ADDR_NO_RANDOMIZE) &&
      personality(static_cast<unsigned long>(persona) | ADDR_NO_RANDOMIZE) != -1) {
    execv("/proc/self/exe", argv);
    personality(static_cast<unsigned long>(persona));  // exec failed: carry on with randomization
  }
#endif
  // Settings that would change meshes behind the input's back: gmsh's ~/.gmshrc and ~/.gmsh-options are not read
  // (gmsh::initialize below), and PETSc's options, which gmsh's elastic pass reads for its linear solve, are cleared.
  unsetenv("PETSC_OPTIONS");
  auto t0 = std::chrono::steady_clock::now();
  json in;
  try {
    std::ifstream f(argv[1]);
    in = json::parse(f);
  } catch (std::exception& e) {
    std::cerr << "axf-mesher: cannot read input: " << e.what() << "\n";
    return 2;
  }
  // A curved 3D mesh is made by the curved-mesh process (run_curved_process), unless this is it. If that process
  // cannot be started, the mesher stops: a curved mesh made here would depend on the output path again.
  // AXF_MESHER_HO_ISOLATE=0 is the way to mesh in one process.
  if (curved_isolated(in)) {
#ifdef __linux__
    if (curved_stage() && std::string(argv[1]) == kCurvedInput) {
      g_canonical = "on";
    } else {
      int rc = 0;
      std::string why;
      bool optout_helps = true;
      bool started = false;
      try {
        started = run_curved_process(in, rc, why, optout_helps);
      } catch (const std::exception& e) {  // what the run made is removed already (CurvedRunFiles)
        why = e.what();
        optout_helps = false;
      }
      if (g_stop_signal) {  // stopped by a signal: raise it again, now that the run has cleaned up
        const int sig = g_stop_signal;
        std::signal(sig, SIG_DFL);
        std::raise(sig);
      }
      if (started) return rc;
      std::string error = "cannot start the curved-mesh process (" + why + ")";
      if (optout_helps)
        error += "; set AXF_MESHER_HO_ISOLATE=0 to mesh in one process (curved meshes then differ from run to run and "
                 "between output directories)";
      std::cerr << "axf-mesher: " << error << "\n";
      const json failed = {{"ok", false}, {"error", error}};
      std::ofstream(in.value("info", std::string("mesh_info.json"))) << json_text(failed, 2);
      return 1;
    }
#else
    g_canonical = "off: Linux only";
#endif
  }
  json info;
  try {
    if (g_canonical == "on") {
      // gmsh's -noenv: no setenv() of PATH and PYTHONPATH at startup (whose array grows with the environment). The
      // API's option parser does not know -noenv and would warn, so "-v 0" comes first and the verbosity is reset.
      static char a0[] = "axf-mesher";
      static char a1[] = "-v";
      static char a2[] = "0";
      static char a3[] = "-noenv";
      static char* gmsh_argv[] = {a0, a1, a2, a3, nullptr};
      gmsh::initialize(4, gmsh_argv, false);
      gmsh::option::setNumber("General.Verbosity", 5);
    } else {
      gmsh::initialize(0, nullptr, false);
    }
    gmsh::option::setNumber("General.Terminal", in.value("verbose", 0));
    gmsh::option::setNumber("General.NumThreads", 1);
    gmsh::model::add("axf");
    if (in.value("plane", false)) {  // (x, y) cross-section of a z-uniform cavity
      mesh_plane(in, info);
      gmsh::finalize();
      std::ofstream(in.value("info", std::string("mesh_info.json")))
          << info.dump(2, ' ', false, json::error_handler_t::replace);
      return 0;
    }
    if (in.value("dim", 2) == 3) {
      mesh3d(in, info);
      gmsh::finalize();
      std::ofstream(in.value("info", std::string("mesh_info.json")))
          << info.dump(2, ' ', false, json::error_handler_t::replace);
      return 0;
    }

    const double Rb = in.at("domain_radius");
    const double z0 = in.value("domain_center_z", 0.0);  // centre of the outer semicircle on the axis
    const int axis_attr = in.value("axis_attr", 900), outer_attr = in.value("outer_attr", 901);
    const int air_attr = in.value("air_attr", 1);

    // Half disk r >= 0, centred at (0, z0).
    int disk = occ::addDisk(0, z0, 0, Rb, Rb);
    int box = occ::addRectangle(0, z0 - Rb, 0, Rb, 2 * Rb);
    DimTags half;
    std::vector<DimTags> hm;
    occ::intersect({{2, disk}}, {{2, box}}, half, hm);

    // Inputs: [half, region shapes..., curves...]; remember which input slots
    // belong to which item.
    DimTags objects = half;
    std::vector<int> owner;  // owner[k] = item index for objects[k]; -1 domain
    owner.assign(objects.size(), -1);
    const json& regions = in.value("regions", json::array());
    const json& curves = in.value("curves", json::array());
    for (size_t i = 0; i < regions.size(); ++i) {
      DimTags s = build_shape(regions[i].at("shape"));
      for (auto& e : s) {
        objects.push_back(e);
        owner.push_back(static_cast<int>(i));
      }
    }
    const int nreg = static_cast<int>(regions.size());
    for (size_t i = 0; i < curves.size(); ++i) {
      DimTags s = build_path(curves[i].at("path"));
      for (auto& e : s) {
        objects.push_back(e);
        owner.push_back(nreg + static_cast<int>(i));
      }
    }

    DimTags frag;
    std::vector<DimTags> fmap;
    occ::fragment(objects, {}, frag, fmap);
    occ::synchronize();

    // Owner of each resulting surface: highest-priority region containing it.
    std::map<int, int> surf_owner;  // surface tag -> region index or -1 (air)
    std::set<int> in_domain;
    std::map<int, std::set<int>> curve_items;  // curve tag -> curve item indices
    for (size_t k = 0; k < fmap.size(); ++k) {
      for (auto& e : fmap[k]) {
        if (e.first == 2) {
          if (owner[k] < 0) in_domain.insert(e.second);
          if (owner[k] >= 0 && owner[k] < nreg) {
            auto it = surf_owner.find(e.second);
            if (it == surf_owner.end() || it->second < owner[k]) surf_owner[e.second] = owner[k];
          } else if (!surf_owner.count(e.second)) {
            surf_owner[e.second] = -1;
          }
        } else if (e.first == 1 && owner[k] >= nreg) {
          curve_items[e.second].insert(owner[k] - nreg);
        }
      }
    }
    DimTags all2;
    gmsh::model::getEntities(all2, 2);
    DimTags drop;
    for (auto& e : all2) {
      if (!in_domain.count(e.second)) {
        double cx, cy, cz;
        occ::getCenterOfMass(2, e.second, cx, cy, cz);
        if (cx < 0) {  // piece at r < 0 (e.g. a disk centred on the axis): clip
          drop.push_back(e);
          surf_owner.erase(e.second);
          continue;
        }
        throw Fail("a region extends outside the computational domain (radius " + std::to_string(Rb) +
                   "); increase mesh.domain_radius or check the geometry");
      }
      if (!surf_owner.count(e.second)) surf_owner[e.second] = -1;
    }
    if (!drop.empty()) {
      occ::remove(drop, true);
      occ::synchronize();
      gmsh::model::getEntities(all2, 2);
    }

    // Physical groups for regions.
    std::map<int, std::vector<int>> by_attr;
    std::map<int, std::string> attr_name;
    for (auto& [tag, ow] : surf_owner) {
      int attr = ow < 0 ? air_attr : regions[ow].at("attr").get<int>();
      by_attr[attr].push_back(tag);
      attr_name[attr] = ow < 0 ? "air" : regions[ow].at("name").get<std::string>();
    }
    json groups = json::array();
    for (auto& [attr, tags] : by_attr) {
      add_physical_group(2, tags, attr, attr_name[attr]);
      groups.push_back({{"dim", 2}, {"attr", attr}, {"name", attr_name[attr]}, {"entities", tags.size()}});
    }
    for (size_t i = 0; i < regions.size(); ++i) {
      int attr = regions[i].at("attr");
      if (!by_attr.count(attr)) {
        throw Fail("region '" + regions[i].at("name").get<std::string>() +
                   "' is completely covered by later regions; reorder or remove it");
      }
    }

    // Boundary of the domain: axis (r ~ 0) and outer arc.
    DimTags bnd;
    gmsh::model::getBoundary(all2, bnd, true, false, false);
    std::vector<int> axis, outer;
    for (auto& e : bnd) {
      // classify by points on the curve: OCC bounding boxes are inflated by the geometric tolerance, so a
      // bounding-box test never recognizes the axis (it would be tagged "outer")
      const int tag = std::abs(e.second);
      std::vector<double> pmin, pmax;
      gmsh::model::getParametrizationBounds(1, tag, pmin, pmax);
      double rmax = 0;
      for (double t : {0.0, 0.25, 0.5, 0.75, 1.0}) {
        std::vector<double> xyz;
        gmsh::model::getValue(1, tag, {pmin[0] + t * (pmax[0] - pmin[0])}, xyz);
        rmax = std::max(rmax, std::fabs(xyz[0]));
      }
      if (rmax < 1e-9 * Rb) axis.push_back(tag);
      else outer.push_back(tag);
    }
    gmsh::model::addPhysicalGroup(1, axis, axis_attr, "axis");
    gmsh::model::addPhysicalGroup(1, outer, outer_attr, "outer");
    groups.push_back({{"dim", 1}, {"attr", axis_attr}, {"name", "axis"}});
    groups.push_back({{"dim", 1}, {"attr", outer_attr}, {"name", "outer"}});
    for (size_t i = 0; i < curves.size(); ++i) {
      std::vector<int> tags;
      for (auto& [tag, items] : curve_items) {
        if (items.count(static_cast<int>(i))) tags.push_back(tag);
      }
      int attr = curves[i].at("attr");
      add_physical_group(1, tags, attr, curves[i].at("name").get<std::string>());
      groups.push_back({{"dim", 1}, {"attr", attr}, {"name", curves[i].at("name")}, {"entities", tags.size()}});
    }

    // ------------------------------------------------------------ mesh size
    const double h_dev = in.at("size");
    const double h_s = in.value("surface_size", h_dev);
    const double h_far = in.value("far_size", Rb / 8);
    const double r_dev = in.value("device_radius", Rb / 4);
    gmsh::option::setNumber("Mesh.MeshSizeExtendFromBoundary", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromPoints", 0);
    gmsh::option::setNumber("Mesh.MeshSizeFromCurvature", in.value("curvature_points", 24));
    gmsh::option::setNumber("Mesh.MeshSizeMax", h_far);
    double floor_2d = 0;
    {  // the floor must not clip the smallest requested size (per-region/curve sizes included)
      double hmin = std::min(h_s, h_dev);
      for (auto& g : regions) {
        for (const char* key : {"size", "surface_size"}) {
          const double v = g.value(key, 0.0);
          if (v > 0) hmin = std::min(hmin, v);
        }
      }
      for (auto& c : curves) {
        for (const char* key : {"size", "surface_size"}) {
          const double v = c.value(key, 0.0);
          if (v > 0) hmin = std::min(hmin, v);
        }
      }
      floor_2d = hmin * 0.25;
      gmsh::option::setNumber("Mesh.MeshSizeMin", floor_2d);
    }
    std::vector<double> fields;
    // far-field growth from the device, measured from the domain centre (0, z0)
    int ff = gmsh::model::mesh::field::add("MathEval");
    char expr[320];
    std::snprintf(expr, sizeof expr, "max(%.17g, %.17g + 0.25*(sqrt(x^2+(y-(%.17g))^2) - %.17g))", h_dev, h_dev, z0,
                  r_dev);
    gmsh::model::mesh::field::setString(ff, "F", expr);
    fields.push_back(ff);
    // per-region sizes
    for (size_t i = 0; i < regions.size(); ++i) {
      double h = regions[i].value("size", 0.0);
      if (h <= 0) continue;
      int attr = regions[i].at("attr");
      std::vector<double> tags;
      for (int t : by_attr[attr]) tags.push_back(t);
      int fc = gmsh::model::mesh::field::add("MathEval");
      std::snprintf(expr, sizeof expr, "%.17g", h);
      gmsh::model::mesh::field::setString(fc, "F", expr);
      int fr = gmsh::model::mesh::field::add("Restrict");
      gmsh::model::mesh::field::setNumber(fr, "InField", fc);
      gmsh::model::mesh::field::setNumbers(fr, "SurfacesList", tags);
      fields.push_back(fr);
    }
    // Distance + Threshold refinement towards a set of curves with a given target size.
    const double layer = in.value("surface_layer", 6.0 * h_dev);
    auto refine_towards = [&](const std::vector<double>& cs, double hmin, double hmax) {
      if (cs.empty() || hmin <= 0) return;
      int fd = gmsh::model::mesh::field::add("Distance");
      gmsh::model::mesh::field::setNumbers(fd, "CurvesList", cs);
      gmsh::model::mesh::field::setNumber(fd, "Sampling", 200);
      int ft = gmsh::model::mesh::field::add("Threshold");
      gmsh::model::mesh::field::setNumber(ft, "InField", fd);
      gmsh::model::mesh::field::setNumber(ft, "SizeMin", hmin);
      gmsh::model::mesh::field::setNumber(ft, "SizeMax", std::max(hmax, hmin));
      gmsh::model::mesh::field::setNumber(ft, "DistMin", 0.5 * hmin);
      gmsh::model::mesh::field::setNumber(ft, "DistMax", std::max(layer, 4.0 * hmin));
      fields.push_back(ft);
    };
    // per-region surface_size: its own boundary refined to its own size
    for (size_t i = 0; i < regions.size(); ++i) {
      const double hr = regions[i].value("surface_size", 0.0);
      if (hr <= 0) continue;
      DimTags sd, cb;
      for (int t : by_attr[regions[i].at("attr").get<int>()]) sd.push_back({2, t});
      gmsh::model::getBoundary(sd, cb, false, false, false);
      std::vector<double> cs;
      for (auto& e : cb) cs.push_back(std::abs(e.second));
      // SizeMax must not undercut the device size: a Threshold field returns SizeMax at every
      // point beyond DistMax, and this one is not Restricted to the region
      refine_towards(cs, hr, std::max(h_dev, hr));
    }
    // refinement towards boundaries of selected regions and curves without their own surface_size
    std::vector<double> refine_curves;
    const std::vector<std::string> refine_list = in.value("refine_regions", std::vector<std::string>{});
    std::set<std::string> refine(refine_list.begin(), refine_list.end());
    for (auto& [attr, tags] : by_attr) {
      if (!refine.count(attr_name[attr])) continue;
      DimTags sd, cb;
      for (int t : tags) sd.push_back({2, t});
      gmsh::model::getBoundary(sd, cb, false, false, false);
      for (auto& e : cb) refine_curves.push_back(std::abs(e.second));
    }
    // per-curve surface_size (sheets); the rest follow the global surface_size
    std::map<int, double> curve_hs;  // curve item index -> its own surface size
    for (size_t i = 0; i < curves.size(); ++i) {
      const double hc = curves[i].value("surface_size", 0.0);
      if (hc > 0) curve_hs[static_cast<int>(i)] = hc;
    }
    std::map<int, std::vector<double>> own_curves;  // item index -> OCC curve tags
    for (auto& [tag, items] : curve_items) {
      int own = -1;
      for (int it : items) if (curve_hs.count(it)) { own = it; break; }
      if (own >= 0) own_curves[own].push_back(tag);
      else refine_curves.push_back(tag);
    }
    for (auto& [item, cs] : own_curves) refine_towards(cs, curve_hs[item], h_dev);
    refine_towards(refine_curves, h_s, h_dev);
    {
      // adaptive cycles: the floor of cycle 0 (per-region and curve sizes included), lowered to the smallest adapted size
      double hmin = floor_2d;
      const int fa = size_field_from_file(in, &hmin);
      if (fa >= 0) {
        fields.push_back(fa);
        gmsh::option::setNumber("Mesh.MeshSizeMin", hmin);
      }
    }
    int fmin = gmsh::model::mesh::field::add("Min");
    gmsh::model::mesh::field::setNumbers(fmin, "FieldsList", fields);
    gmsh::model::mesh::field::setAsBackgroundMesh(fmin);

    gmsh::option::setNumber("Mesh.Algorithm", 6);
    gmsh::model::mesh::generate(2);
    int order = in.value("order", 2);
    if (order > 1) {
      gmsh::option::setNumber("Mesh.SecondOrderLinear", 0);
      gmsh::model::mesh::setOrder(std::min(order, 2));
    }
    // Orient every surface mesh counter-clockwise in (r, z).
    for (auto& e : all2) {
      std::vector<int> et;
      std::vector<std::vector<std::size_t>> etags, enodes;
      gmsh::model::mesh::getElements(et, etags, enodes, 2, e.second);
      if (et.empty() || enodes[0].size() < 3) continue;
      std::vector<double> c[3], pc;
      int dimn, tagn;
      for (int q = 0; q < 3; ++q) gmsh::model::mesh::getNode(enodes[0][q], c[q], pc, dimn, tagn);
      double area = (c[1][0] - c[0][0]) * (c[2][1] - c[0][1]) - (c[2][0] - c[0][0]) * (c[1][1] - c[0][1]);
      if (area < 0) gmsh::model::mesh::reverse({{2, e.second}});
    }

    gmsh::option::setNumber("Mesh.MshFileVersion", 2.2);
    gmsh::option::setNumber("Mesh.Binary", 0);
    gmsh::write(in.at("output").get<std::string>());

    std::vector<std::size_t> ntags;
    std::vector<double> ncoord, npar;
    gmsh::model::mesh::getNodes(ntags, ncoord, npar);
    std::vector<int> et;
    std::vector<std::vector<std::size_t>> etags, enodes;
    gmsh::model::mesh::getElements(et, etags, enodes, 2);
    size_t nel = 0;
    for (auto& v : etags) nel += v.size();
    std::string ver;
    gmsh::option::getString("General.Version", ver);
    info = {{"ok", true},
            {"gmsh_version", ver},
            {"nodes", ntags.size()},
            {"elements_2d", nel},
            {"groups", groups},
            {"domain_radius", Rb},
            {"domain_center_z", z0},
            {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()}};
    gmsh::finalize();
  } catch (std::exception& e) {
    std::string error = e.what();
#ifdef __linux__
    if (g_canonical == "on") error = user_paths(error);  // the user's file names, not the work directory's
#endif
    info = {{"ok", false}, {"error", error}};
    std::cerr << "axf-mesher: " << error << "\n";
  }
  std::ofstream(in.value("info", std::string("mesh_info.json")))
      << info.dump(2, ' ', false, json::error_handler_t::replace);
  return info.value("ok", false) ? 0 : 1;
}
