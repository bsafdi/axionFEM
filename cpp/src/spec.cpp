#include "axionfem/spec.hpp"
#include "axionfem/pml.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>

#include "axionfem/core.hpp"
#include "axionfem/prescribed_source.hpp"
#include "axionfem/surface_resistance.hpp"
#include "axionfem/units.hpp"

namespace axf::spec {
namespace {

// True while a "symmetry": "xsec" design is parsed: shapes and paths live in the (x, y) cross-section plane
// (no r >= 0 restriction; rect also accepts "x"/"y" for "r"/"z"; disks may be placed by pivot, arm and angle).
thread_local bool g_planar = false;
struct PlanarGuard {
  explicit PlanarGuard(bool on) { g_planar = on; }
  ~PlanarGuard() { g_planar = false; }
};

// Reads values from a JSON node while tracking its path for error messages.
struct Node {
  const json& j;
  std::string path;
  const ParamMap* params;

  Node at(const std::string& key) const {
    return Node{j.at(key), path + "/" + key, params};
  }
  Node at(size_t i) const { return Node{j.at(i), strcat(path, "/", i), params}; }
  bool has(const std::string& key) const { return j.is_object() && j.contains(key); }

  double num() const {
    if (j.is_number()) return j.get<double>();
    if (j.is_string()) return eval_expr(j.get<std::string>(), *params, path);
    throw Error("expected a number or an expression string", path, "Example: 2.5 or \"0.6*R\".");
  }
  double num(const std::string& key, double def) const { return has(key) ? at(key).num() : def; }
  double req_num(const std::string& key) const {
    if (!has(key)) throw Error("missing required field '" + key + "'", path);
    return at(key).num();
  }
  std::string str() const {
    if (!j.is_string()) throw Error("expected a string", path);
    return j.get<std::string>();
  }
  std::string str(const std::string& key, const std::string& def) const {
    return has(key) ? at(key).str() : def;
  }
  std::string req_str(const std::string& key) const {
    if (!has(key)) throw Error("missing required field '" + key + "'", path);
    return at(key).str();
  }
  bool boolean(const std::string& key, bool def) const {
    if (!has(key)) return def;
    if (!j.at(key).is_boolean()) throw Error("expected true or false", path + "/" + key);
    return j.at(key).get<bool>();
  }
  Vec2 vec2() const {
    if (!j.is_array() || j.size() != 2) throw Error("expected a point [r, z]", path);
    return Vec2{at(size_t(0)).num(), at(size_t(1)).num()};
  }
  std::pair<double, double> range() const {
    if (!j.is_array() || j.size() != 2) throw Error("expected a range [min, max]", path);
    double a = at(size_t(0)).num(), b = at(size_t(1)).num();
    if (!(b > a)) throw Error(strcat("range must satisfy min < max (got ", a, ", ", b, ")"), path);
    return {a, b};
  }
  void only(std::initializer_list<const char*> allowed) const {
    if (!j.is_object()) throw Error("expected an object", path);
    std::set<std::string> ok(allowed.begin(), allowed.end());
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (!ok.count(it.key())) {
        std::string list;
        for (auto& a : ok) list += (list.empty() ? "" : ", ") + a;
        throw Error("unknown field '" + it.key() + "'", path, "Allowed fields here: " + list + ".");
      }
    }
  }
};

void check_nonneg_r(double r, const std::string& path) {
  if (g_planar) return;  // cross-section designs: (x, y) anywhere in the plane
  if (r < -1e-12) {
    throw Error(strcat("radius r = ", r, " is negative"), path,
                "Axisymmetric geometry lives in the half-plane r >= 0; r is the distance from the symmetry axis.");
  }
}

// Per-vertex polygon fillets: check that the tangent points of neighbouring corners fit on the
// shared edge.  Mirrors polygon_wire() in cpp/apps/axf_mesher.cpp, so that a bad radius is a
// spec error with a path instead of an OpenCASCADE failure.
void check_polygon_fillets(const std::vector<Vec2>& P, const std::vector<double>& f, const std::string& path) {
  const size_t n = P.size();
  if (f.size() != n) return;
  std::vector<double> d(n, 0.0), len(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    const Vec2& p = P[(i + n - 1) % n];
    const Vec2& v = P[i];
    const Vec2& q = P[(i + 1) % n];
    const double ux = p.r - v.r, uy = p.z - v.z, wx = q.r - v.r, wy = q.z - v.z;
    const double lu = std::hypot(ux, uy), lw = std::hypot(wx, wy);
    if (!(lu > 0) || !(lw > 0)) {
      throw Error(strcat("polygon vertex ", i, " coincides with its neighbour"), path,
                  "Remove the duplicate point.");
    }
    len[i] = lw;  // length of edge i -> i+1
    if (f[i] <= 0) continue;
    const double ang = std::acos(std::clamp((ux * wx + uy * wy) / (lu * lw), -1.0, 1.0));
    if (ang < 1e-6 || std::fabs(M_PI - ang) < 1e-6) continue;  // no corner to round
    d[i] = f[i] / std::tan(ang / 2);
  }
  for (size_t i = 0; i < n; ++i) {
    const size_t j = (i + 1) % n;
    if (d[i] + d[j] > len[i] * (1 + 1e-9)) {
      throw Error(strcat("polygon fillets at vertices ", i, " and ", j, " need ", d[i], " + ", d[j], " = ",
                         d[i] + d[j], " m of the edge between them, which is only ", len[i], " m long"),
                  path, "Reduce the fillet radii or move the vertices apart.");
    }
  }
}

Shape parse_shape(const Node& n) {
  if (!n.j.is_object() || n.j.size() != 1) {
    throw Error("a shape must be an object with exactly one key", n.path,
                "Use one of rect, frame, polygon, contour, disk, union, difference.");
  }
  const std::string k = n.j.begin().key();
  Node b = n.at(k);
  Shape s;
  if (k == "rect" || k == "frame") {
    if (g_planar) b.only({"r", "z", "x", "y", "fillet", "thickness"});
    else b.only({"r", "z", "fillet", "thickness"});
    s.kind = (k == "rect") ? Shape::Kind::Rect : Shape::Kind::Frame;
    const char* kr = "r";
    const char* kz = "z";
    if (g_planar) {  // cross-section: "x"/"y" are the names of the two coordinates ("r"/"z" accepted too)
      if (b.has("x") == b.has("r") || b.has("y") == b.has("z"))
        throw Error("a cross-section rect needs exactly one of 'x' or 'r' and one of 'y' or 'z'", b.path,
                    "Use {\"rect\": {\"x\": [x0, x1], \"y\": [y0, y1]}}.");
      if (b.has("x")) kr = "x";
      if (b.has("y")) kz = "y";
    }
    auto [r0, r1] = b.at(kr).range();
    auto [z0, z1] = b.at(kz).range();
    s.r0 = r0; s.r1 = r1; s.z0 = z0; s.z1 = z1;
    check_nonneg_r(r0, b.path + "/r");
    s.fillet = b.num("fillet", 0.0);
    if (s.fillet < 0) throw Error("fillet must be >= 0", b.path + "/fillet");
    double minside = std::min(r1 - r0, z1 - z0);
    if (s.fillet > 0.5 * minside * (1 + 1e-12)) {
      throw Error(strcat("fillet ", s.fillet, " exceeds half the smallest side ", 0.5 * minside), b.path + "/fillet");
    }
    if (s.kind == Shape::Kind::Frame) {
      s.thickness = b.req_num("thickness");
      if (!(s.thickness > 0) || 2 * s.thickness >= minside) {
        throw Error(strcat("frame thickness must be > 0 and < half the smallest side (", 0.5 * minside, ")"),
                    b.path + "/thickness");
      }
    } else if (b.has("thickness")) {
      throw Error("'thickness' only applies to frame shapes", b.path + "/thickness");
    }
  } else if (k == "polygon") {
    s.kind = Shape::Kind::Polygon;
    Node pts = b.has("points") ? b.at("points") : b;
    if (b.j.is_object()) b.only({"points", "fillet", "fillets"});
    if (!pts.j.is_array() || pts.j.size() < 3) throw Error("a polygon needs at least 3 points", pts.path);
    if (b.j.is_object()) s.fillet = b.num("fillet", 0.0);
    if (s.fillet < 0) throw Error("fillet must be >= 0", b.path + "/fillet");
    // Points are [r, z] or [r, z, fillet_radius]; a per-vertex radius overrides the shape's `fillet`.
    std::vector<double> per(pts.j.size(), s.fillet);
    bool any_per = false;
    for (size_t i = 0; i < pts.j.size(); ++i) {
      Node p = pts.at(i);
      if (!p.j.is_array() || p.j.size() < 2 || p.j.size() > 3)
        throw Error("expected a point [r, z] or [r, z, fillet]", p.path);
      s.points.push_back(Vec2{p.at(size_t(0)).num(), p.at(size_t(1)).num()});
      check_nonneg_r(s.points.back().r, p.path);
      if (p.j.size() == 3) {
        per[i] = p.at(size_t(2)).num();
        any_per = true;
        if (per[i] < 0) throw Error("a per-vertex fillet radius must be >= 0", strcat(p.path, "/2"));
      }
    }
    if (b.j.is_object() && b.has("fillets")) {
      if (any_per)
        throw Error("give per-vertex fillet radii either as a third point coordinate or as 'fillets', not both",
                    b.path + "/fillets");
      Node fl = b.at("fillets");
      if (!fl.j.is_array() || fl.j.size() != s.points.size())
        throw Error(strcat("'fillets' must be a list of ", s.points.size(), " radii, one per polygon point"),
                    fl.path, "Omit it, or use a third coordinate [r, z, fillet] on the points instead.");
      for (size_t i = 0; i < fl.j.size(); ++i) {
        per[i] = fl.at(i).num();
        if (per[i] < 0) throw Error("a fillet radius must be >= 0", fl.at(i).path);
      }
      any_per = true;
    }
    if (any_per) {
      check_polygon_fillets(s.points, per, b.path);
      s.fillets = per;
    } else if (s.fillet > 0) {
      check_polygon_fillets(s.points, per, b.path + "/fillet");
    }
  } else if (k == "disk") {
    if (g_planar) b.only({"center", "radius", "pivot", "arm", "angle_deg"});
    else b.only({"center", "radius"});
    s.kind = Shape::Kind::Disk;
    if (b.has("arm") || b.has("angle_deg")) {
      // cross-section rod placement: centre = pivot + arm (cos angle, sin angle), angle from +x towards +y
      if (b.has("center"))
        throw Error("give a disk either 'center' or 'arm'/'angle_deg' (with an optional 'pivot'), not both", b.path);
      const double arm = b.req_num("arm"), ang = b.req_num("angle_deg") * M_PI / 180.0;
      if (!(arm >= 0)) throw Error("arm must be >= 0 (the distance from the pivot to the disk centre)", b.path + "/arm");
      const Vec2 pv = b.has("pivot") ? b.at("pivot").vec2() : Vec2{0.0, 0.0};
      s.center = Vec2{pv.r + arm * std::cos(ang), pv.z + arm * std::sin(ang)};
    } else {
      if (b.has("pivot")) throw Error("'pivot' needs 'arm' and 'angle_deg'", b.path + "/pivot");
      s.center = b.at("center").vec2();
    }
    s.radius = b.req_num("radius");
    if (!(s.radius > 0)) throw Error("disk radius must be > 0", b.path + "/radius");
    check_nonneg_r(s.center.r, b.path + "/center");  // parts at r < 0 are clipped (sphere on the axis)
  } else if (k == "union" || k == "difference") {
    s.kind = (k == "union") ? Shape::Kind::Union : Shape::Kind::Difference;
    if (!b.j.is_array() || b.j.size() < 2) throw Error(k + " needs a list of at least two shapes", b.path);
    for (size_t i = 0; i < b.j.size(); ++i) s.children.push_back(parse_shape(b.at(i)));
  } else if (k == "contour") {
    // closed loop of lines and axis-aligned elliptical arcs; the mesher builds the arcs exactly
    // (rational quadratic B-splines), so a curved wall is smooth instead of faceted (docs/design_format.md)
    b.only({"start", "segments"});
    s.kind = Shape::Kind::Contour;
    s.start = b.at("start").vec2();
    check_nonneg_r(s.start.r, b.path + "/start");
    Node sg = b.at("segments");
    if (!sg.j.is_array() || sg.j.empty()) throw Error("a contour needs a non-empty list of segments", sg.path);
    Vec2 cur = s.start;
    for (size_t i = 0; i < sg.j.size(); ++i) {
      Node q = sg.at(i);
      if (!q.j.is_object() || q.j.size() != 1)
        throw Error("a contour segment is {\"line\": [r, z]} or {\"ellipse\": {\"center\": [r, z], \"axes\": [ar, az], "
                    "\"t\": [t0, t1]}}", q.path);
      const std::string sk = q.j.begin().key();
      Node v = q.at(sk);
      Shape::ContourSeg c;
      if (sk == "line") {
        c.to = v.vec2();
        check_nonneg_r(c.to.r, v.path);
      } else if (sk == "ellipse") {
        v.only({"center", "axes", "t"});
        c.ellipse = true;
        c.center = v.at("center").vec2();
        Node ax = v.at("axes");
        if (!ax.j.is_array() || ax.j.size() != 2) throw Error("expected the semi-axes [ar, az]", ax.path);
        c.ar = ax.at(size_t(0)).num();
        c.az = ax.at(size_t(1)).num();
        if (!(c.ar > 0) || !(c.az > 0)) throw Error("ellipse semi-axes must be > 0", ax.path);
        Node tt = v.at("t");
        if (!tt.j.is_array() || tt.j.size() != 2) throw Error("expected the parameter range [t0, t1] (radians)", tt.path);
        c.t0 = tt.at(size_t(0)).num();
        c.t1 = tt.at(size_t(1)).num();
        if (!(std::fabs(c.t1 - c.t0) > 1e-12) || std::fabs(c.t1 - c.t0) >= 2 * M_PI)
          throw Error(strcat("ellipse arc parameter range [", c.t0, ", ", c.t1, "] must span more than 0 and less than "
                             "2 pi"), tt.path, "point(t) = center + (ar cos t, az sin t), t in radians.");
        const Vec2 p0{c.center.r + c.ar * std::cos(c.t0), c.center.z + c.az * std::sin(c.t0)};
        const double scale = std::max({c.ar, c.az, std::fabs(c.center.r), std::fabs(c.center.z)});
        if (std::hypot(p0.r - cur.r, p0.z - cur.z) > 1e-9 * scale)
          throw Error(strcat("ellipse arc starts at (", p0.r, ", ", p0.z, ") but the contour is at (", cur.r, ", ", cur.z,
                             ")"), v.path + "/t", "Each segment must start where the previous one ends.");
        c.to = Vec2{c.center.r + c.ar * std::cos(c.t1), c.center.z + c.az * std::sin(c.t1)};
        if (std::fabs(c.t1 - c.t0) * std::max(c.ar, c.az) <= 1e-7)
          throw Error("ellipse arc is shorter than 1e-7 m (the CAD kernel's tolerance)", v.path + "/t");
        {  // r >= 0 along the arc: its smallest r is at an end point or at t = pi (mod 2 pi) inside [t0, t1]
          const double lo = std::min(c.t0, c.t1), hi = std::max(c.t0, c.t1);
          const double kpi = std::ceil((lo - M_PI) / (2 * M_PI));
          const bool has_pi = M_PI + 2 * M_PI * kpi <= hi;
          const double rmin = has_pi ? c.center.r - c.ar
                                     : std::min(c.center.r + c.ar * std::cos(c.t0), c.center.r + c.ar * std::cos(c.t1));
          check_nonneg_r(rmin + 1e-12 * scale, v.path);
        }
      } else {
        throw Error("unknown contour segment '" + sk + "'", q.path, "Use line or ellipse.");
      }
      // OpenCASCADE merges points closer than 1e-7 m (its confusion tolerance): shorter segments cannot be built
      if (!c.ellipse && std::hypot(c.to.r - cur.r, c.to.z - cur.z) <= 1e-7)
        throw Error(strcat("contour segment ", i, " is shorter than 1e-7 m (the CAD kernel's tolerance)"), q.path,
                    "Remove the segment or merge it with its neighbour.");
      cur = c.to;
      s.segments.push_back(c);
    }
    if (s.segments.size() < 2) throw Error("a contour needs at least two segments", sg.path);
    {  // the closing segment back to start: either absent (end == start) or a proper line
      const double gap = std::hypot(cur.r - s.start.r, cur.z - s.start.z);
      if (gap > 0 && gap <= 1e-7 && gap > 1e-12)
        throw Error("the contour ends within 1e-7 m of its start but not on it (the closing line would be degenerate)",
                    sg.path, "End the last segment exactly at 'start', or leave a gap > 1e-7 m for a closing line.");
    }
  } else if (k == "imported") {
    // COMSOL 2D import: the region's elements are read from mesh.file (attribute 2 + region index)
    if (!b.j.is_object()) throw Error("imported must be an object (provenance, e.g. {\"comsol_domains\": [3]})", b.path);
    s.kind = Shape::Kind::Imported;
    s.imported = b.j;
  } else {
    throw Error("unknown shape type '" + k + "'", n.path, "Use one of rect, frame, polygon, contour, disk, union, difference.");
  }
  return s;
}

Path parse_path(const Node& n) {
  if (!n.j.is_object() || n.j.size() != 1) {
    throw Error("a path must be an object with exactly one key", n.path, "Use one of segment, polyline, rect, arc.");
  }
  const std::string k = n.j.begin().key();
  Node b = n.at(k);
  Path p;
  if (k == "segment") {
    p.kind = Path::Kind::Segment;
    if (!b.j.is_array() || b.j.size() != 2) throw Error("a segment is [[r0, z0], [r1, z1]]", b.path);
    p.points = {b.at(size_t(0)).vec2(), b.at(size_t(1)).vec2()};
  } else if (k == "polyline") {
    b.only({"points", "closed"});
    p.kind = Path::Kind::Polyline;
    Node pts = b.at("points");
    if (!pts.j.is_array() || pts.j.size() < 2) throw Error("a polyline needs at least 2 points", pts.path);
    for (size_t i = 0; i < pts.j.size(); ++i) p.points.push_back(pts.at(i).vec2());
    p.closed = b.boolean("closed", false);
  } else if (k == "rect") {
    if (g_planar) b.only({"r", "z", "x", "y", "fillet"});
    else b.only({"r", "z", "fillet"});
    p.kind = Path::Kind::Rect;
    const char* kr = (g_planar && b.has("x")) ? "x" : "r";
    const char* kz = (g_planar && b.has("y")) ? "y" : "z";
    auto [r0, r1] = b.at(kr).range();
    auto [z0, z1] = b.at(kz).range();
    p.r0 = r0; p.r1 = r1; p.z0 = z0; p.z1 = z1;
    p.fillet = b.num("fillet", 0.0);
  } else if (k == "arc") {
    b.only({"center", "radius", "angles"});
    p.kind = Path::Kind::Arc;
    p.center = b.at("center").vec2();
    p.radius = b.req_num("radius");
    auto [a0, a1] = b.at("angles").range();
    p.angle0 = a0; p.angle1 = a1;
  } else {
    throw Error("unknown path type '" + k + "'", n.path, "Use one of segment, polyline, rect, arc.");
  }
  for (auto& q : p.points) check_nonneg_r(q.r, b.path);
  return p;
}

Vec3 vec3(const Node& n) {
  if (!n.j.is_array() || n.j.size() != 3) throw Error("expected a point/vector [x, y, z]", n.path);
  return Vec3{n.at(size_t(0)).num(), n.at(size_t(1)).num(), n.at(size_t(2)).num()};
}

Solid parse_solid(const Node& n) {
  if (!n.j.is_object()) throw Error("a solid must be an object", n.path);
  Solid s;
  std::string k;
  for (auto it = n.j.begin(); it != n.j.end(); ++it) {
    if (it.key() == "rotate" || it.key() == "translate") continue;
    if (!k.empty()) throw Error("a solid has exactly one primitive key (plus optional rotate/translate)", n.path);
    k = it.key();
  }
  if (k.empty()) throw Error("missing solid primitive", n.path,
                             "Use box, cylinder, sphere, torus, cone, revolve, extrude, wedge_frame, step, union, difference, "
                             "intersection (or imported, with mesh.file).");
  Node b = n.at(k);
  if (k == "box") {
    b.only({"min", "max"});
    s.kind = Solid::Kind::Box;
    s.p0 = vec3(b.at("min"));
    s.p1 = vec3(b.at("max"));
    if (!(s.p1.x > s.p0.x && s.p1.y > s.p0.y && s.p1.z > s.p0.z)) throw Error("box needs max > min in x, y, z", b.path);
  } else if (k == "cylinder" || k == "cone") {
    s.kind = k == "cylinder" ? Solid::Kind::Cylinder : Solid::Kind::Cone;
    if (k == "cylinder") b.only({"base", "axis", "radius"});
    else b.only({"base", "axis", "r1", "r2"});
    s.p0 = vec3(b.at("base"));
    s.p1 = vec3(b.at("axis"));
    if (k == "cylinder") s.r1 = b.req_num("radius");
    else { s.r1 = b.req_num("r1"); s.r2 = b.req_num("r2"); }
    if (!(s.r1 > 0)) throw Error("radius must be positive", b.path);
  } else if (k == "sphere") {
    b.only({"center", "radius"});
    s.kind = Solid::Kind::Sphere;
    s.center = vec3(b.at("center"));
    s.r1 = b.req_num("radius");
  } else if (k == "torus") {
    b.only({"center", "axis", "major", "minor"});
    s.kind = Solid::Kind::Torus;
    s.center = vec3(b.at("center"));
    if (b.has("axis")) s.axis = vec3(b.at("axis"));
    s.r1 = b.req_num("major");
    s.r2 = b.req_num("minor");
    if (!(s.r1 > s.r2 && s.r2 > 0)) throw Error("torus needs major > minor > 0", b.path);
  } else if (k == "revolve") {
    b.only({"profile", "center", "axis", "angles"});
    s.kind = Solid::Kind::Revolve;
    s.profile = parse_shape(b.at("profile"));
    if (b.has("center")) s.center = vec3(b.at("center"));
    if (b.has("axis")) s.axis = vec3(b.at("axis"));
    if (b.has("angles")) {
      auto [a0, a1] = b.at("angles").range();
      s.angle0 = a0;
      s.angle1 = a1;
      if (a1 - a0 > 360 + 1e-9) throw Error("revolve angle range exceeds 360 degrees", b.path + "/angles");
    }
  } else if (k == "extrude") {
    b.only({"polygon", "z"});
    s.kind = Solid::Kind::Extrude;
    Node pts = b.at("polygon");
    if (!pts.j.is_array() || pts.j.size() < 3) throw Error("extrude polygon needs >= 3 points", pts.path);
    for (size_t i = 0; i < pts.j.size(); ++i) s.polygon.push_back(pts.at(i).vec2());
    auto [z0, z1] = b.at("z").range();
    s.z0 = z0;
    s.z1 = z1;
  } else if (k == "step") {
    b.only({"file", "scale"});
    s.kind = Solid::Kind::Step;
    s.file = b.req_str("file");
    s.scale = b.num("scale", 1.0);
  } else if (k == "wedge_frame") {
    // boolean-free polyhedral coil-case frame (see cpp/apps/axf_mesher.cpp).
    // Every value is evaluated here (expressions of named parameters allowed) and the mesher gets numbers only,
    // like the other solids (axf-mesher does not evaluate expressions).
    b.only({"polygon", "hole", "tan_a", "c_pos", "c_neg"});
    s.kind = Solid::Kind::WedgeFrame;
    json w = json::object();
    w["tan_a"] = b.req_num("tan_a");
    w["c_pos"] = b.req_num("c_pos");
    w["c_neg"] = b.req_num("c_neg");
    const double ta = w["tan_a"], cp = w["c_pos"], cn = w["c_neg"];
    for (const char* key : {"polygon", "hole"}) {
      if (!b.has(key)) { if (std::string(key) == "polygon") throw Error("wedge_frame needs a polygon", b.path); continue; }
      Node pts = b.at(key);
      if (!pts.j.is_array() || pts.j.size() < 3) throw Error(std::string("wedge_frame ") + key + " needs >= 3 points", pts.path);
      json arr = json::array();
      for (size_t i = 0; i < pts.j.size(); ++i) {
        const Vec2 q = pts.at(i).vec2();
        if (!(2 * ta * q.r - cp - cn > 0))
          throw Error(strcat("wedge_frame width 2*tan_a*x - c_pos - c_neg = ", 2 * ta * q.r - cp - cn, " is not positive at x = ", q.r),
                      strcat(pts.path, "/", i), "Every polygon and hole vertex needs x > (c_pos + c_neg) / (2 tan_a).");
        arr.push_back(json::array({q.r, q.z}));
      }
      w[key] = arr;
    }
    s.wedge = w;
  } else if (k == "imported") {
    // COMSOL import (docs/comsol_import.md): the region's elements are read from mesh.file with the
    // region's attribute; there is no CAD.  The object is free-form provenance (e.g. the COMSOL domain numbers).
    if (!b.j.is_object()) throw Error("imported must be an object (provenance, e.g. {\"comsol_domains\": [6]})", b.path);
    if (n.has("rotate") || n.has("translate"))
      throw Error("an imported region cannot be transformed: its elements come from mesh.file as they are", n.path);
    s.kind = Solid::Kind::Imported;
    s.imported = b.j;
  } else if (k == "union" || k == "difference" || k == "intersection") {
    s.kind = k == "union" ? Solid::Kind::Union : k == "difference" ? Solid::Kind::Difference : Solid::Kind::Intersection;
    if (!b.j.is_array() || b.j.size() < 2) throw Error(k + " needs a list of at least two solids", b.path);
    for (size_t i = 0; i < b.j.size(); ++i) s.children.push_back(parse_solid(b.at(i)));
  } else {
    throw Error("unknown solid '" + k + "'", n.path,
                "Use box, cylinder, sphere, torus, cone, revolve, extrude, wedge_frame, step, union, difference, "
                "intersection.");
  }
  if (n.has("rotate")) {
    Node r = n.at("rotate");
    r.only({"axis", "angle_deg", "origin"});
    s.has_rotation = true;
    s.rot_axis = vec3(r.at("axis"));
    s.rot_deg = r.req_num("angle_deg");
    if (r.has("origin")) s.rot_origin = vec3(r.at("origin"));
  }
  if (n.has("translate")) s.translate = vec3(n.at("translate"));
  return s;
}

Constraint parse_constraint(const Node& n) {
  n.only({"net_current", "flux"});
  if (n.j.size() != 1) throw Error("give exactly one of net_current or flux", n.path);
  Constraint c;
  if (n.has("net_current")) {
    c.kind = Constraint::Kind::NetCurrent;
    c.value = n.at("net_current").num();
  } else {
    c.kind = Constraint::Kind::Flux;
    c.value = n.at("flux").num();
  }
  return c;
}

// ---- mechanics / thermal material blocks ----
void parse_mechanical(const Node& n, MechanicalProps& m) {
  n.only({"E_Pa", "nu", "alpha", "notes"});
  m.given = true;
  m.E_Pa = n.req_num("E_Pa");
  m.nu = n.req_num("nu");
  m.alpha = n.num("alpha", 0.0);
  m.notes = n.str("notes", "");
  if (!(m.E_Pa > 0)) throw Error("E_Pa (Young's modulus) must be positive", n.path + "/E_Pa");
  if (!(m.nu > -1.0 && m.nu < 0.5)) {
    throw Error(strcat("Poisson's ratio nu = ", m.nu, " must satisfy -1 < nu < 0.5"), n.path + "/nu",
                "nu = 0.5 is incompressible and makes the displacement formulation singular.");
  }
}

void parse_thermal_material(const Node& n, ThermalProps& t) {
  n.only({"k_W_mK", "notes"});
  t.given = true;
  t.notes = n.str("notes", "");
  Node k = n.at("k_W_mK");
  if (k.j.is_array()) {
    if (k.j.size() < 2) throw Error("a k(T) table needs at least two [T, k] pairs", k.path);
    for (size_t i = 0; i < k.j.size(); ++i) {
      Node row = k.at(i);
      if (!row.j.is_array() || row.j.size() != 2) throw Error("each k(T) entry is [T_K, k_W_mK]", row.path);
      t.kT_T.push_back(row.at(size_t(0)).num());
      t.kT_k.push_back(row.at(size_t(1)).num());
    }
    for (size_t i = 0; i < t.kT_T.size(); ++i) {
      if (!(t.kT_k[i] > 0)) throw Error("k must be positive", k.path);
      if (i > 0 && !(t.kT_T[i] > t.kT_T[i - 1]))
        throw Error("the k(T) table must be strictly increasing in T", k.path);
    }
    t.k_const = t.kT_k.front();
  } else {
    t.k_const = k.num();
    if (!(t.k_const > 0)) throw Error("k_W_mK must be positive", k.path);
  }
}
// ---- end mechanics / thermal material blocks ----

Material parse_material(const std::string& name, const Node& n) {
  n.only({"kind", "mu_r", "eps_r", "tan_delta", "sigma_S_m", "bh", "B_sat", "B", "H", "density", "warm", "cost_per_kg",
          "rf", "mechanical", "thermal"});
  Material m;
  m.name = name;
  std::string k = n.req_str("kind");
  if (k == "vacuum") m.kind = MaterialKind::Vacuum;
  else if (k == "linear") m.kind = MaterialKind::Linear;
  else if (k == "iron") m.kind = MaterialKind::Iron;
  else if (k == "superconductor") m.kind = MaterialKind::Superconductor;
  else if (k == "conductor") m.kind = MaterialKind::Conductor;
  else throw Error("unknown material kind '" + k + "'", n.path + "/kind",
                   "Use vacuum, linear, iron, superconductor or conductor.");
  m.mu_r = n.num("mu_r", m.kind == MaterialKind::Iron ? 1000.0 : 1.0);
  if (!(m.mu_r >= 1.0)) throw Error("mu_r must be >= 1", n.path + "/mu_r");
  m.eps_r = n.num("eps_r", 1.0);
  if (!(m.eps_r >= 1.0)) throw Error("eps_r must be >= 1", n.path + "/eps_r");
  m.tan_delta = n.num("tan_delta", 0.0);
  if (!(m.tan_delta >= 0.0)) throw Error("tan_delta must be >= 0", n.path + "/tan_delta");
  m.sigma_S_m = n.num("sigma_S_m", 0.0);
  if (!(m.sigma_S_m >= 0.0)) throw Error("sigma_S_m must be >= 0", n.path + "/sigma_S_m");
  if (m.sigma_S_m > 0 && (m.kind == MaterialKind::Superconductor || m.kind == MaterialKind::Conductor))
    throw Error("sigma_S_m applies to non-PEC media (vacuum, linear, iron)", n.path + "/sigma_S_m",
                "'conductor' and 'superconductor' are perfect conductors at RF; use kind 'linear' (mu_r 1) for a "
                "resistive metal whose eddy currents the full-wave solver should resolve.");
  if (m.kind == MaterialKind::Linear && n.has("B_sat")) {
    // a linear material's saturation flux density: used only as the saturation check of thin layers
    m.bh.B_sat = n.num("B_sat", 0.0);
    m.bh.model = "given";
    if (!(m.bh.B_sat > 0)) throw Error("B_sat must be > 0 (T)", n.path + "/B_sat");
  }
  if (m.kind == MaterialKind::Iron) {
    m.bh.model = n.str("bh", "knee");
    m.bh.mu_r = m.mu_r;
    m.bh.B_sat = n.num("B_sat", 2.0);
    if (m.bh.model == "table") {
      Node B = n.at("B"), H = n.at("H");
      if (!B.j.is_array() || !H.j.is_array() || B.j.size() != H.j.size() || B.j.size() < 3) {
        throw Error("a B-H table needs arrays B and H of equal length >= 3", n.path);
      }
      for (size_t i = 0; i < B.j.size(); ++i) {
        m.bh.B.push_back(B.at(i).num());
        m.bh.H.push_back(H.at(i).num());
      }
      for (size_t i = 1; i < m.bh.B.size(); ++i) {
        if (!(m.bh.B[i] > m.bh.B[i - 1]) || !(m.bh.H[i] > m.bh.H[i - 1])) {
          throw Error("B-H table must be strictly increasing in both B and H", n.path,
                      "Start at (0, 0); the solver assumes a monotone (lossless, anhysteretic) curve.");
        }
      }
    } else if (m.bh.model != "knee") {
      throw Error("unknown B-H model '" + m.bh.model + "'", n.path + "/bh", "Use \"knee\" or \"table\".");
    }
    m.density = n.num("density", 7874.0);
    m.warm = n.boolean("warm", true);
  } else {
    m.density = n.num("density", 0.0);
    m.warm = n.boolean("warm", false);
  }
  m.cost_per_kg = n.num("cost_per_kg", 0.0);
  if (n.has("rf")) {
    Node f = n.at("rf");
    f.only({"model", "Rs_ohm", "f_ref", "exponent", "rho_ohm_m", "T_K", "magnetoresistance", "f_Hz", "notes", "RRR",
            "rho_300K_ohm_m", "rho_l_ohm_m2", "specularity", "v_F_m_s", "A_ohm_K", "Delta_over_kTc", "Tc_K",
            "R_res_ohm", "T_grid_K", "Rs_bcs_ohm", "flux_flow", "log_factor"});
    m.rf.model = f.str("model", f.has("rho_ohm_m") ? "skin" : (f.has("f_Hz") ? "table" : "power"));
    m.rf.owner = m.name;
    const std::string& model = m.rf.model;
    if (model != "skin" && model != "power" && model != "table" && model != "anomalous" && model != "bcs")
      throw Error("rf.model must be 'power', 'skin', 'table', 'anomalous' or 'bcs'", f.path + "/model",
                  "See docs/design_format.md (rf surface-resistance models).");
    // fields that belong to one model only: reject them elsewhere (a silently ignored parameter is worse than an error)
    const std::map<std::string, std::set<std::string>> owners = {
        {"Rs_ohm", {"power", "table"}}, {"exponent", {"power"}}, {"f_ref", {"power", "bcs"}},
        {"rho_ohm_m", {"skin", "anomalous"}}, {"f_Hz", {"table", "bcs"}}, {"RRR", {"anomalous"}},
        {"rho_300K_ohm_m", {"anomalous"}}, {"rho_l_ohm_m2", {"anomalous"}}, {"specularity", {"anomalous"}},
        {"v_F_m_s", {"anomalous"}}, {"A_ohm_K", {"bcs"}}, {"Delta_over_kTc", {"bcs"}}, {"Tc_K", {"bcs"}},
        {"R_res_ohm", {"bcs"}}, {"T_grid_K", {"bcs"}}, {"Rs_bcs_ohm", {"bcs"}}, {"log_factor", {"bcs"}},
        {"flux_flow", {"bcs", "power", "table"}}};
    for (auto& [key, ok] : owners)
      if (f.has(key) && !ok.count(model)) {
        std::string list;
        for (auto& o : ok) list += (list.empty() ? "'" : ", '") + o + "'";
        throw Error("rf field '" + key + "' does not belong to the '" + model + "' model", f.path + "/" + key,
                    "It is used by the model(s) " + list + "; remove it or change \"model\".");
      }
    if (model != "table" && f.has("Rs_ohm") && f.at("Rs_ohm").j.is_array())
      throw Error("Rs_ohm is a list: did you mean \"model\": \"table\" with f_Hz?", f.path + "/Rs_ohm");
    m.rf.Rs_ohm = model == "power" ? f.num("Rs_ohm", 0.0) : 0.0;
    m.rf.f_ref = f.num("f_ref", 1e6);
    m.rf.exponent = f.num("exponent", 0.0);
    m.rf.rho_ohm_m = f.num("rho_ohm_m", 0.0);
    m.rf.T_K = f.req_num("T_K");
    if (!(m.rf.T_K > 0)) throw Error("rf T_K must be > 0", f.path + "/T_K");
    if (model == "skin" && !(m.rf.rho_ohm_m > 0)) throw Error("skin model needs rho_ohm_m > 0", f.path);
    if (model == "power" && !(m.rf.Rs_ohm > 0)) throw Error("power model needs Rs_ohm > 0", f.path);
    // log-log (f, R_s) table from the arrays f_Hz and `rkey`
    auto read_f_table = [&]() {
      if (!f.has("f_Hz") || !f.at("f_Hz").j.is_array() || f.at("f_Hz").j.size() < 2)
        throw Error(strcat("the ", model, " model's table needs an array f_Hz of length >= 2"), f.path,
                    "Example: \"rf\": {\"model\": \"table\", \"f_Hz\": [1e9, 1e10], \"Rs_ohm\": [2e-3, 8e-3], \"T_K\": 4}");
      Node F = f.at("f_Hz");
      for (size_t i = 0; i < F.j.size(); ++i) {
        m.rf.tab_f_Hz.push_back(F.at(i).num());
        if (!(m.rf.tab_f_Hz[i] > 0) || (i > 0 && !(m.rf.tab_f_Hz[i] > m.rf.tab_f_Hz[i - 1])))
          throw Error("table f_Hz must be positive and strictly increasing", F.at(i).path);
      }
    };
    if (model == "table") {
      if (!f.has("f_Hz") || !f.has("Rs_ohm") || !f.at("f_Hz").j.is_array() || !f.at("Rs_ohm").j.is_array() ||
          f.at("f_Hz").j.size() != f.at("Rs_ohm").j.size() || f.at("f_Hz").j.size() < 2)
        throw Error("table model needs arrays f_Hz and Rs_ohm of equal length >= 2", f.path,
                    "Example: \"rf\": {\"model\": \"table\", \"f_Hz\": [1e9, 1e10], \"Rs_ohm\": [2e-3, 8e-3], \"T_K\": 4}");
      read_f_table();
      Node Rv = f.at("Rs_ohm");
      for (size_t i = 0; i < Rv.j.size(); ++i) {
        m.rf.tab_Rs_ohm.push_back(Rv.at(i).num());
        if (!(m.rf.tab_Rs_ohm[i] > 0)) throw Error("table Rs_ohm values must be positive", Rv.at(i).path);
      }
    }
    if (model == "anomalous") {
      // resistivity at T_K: rho_ohm_m, or rho_300K_ohm_m / RRR. RRR is the standard residual-resistivity ratio
      // rho(room temperature) / rho(4.2 K): it gives the residual resistivity, so it is only accepted for
      // T_K <= 20 K, where pure copper (and other pure metals) are at their residual value (say which room
      // temperature your RRR refers to; copper: rho(293 K) = 1.678e-8, rho(300 K) = 1.725e-8 ohm m, CRC)
      const bool by_rho = f.has("rho_ohm_m"), by_rrr = f.has("RRR") || f.has("rho_300K_ohm_m");
      if (by_rho == by_rrr)
        throw Error("anomalous model needs either rho_ohm_m (resistivity at T_K) or RRR together with rho_300K_ohm_m",
                    f.path, "Example (copper, RRR 100): \"rf\": {\"model\": \"anomalous\", \"RRR\": 100, "
                    "\"rho_300K_ohm_m\": 1.725e-8, \"rho_l_ohm_m2\": 6.6e-16, \"specularity\": 0, \"T_K\": 4}");
      if (by_rrr) {
        const double rrr = f.req_num("RRR"), r300 = f.req_num("rho_300K_ohm_m");
        if (!(rrr >= 1)) throw Error("RRR must be >= 1", f.path + "/RRR");
        if (!(r300 > 0)) throw Error("rho_300K_ohm_m must be > 0", f.path + "/rho_300K_ohm_m");
        if (m.rf.T_K > 20)
          throw Error(strcat("RRR gives the residual (low-temperature) resistivity rho_300K / RRR, valid below ~20 K; "
                             "at T_K = ", m.rf.T_K, " K the phonon part of rho is not negligible"),
                      f.path + "/RRR",
                      "Give rho_ohm_m at T_K instead (copper: ~2e-9 ohm m at 77 K, 1.725e-8 at 300 K for any RRR).");
        m.rf.rho_ohm_m = r300 / rrr;
      }
      if (!(m.rf.rho_ohm_m > 0)) throw Error("anomalous model needs rho_ohm_m > 0", f.path + "/rho_ohm_m");
      m.rf.rho_l = f.req_num("rho_l_ohm_m2");
      if (!(m.rf.rho_l > 0))
        throw Error("rho_l_ohm_m2 (resistivity x mean free path) must be > 0", f.path + "/rho_l_ohm_m2",
                    "Copper: 6.6e-16 ohm m^2 (free-electron m v_F / n e^2; see docs/physics/surface_resistance.md).");
      m.rf.specularity = f.num("specularity", 0.0);
      if (!(m.rf.specularity >= 0 && m.rf.specularity <= 1))
        throw Error("specularity must be in [0, 1] (0 diffuse, 1 specular)", f.path + "/specularity");
      m.rf.v_F = f.num("v_F_m_s", 0.0);
      if (!(m.rf.v_F >= 0)) throw Error("v_F_m_s must be >= 0 (0: quasi-static, omega tau -> 0)", f.path + "/v_F_m_s");
    }
    if (model == "bcs") {
      m.rf.R_res = f.req_num("R_res_ohm");
      if (!(m.rf.R_res >= 0)) throw Error("R_res_ohm must be >= 0", f.path + "/R_res_ohm");
      m.rf.bcs_table = f.has("Rs_bcs_ohm") || f.has("T_grid_K") || f.has("f_Hz");
      if (m.rf.bcs_table) {
        for (const char* k : {"A_ohm_K", "Delta_over_kTc", "Tc_K", "f_ref", "log_factor"})
          if (f.has(k))
            throw Error(strcat("bcs: '", k, "' belongs to the analytic form, not to a tabulated R_BCS(f, T)"),
                        f.path + "/" + k, "Give either A_ohm_K, f_ref, Delta_over_kTc, Tc_K or f_Hz, T_grid_K, Rs_bcs_ohm.");
        if (!f.has("f_Hz") || !f.has("T_grid_K") || !f.has("Rs_bcs_ohm"))
          throw Error("a tabulated R_BCS needs f_Hz, T_grid_K and Rs_bcs_ohm (rows per T)", f.path,
                      "Example: \"T_grid_K\": [2, 4.2], \"f_Hz\": [1e9, 2e9], \"Rs_bcs_ohm\": [[5e-9, 2e-8], [3e-7, 1.2e-6]]");
        read_f_table();
        Node Tg = f.at("T_grid_K"), Rb = f.at("Rs_bcs_ohm");
        std::vector<double> Ts;
        if (!Tg.j.is_array() || Tg.j.empty()) throw Error("T_grid_K must be a non-empty list", Tg.path);
        for (size_t i = 0; i < Tg.j.size(); ++i) {
          Ts.push_back(Tg.at(i).num());
          if (!(Ts[i] > 0) || (i > 0 && !(Ts[i] > Ts[i - 1])))
            throw Error("T_grid_K must be positive and strictly increasing", Tg.at(i).path);
        }
        const size_t nf = m.rf.tab_f_Hz.size();
        if (Ts.empty() || !Rb.j.is_array() || Rb.j.size() != Ts.size())
          throw Error("bcs table: Rs_bcs_ohm must be a list of rows, one per T_grid_K entry, each with one value per f_Hz",
                      f.path + "/Rs_bcs_ohm", "Example: \"T_grid_K\": [2, 4.2], \"f_Hz\": [1e9, 2e9], "
                      "\"Rs_bcs_ohm\": [[5e-9, 2e-8], [3e-7, 1.2e-6]]");
        std::vector<std::vector<double>> R(Ts.size());
        for (size_t i = 0; i < Ts.size(); ++i) {
          Node row = Rb.at(i);
          if (!row.j.is_array() || row.j.size() != nf)
            throw Error(strcat("bcs table row ", i, " must have ", nf, " values (one per f_Hz)"), row.path);
          for (size_t k = 0; k < nf; ++k) {
            R[i].push_back(row.at(k).num());
            if (!(R[i][k] > 0)) throw Error("Rs_bcs_ohm values must be positive", row.at(k).path);
          }
        }
        // reduce to the material temperature T_K: ln R linear in 1/T (Arrhenius), no extrapolation
        const double T = m.rf.T_K;
        if (!(T >= Ts.front() * (1 - 1e-12) && T <= Ts.back() * (1 + 1e-12)))
          throw Error(strcat("bcs table covers T = ", Ts.front(), " .. ", Ts.back(), " K, not T_K = ", T, " K (no extrapolation)"),
                      f.path + "/T_grid_K");
        size_t i = 0;
        while (i + 2 < Ts.size() && T > Ts[i + 1]) ++i;
        for (size_t k = 0; k < nf; ++k) {
          double r = R[i][k];
          if (Ts.size() > 1) {
            const double t = (1 / T - 1 / Ts[i]) / (1 / Ts[i + 1] - 1 / Ts[i]);
            r = std::exp(std::log(R[i][k]) + t * std::log(R[i + 1][k] / R[i][k]));
          }
          m.rf.tab_Rs_ohm.push_back(r);
        }
      } else {
        m.rf.bcs_A = f.req_num("A_ohm_K");
        m.rf.f_ref = f.req_num("f_ref");
        m.rf.bcs_gap = f.req_num("Delta_over_kTc");
        m.rf.bcs_Tc = f.req_num("Tc_K");
        if (!(m.rf.bcs_A > 0)) throw Error("A_ohm_K must be > 0", f.path + "/A_ohm_K",
                                             "Niobium (Padamsee-style fit): A_ohm_K = 2e-4 with f_ref = 1.5e9.");
        if (!(m.rf.f_ref > 0)) throw Error("f_ref must be > 0", f.path + "/f_ref");
        if (!(m.rf.bcs_gap > 0)) throw Error("Delta_over_kTc must be > 0", f.path + "/Delta_over_kTc");
        if (!(m.rf.bcs_Tc > 0)) throw Error("Tc_K must be > 0", f.path + "/Tc_K");
        m.rf.bcs_log = f.boolean("log_factor", false);
        if (m.rf.bcs_log && !(units::h * m.rf.f_ref < units::kB * m.rf.T_K))
          throw Error(strcat("bcs log_factor normalizes ln(2.246 kT/hf) at f_ref, which needs h f_ref < k T_K "
                             "(f_ref = ", m.rf.f_ref, " Hz, T_K = ", m.rf.T_K, " K)"),
                      f.path + "/log_factor", "Use a tabulated R_BCS(f, T) at this temperature.");
        if (!(m.rf.T_K < 0.5 * m.rf.bcs_Tc))
          throw Error(strcat("the analytic BCS form A/T (f/f_ref)^2 exp(-Delta/kT) holds for T < Tc/2; T_K = ", m.rf.T_K,
                             " K, Tc_K = ", m.rf.bcs_Tc, " K"),
                      f.path + "/T_K", "Use a tabulated R_BCS(f, T) (f_Hz, T_grid_K, Rs_bcs_ohm) from a full BCS code.");
      }
    }
    if (f.has("flux_flow")) {
      Node ff = f.at("flux_flow");
      if (ff.has("Rn_ohm"))
        throw Error("flux_flow: 'Rn_ohm' is not an input; give the physical inputs rho_n_ohm_m and lambda_m",
                    ff.path + "/Rn_ohm",
                    "The vortex term is the full local impedance Re sqrt(i mu0 w (i mu0 w lambda^2 + rho_v)), valid "
                    "also when the vortex resistivity is not small (docs/physics/surface_resistance.md Sec. 4).");
      ff.only({"B_T", "Bc2_T", "rho_n_ohm_m", "lambda_m", "f_depin_Hz", "creep_epsilon", "orientation", "notes"});
      m.rf.flux_flow = true;
      // B_T: a number, or "local" / absent: the local static field at the wall (cavity wall losses)
      m.rf.ff_B = std::numeric_limits<double>::quiet_NaN();
      if (ff.has("B_T") && !(ff.at("B_T").j.is_string() && ff.at("B_T").j.get<std::string>() == "local"))
        m.rf.ff_B = ff.req_num("B_T");
      m.rf.ff_Bc2 = ff.req_num("Bc2_T");
      m.rf.ff_rho_n = ff.req_num("rho_n_ohm_m");
      m.rf.ff_lambda = ff.req_num("lambda_m");
      m.rf.ff_f0 = ff.req_num("f_depin_Hz");
      m.rf.ff_eps = ff.num("creep_epsilon", 0.0);
      if (!(m.rf.ff_Bc2 > 0)) throw Error("flux_flow Bc2_T must be > 0", ff.path + "/Bc2_T");
      if (!std::isnan(m.rf.ff_B) && !(std::fabs(m.rf.ff_B) < m.rf.ff_Bc2))
        throw Error("flux_flow needs |B_T| < Bc2_T (mixed state); above B_c2 the material is normal", ff.path + "/B_T",
                    "Use a normal-metal model (skin, anomalous) for the wall instead.");
      if (!(m.rf.ff_rho_n > 0))
        throw Error("flux_flow rho_n_ohm_m (normal-state resistivity just above Tc) must be > 0", ff.path + "/rho_n_ohm_m",
                    "Typical: Nb3Sn ~1.5e-7, NbTi ~6e-7, Nb (RRR 300) ~5e-10 ohm m.");
      if (!(m.rf.ff_lambda > 0))
        throw Error("flux_flow lambda_m (London penetration depth) must be > 0", ff.path + "/lambda_m",
                    "Typical: Nb ~40e-9, Nb3Sn ~110e-9, NbTi ~300e-9 m.");
      if (!(m.rf.ff_f0 >= 0)) throw Error("flux_flow f_depin_Hz must be >= 0 (0: free flux flow)", ff.path + "/f_depin_Hz");
      if (!(m.rf.ff_eps >= 0 && m.rf.ff_eps <= 1))
        throw Error("flux_flow creep_epsilon must be in [0, 1] (0: no creep, 1: free flux flow)", ff.path + "/creep_epsilon");
      m.rf.ff_orientation = -1;
      if (ff.has("orientation")) {
        Node o = ff.at("orientation");
        if (o.j.is_string()) {
          if (o.j.get<std::string>() != "field")
            throw Error("flux_flow orientation must be \"field\" or a number in [0, 1]", o.path);
        } else {
          m.rf.ff_orientation = o.num();
          if (!(m.rf.ff_orientation >= 0 && m.rf.ff_orientation <= 1))
            throw Error("flux_flow orientation factor must be in [0, 1] (sin^2 of the angle between RF current and "
                        "vortices)", o.path);
        }
      }
    }
    if (f.has("magnetoresistance")) {
      if (model == "anomalous" || model == "bcs")
        throw Error("magnetoresistance is not combined with the '" + model + "' model", f.path + "/magnetoresistance",
                    model == "anomalous"
                        ? "At high field the cyclotron radius falls below the mean free path and the RS model no "
                          "longer applies (magnetoresistance changes the orbits, not the scattering time). Use a "
                          "measured R_s(f) at the operating field (\"table\"); \"anomalous\" with a zero-field rho "
                          "is then an uncontrolled estimate (docs/physics/surface_resistance.md Sec. 2.3)."
                        : "Vortex losses of a superconductor in a static field: use \"flux_flow\".");
      if (m.rf.flux_flow)
        throw Error("magnetoresistance and flux_flow cannot be combined", f.path + "/magnetoresistance");
      Node mr = f.at("magnetoresistance");
      mr.only({"model", "B0_T", "n", "B_T", "factor", "notes"});
      m.rf.mr_model = mr.str("model", mr.has("B_T") ? "table" : "kohler");
      if (m.rf.mr_model == "kohler") {
        m.rf.mr_B0 = mr.req_num("B0_T");
        m.rf.mr_n = mr.num("n", 2.0);
        if (!(m.rf.mr_B0 > 0)) throw Error("magnetoresistance B0_T must be > 0", mr.path + "/B0_T");
      } else if (m.rf.mr_model == "table") {
        Node B = mr.at("B_T"), F = mr.at("factor");
        if (!B.j.is_array() || !F.j.is_array() || B.j.size() != F.j.size() || B.j.size() < 2)
          throw Error("magnetoresistance table needs arrays B_T and factor of equal length >= 2", mr.path);
        for (size_t i = 0; i < B.j.size(); ++i) {
          m.rf.mr_B.push_back(B.at(i).num());
          m.rf.mr_factor.push_back(F.at(i).num());
          if (i > 0 && !(m.rf.mr_B[i] > m.rf.mr_B[i - 1]))
            throw Error("magnetoresistance B_T must be strictly increasing", B.path);
          if (!(m.rf.mr_factor[i] > 0)) throw Error("magnetoresistance factors must be positive", F.path);
        }
      } else {
        throw Error("magnetoresistance model must be 'kohler' or 'table'", mr.path + "/model");
      }
    }
  }
  // mechanics / thermal properties
  if (n.has("mechanical")) parse_mechanical(n.at("mechanical"), m.mech);
  if (n.has("thermal")) parse_thermal_material(n.at("thermal"), m.therm);
  return m;
}

Conductor parse_conductor(const std::string& name, const Node& n, const Conductor& base) {
  n.only({"law", "T_op", "fill_factor", "C0", "alpha", "beta", "gamma", "p", "q", "Bc20", "Tc0", "strain",
          "Ca1", "Ca2", "eps_0a", "Jc0", "B0", "b_exp", "k_anis", "T_ref", "Tc", "m_T", "B_T", "T_K", "Jc",
          "n_value", "E0", "tape_normal", "B_min", "notes"});
  // Defaults: overriding a built-in by name starts from it; otherwise the published parameter set of
  // the requested law supplies the defaults, so a design only gives what it changes.
  const std::string law = n.str("law", base.name.empty() ? std::string("nbti_bottura") : base.law);
  Conductor c;
  if (!base.name.empty() && base.law == law) {
    c = base;
  } else {
    bool found = false;
    for (auto& [bn, bc] : builtin_conductors()) {
      (void)bn;
      if (bc.law != law) continue;
      c = bc;
      found = true;
      break;
    }
    if (!found) c.law = law;
  }
  c.name = name;
  if (law != "nbti_bottura" && law != "nb3sn_durham" && law != "rebco_kim" && law != "table") {
    throw Error("unknown conductor law '" + law + "'", n.path + "/law",
                "Use nbti_bottura (NbTi, Bottura fit), nb3sn_durham (Nb3Sn, Durham/ITER scaling at fixed "
                "strain), rebco_kim (REBCO, anisotropic Kim) or table (Jc on a (B, T) grid).");
  }
  c.T_op = n.num("T_op", c.T_op);
  if (!(c.T_op > 0)) throw Error("T_op must be positive (K)", n.path + "/T_op");
  c.fill_factor = n.num("fill_factor", c.fill_factor);
  if (!(c.fill_factor > 0 && c.fill_factor <= 1))
    throw Error("fill_factor must be in (0, 1]", n.path + "/fill_factor",
                "It is the superconductor cross-section divided by the winding (or tape stack) cross-section, "
                "so the engineering current density is fill_factor times the superconductor's Jc.");
  c.C0 = n.num("C0", c.C0);
  c.alpha = n.num("alpha", c.alpha);
  c.beta = n.num("beta", c.beta);
  c.gamma = n.num("gamma", c.gamma);
  c.p = n.num("p", c.p);
  c.q = n.num("q", c.q);
  c.Bc20 = n.num("Bc20", c.Bc20);
  c.Tc0 = n.num("Tc0", c.Tc0);
  c.strain = n.num("strain", c.strain);
  c.Ca1 = n.num("Ca1", c.Ca1);
  c.Ca2 = n.num("Ca2", c.Ca2);
  c.eps_0a = n.num("eps_0a", c.eps_0a);
  c.Jc0 = n.num("Jc0", c.Jc0);
  c.B0 = n.num("B0", c.B0);
  c.b_exp = n.num("b_exp", c.b_exp);
  c.k_anis = n.num("k_anis", c.k_anis);
  c.T_ref = n.num("T_ref", c.T_ref);
  c.Tc = n.num("Tc", c.Tc);
  c.m_T = n.num("m_T", c.m_T);
  c.n_value = n.num("n_value", c.n_value);
  c.E0 = n.num("E0", c.E0);
  c.B_min = n.num("B_min", c.B_min);
  if (!(c.n_value > 1)) throw Error("n_value must be > 1 (E-J power-law exponent)", n.path + "/n_value");
  if (!(c.E0 > 0)) throw Error("E0 must be positive (V/m, the critical-current criterion)", n.path + "/E0");
  if (c.B_min < 0) throw Error("B_min must be >= 0 (T)", n.path + "/B_min");
  if (n.has("tape_normal")) {
    Vec2 v = n.at("tape_normal").vec2();
    const double L = std::hypot(v.r, v.z);
    if (!(L > 0)) throw Error("tape_normal must be a nonzero direction [n_r, n_z]", n.path + "/tape_normal");
    c.normal_r = v.r / L;
    c.normal_z = v.z / L;
  }
  if (law == "table") {
    Node B = n.at("B_T"), J = n.at("Jc");
    if (!B.j.is_array() || B.j.size() < 2) throw Error("a Jc table needs B_T with at least 2 values [T]", B.path);
    c.tab_B.clear();
    for (size_t i = 0; i < B.j.size(); ++i) {
      c.tab_B.push_back(B.at(i).num());
      if (i > 0 && !(c.tab_B[i] > c.tab_B[i - 1])) throw Error("B_T must be strictly increasing", B.path);
    }
    c.tab_T.clear();
    if (n.has("T_K")) {
      Node T = n.at("T_K");
      if (!T.j.is_array() || T.j.empty()) throw Error("T_K must be a non-empty array [K]", T.path);
      for (size_t i = 0; i < T.j.size(); ++i) {
        c.tab_T.push_back(T.at(i).num());
        if (i > 0 && !(c.tab_T[i] > c.tab_T[i - 1])) throw Error("T_K must be strictly increasing", T.path);
      }
    } else {
      c.tab_T.push_back(c.T_op);  // field-only table: valid at the operating temperature
    }
    const size_t nB = c.tab_B.size(), nT = c.tab_T.size();
    c.tab_Jc.clear();
    if (!J.j.is_array() || J.j.size() != nB * nT) {
      throw Error(strcat("Jc must list ", nB * nT, " values (len(B_T) * len(T_K), row-major with B outer)"), J.path,
                  "Without T_K the table is a function of the field alone, valid at T_op.");
    }
    for (size_t i = 0; i < J.j.size(); ++i) {
      c.tab_Jc.push_back(J.at(i).num());
      if (!(c.tab_Jc.back() >= 0)) throw Error("Jc values must be >= 0 (A/m^2)", J.path);
    }
    if (nT > 1 && c.tab_T.back() <= c.T_op)
      throw Error("the Jc table must reach above T_op for a temperature margin", n.path + "/T_K");
  } else if (n.has("B_T") || n.has("T_K") || n.has("Jc")) {
    throw Error("B_T / T_K / Jc belong to the 'table' law", n.path, "Set \"law\": \"table\".");
  }
  c.anisotropic = law == "rebco_kim" || (law == "table" && c.k_anis != 1.0);
  return c;
}

}  // namespace

double RFLoss::Rs_flux_flow(double f, double B, double s) const {
  if (!flux_flow) return 0.0;
  if (std::isnan(B)) B = ff_B;
  if (std::isnan(B))
    throw Error(strcat("rf flux_flow of material '", owner, "' takes |B| from the local static field, which only the "
                       "cavity wall-loss evaluation provides"),
                "/materials/" + owner + "/rf/flux_flow/B_T", "Give B_T explicitly for this analysis.");
  if (std::isnan(s)) s = ff_orientation >= 0 ? ff_orientation : 1.0;
  return axf::rs::flux_flow(f, ff_rho_n, ff_lambda, B, ff_Bc2, ff_f0, ff_eps, s);
}

double RFLoss::Rs(double f) const { return Rs_base(f) + Rs_flux_flow(f); }

double RFLoss::Rs_base(double f) const {
  if (model == "skin") return axf::rs::skin(f, rho_ohm_m);
  if (model == "anomalous") return axf::rs::anomalous(f, rho_ohm_m, rho_l, specularity, v_F);
  if (model == "power") return Rs_ohm * std::pow(f / f_ref, exponent);
  if (model == "bcs" && !bcs_table) return axf::rs::bcs(f, T_K, bcs_A, f_ref, bcs_gap, bcs_Tc, bcs_log) + R_res;
  if (model == "table" || model == "bcs") {
    const auto& F = tab_f_Hz;
    // relative tolerance at each end (an absolute one scaled by f_max could exceed f_min of a wide table)
    if (!(f >= F.front() * (1 - 1e-12) && f <= F.back() * (1 + 1e-12)))
      throw Error(strcat("rf ", model == "bcs" ? "R_BCS table" : "table", " of material '", owner, "' covers ",
                         F.front(), " to ", F.back(), " Hz, not f = ", f, " Hz (no extrapolation)"),
                  "/materials/" + owner + "/rf", "Extend the table to the frequencies of the analysis.");
    double r;
    if (f <= F.front()) {
      r = tab_Rs_ohm.front();
    } else if (f >= F.back()) {
      r = tab_Rs_ohm.back();
    } else {
      const size_t i = static_cast<size_t>(std::upper_bound(F.begin(), F.end(), f) - F.begin());
      const double t = std::log(f / F[i - 1]) / std::log(F[i] / F[i - 1]);
      r = std::exp(std::log(tab_Rs_ohm[i - 1]) + t * std::log(tab_Rs_ohm[i] / tab_Rs_ohm[i - 1]));
    }
    return r + (model == "bcs" ? R_res : 0.0);
  }
  return 0.0;
}

std::vector<std::pair<std::string, std::string>> RFLoss::warnings(double f) const {
  std::vector<std::pair<std::string, std::string>> w;
  const std::string who = "material '" + owner + "'";
  if (flux_flow)
    w.push_back({"flux_flow",
                 who + ": R_s includes a flux_flow (vortex) term: an order-of-magnitude model with large "
                       "uncertainty (pinning, creep, B_c2(T), lambda and rho_n of real walls; "
                       "docs/physics/surface_resistance.md Sec. 4)" +
                     std::string(ff_orientation >= 0
                                     ? strcat("; fixed orientation factor ", ff_orientation)
                                     : "; orientation factor sin^2(J, B0) per wall point where the engine computes it "
                                       "(cavity wall losses: axisymmetric, cross-section and 3D), else 1 (current perpendicular to the "
                                       "vortices: upper bound)")});
  if (model == "skin") {
    // classical skin in the anomalous regime: alpha with the copper rho*l (6.6e-16 ohm m^2; good metals 4e-16 .. 1e-15)
    const double a = axf::rs::anomalous_alpha(f, rho_ohm_m, 6.6e-16);
    if (a > 1)
      w.push_back({"skin_anomalous",
                   strcat(who, ": classical skin model at f = ", f, " Hz with rho = ", rho_ohm_m,
                          " ohm m is in the anomalous-skin regime (alpha = ", a, " > 1 with the copper rho*l "
                          "6.6e-16 ohm m^2): it underestimates R_s (by ~4 % at alpha = 1, 13 % at 10, 40 % at 1e3, "
                          "diffuse) and overestimates Q. Use rf.model \"anomalous\" (e.g. the builtin 'copper'; "
                          "docs/physics/surface_resistance.md)")});
  }
  if (model == "bcs" && !bcs_table) {
    if (units::h * f > units::kB * T_K)
      w.push_back({"bcs_hf_kT", strcat(who, ": analytic BCS evaluated at h f > k T (f = ", f, " Hz, T_K = ", T_K,
                                       " K), outside its validity (hbar omega << k T); R_BCS there is indicative only")});
    else if (!bcs_log) {
      const double r = axf::rs::bcs_log_ratio(f, f_ref, T_K);
      if (std::fabs(r - 1) > 0.1)
        w.push_back({"bcs_log",
                     strcat(who, ": analytic BCS scaled by f^2 from f_ref = ", f_ref, " Hz to ", f,
                            " Hz omits the Mattis-Bardeen factor ln(2.246 kT/hf), which changes R_BCS by x", r,
                            " here; set \"log_factor\": true or give a tabulated R_BCS(f, T)")});
    }
  }
  return w;
}

double RFLoss::mr(double B) const {
  B = std::fabs(B);
  if (mr_model == "kohler") return 1.0 + std::pow(B / mr_B0, mr_n);
  if (mr_model == "table" && !mr_B.empty()) {
    if (B <= mr_B.front()) return mr_factor.front();
    if (B >= mr_B.back()) return mr_factor.back();
    auto it = std::upper_bound(mr_B.begin(), mr_B.end(), B);
    const size_t i = static_cast<size_t>(it - mr_B.begin());
    const double t = (B - mr_B[i - 1]) / (mr_B[i] - mr_B[i - 1]);
    return mr_factor[i - 1] + t * (mr_factor[i] - mr_factor[i - 1]);
  }
  return 1.0;
}

double RFLoss::Rs_factor(double B) const {
  if (mr_model.empty()) return 1.0;
  return model == "skin" ? std::sqrt(mr(B)) : mr(B);
}

double RFLoss::Rs(double f, double B) const { return Rs(f) * Rs_factor(B); }

// ---------------------------------------------------------------- builtins
std::map<std::string, Material> builtin_materials() {
  std::map<std::string, Material> m;
  auto add = [&](const std::string& name, MaterialKind k) {
    Material x;
    x.name = name;
    x.kind = k;
    m[name] = x;
    return &m[name];
  };
  // ---- structural and thermal defaults (order-of-magnitude handbook values at the stated
  // temperature; see docs/physics/mechanics_thermal.md). Override them per design with the
  // "mechanical" / "thermal" material blocks before quoting engineering numbers.
  auto mech = [](Material* x, double E, double nu, double alpha, const char* notes) {
    x->mech.given = true;
    x->mech.E_Pa = E;
    x->mech.nu = nu;
    x->mech.alpha = alpha;
    x->mech.notes = notes;
  };
  auto therm = [](Material* x, double k, const char* notes) {
    x->therm.given = true;
    x->therm.k_const = k;
    x->therm.notes = notes;
  };
  add("vacuum", MaterialKind::Vacuum);
  add("air", MaterialKind::Vacuum);
  {
    // Stranded, epoxy-impregnated superconducting winding treated as an isotropic smeared solid:
    // roughly half metal (NbTi/Cu or Nb3Sn/Cu) and half epoxy + insulation by volume. Real windings
    // are strongly orthotropic (the hoop direction is stiff, the transverse directions are not), so
    // these numbers give the right order of magnitude only.
    auto* w = add("winding", MaterialKind::Vacuum);  // stranded winding: magnetically vacuum
    mech(w, 80.0e9, 0.30, 3.0e-3,
         "smeared isotropic impregnated coil, ~50 vol% metal + 50 vol% epoxy/insulation; real "
         "windings are orthotropic");
    therm(w, 1.0, "transverse (epoxy-dominated) conductivity of an impregnated coil near 4 K; "
                  "strongly anisotropic in reality");
  }
  {
    auto* sc = add("superconductor", MaterialKind::Superconductor);
    mech(sc, 105.0e9, 0.40, 1.43e-3, "niobium: E and nu near 4 K, alpha = integrated contraction 293 K -> 4 K");
    therm(sc, 5.0, "superconducting niobium near 4 K (electronic conduction suppressed; RRR dependent)");
  }
  {
    auto* nb = add("nb", MaterialKind::Superconductor);
    nb->density = 8570.0;
    mech(nb, 105.0e9, 0.40, 1.43e-3, "niobium: E and nu near 4 K, alpha = integrated contraction 293 K -> 4 K");
    therm(nb, 5.0, "superconducting niobium near 4 K (electronic conduction suppressed; RRR dependent)");
  }
  for (const char* name : {"copper", "copper_classical"}) {
    auto* cu = add(name, MaterialKind::Conductor);
    cu->density = 8960.0;
    // Cold copper, RRR ~ 100: residual resistivity rho = 1.7e-10 ohm m at T_K = 0.03 K.
    // `copper`: Reuter-Sondheimer anomalous skin effect, diffuse (p = 0), quasi-static, with the
    // copper rho*l = 6.6e-16 ohm m^2 (mean free path 3.9 um). It equals the classical value below ~100 kHz
    // (alpha << 1) and is 2.4 % above it at 1 MHz, 10 % at 10 MHz, 30 % at 100 MHz, 1.65x at 1 GHz.
    // `copper_classical`: the classical skin sqrt(pi f mu0 rho) with the same rho, for like-for-like comparisons
    // with classical impedance boundary conditions (COMSOL).
    // docs/physics/surface_resistance.md Sec. 1.1.
    cu->rf.owner = name;
    cu->rf.rho_ohm_m = 1.7e-10;
    cu->rf.T_K = 0.03;
    if (std::string(name) == "copper") {
      cu->rf.model = "anomalous";
      cu->rf.rho_l = 6.6e-16;
      cu->rf.specularity = 0.0;
      cu->rf.v_F = 0.0;
    } else {
      cu->rf.model = "skin";
    }
    mech(cu, 130.0e9, 0.34, 3.26e-3, "OFHC copper near 4 K; alpha = integrated contraction 293 K -> 4 K");
    therm(cu, 600.0, "OFHC copper, RRR ~ 100, near 4 K");
  }
  auto* fe = add("iron", MaterialKind::Iron);
  fe->mu_r = 1000.0;
  fe->bh.model = "knee";
  fe->bh.mu_r = 1000.0;
  fe->bh.B_sat = 2.0;
  fe->density = 7874.0;
  fe->warm = true;
  mech(fe, 211.0e9, 0.29, 1.98e-3,
       "soft iron at room temperature; alpha (contraction 293 K -> 4 K) is only applied when the material is "
       "declared cold (warm: false)");
  therm(fe, 80.0, "pure iron at room temperature (~30 W/(m K) near 4 K)");
  m["iron_knee"] = *fe;
  m["iron_knee"].name = "iron_knee";
  // ---- end structural and thermal defaults ----
  return m;
}

std::map<std::string, Conductor> builtin_conductors() {
  std::map<std::string, Conductor> m;
  {
    // NbTi, Bottura's practical fit with the LHC strand parameters (IEEE TAS 10 (2000) 1054):
    // C0 = 90.3 kA T/mm^2, Bc20 = 14.5 T, Tc0 = 9.2 K, alpha = 0.57, beta = 0.90, gamma = 2.32.
    Conductor c;
    c.name = "nbti";
    c.law = "nbti_bottura";
    c.C0 = 90.3e3 * 1e6;  // kA T/mm^2 -> A T/m^2
    c.alpha = 0.57;
    c.beta = 0.90;
    c.gamma = 2.32;
    c.Bc20 = 14.5;
    c.Tc0 = 9.2;
    c.T_op = 4.2;
    c.B_min = 0.05;  // the fit diverges as B^-0.43 at low field
    c.n_value = 40;
    m["nbti"] = c;
  }
  {
    // Nb3Sn, Durham/ITER unified scaling law with the EU-DEMO/ITER-type strand parameters
    // (C = 83075 A T/mm^2, Bc20 = 33.24 T, Tc0 = 16.34 K, p = 0.593, q = 2.156, Ca1 = 50.06,
    // Ca2 = 0, eps_0a = 3.12e-3); `strain` is the total axial strain (applied + pre-compression).
    Conductor c;
    c.name = "nb3sn";
    c.law = "nb3sn_durham";
    c.C0 = 83075.0 * 1e6;  // A T/mm^2 -> A T/m^2
    c.p = 0.593;
    c.q = 2.156;
    c.Bc20 = 33.24;
    c.Tc0 = 16.34;
    c.Ca1 = 50.06;
    c.Ca2 = 0.0;
    c.eps_0a = 3.12e-3;
    c.strain = 0.0;
    c.T_op = 4.2;
    c.B_min = 0.05;
    c.n_value = 30;
    m["nb3sn"] = c;
  }
  {
    // REBCO coated conductor, anisotropic Kim fit of Grilli et al., IEEE TAS 24 (2014) 8000508,
    // Table II set #1 (77 K): Jc0 = 4.75e10 A/m^2, B0 = 35 mT, b_exp = 0.6, k = 0.25.  The
    // reduced-temperature factor is a convenience extension (see docs/validation/hts.md).
    Conductor c;
    c.name = "rebco";
    c.law = "rebco_kim";
    c.Jc0 = 4.75e10;
    c.B0 = 0.035;
    c.b_exp = 0.6;
    c.k_anis = 0.25;
    c.T_ref = 77.0;
    c.Tc = 92.0;
    c.m_T = 1.5;
    c.T_op = 77.0;
    c.n_value = 25;
    c.normal_r = 0.0;
    c.normal_z = 1.0;  // pancake winding: the harmful field component is the axial one
    c.anisotropic = true;
    m["rebco"] = c;
  }
  return m;
}

Benchmark builtin_benchmark(const std::string& name) {
  // Built-in readout benchmark sets {name, Q, T [K], eta_A}: one optimistic and two conservative choices.
  if (name == "optimistic") return {"optimistic", 2e7, 0.010, 0.1};
  if (name == "conservative") return {"conservative", 1e6, 0.030, 10.0};
  if (name == "conservative_solenoid") return {"conservative_solenoid", 1e5, 0.030, 10.0};
  throw Error("unknown benchmark '" + name + "'", "/benchmarks",
              "Built-in benchmarks: optimistic, conservative, conservative_solenoid; or give {name, Q, T, eta_A}.");
}

// ---------------------------------------------------------------- Shape/Path
json Shape::resolved() const {
  switch (kind) {
    case Kind::Rect:
      return {{"rect", {{"r", {r0, r1}}, {"z", {z0, z1}}, {"fillet", fillet}}}};
    case Kind::Frame:
      return {{"frame", {{"r", {r0, r1}}, {"z", {z0, z1}}, {"fillet", fillet}, {"thickness", thickness}}}};
    case Kind::Polygon: {
      json pts = json::array();
      for (auto& p : points) pts.push_back({p.r, p.z});
      json b = {{"points", pts}, {"fillet", fillet}};
      if (!fillets.empty()) b["fillets"] = fillets;
      return {{"polygon", b}};
    }
    case Kind::Disk:
      return {{"disk", {{"center", {center.r, center.z}}, {"radius", radius}}}};
    case Kind::Contour: {
      json sg = json::array();
      for (auto& c : segments) {
        if (c.ellipse)
          sg.push_back({{"ellipse", {{"center", {c.center.r, c.center.z}}, {"axes", {c.ar, c.az}}, {"t", {c.t0, c.t1}}}}});
        else
          sg.push_back({{"line", {c.to.r, c.to.z}}});
      }
      return {{"contour", {{"start", {start.r, start.z}}, {"segments", sg}}}};
    }
    case Kind::Imported:
      return {{"imported", imported}};
    case Kind::Union:
    case Kind::Difference: {
      json a = json::array();
      for (auto& c : children) a.push_back(c.resolved());
      return {{kind == Kind::Union ? "union" : "difference", a}};
    }
  }
  return {};
}

void Shape::bbox(double& rmin, double& rmax, double& zmin, double& zmax) const {
  rmin = zmin = 1e300;
  rmax = zmax = -1e300;
  auto grow = [&](double r, double z) {
    rmin = std::min(rmin, r); rmax = std::max(rmax, r);
    zmin = std::min(zmin, z); zmax = std::max(zmax, z);
  };
  switch (kind) {
    case Kind::Rect:
    case Kind::Frame: grow(r0, z0); grow(r1, z1); break;
    case Kind::Polygon: for (auto& p : points) grow(p.r, p.z); break;
    case Kind::Contour:
      grow(start.r, start.z);
      for (auto& c : segments) {
        grow(c.to.r, c.to.z);
        if (c.ellipse)
          for (int m = 1; m < 256; ++m) {
            const double t = c.t0 + (c.t1 - c.t0) * m / 256.0;
            grow(c.center.r + c.ar * std::cos(t), c.center.z + c.az * std::sin(t));
          }
      }
      break;
    case Kind::Disk: grow(center.r - radius, center.z - radius); grow(center.r + radius, center.z + radius); break;
    case Kind::Union:
      for (auto& c : children) {
        double a, b, d, e;
        c.bbox(a, b, d, e);
        grow(a, d); grow(b, e);
      }
      break;
    case Kind::Difference: children.front().bbox(rmin, rmax, zmin, zmax); break;
    case Kind::Imported: break;  // extent known only from mesh.file
  }
}

json Path::resolved() const {
  auto pts = [&]() {
    json a = json::array();
    for (auto& p : points) a.push_back({p.r, p.z});
    return a;
  };
  switch (kind) {
    case Kind::Segment: return {{"segment", pts()}};
    case Kind::Polyline: return {{"polyline", {{"points", pts()}, {"closed", closed}}}};
    case Kind::Rect: return {{"rect", {{"r", {r0, r1}}, {"z", {z0, z1}}, {"fillet", fillet}}}};
    case Kind::Arc:
      return {{"arc", {{"center", {center.r, center.z}}, {"radius", radius}, {"angles", {angle0, angle1}}}}};
  }
  return {};
}

void Path::bbox(double& rmin, double& rmax, double& zmin, double& zmax) const {
  rmin = zmin = 1e300;
  rmax = zmax = -1e300;
  auto grow = [&](double r, double z) {
    rmin = std::min(rmin, r); rmax = std::max(rmax, r);
    zmin = std::min(zmin, z); zmax = std::max(zmax, z);
  };
  if (kind == Kind::Rect) {
    grow(r0, z0); grow(r1, z1);
  } else if (kind == Kind::Arc) {
    grow(center.r - radius, center.z - radius); grow(center.r + radius, center.z + radius);
  } else {
    for (auto& p : points) grow(p.r, p.z);
  }
}

json Solid::resolved() const {
  auto v3 = [](const Vec3& v) { return json::array({v.x, v.y, v.z}); };
  json out;
  switch (kind) {
    case Kind::Box: out = {{"box", {{"min", v3(p0)}, {"max", v3(p1)}}}}; break;
    case Kind::Cylinder: out = {{"cylinder", {{"base", v3(p0)}, {"axis", v3(p1)}, {"radius", r1}}}}; break;
    case Kind::Cone: out = {{"cone", {{"base", v3(p0)}, {"axis", v3(p1)}, {"r1", r1}, {"r2", r2}}}}; break;
    case Kind::Sphere: out = {{"sphere", {{"center", v3(center)}, {"radius", r1}}}}; break;
    case Kind::Torus: out = {{"torus", {{"center", v3(center)}, {"axis", v3(axis)}, {"major", r1}, {"minor", r2}}}}; break;
    case Kind::Revolve:
      out = {{"revolve", {{"profile", profile.resolved()}, {"center", v3(center)}, {"axis", v3(axis)},
                          {"angles", {angle0, angle1}}}}};
      break;
    case Kind::Extrude: {
      json pts = json::array();
      for (auto& q : polygon) pts.push_back({q.r, q.z});
      out = {{"extrude", {{"polygon", pts}, {"z", {z0, z1}}}}};
      break;
    }
    case Kind::Step: out = {{"step", {{"file", file}, {"scale", scale}}}}; break;
    case Kind::WedgeFrame: out = {{"wedge_frame", wedge}}; break;
    case Kind::Imported: out = {{"imported", imported}}; break;
    case Kind::Union:
    case Kind::Difference:
    case Kind::Intersection: {
      json a = json::array();
      for (auto& c : children) a.push_back(c.resolved());
      out = {{kind == Kind::Union ? "union" : kind == Kind::Difference ? "difference" : "intersection", a}};
      break;
    }
  }
  if (has_rotation) out["rotate"] = {{"axis", v3(rot_axis)}, {"angle_deg", rot_deg}, {"origin", v3(rot_origin)}};
  if (translate.x != 0 || translate.y != 0 || translate.z != 0) out["translate"] = v3(translate);
  return out;
}

// ---------------------------------------------------------------- Design
const Material& Design::material_of_region(const std::string& region) const {
  for (auto& r : regions) {
    if (r.name == region) return materials.at(r.material);
  }
  for (auto& c : curves) {
    if (c.name == region) return materials.at(c.material);
  }
  throw Error("no region or curve named '" + region + "'");
}

const Region* Design::find_region(const std::string& name) const {
  for (auto& r : regions) if (r.name == name) return &r;
  return nullptr;
}

const Curve* Design::find_curve(const std::string& name) const {
  for (auto& c : curves) if (c.name == name) return &c;
  return nullptr;
}

// fullwave.adaptive (3D): the band, tolerance and budget of an adaptive sweep. `fw` is the fullwave node; the initial
// points become f.frequencies, so everything that reads the sweep's frequencies (mesh sizing, the exterior map's
// multipole order) sees the band.
static void parse_adaptive_sweep(const Node& fw, FullWaveSpec& f, bool is3d) {
  const Node n = fw.at("adaptive");
  n.only({"f_min_Hz", "f_max_Hz", "tolerance", "max_solves", "initial_points", "spacing", "dense_points", "notes"});
  if (!is3d)
    throw Error("fullwave.adaptive is implemented for 3D full-wave sweeps only", n.path,
                "Give the axisymmetric sweep's frequencies as fullwave.frequencies or f_min/f_max/points.");
  if (fw.has("frequencies") || fw.has("f_min") || fw.has("f_max") || fw.has("points"))
    throw Error("give either fullwave.adaptive or 'frequencies'/'f_min'/'f_max'/'points', not both", n.path,
                "The adaptive sweep chooses its own frequencies in [f_min_Hz, f_max_Hz].");
  auto& a = f.adaptive;
  a.enabled = true;
  a.f_min_Hz = n.req_num("f_min_Hz");
  a.f_max_Hz = n.req_num("f_max_Hz");
  if (!(std::isfinite(a.f_min_Hz) && std::isfinite(a.f_max_Hz) && a.f_min_Hz > 0 && a.f_max_Hz > a.f_min_Hz))
    throw Error("fullwave.adaptive needs 0 < f_min_Hz < f_max_Hz (Hz)", n.path);
  // the band must hold distinct frequencies at the candidate grid's resolution
  if (!(a.f_max_Hz - a.f_min_Hz > 1e-9 * a.f_max_Hz))
    throw Error("fullwave.adaptive band is narrower than 1e-9 of f_max_Hz", n.path,
                "Widen the band, or list the frequencies in fullwave.frequencies.");
  a.tolerance = n.num("tolerance", a.tolerance);
  if (!(a.tolerance >= 1e-6 && a.tolerance <= 0.1))
    throw Error("fullwave.adaptive.tolerance must lie in [1e-6, 0.1] (relative)", n.path + "/tolerance",
                "Below 1e-6 the fit tolerance (1e-3 x the tolerance) reaches the solves' own accuracy (true residual "
                "about 1e-10, gated at 1e-6) and the fit would chase noise; 1e-3 is the default.");
  const double ip = n.num("initial_points", a.initial_points);
  if (!(ip >= 3 && ip <= 50 && std::floor(ip) == ip))
    throw Error("fullwave.adaptive.initial_points must be an integer between 3 and 50 (the band ends included)",
                n.path + "/initial_points");
  a.initial_points = static_cast<int>(ip);
  const double ms = n.num("max_solves", std::max<double>(a.max_solves, ip));
  if (!(ms >= ip && ms <= 500 && std::floor(ms) == ms))
    throw Error("fullwave.adaptive.max_solves must be an integer between initial_points and 500",
                n.path + "/max_solves");
  a.max_solves = static_cast<int>(ms);
  a.spacing = n.str("spacing", a.spacing);
  if (a.spacing != "log" && a.spacing != "linear")
    throw Error("fullwave.adaptive.spacing must be \"log\" or \"linear\"", n.path + "/spacing");
  const double dp = n.num("dense_points", a.dense_points);
  if (!(dp >= 2 && dp <= 20000 && std::floor(dp) == dp))
    throw Error("fullwave.adaptive.dense_points must be an integer between 2 and 20000", n.path + "/dense_points");
  a.dense_points = static_cast<int>(dp);
  // the initial points: the band ends exactly, the rest log- or linearly spaced
  const int np = a.initial_points;
  for (int i = 0; i < np; ++i) {
    const double t = double(i) / (np - 1);
    double x = a.spacing == "log" ? a.f_min_Hz * std::pow(a.f_max_Hz / a.f_min_Hz, t)
                                  : a.f_min_Hz + t * (a.f_max_Hz - a.f_min_Hz);
    if (i == 0) x = a.f_min_Hz;
    if (i == np - 1) x = a.f_max_Hz;
    f.frequencies.push_back(x);
  }
}

// ---- 3D cavity haloscopes (beta; kept apart from the 2D cavity block)
static void parse_eigensolver_3d(const Node& n, FullWaveSpec& f, bool is3d) {
  n.only({"method", "max_iter", "tol", "max_steps", "completeness_steps", "notes"});
  if (!is3d)
    throw Error("fullwave.eigensolver selects the 3D eigensolver; axisymmetric designs always use shift-invert "
                "Lanczos", n.path, "Remove fullwave.eigensolver (use fullwave.target_f_Hz to target modes in 2D).");
  auto& e = f.eigensolver;
  e.method = n.str("method", "");
  if (!e.method.empty() && e.method != "ame" && e.method != "shift_invert")
    throw Error("fullwave.eigensolver.method must be \"ame\" or \"shift_invert\"", n.path + "/method");
  const double it = n.num("max_iter", 500.0), tol = n.num("tol", 1e-8), ms = n.num("max_steps", 0.0);
  if (!(it >= 1 && it <= 100000 && std::floor(it) == it))
    throw Error("max_iter must be an integer between 1 and 100000", n.path + "/max_iter");
  if (!(tol > 0 && tol < 1)) throw Error("tol must lie in (0, 1)", n.path + "/tol");
  if (!(ms >= 0 && ms <= 5000 && std::floor(ms) == ms))
    throw Error("max_steps must be an integer between 0 (automatic) and 5000", n.path + "/max_steps");
  e.max_iter = static_cast<int>(it);
  e.tol = tol;
  e.max_steps = static_cast<int>(ms);
  const double cs = n.num("completeness_steps", 20.0);
  if (!(cs >= 0 && cs <= 500 && std::floor(cs) == cs))
    throw Error("completeness_steps must be an integer between 0 (no check) and 500", n.path + "/completeness_steps");
  e.completeness_steps = static_cast<int>(cs);
  if (e.method == "ame" && f.target_f_Hz > 0)
    throw Error("AME returns the lowest modes and cannot target a frequency", n.path + "/method",
                "Use \"method\": \"shift_invert\" (the default with target_f_Hz) or remove target_f_Hz.");
  if (e.method == "shift_invert" && !(f.target_f_Hz > 0))
    throw Error("shift_invert needs fullwave.target_f_Hz (the shift)", n.path + "/method",
                "Give target_f_Hz; for the lowest modes use \"ame\".");
}

static void parse_cavity_3d(const Node& n, Design& d) {
  n.only({"interior_point", "form_factor", "losses", "wall_material", "allow_lossless_walls", "mode_overlaps",
          "wall_samples", "notes"});
  auto& c = d.cavity3d;
  c.enabled = true;
  if (!d.fullwave.enabled || d.fullwave.modes == 0)
    throw Error("the cavity analysis post-processes eigenmodes: it needs fullwave.modes > 0", n.path,
                "Add \"fullwave\": {\"modes\": 6, \"boundary\": \"pec\", \"rf_only\": true}.");
  if (!d.fullwave.frequencies.empty())
    throw Error("a cavity design cannot have a full-wave frequency sweep yet", "/fullwave/frequencies",
                "Remove the frequencies, or run the sweep in a separate design without the cavity section.");
  if (!d.fullwave.rf_only)
    throw Error("a 3D cavity design needs fullwave.rf_only (closed PEC domain, no static solve)", n.path,
                "Set \"fullwave\": {\"rf_only\": true, \"boundary\": \"pec\", ...}; the static field B0 for the form "
                "factor is fullwave.source (uniform or filament segments: the applied field, not a Meissner-screened one).");
  if (!n.has("interior_point"))
    throw Error("missing required field 'interior_point' (a point [x, y, z] inside the cavity)", n.path);
  c.interior_point = vec3(n.at("interior_point"));
  if (n.has("form_factor")) {
    Node ff = n.at("form_factor");
    ff.only({"B_ref_point", "notes"});
    if (ff.has("B_ref_point")) c.B_ref_point = vec3(ff.at("B_ref_point"));
  }
  c.losses = n.boolean("losses", true);
  c.mode_overlaps = n.boolean("mode_overlaps", false);
  c.wall_samples = n.boolean("wall_samples", false);
  if (c.wall_samples && !c.losses)
    throw Error("cavity.wall_samples needs cavity.losses (the wall quadrature)", n.path + "/wall_samples");
  c.wall_material = n.str("wall_material", "");
  c.allow_lossless_walls = n.boolean("allow_lossless_walls", false);
  if (!c.wall_material.empty()) {
    auto it = d.materials.find(c.wall_material);
    if (it == d.materials.end())
      throw Error("cavity.wall_material '" + c.wall_material + "' is not a material", n.path + "/wall_material",
                  "Define it under \"materials\" (a conductor with an \"rf\" block).");
    if (c.losses && it->second.rf.model.empty() && !c.allow_lossless_walls)
      throw Error("cavity.wall_material '" + c.wall_material + "' has no rf data: the wall would be lossless and "
                  "Q0 would leave out the wall loss", n.path + "/wall_material",
                  "Give the material an \"rf\" block (e.g. {\"model\": \"skin\", \"rho_ohm_m\": 1.7e-8, \"T_K\": 4}), "
                  "or set \"cavity\": {\"allow_lossless_walls\": true} if lossless walls are intended.");
  }
}

Design parse_design(const json& doc) {
  Design d;
  d.source = doc;
  ParamMap none;
  Node root{doc, "", &none};
  root.only({"schema", "name", "description", "symmetry", "parameters", "materials", "conductors", "regions",
             "curves", "windings", "circuits", "superconductors", "pickup", "background", "fullwave", "cavity", "hts",
             "electrostatics", "benchmarks", "scan", "cost", "mesh", "mechanics", "thermal", "notes"});
  d.schema = root.str("schema", d.schema);
  if (d.schema != "axionfem.design/1") {
    throw Error("unsupported schema '" + d.schema + "'", "/schema", "This version reads \"axionfem.design/1\".");
  }
  d.name = root.req_str("name");
  d.description = root.str("description", "");
  d.symmetry = root.str("symmetry", "axisymmetric");
  if (d.symmetry != "axisymmetric" && d.symmetry != "3d" && d.symmetry != "xsec" && d.symmetry != "planar") {
    throw Error("symmetry must be 'axisymmetric', '3d', 'xsec' or 'planar'", "/symmetry",
                "\"xsec\": the (x, y) cross-section of a z-uniform cavity (TM_z/TE_z modes; docs/design_format.md); "
                "\"planar\": a 2D (x, y) full-wave model on an imported mesh (e.g. a COMSOL 2D import).");
  }
  if (d.symmetry == "planar" && !(root.has("mesh") && doc.at("mesh").contains("file")))
    throw Error("a planar 2D design needs an imported mesh (mesh.file and mesh.info_file)", "/symmetry",
                "Planar designs are produced by axf import-comsol from COMSOL 2D models; there is no planar mesher.");
  const bool is3d = d.symmetry == "3d";
  const bool isxs = d.symmetry == "xsec";
  PlanarGuard planar_guard(isxs);
  if (isxs) {
    for (const char* key : {"windings", "circuits", "superconductors", "pickup", "hts", "electrostatics", "mechanics",
                            "thermal", "benchmarks", "scan", "cost"}) {
      if (root.has(key))
        throw Error(strcat("'", key, "' is not supported in a cross-section (\"symmetry\": \"xsec\") design"),
                    strcat("/", key),
                    "A cross-section design is a z-uniform cavity: regions, curves, materials, fullwave (modes, "
                    "families, target_f_Hz), cavity, background (uniform Bz) and mesh.");
    }
    if (!root.has("cavity"))
      throw Error("a cross-section (\"symmetry\": \"xsec\") design needs a 'cavity' section", "/cavity",
                  "Add \"cavity\": {} (optionally interior_point [x, y], wall_material, length).");
    if (!root.has("fullwave"))
      throw Error("a cross-section design needs 'fullwave' with the number of modes", "/fullwave",
                  "Add \"fullwave\": {\"modes\": 6}.");
  }

  // parameters
  if (root.has("parameters")) {
    const json& p = doc.at("parameters");
    if (!p.is_object()) throw Error("parameters must be an object of name: value", "/parameters");
    std::map<std::string, std::string> defs;
    for (auto it = p.begin(); it != p.end(); ++it) {
      if (it.value().is_number()) {
        std::ostringstream os;
        os.precision(17);
        os << it.value().get<double>();
        defs[it.key()] = os.str();
      } else if (it.value().is_string()) {
        defs[it.key()] = it.value().get<std::string>();
      } else {
        throw Error("parameter values must be numbers or expression strings", "/parameters/" + it.key());
      }
    }
    d.parameters = resolve_params(defs, "/parameters");
  }
  const ParamMap* P = &d.parameters;
  Node r{doc, "", P};

  // materials
  d.materials = builtin_materials();
  if (r.has("materials")) {
    Node m = r.at("materials");
    if (!m.j.is_object()) throw Error("materials must be an object of name: definition", m.path);
    for (auto it = m.j.begin(); it != m.j.end(); ++it) {
      d.materials[it.key()] = parse_material(it.key(), m.at(it.key()));
    }
  }

  // conductors (critical surfaces); built-ins are always available and may be overridden by name
  d.conductors = builtin_conductors();
  if (r.has("conductors")) {
    Node cs = r.at("conductors");
    if (!cs.j.is_object()) throw Error("conductors must be an object of name: definition", cs.path);
    for (auto it = cs.j.begin(); it != cs.j.end(); ++it) {
      auto bi = d.conductors.find(it.key());
      Conductor base = bi != d.conductors.end() ? bi->second : Conductor();
      d.conductors[it.key()] = parse_conductor(it.key(), cs.at(it.key()), base);
    }
  }
  auto known_conductor = [&](const std::string& n, const std::string& path) {
    if (!d.conductors.count(n)) {
      std::string list;
      for (auto& kv : d.conductors) list += (list.empty() ? "" : ", ") + kv.first;
      throw Error("unknown conductor '" + n + "'", path,
                  "Define it under \"conductors\" or use one of: " + list + ".");
    }
  };

  std::set<std::string> names;
  // Names the mesher and the solvers use for the parts of the domain that no region describes.  A user
  // region of the same name would shadow them: without this check, a dielectric shell named "air" would be
  // silently treated as vacuum by the electrostatic and mechanical analyses (-30 % on the capacitance
  // of a sphere inside such a shell, with no warning).
  static const char* kReservedRegionNames[] = {"air", "infinity_shell"};
  auto unique_name = [&](const std::string& n, const std::string& path) {
    if (n.empty()) throw Error("names must be non-empty", path);
    for (const char* rsv : kReservedRegionNames) {
      if (n == rsv) {
        throw Error("'" + n + "' is a reserved region name", path,
                    std::string("The solvers use \"air\" for the background outside every named region "
                                "and \"infinity_shell\" for the 3D open-boundary shell; a region of that "
                                "name would be treated as vacuum. Rename it (e.g. \"") + n + "_region\").");
      }
    }
    if (!names.insert(n).second) throw Error("duplicate name '" + n + "'", path, "Region and curve names must be unique.");
  };
  auto known_material = [&](const std::string& n, const std::string& path) {
    if (!d.materials.count(n)) {
      std::string list;
      for (auto& kv : d.materials) list += (list.empty() ? "" : ", ") + kv.first;
      throw Error("unknown material '" + n + "'", path, "Define it under \"materials\" or use one of: " + list + ".");
    }
  };

  // regions
  if (r.has("regions")) {
    Node rs = r.at("regions");
    if (!rs.j.is_array()) throw Error("regions must be a list", rs.path);
    for (size_t i = 0; i < rs.j.size(); ++i) {
      Node n = rs.at(i);
      n.only({"name", "shape", "solid", "material", "mesh_size", "surface_size", "enabled", "notes"});
      Region g;
      g.name = n.req_str("name");
      unique_name(g.name, n.path + "/name");
      // optional parametric switch: the region exists only if `enabled` evaluates to nonzero
      if (n.has("enabled")) {
        const Node en = n.at("enabled");
        const bool on = en.j.is_boolean() ? en.j.get<bool>() : en.num() != 0.0;
        if (!on) {
          d.disabled_regions.push_back(g.name);
          continue;
        }
      }
      if (is3d) {
        if (!n.has("solid")) throw Error("3D regions need a 'solid'", n.path, "e.g. {\"cylinder\": {...}} or {\"revolve\": {\"profile\": SHAPE}}.");
        g.solid = parse_solid(n.at("solid"));
      } else {
        if (!n.has("shape"))
          throw Error(isxs ? "cross-section regions need a 'shape' in the (x, y) plane"
                           : "axisymmetric regions need a 'shape' in the (r, z) half-plane", n.path);
        g.shape = parse_shape(n.at("shape"));
      }
      g.material = n.str("material", "vacuum");
      known_material(g.material, n.path + "/material");
      if (isxs && d.materials.at(g.material).kind == MaterialKind::Iron)
        throw Error("iron is not supported in a cross-section (xsec) cavity design", n.path + "/material",
                    "Use a conductor (PEC rod or wall), or a linear material (eps_r, mu_r, tan_delta).");
      g.mesh_size = n.num("mesh_size", 0.0);
      if (g.mesh_size < 0) throw Error("mesh_size must be > 0 (or absent for automatic)", n.path + "/mesh_size");
      g.surface_size = n.num("surface_size", 0.0);
      if (g.surface_size < 0)
        throw Error("surface_size must be > 0 (or absent for automatic)", n.path + "/surface_size",
                    "surface_size refines the mesh at and near this region's own boundary.");
      d.regions.push_back(std::move(g));
    }
  }
  // curves
  if (r.has("curves")) {
    Node cs = r.at("curves");
    if (!cs.j.is_array()) throw Error("curves must be a list", cs.path);
    for (size_t i = 0; i < cs.j.size(); ++i) {
      Node n = cs.at(i);
      n.only({"name", "path", "material", "thickness", "mesh_size", "surface_size", "notes"});
      Curve c;
      c.name = n.req_str("name");
      unique_name(c.name, n.path + "/name");
      c.path = parse_path(n.at("path"));
      c.material = n.str("material", "superconductor");
      known_material(c.material, n.path + "/material");
      const Material& cm = d.materials.at(c.material);
      const auto k = cm.kind;
      if (n.has("thickness")) {
        // a thin magnetic layer (docs/physics/thin_magnetic_layer.md)
        c.thickness = n.num("thickness", 0.0);
        if (!(c.thickness > 0) || !std::isfinite(c.thickness))
          throw Error(strcat("curve '", c.name, "': thickness must be a positive length in metres (got ",
                             c.thickness, ")"),
                      n.path + "/thickness",
                      "Give the layer's physical thickness d, e.g. \"thickness\": 0.002 for a 2 mm shield.");
        if (k == MaterialKind::Iron)
          throw Error(strcat("curve '", c.name, "': a thin magnetic layer must be a linear material; '", c.material,
                             "' is iron with a B-H law, and saturating layers are not modelled"),
                      n.path + "/material",
                      "Use a material {\"kind\": \"linear\", \"mu_r\": ...} (the engine reports the layer's "
                      "peak B for a saturation check), or mesh the shield through its thickness as an iron region.");
        if (k != MaterialKind::Linear)
          throw Error(strcat("curve '", c.name, "' has a thickness, so it is a thin magnetic layer, but its material '",
                             c.material, "' is not a linear magnetic material"),
                      n.path + "/material",
                      "A thin layer needs {\"kind\": \"linear\", \"mu_r\": ...} with mu_r > 1. Superconducting and "
                      "conducting sheets take no thickness: remove it.");
        if (!(cm.mu_r > 1))
          throw Error(strcat("curve '", c.name, "': the thin-layer material '", c.material, "' has mu_r = ", cm.mu_r,
                             ", but a thin magnetic layer needs mu_r > 1"),
                      n.path + "/material",
                      "mu_r = 1 is no layer at all (remove the curve); a diamagnetic layer is not modelled.");
      } else if (k != MaterialKind::Superconductor && k != MaterialKind::Conductor) {
        throw Error(k == MaterialKind::Linear
                        ? strcat("curve '", c.name, "' is made of the linear material '", c.material,
                                 "' but has no thickness")
                        : std::string("a zero-thickness curve must be a superconductor or conductor sheet"),
                    n.path + "/material",
                    k == MaterialKind::Linear
                        ? "Give \"thickness\" (m) to make it a thin magnetic layer (docs/physics/thin_magnetic_layer.md), "
                          "or model the part as a region with a shape."
                        : "Model magnetic or vacuum parts as regions with a shape, or as a thin layer (a linear "
                          "material with \"thickness\").");
      }
      c.mesh_size = n.num("mesh_size", 0.0);
      if (c.mesh_size < 0) throw Error("mesh_size must be > 0 (or absent for automatic)", n.path + "/mesh_size");
      c.surface_size = n.num("surface_size", 0.0);
      if (c.surface_size < 0)
        throw Error("surface_size must be > 0 (or absent for automatic)", n.path + "/surface_size",
                    "surface_size refines the mesh at and near this sheet.");
      d.curves.push_back(std::move(c));
    }
  }
  if (d.regions.empty() && d.curves.empty()) {
    throw Error("the design has no regions or curves", "/regions", "Add at least one region, e.g. a winding.");
  }

  // windings
  if (r.has("windings")) {
    Node ws = r.at("windings");
    for (size_t i = 0; i < ws.j.size(); ++i) {
      Node n = ws.at(i);
      n.only({"name", "region", "type", "ampere_turns", "circuit", "turns", "axis", "conductor", "notes"});
      Winding w;
      w.region = n.req_str("region");
      w.name = n.str("name", w.region);
      w.circuit = n.str("circuit", "");
      w.conductor = n.str("conductor", "");
      if (!w.conductor.empty()) known_conductor(w.conductor, n.path + "/conductor");
      if (!d.find_region(w.region)) {
        const bool off = std::count(d.disabled_regions.begin(), d.disabled_regions.end(), w.region) > 0;
        throw Error("winding refers to " + std::string(off ? "disabled" : "unknown") + " region '" + w.region + "'",
                    n.path + "/region", off ? "The region's 'enabled' expression evaluates to 0." : "");
      }
      std::string t = n.req_str("type");
      if (t == "azimuthal") w.type = WindingType::Azimuthal;
      else if (t == "poloidal") w.type = WindingType::Poloidal;
      else if (t == "loop" && is3d) w.type = WindingType::Loop;
      else throw Error(std::string("winding type must be 'azimuthal' or 'poloidal'") + (is3d ? " or 'loop'" : ""),
                       n.path + "/type",
                       "azimuthal: current circulates around the symmetry axis (solenoid/CORE magnet). "
                       "poloidal: current loops around the region's cross-section (toroid magnet). "
                       "loop (3D): a harmonic current around the region's single loop (racetrack, saddle or any "
                       "closed coil), right-handed about axis.direction.");
      if (n.has("axis")) {  // also for windings in a circuit
        Node ax = n.at("axis");
        ax.only({"origin", "direction"});
        if (ax.has("origin")) w.axis_origin = vec3(ax.at("origin"));
        if (ax.has("direction")) w.axis_direction = vec3(ax.at("direction"));
      }
      if (w.circuit.empty()) {
      w.ampere_turns = n.req_num("ampere_turns");
        if (n.has("turns")) throw Error("'turns' requires 'circuit'", n.path + "/turns",
                                        "A standalone winding takes ampere_turns; windings in a circuit take turns.");
      } else {
        w.turns = n.req_num("turns");
        if (n.has("ampere_turns")) throw Error("a winding in a circuit takes 'turns', not 'ampere_turns'",
                                               n.path + "/ampere_turns", "The circuit sets the current.");
        if (w.turns == 0) throw Error("turns must be nonzero", n.path + "/turns");
      }
      auto mk = d.material_of_region(w.region).kind;
      if (mk == MaterialKind::Superconductor || mk == MaterialKind::Iron) {
        throw Error("a winding region must be magnetically vacuum-like (material 'winding')", n.path + "/region");
      }
      d.windings.push_back(w);
    }
  }

  // circuits
  if (r.has("circuits")) {
    Node cs = r.at("circuits");
    std::set<std::string> cn;
    for (size_t i = 0; i < cs.j.size(); ++i) {
      Node n = cs.at(i);
      n.only({"name", "type", "current", "flux_linkage", "notes"});
      Circuit c;
      c.name = n.req_str("name");
      if (!cn.insert(c.name).second) throw Error("duplicate circuit name '" + c.name + "'", n.path + "/name");
      std::string t = n.str("type", "persistent");
      if (t == "persistent") {
        c.kind = Circuit::Kind::Persistent;
        c.flux_linkage = n.num("flux_linkage", 0.0);
      } else if (t == "current") {
        c.kind = Circuit::Kind::Current;
        c.current = n.req_num("current");
      } else {
        throw Error("circuit type must be 'persistent' or 'current'", n.path + "/type",
                    "persistent: closed superconducting loop that conserves its flux linkage (passive "
                    "cancellation coils); current: powered by a supply at a fixed current.");
      }
      d.circuits.push_back(c);
    }
  }
  for (size_t i = 0; i < d.windings.size(); ++i) {
    auto& w = d.windings[i];
    if (w.circuit.empty()) continue;
    bool found = false;
    for (auto& c : d.circuits) {
      if (c.name != w.circuit) continue;
      found = true;
      if (c.kind == Circuit::Kind::Persistent && w.type != WindingType::Azimuthal) {
        throw Error("persistent circuits currently support azimuthal windings only", strcat("/windings/", i),
                    "Toroidal (poloidal) persistent windings arrive with the 3D solver.");
      }
    }
    if (!found) throw Error("winding refers to unknown circuit '" + w.circuit + "'", strcat("/windings/", i, "/circuit"),
                            "Define it under \"circuits\".");
  }

  // superconductor states
  std::set<std::string> sc_named;
  if (r.has("superconductors")) {
    Node ss = r.at("superconductors");
    for (size_t i = 0; i < ss.j.size(); ++i) {
      Node n = ss.at(i);
      n.only({"body", "azimuthal", "poloidal", "azimuthal_slit", "axis", "notes"});
      Superconductor s;
      s.body = n.req_str("body");
      if (!d.find_region(s.body) && !d.find_curve(s.body)) {
        throw Error("unknown superconducting body '" + s.body + "'", n.path + "/body");
      }
      if (d.material_of_region(s.body).kind != MaterialKind::Superconductor) {
        throw Error("body '" + s.body + "' is not made of a superconductor", n.path + "/body");
      }
      if (n.has("azimuthal")) s.azimuthal = parse_constraint(n.at("azimuthal"));
      if (n.has("poloidal")) s.poloidal = parse_constraint(n.at("poloidal"));
      s.azimuthal_slit = n.boolean("azimuthal_slit", false);
      if (n.has("axis")) {
        if (!is3d) throw Error("superconductor axis is for 3D designs (axisymmetric bodies use the z axis)", n.path + "/axis");
        Node ax = n.at("axis");
        ax.only({"origin", "direction"});
        s.has_axis = true;
        if (ax.has("origin")) s.axis_origin = vec3(ax.at("origin"));
        if (ax.has("direction")) s.axis_direction = vec3(ax.at("direction"));
        const auto& a = s.axis_direction;
        if (!(a.x * a.x + a.y * a.y + a.z * a.z > 0)) throw Error("axis direction must be nonzero", ax.path + "/direction");
      }
      if (s.azimuthal_slit && s.azimuthal.kind == Constraint::Kind::Flux) {
        throw Error("a body with an azimuthal slit cannot hold a fixed azimuthal flux", n.path,
                    "A slit body carries no net azimuthal current: use {\"net_current\": 0}.");
      }
      sc_named.insert(s.body);
      d.superconductors.push_back(s);
    }
  }
  // Every superconducting region/curve gets a default state.
  for (auto& g : d.regions) {
    if (d.materials.at(g.material).kind == MaterialKind::Superconductor && !sc_named.count(g.name)) {
      Superconductor s;
      s.body = g.name;
      d.superconductors.push_back(s);
    }
  }
  for (auto& c : d.curves) {
    if (d.materials.at(c.material).kind == MaterialKind::Superconductor && !sc_named.count(c.name)) {
      Superconductor s;
      s.body = c.name;
      d.superconductors.push_back(s);
    }
  }

  // pickup
  if (r.has("pickup")) {
    Node n = r.at("pickup");
    n.only({"type", "conductor", "port", "port_boundary", "inside_point", "toward", "notes"});
    Pickup p;
    std::string t = n.req_str("type");
    if (t == "azimuthal") {
      p.kind = Pickup::Kind::Azimuthal;
      p.conductor = n.req_str("conductor");
      if (!d.find_region(p.conductor) && !d.find_curve(p.conductor)) {
        throw Error("pickup conductor '" + p.conductor + "' is not a region or curve", n.path + "/conductor");
      }
      auto mk = d.material_of_region(p.conductor).kind;
      if (mk != MaterialKind::Superconductor && mk != MaterialKind::Conductor) {
        throw Error("pickup conductor must be a superconductor or conductor", n.path + "/conductor");
      }
    } else if (t == "poloidal") {
      p.kind = Pickup::Kind::Poloidal;
      p.port = n.req_str("port");
      if (!d.find_region(p.port)) {
        throw Error("pickup port '" + p.port + "' is not a region", n.path + "/port",
                    "The port is a small vacuum region spanning the readout gap between two conductors.");
      }
      if (n.has("inside_point")) p.inside_point = n.at("inside_point").vec2();
    } else if (t == "port") {
      p.kind = Pickup::Kind::Port;
      if (n.has("port") == n.has("port_boundary"))
        throw Error("a port pickup needs exactly one of 'port' (a gap region) and 'port_boundary' (a boundary group "
                    "of type \"port\" in mesh.info_file)", n.path,
                    "port_boundary is the surface (lumped-port) feed of imported meshes, e.g. a COMSOL import.");
      if (n.has("port")) {
        p.port = n.req_str("port");
        if (!d.find_region(p.port)) throw Error("pickup port '" + p.port + "' is not a region", n.path + "/port");
      } else {
        p.port_boundary = n.req_str("port_boundary");
        if (p.port_boundary.empty()) throw Error("port_boundary must name a boundary group", n.path + "/port_boundary");
      }
      if (n.has("toward")) p.toward = vec3(n.at("toward"));
    } else {
      throw Error("pickup type must be 'azimuthal', 'poloidal' or 'port'", n.path + "/type",
                  "azimuthal: readout current circulates around the axis on a slit superconductor (toroid sheath). "
                  "poloidal: readout current flows across a gap between conductors (coax, CORE neck).");
    }
    d.pickup = p;
  }

  // background field
  if (r.has("background")) {
    Node n = r.at("background");
    n.only({"type", "Bz", "r", "z", "Br", "Bz_grid", "scale", "interpolation", "source", "notes"});
    auto& b = d.background;
    b.type = n.req_str("type");
    b.scale = n.num("scale", 1.0);
    b.interpolation = n.str("interpolation", "hermite");
    if (b.interpolation != "hermite" && b.interpolation != "bilinear") {
      throw Error("background.interpolation must be \"hermite\" or \"bilinear\"", n.path + "/interpolation");
    }
    if (b.type == "uniform") {
      b.Bz = n.req_num("Bz");
    } else if (b.type == "grid") {
      auto vec = [&](const char* key) {
        std::vector<double> v;
        Node a = n.at(key);
        if (!a.j.is_array()) throw Error(std::string(key) + " must be an array", a.path);
        for (size_t i = 0; i < a.j.size(); ++i) v.push_back(a.at(i).num());
        return v;
      };
      b.r = vec("r");
      b.z = vec("z");
      b.Br = vec("Br");
      b.Bz_grid = vec("Bz_grid");
      const size_t nr = b.r.size(), nz = b.z.size();
      if (nr < 2 || nz < 2 || b.Br.size() != nr * nz || b.Bz_grid.size() != nr * nz) {
        throw Error("background grid needs r (nr >= 2), z (nz >= 2) and Br, Bz_grid with nr*nz values (row-major, z outer)",
                    n.path, "axionfem.background.from_comsol_table() builds this from a COMSOL (r, z, Br, Bz) export.");
      }
      if (std::fabs(b.r[0]) > 1e-12) throw Error("background grid must start at r = 0", n.path + "/r");
      for (size_t i = 1; i < nr; ++i) if (!(b.r[i] > b.r[i - 1])) throw Error("r must increase", n.path + "/r");
      for (size_t i = 1; i < nz; ++i) if (!(b.z[i] > b.z[i - 1])) throw Error("z must increase", n.path + "/z");
    } else {
      throw Error("background type must be 'uniform' or 'grid'", n.path + "/type");
    }
    // 3D designs: the background is axisymmetric about the z axis through the origin
  }

  // benchmarks
  if (r.has("benchmarks")) {
    Node bs = r.at("benchmarks");
    for (size_t i = 0; i < bs.j.size(); ++i) {
      Node n = bs.at(i);
      if (n.j.is_string()) {
        d.benchmarks.push_back(builtin_benchmark(n.str()));
      } else {
        n.only({"name", "Q", "T", "eta_A"});
        Benchmark b;
        b.name = n.req_str("name");
        b.Q = n.req_num("Q");
        b.T = n.req_num("T");
        b.eta_A = n.req_num("eta_A");
        if (!(b.Q > 0 && b.T > 0 && b.eta_A > 0)) throw Error("Q, T and eta_A must be positive", n.path);
        d.benchmarks.push_back(b);
      }
    }
  } else {
    d.benchmarks = {builtin_benchmark("optimistic"), builtin_benchmark("conservative")};
  }

  // scan
  if (r.has("scan")) {
    Node n = r.at("scan");
    n.only({"f_min", "f_max", "live_time", "model", "E_over_N", "C_QCD", "rho_DM", "Q_axion", "Theta95", "ma_fa"});
    auto& s = d.scan;
    s.f_min = n.num("f_min", s.f_min);
    s.f_max = n.num("f_max", s.f_max);
    s.live_time = n.num("live_time", s.live_time);
    s.model = n.str("model", s.model);
    if (s.model == "DFSZ") s.E_over_N = 8.0 / 3.0;
    else if (s.model == "KSVZ") s.E_over_N = 0.0;
    else if (s.model != "custom") throw Error("scan model must be DFSZ, KSVZ or custom", n.path + "/model");
    s.E_over_N = n.num("E_over_N", s.E_over_N);
    s.C_QCD = n.num("C_QCD", s.C_QCD);
    s.rho_DM = n.num("rho_DM", s.rho_DM);
    s.Q_axion = n.num("Q_axion", s.Q_axion);
    s.Theta95 = n.num("Theta95", s.Theta95);
    s.ma_fa = n.num("ma_fa", s.ma_fa);
    if (!(s.f_max > s.f_min && s.f_min > 0)) throw Error("need 0 < f_min < f_max", n.path);
    if (!(s.live_time > 0)) throw Error("live_time must be positive (years)", n.path + "/live_time");
  }
  // full-wave analysis
  if (r.has("fullwave")) {
    Node n = r.at("fullwave");
    if (isxs)
      n.only({"modes", "families", "target_f_Hz", "boundary", "notes"});
    n.only({"frequencies", "f_min", "f_max", "points", "boundary", "modes", "families", "dtn_modes", "scene_frequency",
            "ports", "source", "source_check", "source_quadrature_order", "validation_shunt_R_ohm", "notes",
            "rf_only", "validation_loads", "field_export_directory", "field_currents", "field_probes", "source_controls",
            "source_regions", "target_f_Hz", "eigensolver", "pml", "thickness_m", "adaptive",
            "surface_quadrature_file", "surface_quadrature_order"});
    auto& f = d.fullwave;
    f.enabled = true;
    f.rf_only = n.boolean("rf_only", false);
    f.field_export_directory = n.str("field_export_directory", "");
    f.surface_quadrature_file = n.str("surface_quadrature_file", "");
    f.surface_quadrature_order = static_cast<int>(n.num("surface_quadrature_order", 2));
    if (!is3d && (n.has("surface_quadrature_file") || n.has("surface_quadrature_order")))
      throw Error("surface_quadrature_file is for 3D full-wave designs (\"symmetry\": \"3d\"); the axisymmetric "
                  "engine has no conductor-surface export", n.path + "/surface_quadrature_file");
    if (is3d && !f.surface_quadrature_file.empty() && !f.rf_only)
      throw Error("surface_quadrature_file needs fullwave.rf_only: only then can conductors be PEC boundary faces "
                  "(mesh.remove_pec or an imported mesh); in other 3D designs they are volume elements and nothing "
                  "would be exported", n.path + "/surface_quadrature_file");
    if (f.surface_quadrature_order < 0 || f.surface_quadrature_order > 20)
      throw Error("surface_quadrature_order must be between 0 and 20", n.path + "/surface_quadrature_order");
    if (n.has("field_currents")) f.field_currents = n.at("field_currents").j;
    if (n.has("field_probes")) {
      Node ps=n.at("field_probes");
      if(!ps.j.is_array()) throw Error("field_probes must be an array of [x,y,z]",ps.path);
      for(size_t i=0;i<ps.j.size();++i) f.field_probes.push_back(vec3(ps.at(i)));
    }
    if (n.has("frequencies")) {
      if (n.has("f_min") || n.has("f_max") || n.has("points"))
        throw Error("give either 'frequencies' or 'f_min'/'f_max'/'points', not both", n.path);
      Node fl = n.at("frequencies");
      if (!fl.j.is_array()) throw Error("expected a list of frequencies in Hz", fl.path);
      for (size_t i = 0; i < fl.j.size(); ++i) f.frequencies.push_back(fl.at(i).num());
    } else if (n.has("f_min") || n.has("f_max")) {
      const double a = n.req_num("f_min"), b = n.req_num("f_max");
      const int np = static_cast<int>(n.num("points", 20.0));
      if (!(a > 0 && b >= a)) throw Error("need 0 < f_min <= f_max", n.path);
      if (np < 1 || np > 2000) throw Error("points must be between 1 and 2000", n.path + "/points");
      for (int i = 0; i < np; ++i) f.frequencies.push_back(np == 1 ? a : a * std::pow(b / a, double(i) / (np - 1)));
    }
    if (n.has("adaptive")) parse_adaptive_sweep(n, f, is3d);
    for (double x : f.frequencies)
      if (!(x > 0)) throw Error("full-wave frequencies must be positive (Hz)", n.path + "/frequencies");
    f.boundary = n.str("boundary", f.boundary);
    if (f.boundary != "radiation" && f.boundary != "pec")
      throw Error("fullwave.boundary must be \"radiation\" or \"pec\"", n.path + "/boundary");
    f.modes = static_cast<int>(n.num("modes", 0.0));
    if (f.modes < 0 || f.modes > 200) throw Error("modes must be between 0 and 200", n.path + "/modes");
    if (n.has("families")) {
      Node fa = n.at("families");
      if (!fa.j.is_array()) throw Error("expected a list such as [\"TE\", \"TM\"]", fa.path);
      for (size_t i = 0; i < fa.j.size(); ++i) {
        std::string x = fa.at(i).str();
        if (x != "TE" && x != "TM")
          throw Error(isxs ? "families are \"TM\" (E along z) or \"TE\" (H along z)"
                           : "families are \"TE\" (azimuthal E) or \"TM\" (azimuthal H)", fa.at(i).path);
        f.families.push_back(x);
      }
    }
    if (isxs && f.boundary != "pec" && n.has("boundary"))
      throw Error("a cross-section cavity is closed by its conducting wall: fullwave.boundary must be \"pec\" (or absent)",
                  n.path + "/boundary");
    if (isxs && f.modes == 0) throw Error("a cross-section design needs fullwave.modes > 0", n.path + "/modes");
    if (isxs && f.families.empty()) f.families = {"TM"};  // TM_z: the modes that couple to an axial B0
    f.dtn_modes = static_cast<int>(n.num("dtn_modes", 80.0));
    if (f.dtn_modes < 4 || f.dtn_modes > 400) throw Error("dtn_modes must be between 4 and 400", n.path + "/dtn_modes");
    if (f.frequencies.empty() && f.modes == 0)
      throw Error("fullwave needs frequencies (or f_min/f_max) and/or modes > 0", n.path);
    if (n.has("target_f_Hz")) {
      f.target_f_Hz = n.at("target_f_Hz").num();
      if (!(std::isfinite(f.target_f_Hz) && f.target_f_Hz > 0))
        throw Error("target_f_Hz must be a positive frequency in Hz", n.path + "/target_f_Hz");
      if (f.modes == 0)
        throw Error("target_f_Hz selects eigenmodes: set fullwave.modes > 0 as well", n.path + "/target_f_Hz");
    }
    if (n.has("eigensolver")) parse_eigensolver_3d(n.at("eigensolver"), f, is3d);
    // a prescribed source replaces the static field of a 3D sweep altogether, so it cannot come with a background
    if (is3d && r.has("background") && !f.frequencies.empty() && n.has("source"))
      throw Error("a 3D full-wave sweep cannot combine an imposed background with a prescribed fullwave.source",
                  n.path + "/source",
                  "Give the static field once: as `background` (the solver adds the reaction of superconductors and "
                  "magnetic materials) or as fullwave.source (the applied field alone).");
    if (is3d && r.has("background") && !f.frequencies.empty() && f.rf_only)
      throw Error("fullwave.rf_only skips the static solve, so a 3D sweep would see the background without the "
                  "reaction of its superconductors and magnetic materials", "/background",
                  "Drop rf_only to couple the background with its reaction field, or give the applied field as "
                  "fullwave.source.");
    f.scene_frequency = n.num("scene_frequency", 0.0);
    if (f.scene_frequency < 0) throw Error("scene_frequency must be positive (Hz)", n.path + "/scene_frequency");
    if (n.has("ports")) {
      Node ps = n.at("ports");
      if (!is3d || !d.pickup || d.pickup->kind != Pickup::Kind::Port)
        throw Error("fullwave.ports needs a 3D design with a primary port pickup", ps.path);
      if (!ps.j.is_array()) throw Error("expected an array of additional current ports", ps.path);
      if (!d.pickup->port_boundary.empty())
        throw Error("fullwave.ports with a surface (port_boundary) pickup is not implemented", ps.path);
      std::set<std::string> names{"pickup"}, regions{d.pickup->port};
      for (size_t i = 0; i < ps.j.size(); ++i) {
        Node p = ps.at(i);
        p.only({"name", "region", "toward", "notes"});
        FullWaveSpec::Port q;
        q.name = p.req_str("name");
        q.region = p.req_str("region");
        if (q.name.empty() || !names.insert(q.name).second)
          throw Error("port names must be nonempty and unique ('pickup' is reserved)", p.path + "/name");
        if (!d.find_region(q.region) || !regions.insert(q.region).second)
          throw Error("each port must use a distinct existing region", p.path + "/region");
        const auto mk = d.material_of_region(q.region).kind;
        if (mk == MaterialKind::Superconductor || mk == MaterialKind::Conductor)
          throw Error("the current-port gap must be an RF field region, not a PEC conductor", p.path + "/region");
        if (!p.has("toward")) throw Error("additional ports need an explicit nonzero toward direction", p.path);
        q.toward = vec3(p.at("toward"));
        if (!(q.toward.x*q.toward.x + q.toward.y*q.toward.y + q.toward.z*q.toward.z > 0))
          throw Error("toward must be nonzero so circuit voltage/current signs are explicit", p.path + "/toward");
        f.ports.push_back(q);
      }
    }
    const bool imported2d = !is3d && !isxs && r.has("mesh") && doc.at("mesh").contains("file");
    if (n.has("pml")) {
      Node pl = n.at("pml");
      if (!pl.j.is_array() || pl.j.empty()) throw Error("fullwave.pml must be a nonempty list of layers", pl.path);
      if (!is3d && !imported2d)
        throw Error("fullwave.pml needs a 3D design or a 2D design on an imported mesh", pl.path);
      if (f.boundary != "pec")
        throw Error("fullwave.pml needs fullwave.boundary \"pec\" (the layers terminate a closed domain)", pl.path);
      for (size_t i = 0; i < pl.j.size(); ++i) {
        Node e = pl.at(i);
        e.only({"regions", "type", "center", "axis", "scaling_factor", "curvature", "wavelength_m", "notes"});
        if (!e.has("regions") || !e.at("regions").j.is_array() || e.at("regions").j.empty())
          throw Error("each PML layer needs a nonempty 'regions' list", e.path + "/regions");
        for (size_t k = 0; k < e.at("regions").j.size(); ++k) {
          const std::string rn = e.at("regions").at(k).str();
          if (!d.find_region(rn)) throw Error("PML region '" + rn + "' is not a region", e.at("regions").at(k).path);
        }
        json lj = e.j;
        for (const char* key : {"scaling_factor", "curvature", "wavelength_m"})
          if (e.has(key)) lj[key] = e.at(key).num();
        (void)pml::from_json(lj, is3d ? 3 : (d.symmetry == "planar" ? 2 : 1), e.path);
        f.pml.push_back(lj);
      }
    }
    if (n.has("thickness_m")) {
      if (d.symmetry != "planar") throw Error("fullwave.thickness_m is for planar 2D designs", n.path + "/thickness_m");
      f.thickness_m = n.at("thickness_m").num();
      if (!(std::isfinite(f.thickness_m) && f.thickness_m > 0))
        throw Error("thickness_m must be positive (m)", n.path + "/thickness_m");
    }
    if (n.has("source")) {
      if (!is3d && !imported2d)
        throw Error("prescribed fullwave.source is supported only for 3D full-wave models and 2D designs on an "
                    "imported mesh", n.path + "/source");
      f.source = n.at("source").j;
      const auto parsed = PrescribedMagneticSource::from_json(f.source);
      // radiation: the overlap outside R1 uses the exterior multipoles of B0, which exist only for a compact,
      // current-conserving source (checked against R1 at solve time); a uniform field does not decay
      if (f.boundary == "radiation" && parsed.is_uniform())
        throw Error("a uniform prescribed source has no exterior multipole expansion and cannot be used with "
                    "fullwave.boundary = \"radiation\"", n.path + "/source",
                    "Use closed current segments inside the domain sphere, or boundary \"pec\".");
      if (f.boundary == "radiation") {
        const double extent = parsed.max_distance_m({0., 0., 0.});
        const double imbalance = parsed.max_relative_node_imbalance(1e-9 * std::max(extent, 1e-300));
        if (imbalance > 1e-6)
          throw Error("prescribed source segments do not form closed circuits (relative node current imbalance " +
                      std::to_string(imbalance) + "); with fullwave.boundary = \"radiation\" the field outside the "
                      "domain sphere must be curl-free", n.path + "/source",
                      "Close every filament: endpoints must coincide to 1e-9 of the source extent (write coordinates "
                      "at full precision) with balanced currents.");
      }
    }
    if (n.has("source_controls")) {
      Node controls = n.at("source_controls");
      if (!is3d || f.boundary != "pec" || f.frequencies.empty() || !controls.j.is_array())
        throw Error("source_controls needs an array in a 3D PEC frequency sweep (with a radiation boundary the "
                    "controls would also need their own exterior overlap, which is not implemented)", controls.path);
      std::set<std::string> names;
      for (size_t i = 0; i < controls.j.size(); ++i) {
        Node item = controls.at(i);
        item.only({"name","source","quadrature_order"});
        const std::string name = item.req_str("name");
        if (name.empty() || !names.insert(name).second)
          throw Error("source-control names must be nonempty and unique",item.path+"/name");
        const json source = item.at("source").j;
        (void)PrescribedMagneticSource::from_json(source);
        const double order = item.num("quadrature_order",0.0);
        if (!(std::isfinite(order) && order >= 0 && order <= 30 && std::floor(order) == order))
          throw Error("source-control quadrature_order must be an integer between 0 and 30",item.path+"/quadrature_order");
        f.source_controls.push_back({{"name",name},{"source",source},{"quadrature_order",static_cast<int>(order)}});
      }
    }
    if (n.has("source_regions")) {
      Node sr = n.at("source_regions");
      if (f.boundary != "pec")
        throw Error("source_regions needs fullwave.boundary \"pec\" (the exterior term of a radiation boundary uses the "
                    "unrestricted source)", sr.path);
      if (!n.has("source") || !sr.j.is_array() || sr.j.empty())
        throw Error("source_regions needs fullwave.source and a nonempty list of region names", sr.path,
                    "B0 is set to zero outside the listed regions (a COMSOL External Current Density on some domains).");
      for (size_t i = 0; i < sr.j.size(); ++i) {
        const std::string name = sr.at(i).str();
        if (!d.find_region(name)) throw Error("source region '" + name + "' is not a region", sr.at(i).path);
        f.source_regions.push_back(name);
      }
    }
    f.source_check = n.boolean("source_check", false);
    const double qorder = n.num("source_quadrature_order", 0.0);
    if (!(std::isfinite(qorder) && qorder >= 0 && qorder <= 30 && std::floor(qorder) == qorder))
      throw Error("source_quadrature_order must be an integer between 0 and 30", n.path + "/source_quadrature_order");
    f.source_quadrature_order = static_cast<int>(qorder);
    if (f.source_check && (!is3d || f.boundary != "pec" || f.frequencies.empty()))
      throw Error("source_check needs a 3D PEC frequency sweep (with a radiation boundary the exterior part of the "
                  "source has no finite-element right-hand side)", n.path + "/source_check");
    if (n.has("validation_shunt_R_ohm")) {
      Node rs = n.at("validation_shunt_R_ohm");
      if (!is3d || f.boundary != "pec" || f.frequencies.empty() || !rs.j.is_array() || rs.j.size() != f.ports.size() + 1)
        throw Error("validation_shunt_R_ohm needs one positive resistor per port in a 3D PEC sweep", rs.path);
      for (size_t i = 0; i < rs.j.size(); ++i) {
        const double R = rs.at(i).num();
        if (!(std::isfinite(R) && R > 0)) throw Error("shunt resistances must be finite and positive", rs.at(i).path);
        f.validation_shunt_R_ohm.push_back(R);
      }
    }
    if (n.has("validation_loads")) {
      Node ls = n.at("validation_loads");
      if (!is3d || f.boundary != "pec" || f.frequencies.empty() || !ls.j.is_array() || ls.j.size() != f.ports.size()+1 ||
          !f.validation_shunt_R_ohm.empty())
        throw Error("validation_loads needs one series R/L/C object per 3D PEC port, without legacy shunts", ls.path);
      f.validation_loads = json::array();
      for (size_t i=0; i<ls.j.size(); ++i) {
        Node l=ls.at(i); l.only({"R_ohm","L_H","C_F"}); json v;
        for (const char* key : {"R_ohm","L_H","C_F"}) {
          const double x=l.num(key,0.0);
          if (!std::isfinite(x) || x<0) throw Error("load values must be finite and nonnegative",l.path);
          v[key]=x;
        }
        f.validation_loads.push_back(v);
      }
    }
    // modes-only RF designs (3D cavities) need no port: the closed PEC structure alone defines the eigenproblem
    const bool modes_only = f.frequencies.empty() && f.modes > 0 && !d.pickup;
    if (f.rf_only && imported2d) {
      if (f.boundary != "pec" || !d.windings.empty() || d.electrostatics.enabled ||
          (!f.frequencies.empty() && (!d.pickup || d.pickup->kind != Pickup::Kind::Port || d.pickup->port_boundary.empty())))
        throw Error("a 2D design on an imported mesh (rf_only) needs boundary \"pec\", no windings/electrostatics, and "
                    "for a frequency sweep a port pickup on a boundary (pickup.port_boundary)", n.path);
    } else if (f.rf_only && (!is3d || f.boundary!="pec" || (!modes_only && (!d.pickup || d.pickup->kind!=Pickup::Kind::Port)) ||
                      !d.windings.empty() || d.electrostatics.enabled))
      throw Error("rf_only requires a 3D PEC model (boundary \"pec\") with a port pickup, or without a pickup and "
                  "with eigenmodes only (modes > 0, no frequencies), and without static windings/electrostatics", n.path);
    if (!f.field_currents.is_null()) {
      Node cs=n.at("field_currents");cs.only({"re","im"});
      for(const char* key:{"re","im"}) {
        Node vs=cs.at(key);
        if(!vs.j.is_array() || vs.j.size()!=f.ports.size()+1)
          throw Error("field_currents needs re and im arrays with one finite value per port",vs.path);
        for(size_t i=0;i<vs.j.size();++i) {
          const double v=vs.at(i).num();
          if(!std::isfinite(v)) throw Error("field current must be finite",vs.at(i).path);
          f.field_currents[key][i]=v;
        }
      }
    }
  }
  // cavity haloscope analysis of the eigenmodes (beta)
  if (r.has("cavity") && is3d) {
    parse_cavity_3d(r.at("cavity"), d);
  } else if (r.has("cavity")) {
    Node n = r.at("cavity");
    auto& c = d.cavity;
    if (isxs) {
      // z-uniform cross-section: the domain boundary is a conducting wall (`wall_material`), conductor
      // regions and curves are PEC posts/partitions; optional end caps at distance `length`
      n.only({"interior_point", "losses", "wall_material", "length", "end_cap_material", "allow_lossless_walls",
              "notes"});
      c.enabled = true;
      c.allow_lossless_walls = n.boolean("allow_lossless_walls", false);
      if (n.has("interior_point")) {
        c.interior_point = n.at("interior_point").vec2();
        c.interior_given = true;
      }
      c.losses = n.boolean("losses", true);
      auto conductor_material = [&](const std::string& key) {
        const std::string nm = n.req_str(key);
        known_material(nm, n.path + "/" + key);
        const auto k = d.materials.at(nm).kind;
        if (k != MaterialKind::Conductor && k != MaterialKind::Superconductor)
          throw Error(strcat("cavity.", key, " '", nm, "' must be a conductor or superconductor material"),
                      n.path + "/" + key, "Use e.g. \"copper\" or a material of kind \"conductor\" with an rf model.");
        return nm;
      };
      if (n.has("wall_material")) c.wall_material = conductor_material("wall_material");
      if (n.has("length")) {
        c.length = n.at("length").num();
        if (!(std::isfinite(c.length) && c.length > 0))
          throw Error("cavity.length must be a positive length in metres (the distance between the end caps)",
                      n.path + "/length");
      }
      if (n.has("end_cap_material")) {
        if (!(c.length > 0))
          throw Error("end_cap_material needs cavity.length (the end caps are at z = 0 and z = length)",
                      n.path + "/end_cap_material");
        c.end_cap_material = conductor_material("end_cap_material");
      } else {
        c.end_cap_material = c.wall_material;
      }
      // as in 3D: a named wall material without rf data would make that wall silently lossless
      for (const std::string* key : {&c.wall_material, &c.end_cap_material}) {
        if (key->empty() || !c.losses || c.allow_lossless_walls || !d.materials.at(*key).rf.model.empty()) continue;
        const std::string fld = key == &c.wall_material ? "wall_material" : "end_cap_material";
        throw Error("cavity." + fld + " '" + *key + "' has no rf data: the wall would be lossless and Q0 would leave "
                    "out its loss", n.path + "/" + fld,
                    "Give the material an \"rf\" block (e.g. the builtin \"copper\"), or set \"cavity\": "
                    "{\"allow_lossless_walls\": true} if lossless walls are intended.");
      }
      if (d.background.type == "grid")
        throw Error("a cross-section design takes only a uniform axial background (it sets |B0| for "
                    "magnetoresistance and flux-flow wall losses; C does not depend on it)", "/background");
    } else {
      n.only({"interior_point", "form_factor", "losses", "allow_lossless_walls", "notes"});
      c.enabled = true;
      c.allow_lossless_walls = n.boolean("allow_lossless_walls", false);
      if (!d.fullwave.enabled || d.fullwave.modes == 0)
        throw Error("the cavity analysis post-processes eigenmodes: it needs fullwave.modes > 0", n.path,
                    "Add \"fullwave\": {\"modes\": 6, \"boundary\": \"pec\"}.");
      if (!d.fullwave.frequencies.empty())
        throw Error("a cavity design cannot have a full-wave frequency sweep yet (the sweep would be restricted to the "
                    "cavity as well)", "/fullwave/frequencies", "Remove the frequencies, or run the sweep in a separate design "
                    "without the cavity section.");
      if (!n.has("interior_point"))
        throw Error("missing required field 'interior_point' (a point [r, z] inside the cavity)", n.path);
      c.interior_point = n.at("interior_point").vec2();
      if (!(c.interior_point.r >= 0)) throw Error("interior_point needs r >= 0", n.path + "/interior_point");
      if (n.has("form_factor")) {
        Node ff = n.at("form_factor");
        ff.only({"B_ref_point", "B0", "notes"});
        if (ff.has("B_ref_point")) {
          c.B_ref_point = ff.at("B_ref_point").vec2();
          if (!(c.B_ref_point->r >= 0)) throw Error("B_ref_point needs r >= 0", ff.path + "/B_ref_point");
        }
        if (ff.has("B0")) {
          c.B0_field = ff.at("B0").str();
          if (c.B0_field != "applied" && c.B0_field != "static")
            throw Error(strcat("unknown form_factor.B0 '", c.B0_field, "'"), ff.path + "/B0",
                        "Use \"applied\" (magnet field, superconducting cavity walls transparent; default) or \"static\" "
                        "(full static solution: closed superconducting walls expel B0).");
        }
      }
      c.losses = n.boolean("losses", true);
      c.interior_given = true;
    }
  }
  // critical-state (screening-current) analysis
  if (r.has("hts")) {
    Node n = r.at("hts");
    n.only({"regions", "drive", "waveform", "steps_per_segment", "field_of_view", "newton_tol", "max_newton",
            "e_floor", "profile_points", "notes"});
    auto& h = d.hts;
    h.enabled = true;
    if (is3d) throw Error("the critical-state model is axisymmetric", n.path,
                          "Solve the HTS screening currents on the axisymmetric design.");
    Node rs = n.at("regions");
    if (!rs.j.is_array() || rs.j.empty()) throw Error("hts.regions must list at least one region", rs.path);
    for (size_t i = 0; i < rs.j.size(); ++i) {
      Node x = rs.at(i);
      HTSRegion g;
      if (x.j.is_string()) {
        g.region = x.str();
      } else {
        x.only({"region", "conductor", "initial", "notes"});
        g.region = x.req_str("region");
        g.conductor = x.str("conductor", "");
        g.initial = x.str("initial", g.initial);
      }
      if (g.conductor.empty()) g.conductor = "rebco";
      known_conductor(g.conductor, x.path + "/conductor");
      if (!d.find_region(g.region)) {
        const bool off = std::count(d.disabled_regions.begin(), d.disabled_regions.end(), g.region) > 0;
        throw Error("hts region '" + g.region + "' is " + (off ? "disabled" : "not a region"), x.path + "/region",
                    "Critical-state regions are bulk/homogenized-stack regions with a superconductor material.");
      }
      if (d.material_of_region(g.region).kind != MaterialKind::Superconductor) {
        throw Error("hts region '" + g.region + "' is not made of a superconductor", x.path + "/region",
                    "Give it material \"superconductor\" (or \"nb\"): the static and pickup solves keep the ideal "
                    "(Meissner) model, the critical-state analysis replaces it by the power law.");
      }
      if (g.initial != "zero_field_cooled" && g.initial != "field_cooled")
        throw Error("hts initial state must be \"zero_field_cooled\" or \"field_cooled\"", x.path + "/initial",
                    "zero_field_cooled: cooled in zero field, then the drive ramps up (the region screens). "
                    "field_cooled: cooled with the drive already at scale(t_0) and no screening currents.");
      for (auto& other : h.regions)
        if (other.region == g.region) throw Error("region '" + g.region + "' is listed twice", x.path + "/region");
      h.regions.push_back(g);
    }
    h.steps_per_segment = static_cast<int>(n.num("steps_per_segment", 20.0));
    if (h.steps_per_segment < 1 || h.steps_per_segment > 100000)
      throw Error("steps_per_segment must be between 1 and 100000", n.path + "/steps_per_segment");
    if (n.has("waveform") && n.has("drive"))
      throw Error("give either 'drive' (ramp/hold/cycles) or 'waveform' (times and scales), not both", n.path);
    if (n.has("waveform")) {
      Node w = n.at("waveform");
      w.only({"times", "scales"});
      Node ts = w.at("times"), ss = w.at("scales");
      if (!ts.j.is_array() || !ss.j.is_array() || ts.j.size() != ss.j.size() || ts.j.size() < 2)
        throw Error("waveform needs 'times' [s] and 'scales' of equal length >= 2", w.path);
      for (size_t i = 0; i < ts.j.size(); ++i) {
        h.times.push_back(ts.at(i).num());
        h.scales.push_back(ss.at(i).num());
        if (i > 0 && !(h.times[i] > h.times[i - 1])) throw Error("waveform times must increase", ts.path);
      }
    } else {
      Node w = n.has("drive") ? n.at("drive") : n;
      if (n.has("drive")) w.only({"t_ramp", "t_hold", "ramp_down", "t_hold_end", "scale", "cycles"});
      const double tr = w.num("t_ramp", 1.0), th = w.num("t_hold", 0.0), te = w.num("t_hold_end", 0.0);
      const double s1 = w.num("scale", 1.0);
      const bool down = w.boolean("ramp_down", true);
      const int cycles = static_cast<int>(w.num("cycles", 1.0));
      if (!(tr > 0)) throw Error("t_ramp must be positive (s)", w.path + "/t_ramp");
      if (th < 0 || te < 0) throw Error("hold times must be >= 0 (s)", w.path);
      if (cycles < 1 || cycles > 1000) throw Error("cycles must be between 1 and 1000", w.path + "/cycles");
      double t = 0;
      h.times.push_back(0.0);
      h.scales.push_back(0.0);
      for (int k = 0; k < cycles; ++k) {
        t += tr; h.times.push_back(t); h.scales.push_back(s1);
        if (th > 0) { t += th; h.times.push_back(t); h.scales.push_back(s1); }
        if (!down) break;
        t += tr; h.times.push_back(t); h.scales.push_back(0.0);
        if (te > 0) { t += te; h.times.push_back(t); h.scales.push_back(0.0); }
      }
    }
    if (n.has("field_of_view")) {
      Node f = n.at("field_of_view");
      if (!f.j.is_array() || f.j.empty()) throw Error("field_of_view must list points [[r, z], ...]", f.path);
      for (size_t i = 0; i < f.j.size(); ++i) {
        h.field_of_view.push_back(f.at(i).vec2());
        check_nonneg_r(h.field_of_view.back().r, f.at(i).path);
      }
    }
    h.newton_tol = n.num("newton_tol", h.newton_tol);
    h.max_newton = static_cast<int>(n.num("max_newton", h.max_newton));
    h.e_floor = n.num("e_floor", h.e_floor);
    h.profile_points = static_cast<int>(n.num("profile_points", h.profile_points));
    if (!(h.newton_tol > 0 && h.newton_tol < 1)) throw Error("newton_tol must be in (0, 1)", n.path + "/newton_tol");
    if (h.max_newton < 1 || h.max_newton > 1000) throw Error("max_newton must be between 1 and 1000", n.path + "/max_newton");
    if (!(h.e_floor > 0 && h.e_floor < 1)) throw Error("e_floor must be in (0, 1)", n.path + "/e_floor");
    if (h.profile_points < 0 || h.profile_points > 10000)
      throw Error("profile_points must be between 0 and 10000", n.path + "/profile_points");
  }
  // electrostatics (capacitance matrices, parasitic and neck capacitance, self-resonance)
  if (r.has("electrostatics")) {
    Node n = r.at("electrostatics");
    n.only({"boundary", "notes"});
    auto& f = d.electrostatics;
    f.enabled = true;
    f.boundary = n.str("boundary", f.boundary);
    if (f.boundary != "open" && f.boundary != "grounded") {
      throw Error("electrostatics.boundary must be \"open\" or \"grounded\"", n.path + "/boundary",
                  "\"open\": the exact Laplace exterior map (ground at infinity); \"grounded\": phi = 0 on the "
                  "outer sphere at mesh.domain_radius.");
    }
    f.notes = n.str("notes", "");
  }
  // ---------------------------------------------------------------------------------------------
  // cost
  if (r.has("cost")) {
    Node n = r.at("cost");
    n.only({"model", "a", "kappa", "p_energy", "p_area", "iron_cost_per_kg", "cold_envelope", "B_peak_cap",
            "sc_wall_cap", "load_line_cap"});
    auto& c = d.cost;
    c.model = n.str("model", c.model);
    c.a = n.num("a", c.a);
    c.kappa = n.num("kappa", c.kappa);
    c.p_energy = n.num("p_energy", c.p_energy);
    c.p_area = n.num("p_area", c.p_area);
    c.iron_cost_per_kg = n.num("iron_cost_per_kg", c.iron_cost_per_kg);
    c.cold_envelope = n.str("cold_envelope", c.cold_envelope);
    if (c.cold_envelope != "bounding_cylinder") {
      throw Error("unknown cost.cold_envelope '" + c.cold_envelope + "'", n.path + "/cold_envelope",
                  "Only \"bounding_cylinder\" (A_cold = 2 pi R (R + H) of the cold parts) is implemented.");
    }
    c.B_peak_cap = n.num("B_peak_cap", 0.0);
    c.sc_wall_cap = n.num("sc_wall_cap", 0.0);
    c.load_line_cap = n.num("load_line_cap", 0.0);
    if (c.load_line_cap < 0) throw Error("load_line_cap must be >= 0 (fraction of Jc)", n.path + "/load_line_cap");
  }
  // ---- mechanics and thermal analyses ----
  if (r.has("mechanics")) {
    Node n = r.at("mechanics");
    n.only({"supports", "pressures", "cooldown", "symmetry_planes", "probes", "external_field", "notes"});
    auto& mm = d.mechanics;
    mm.enabled = true;
    if (d.fullwave.rf_only) throw Error("mechanics needs the static solve, which fullwave.rf_only skips", n.path);
    for (const char* k : {"symmetry_planes", "probes", "external_field"})
      if (n.has(k) && d.symmetry != "3d") throw Error(std::string("mechanics.") + k + " is for 3D designs", n.path + "/" + k);
    mm.cooldown = n.boolean("cooldown", false);
    if (n.has("supports")) {
      Node ss = n.at("supports");
      if (!ss.j.is_array()) throw Error("supports must be a list", ss.path);
      for (size_t i = 0; i < ss.j.size(); ++i) {
        Node q = ss.at(i);
        q.only({"region", "fix", "notes"});
        Support sup;
        sup.region = q.req_str("region");
        sup.fix = q.str("fix", "all");
        const bool fix3 = d.symmetry == "3d";  // 3D fixes Cartesian components
        if (sup.fix != "all" && sup.fix != (fix3 ? "x" : "r") && sup.fix != "z" && !(fix3 && sup.fix == "y")) {
          throw Error(fix3 ? "support 'fix' must be \"all\", \"x\", \"y\" or \"z\" in a 3D design"
                           : "support 'fix' must be \"all\", \"r\" or \"z\"", q.path + "/fix",
                      fix3 ? "A component fixes that Cartesian displacement; inclined constraints: mechanics.symmetry_planes."
                           : "\"r\" fixes the radial displacement only, \"z\" the axial one, \"all\" both.");
        }
        if (!d.find_region(sup.region)) {
          const bool off = std::count(d.disabled_regions.begin(), d.disabled_regions.end(), sup.region) > 0;
          throw Error("support refers to " + std::string(off ? "disabled" : "unknown") + " region '" + sup.region + "'",
                      q.path + "/region");
        }
        mm.supports.push_back(sup);
      }
    }
    if (n.has("pressures")) {
      Node ps = n.at("pressures");
      if (!ps.j.is_array()) throw Error("pressures must be a list", ps.path);
      for (size_t i = 0; i < ps.j.size(); ++i) {
        Node q = ps.at(i);
        q.only({"on", "from", "p_Pa", "notes"});
        PressureLoad pl;
        pl.on = q.req_str("on");
        pl.from = q.str("from", "vacuum");
        pl.p_Pa = q.req_num("p_Pa");
        if (!d.find_region(pl.on)) throw Error("pressure load 'on' is not a region: '" + pl.on + "'", q.path + "/on");
        if (pl.from != "vacuum" && !d.find_region(pl.from)) {
          throw Error("pressure load 'from' is neither a region nor \"vacuum\": '" + pl.from + "'", q.path + "/from",
                      "\"vacuum\" means every face of 'on' that borders a non-structural element (air, a gap, a port).");
        }
        if (pl.on == pl.from) throw Error("a pressure load needs two different sides", q.path);
        mm.pressures.push_back(pl);
      }
    }
  }
  if (r.has("thermal")) {
    Node n = r.at("thermal");
    n.only({"fixed", "loads", "probes", "notes"});  // probes: 3D only (cpp/src/model3d_mech.cpp)
    auto& tt = d.thermal;
    tt.enabled = true;
    if (n.has("probes") && d.symmetry != "3d") throw Error("thermal.probes is for 3D designs", n.path + "/probes");
    if (d.fullwave.rf_only) throw Error("the thermal analysis does not run with fullwave.rf_only", n.path);
    if (n.has("fixed")) {
      Node fs = n.at("fixed");
      if (!fs.j.is_array()) throw Error("thermal.fixed must be a list", fs.path);
      for (size_t i = 0; i < fs.j.size(); ++i) {
        Node q = fs.at(i);
        q.only({"region", "curve", "T_K", "notes"});
        ThermalFix f;
        f.T_K = q.req_num("T_K");
        if (q.has("region") == q.has("curve"))
          throw Error("give exactly one of 'region' or 'curve'", q.path);
        if (q.has("region")) {
          f.region = q.req_str("region");
          if (!d.find_region(f.region)) throw Error("unknown region '" + f.region + "'", q.path + "/region");
        } else {
          f.curve = q.req_str("curve");
          if (!d.find_curve(f.curve)) throw Error("unknown curve '" + f.curve + "'", q.path + "/curve");
        }
        if (!(f.T_K > 0)) throw Error("T_K must be positive (kelvin)", q.path + "/T_K");
        tt.fixed.push_back(f);
      }
    }
    if (n.has("loads")) {
      Node ls = n.at("loads");
      if (!ls.j.is_array()) throw Error("thermal.loads must be a list", ls.path);
      for (size_t i = 0; i < ls.j.size(); ++i) {
        Node q = ls.at(i);
        q.only({"region", "W", "W_m3", "notes"});
        ThermalLoad l;
        l.region = q.req_str("region");
        if (!d.find_region(l.region)) throw Error("unknown region '" + l.region + "'", q.path + "/region");
        if (q.has("W") == q.has("W_m3"))
          throw Error("give exactly one of 'W' (total power in the region) or 'W_m3' (power density)", q.path);
        if (q.has("W")) {
          l.total = true;
          l.W = q.at("W").num();
        } else {
          l.W_m3 = q.at("W_m3").num();
        }
        tt.loads.push_back(l);
      }
    }
    if (tt.fixed.empty())
      throw Error("the thermal analysis needs at least one prescribed temperature", n.path + "/fixed",
                  "Add {\"region\": NAME, \"T_K\": value} (a heat sink) so the temperature level is set.");
  }
  // ---- end mechanics and thermal analyses ----
  // mesh
  if (r.has("mesh")) {
    Node n = r.at("mesh");
    n.only({"order", "size", "surface_size", "domain_radius", "domain_center_z", "refinements", "adapt",
            "domain_solid", "symmetry", "remove_pec", "file", "info_file", "far_size", "surface_layer",
            "threads", "algorithm3d", "curvature_points", "geometry_order", "extend_from_boundary",
            "optimize_netgen"});
    if (isxs) n.only({"order", "size", "surface_size", "refinements"});
    auto& m = d.mesh;
    for (const char* key : {"far_size","surface_layer","threads","algorithm3d","curvature_points"}) if(n.has(key)) {
      Node v=n.at(key);
      if(!is3d || !v.j.is_number() || !std::isfinite(v.num()) || v.num()<=0)
        throw Error("3D mesher control must be a positive finite numeric literal",v.path);
      if((std::string(key)=="threads" || std::string(key)=="algorithm3d" || std::string(key)=="curvature_points") &&
          std::floor(v.num())!=v.num()) throw Error("mesher count must be an integer",v.path);
    }
    if (n.has("extend_from_boundary")) {  // gmsh Mesh.MeshSizeExtendFromBoundary (3D mesher)
      Node v = n.at("extend_from_boundary");
      if (!is3d || !v.j.is_number() || std::floor(v.num()) != v.num() || v.num() < 0 || v.num() > 2)
        throw Error("mesh.extend_from_boundary is 0, 1 or 2 (gmsh Mesh.MeshSizeExtendFromBoundary), for 3D designs",
                    v.path, "1 grades the volume mesh from the boundary sizes (e.g. curvature-refined walls).");
    }
    if (n.has("optimize_netgen") && (!is3d || !n.at("optimize_netgen").j.is_boolean()))
      throw Error("mesh.optimize_netgen is true or false, for 3D designs", n.path + "/optimize_netgen");
    m.remove_pec=n.boolean("remove_pec",false);
    m.file=n.str("file",""); m.info_file=n.str("info_file","");
    if (m.file.empty()!=m.info_file.empty()) throw Error("mesh.file and mesh.info_file must be supplied together",n.path);
    if (n.has("domain_solid")) m.domain_solid=n.at("domain_solid").j;
    if (n.has("symmetry")) {
      Node ss=n.at("symmetry");
      if (!ss.j.is_array()) throw Error("mesh.symmetry must be an array of named plane cuts",ss.path);
      std::set<std::string> axes,names;
      for (size_t i=0;i<ss.j.size();++i) {
        Node p=ss.at(i); p.only({"name","axis","side","type"});
        const std::string name=p.req_str("name"),axis=p.req_str("axis"),type=p.req_str("type");
        const double side=p.num("side",1);
        if (name.empty() || !names.insert(name).second || !axes.insert(axis).second ||
            (axis!="x" && axis!="y" && axis!="z") || (type!="pec" && type!="pmc") || (side!=1 && side!=-1))
          throw Error("symmetry needs unique name/axis, x|y|z, side +/-1 and type pec|pmc",p.path);
        m.symmetry.push_back({{"name",name},{"axis",axis},{"side",side},{"type",type}});
      }
    }
    if ((m.remove_pec || !m.symmetry.empty() || !m.domain_solid.is_null() || !m.file.empty()) && !d.fullwave.rf_only)
      throw Error("PEC holes, named symmetry, custom domains and mesh import require fullwave.rf_only",n.path);
    m.order = static_cast<int>(n.num("order", m.order));
    m.geometry_order = static_cast<int>(n.num("geometry_order", 0));
    if (n.has("geometry_order") && (!is3d || (m.geometry_order != 1 && m.geometry_order != 2) || m.geometry_order > m.order))
      throw Error("mesh.geometry_order is 1 or 2 and at most mesh.order, for 3D designs", n.path + "/geometry_order",
                  "Element geometry of higher order than the fields breaks the exactness of rigid motions.");
    if (m.order < 1 || m.order > 8) throw Error("mesh order must be between 1 and 8", n.path + "/order");
    m.size = n.num("size", 0.0);
    m.surface_size = n.num("surface_size", 0.0);
    m.domain_radius = n.num("domain_radius", 0.0);
    m.domain_center_z = n.num("domain_center_z", 0.0);
    m.refinements = static_cast<int>(n.num("refinements", 0.0));
    if (n.has("adapt")) {
      Node a = n.at("adapt");
      a.only({"cycles", "tol", "reduction", "max_elements", "watch"});
      m.adapt.cycles = static_cast<int>(a.num("cycles", 4.0));
      if (a.has("watch")) {
        Node w = a.at("watch");
        if (!w.j.is_array() || w.j.empty())
          throw Error("adapt.watch must be a non-empty array of \"Psi\", \"L_p\", \"W_mag\", \"Gamma\"", w.path);
        m.adapt.watch.clear();
        for (size_t i = 0; i < w.j.size(); ++i) {
          const std::string q = w.at(i).str();
          if (q != "Psi" && q != "L_p" && q != "W_mag" && q != "Gamma")
            throw Error("unknown adapt.watch quantity '" + q + "'", w.at(i).path,
                        "Use Psi, L_p, W_mag or Gamma (the pickup surface-loss factors, pickup.losses).");
          m.adapt.watch.push_back(q);
        }
        if (std::find(m.adapt.watch.begin(), m.adapt.watch.end(), "Gamma") != m.adapt.watch.end()) {
          auto conducting = [&](const std::string& name) {
            auto it = d.materials.find(name);
            return it != d.materials.end() && (it->second.kind == MaterialKind::Superconductor ||
                                               it->second.kind == MaterialKind::Conductor);
          };
          bool surfaces = false;
          for (auto& g : d.regions) surfaces = surfaces || conducting(g.material);
          for (auto& c : d.curves) surfaces = surfaces || conducting(c.material);
          if (!d.pickup || !surfaces)
            throw Error("adapt.watch \"Gamma\" needs a pickup and a conductor or superconductor surface", w.path,
                        "Gamma is the pickup's surface-loss factor (pickup.losses); without them there is nothing "
                        "to watch.");
          // Q = 2 pi f L_p / R_eff: watching the loss implies watching L_p
          if (std::find(m.adapt.watch.begin(), m.adapt.watch.end(), "L_p") == m.adapt.watch.end())
            m.adapt.watch.push_back("L_p");
        }
      }
      m.adapt.tol = a.num("tol", m.adapt.tol);
      m.adapt.reduction = a.num("reduction", m.adapt.reduction);
      m.adapt.max_elements = static_cast<long>(a.num("max_elements", static_cast<double>(m.adapt.max_elements)));
      if (m.adapt.cycles < 0 || m.adapt.cycles > 20) throw Error("adapt.cycles must be between 0 and 20", a.path + "/cycles");
      if (!(m.adapt.tol > 0)) throw Error("adapt.tol must be positive", a.path + "/tol");
      if (!(m.adapt.reduction > 0 && m.adapt.reduction < 1))
        throw Error("adapt.reduction must be between 0 and 1", a.path + "/reduction");
      // every cycle meshes anew and would be refined uniformly again, on top of sizes measured on a refined mesh
      if (m.adapt.cycles > 0 && m.refinements > 0)
        throw Error("mesh.refinements cannot be combined with mesh.adapt", n.path + "/refinements",
                    "Each adaptive cycle would refine its new mesh again. Use mesh.adapt alone, or a smaller mesh.size.");
    }
    if (!std::isfinite(m.domain_center_z))
      throw Error("domain_center_z must be a finite number", n.path + "/domain_center_z");
    if (!m.file.empty() && (!m.symmetry.empty() || !m.domain_solid.is_null() || m.adapt.cycles > 0))
      throw Error("an imported mesh (mesh.file) cannot be combined with mesh.symmetry, mesh.domain_solid or mesh.adapt: "
                  "the mesher does not run", n.path);
  }
  // imported regions and surface ports exist only with an imported mesh
  {
    std::function<bool(const Solid&)> imported = [&](const Solid& s) {
      if (s.kind == Solid::Kind::Imported) return true;
      for (const auto& c : s.children) if (imported(c)) return true;
      return false;
    };
    for (size_t i = 0; i < d.regions.size(); ++i)
      if (is3d && imported(d.regions[i].solid) &&
          (d.mesh.file.empty() || d.regions[i].solid.kind != Solid::Kind::Imported))
        throw Error("region '" + d.regions[i].name + "' uses an 'imported' solid, which needs mesh.file and "
                    "cannot be part of a boolean", "/regions",
                    "Imported regions take their elements from mesh.file (attribute 2 + region index).");
    for (size_t i = 0; i < d.regions.size(); ++i)
      if (!is3d && d.regions[i].shape.kind == Shape::Kind::Imported && d.mesh.file.empty())
        throw Error("region '" + d.regions[i].name + "' uses an 'imported' shape, which needs mesh.file and "
                    "mesh.info_file", "/regions", "Imported regions take their elements from mesh.file (attribute 2 + k).");
    if (!is3d && !isxs && !d.mesh.file.empty())
      for (size_t i = 0; i < d.regions.size(); ++i)
        if (d.regions[i].shape.kind != Shape::Kind::Imported)
          throw Error("region '" + d.regions[i].name + "': a 2D design on an imported mesh needs 'imported' shapes "
                      "(the elements come from mesh.file)", "/regions");
    if (d.pickup && !d.pickup->port_boundary.empty() && d.mesh.file.empty())
      throw Error("pickup.port_boundary needs an imported mesh (mesh.file and mesh.info_file)", "/pickup/port_boundary");
  }
  // The critical-state (HTS) analysis does not drive linear magnetic regions by a background yet:
  // the static solve does, so the two would disagree. Refuse the combination.
  if (d.hts.enabled && !d.background.type.empty()) {
    for (size_t i = 0; i < d.regions.size(); ++i) {
      const Material& m = d.materials.at(d.regions[i].material);
      if (m.kind == MaterialKind::Linear && m.mu_r != 1.0)
        throw Error(strcat("region '", d.regions[i].name, "' is a linear magnetic material (mu_r = ", m.mu_r,
                           "), and the critical-state ('hts') analysis does not yet include a background field's "
                           "drive of magnetic regions"),
                    strcat("/regions/", i, "/material"),
                    "Give the background's magnet as windings instead of a 'background', set mu_r = 1, or drop "
                    "the 'hts' section.");
    }
  }
  // thin magnetic layers: the axisymmetric magnetostatic (A_phi) problems only
  for (size_t i = 0; i < d.curves.size(); ++i) {
    const Curve& c = d.curves[i];
    if (!c.thin_layer()) continue;
    const std::string where = strcat("/curves/", i, "/thickness");
    if (d.symmetry != "axisymmetric")
      throw Error(strcat("curve '", c.name, "' is a thin magnetic layer, which only axisymmetric designs support (this "
                         "design is '", d.symmetry, "')"),
                  where, "Mesh the shield through its thickness as a region with a linear or iron material.");
    if (d.mesh.adapt.cycles > 0)
      throw Error(strcat("curve '", c.name, "' is a thin magnetic layer, and mesh.adapt cannot be used with thin "
                         "layers: its error indicator counts the layer's jump in B_n as error and refines the mesh "
                         "at the layer until it is too thin for the layer"),
                  "/mesh/adapt", "Remove mesh.adapt and refine by hand (mesh.size, the curve's surface_size), keeping "
                                 "the elements next to the layer several layer thicknesses high.");
    std::string other;
    if (d.fullwave.enabled) other = "fullwave";
    else if (d.cavity.enabled) other = "cavity";
    else if (d.hts.enabled) other = "hts";
    else if (d.electrostatics.enabled) other = "electrostatics";
    else if (d.mechanics.enabled) other = "mechanics";
    if (!other.empty())
      throw Error(strcat("curve '", c.name, "' is a thin magnetic layer, which the '", other,
                         "' analysis does not model (only the magnetostatic poloidal and azimuthal-pickup problems do)"),
                  where,
                  strcat("Remove the '", other, "' section, or mesh the shield through its thickness as a region "
                         "(docs/physics/thin_magnetic_layer.md, \"Not modelled\")."));
  }
  return d;
}

json design_schema() {
  // Hand-maintained; kept in sync with parse_design by tests/test_schema.py.
  json num = {{"oneOf", {{{"type", "number"}}, {{"type", "string"}, {"description", "expression over parameters"}}}}};
  json pt = {{"type", "array"}, {"items", num}, {"minItems", 2}, {"maxItems", 2}};
  json pt3 = {{"type", "array"}, {"items", num}, {"minItems", 3}, {"maxItems", 3}};
  json pt23 = {{"type", "array"}, {"items", num}, {"minItems", 2}, {"maxItems", 3}};  // (r, z) in 2D, (x, y, z) in 3D
  json str = {{"type", "string"}};
  json arr = {{"type", "array"}};
  json shape = {{"type", "object"},
                {"description", "one of rect{r,z,fillet}, frame{r,z,thickness,fillet}, "
                                "polygon{points,fillet,fillets} (points are [r,z] or [r,z,fillet]; 'fillets' is a "
                                "parallel list of per-vertex radii), contour{start,segments: [{line:[r,z]} | {ellipse:{center,axes,t}}]} "
                                "(exact elliptical arcs), disk{center,radius}, union[shapes], "
                                "difference[shapes], imported{} (2D imported meshes, mesh.file); cross-section (xsec) designs: (x, y) coordinates, rect{x,y}, "
                                "disk{pivot,arm,angle_deg,radius}"}};
  json solid = {{"type", "object"},
                {"description", "3D: box, cylinder, sphere, torus, cone, revolve{profile,angles}, extrude, step, "
                                "union, difference, intersection, imported (mesh.file); optional rotate/translate"}};
  json path = {{"type", "object"},
               {"description", "one of segment[[r,z],[r,z]], polyline{points,closed}, rect{r,z,fillet}, "
                               "arc{center,radius,angles}"}};
  json constraint = {{"type", "object"},
                     {"properties", {{"net_current", num}, {"flux", num}}},
                     {"additionalProperties", false},
                     {"minProperties", 1}, {"maxProperties", 1}};
  auto obj = [](json props, json required = json::array()) {
    json o = {{"type", "object"}, {"properties", props}, {"additionalProperties", false}};
    if (!required.empty()) o["required"] = required;
    return o;
  };
  json enabled = {{"oneOf", json::array({json{{"type", "boolean"}}, num})},
                  {"description", "the region exists only if this evaluates to nonzero"}};
  json region = obj({{"name", str}, {"shape", shape}, {"solid", solid}, {"material", str}, {"mesh_size", num},
                     {"surface_size", num}, {"enabled", enabled}, {"notes", str}},
                    {"name"});
  region["anyOf"] = {{{"required", {"shape"}}}, {{"required", {"solid"}}}};
  json winding = obj({{"name", str}, {"region", str}, {"type", {{"enum", {"azimuthal", "poloidal", "loop"}}}},
                      {"ampere_turns", num}, {"circuit", str}, {"turns", num}, {"axis", {{"type", "object"}}},
                      {"conductor", str}, {"notes", str}},
                     {"region", "type"});
  // powered by ampere_turns, or a member of a circuit with a number of turns
  winding["oneOf"] = {{{"required", {"ampere_turns"}}}, {{"required", {"circuit", "turns"}}}};
  json circuit = obj({{"name", str}, {"type", {{"enum", {"persistent", "current"}}}}, {"current", num},
                      {"flux_linkage", num}, {"notes", str}},
                     {"name", "type"});
  // ---- mechanical / thermal material blocks and analyses ----
  json mechanical = obj({{"E_Pa", num}, {"nu", num}, {"alpha", num}, {"notes", str}}, {"E_Pa", "nu"});
  json thermal_mat = obj({{"k_W_mK", {{"oneOf", json::array({num, json{{"type", "array"}}})}}}, {"notes", str}},
                         {"k_W_mK"});
  json mechanics =
      obj({{"supports", {{"type", "array"},
                         {"items", obj({{"region", str}, {"fix", {{"enum", {"all", "r", "z", "x", "y"}}}}, {"notes", str}},
                                       {"region"})}}},
           {"symmetry_planes", {{"type", "array"}}},  // 3D: [{"point": [x,y,z], "normal": [nx,ny,nz], "name"}]
           {"probes", {{"type", "array"}}},           // 3D: [[x, y, z], ...]
           {"external_field", {{"type", "object"}}},  // 3D: prescribed source (fullwave.source format)
           {"pressures", {{"type", "array"},
                          {"items", obj({{"on", str}, {"from", str}, {"p_Pa", num}, {"notes", str}},
                                        {"on", "p_Pa"})}}},
           {"cooldown", {{"type", "boolean"}}},
           {"notes", str}});
  json thermal_spec =
      obj({{"fixed", {{"type", "array"},
                      {"items", obj({{"region", str}, {"curve", str}, {"T_K", num}, {"notes", str}}, {"T_K"})}}},
           {"loads", {{"type", "array"},
                      {"items", obj({{"region", str}, {"W", num}, {"W_m3", num}, {"notes", str}}, {"region"})}}},
           {"probes", {{"type", "array"}}},  // 3D
           {"notes", str}},
          {"fixed"});
  // ---- end mechanical / thermal blocks ----
  json material = obj({{"kind", {{"enum", {"vacuum", "linear", "iron", "superconductor", "conductor"}}}},
                       {"mu_r", num}, {"eps_r", num}, {"tan_delta", num}, {"sigma_S_m", num}, {"bh", {{"enum", {"knee", "table"}}}}, {"B_sat", num}, {"B", arr}, {"H", arr},
                       {"density", num}, {"warm", {{"type", "boolean"}}}, {"cost_per_kg", num},
                       {"rf", {{"type", "object"}}},
                       {"mechanical", mechanical}, {"thermal", thermal_mat}},
                      {"kind"});
  json conductor = obj({{"law", {{"enum", {"nbti_bottura", "nb3sn_durham", "rebco_kim", "table"}}}},
                        {"T_op", num}, {"fill_factor", num}, {"C0", num}, {"alpha", num}, {"beta", num},
                        {"gamma", num}, {"p", num}, {"q", num}, {"Bc20", num}, {"Tc0", num}, {"strain", num},
                        {"Ca1", num}, {"Ca2", num}, {"eps_0a", num}, {"Jc0", num}, {"B0", num}, {"b_exp", num},
                        {"k_anis", num}, {"T_ref", num}, {"Tc", num}, {"m_T", num}, {"B_T", arr}, {"T_K", arr},
                        {"Jc", arr}, {"n_value", num}, {"E0", num}, {"tape_normal", pt}, {"B_min", num},
                        {"notes", str}});
  json hts_region = {{"oneOf", json::array({str,
                      obj({{"region", str}, {"conductor", str},
                           {"initial", {{"enum", {"zero_field_cooled", "field_cooled"}}}}, {"notes", str}},
                          {"region"})})}};
  return {
      {"$schema", "https://json-schema.org/draft/2020-12/schema"},
      {"title", "axionFEM design (axionfem.design/1)"},
      {"type", "object"},
      {"required", {"name", "regions"}},
      {"additionalProperties", false},
      {"properties",
       {{"schema", {{"const", "axionfem.design/1"}}},
        {"name", str},
        {"description", str},
        {"notes", str},
        {"symmetry", {{"enum", {"axisymmetric", "3d", "xsec", "planar"}}}},
        {"parameters", {{"type", "object"}, {"additionalProperties", num}}},
        {"materials", {{"type", "object"}, {"additionalProperties", material}}},
        {"conductors", {{"type", "object"}, {"additionalProperties", conductor}}},
        {"regions", {{"type", "array"}, {"items", region}}},
        {"curves", {{"type", "array"},
                    {"items", obj({{"name", str}, {"path", path}, {"material", str}, {"thickness", num},
                                   {"mesh_size", num},
                                   {"surface_size", num}, {"notes", str}},
                                  {"name", "path"})}}},
        {"windings", {{"type", "array"}, {"items", winding}}},
        {"circuits", {{"type", "array"}, {"items", circuit}}},
        {"superconductors",
         {{"type", "array"},
          {"items", obj({{"body", str}, {"azimuthal", constraint}, {"poloidal", constraint},
                         {"azimuthal_slit", {{"type", "boolean"}}},
                         {"axis", obj({{"origin", arr}, {"direction", arr}})}, {"notes", str}},
                        {"body"})}}},
        {"pickup", obj({{"type", {{"enum", {"azimuthal", "poloidal", "port"}}}}, {"conductor", str}, {"port", str},
                        {"port_boundary", str},
                        {"inside_point", pt}, {"toward", pt3}, {"notes", str}},
                       {"type"})},
        {"background", obj({{"type", {{"enum", {"uniform", "grid"}}}}, {"Bz", num}, {"r", arr}, {"z", arr},
                            {"Br", arr}, {"Bz_grid", arr}, {"scale", num},
                            {"interpolation", {{"enum", {"hermite", "bilinear"}}}}, {"source", str}, {"notes", str}},
                           {"type"})},
        {"fullwave", obj({{"frequencies", {{"type", "array"}, {"items", num}}}, {"f_min", num}, {"f_max", num},
                          {"points", num},
                          {"adaptive", obj({{"f_min_Hz", num}, {"f_max_Hz", num}, {"tolerance", num},
                                            {"max_solves", num}, {"initial_points", num},
                                            {"spacing", {{"enum", {"log", "linear"}}}}, {"dense_points", num},
                                            {"notes", str}}, {"f_min_Hz", "f_max_Hz"})},
                          {"boundary", {{"enum", {"radiation", "pec"}}}}, {"modes", num},
                          {"families", {{"type", "array"}, {"items", {{"enum", {"TE", "TM"}}}}}},
                          {"dtn_modes", num}, {"scene_frequency", num},
                          {"ports", {{"type", "array"}, {"items", obj({{"name", str}, {"region", str},
                            {"toward", pt3}, {"notes", str}}, {"name", "region", "toward"})}}},
                          {"source", {{"type", "object"}}}, {"source_check", {{"type", "boolean"}}},
                          {"source_regions", {{"type", "array"}, {"items", str}}},
                          {"source_controls",{{"type","array"},{"items",obj({{"name",str},
                            {"source",{{"type","object"}}},{"quadrature_order",num}},{"name","source"})}}},
                          {"source_quadrature_order", num},
                          {"pml", {{"type", "array"}, {"items", obj({{"regions", {{"type", "array"}, {"items", str}}},
                            {"type", {{"enum", {"cartesian", "cylindrical", "spherical"}}}}, {"center", arr},
                            {"axis", pt3}, {"scaling_factor", num}, {"curvature", num}, {"wavelength_m", num},
                            {"notes", str}}, {"regions"})}}},
                          {"thickness_m", num},
                          {"rf_only", {{"type","boolean"}}}, {"validation_loads",arr},
                          {"field_export_directory",str}, {"field_currents",{{"type","object"}}},
                          {"field_probes",{{"type","array"},{"items",pt3}}},
                          {"surface_quadrature_file",{{"type","string"}}},
                          {"surface_quadrature_order",{{"type","number"}}},
                          {"validation_shunt_R_ohm", arr}, {"target_f_Hz", num},
                          {"eigensolver", obj({{"method", {{"enum", {"ame", "shift_invert"}}}}, {"max_iter", num},
                                               {"tol", num}, {"max_steps", num}, {"completeness_steps", num},
                                               {"notes", str}})},
                          {"notes", str}})},
        // (r, z) in 2D, (x, y) in xsec, (x, y, z) in 3D; form_factor.B0 is 2D only, length/end_cap_material xsec only,
        // mode_overlaps 3D only (the parser enforces this per symmetry)
        {"cavity", obj({{"interior_point", pt23}, {"form_factor", obj({{"B_ref_point", pt23},
                                                                       {"B0", {{"enum", {"applied", "static"}}}},
                                                                       {"notes", str}})},
                        {"losses", {{"type", "boolean"}}}, {"wall_material", str}, {"length", num},
                        {"end_cap_material", str}, {"allow_lossless_walls", {{"type", "boolean"}}},
                        {"mode_overlaps", {{"type", "boolean"}}}, {"wall_samples", {{"type", "boolean"}}},
                        {"notes", str}})},
        {"hts", obj({{"regions", {{"type", "array"}, {"items", hts_region}, {"minItems", 1}}},
                     {"drive", obj({{"t_ramp", num}, {"t_hold", num}, {"t_hold_end", num},
                                    {"ramp_down", {{"type", "boolean"}}}, {"scale", num}, {"cycles", num}})},
                     {"waveform", obj({{"times", {{"type", "array"}, {"items", num}}},
                                       {"scales", {{"type", "array"}, {"items", num}}}},
                                      {"times", "scales"})},
                     {"steps_per_segment", num},
                     {"field_of_view", {{"type", "array"}, {"items", pt}}},
                     {"newton_tol", num}, {"max_newton", num}, {"e_floor", num}, {"profile_points", num},
                     {"notes", str}},
                    {"regions"})},
        {"electrostatics", obj({{"boundary", {{"enum", {"open", "grounded"}}}}, {"notes", str}})},
        {"benchmarks", arr},
        {"scan", obj({{"f_min", num}, {"f_max", num}, {"live_time", num}, {"model", str}, {"E_over_N", num},
                      {"C_QCD", num}, {"rho_DM", num}, {"Q_axion", num}, {"Theta95", num}, {"ma_fa", num}})},
        {"cost", obj({{"model", str}, {"a", num}, {"kappa", num}, {"p_energy", num}, {"p_area", num},
                      {"iron_cost_per_kg", num}, {"cold_envelope", {{"enum", {"bounding_cylinder"}}}},
                      {"B_peak_cap", num}, {"sc_wall_cap", num}, {"load_line_cap", num}})},
        {"mesh", obj({{"order", num}, {"size", num}, {"surface_size", num}, {"domain_radius", num},
                      {"domain_solid",solid}, {"symmetry",arr}, {"remove_pec",{{"type","boolean"}}},
                      {"file",str}, {"info_file",str},
                      {"far_size",num},{"surface_layer",num},{"threads",num},{"algorithm3d",num},{"curvature_points",num},{"geometry_order",num},
                      {"extend_from_boundary",num},{"optimize_netgen",{{"type","boolean"}}},
                      {"domain_center_z", num},
                      {"refinements", {{"oneOf", num["oneOf"]},
                                       {"description", "uniform refinements of the generated mesh; the parser refuses them with mesh.adapt"}}},
                      {"adapt", obj({{"cycles", num}, {"tol", num}, {"reduction", num}, {"max_elements", num},
                                     {"watch", {{"type", "array"}, {"minItems", 1},
                                                {"items", {{"enum", {"Psi", "L_p", "W_mag", "Gamma"}}}}}}})}})},
        {"mechanics", mechanics},
        {"thermal", thermal_spec}}}};
}

json resolved_geometry(const Design& d) {
  auto kind_name = [](MaterialKind k) -> const char* {
    switch (k) {
      case MaterialKind::Vacuum: return "vacuum";
      case MaterialKind::Linear: return "linear";
      case MaterialKind::Iron: return "iron";
      case MaterialKind::Superconductor: return "superconductor";
      case MaterialKind::Conductor: return "conductor";
    }
    return "vacuum";
  };
  const bool is3d = d.symmetry == "3d";
  json regions = json::array(), curves = json::array(), materials = json::object();
  for (const auto& g : d.regions) {
    const Material& m = d.materials.at(g.material);
    json r = {{"name", g.name}, {"material", g.material}, {"kind", kind_name(m.kind)}};
    if (is3d) r["solid"] = g.solid.resolved();
    else r["shape"] = g.shape.resolved();
    if (g.mesh_size > 0) r["mesh_size"] = g.mesh_size;
    if (g.surface_size > 0) r["surface_size"] = g.surface_size;
    regions.push_back(r);
  }
  for (const auto& c : d.curves) {
    const Material& m = d.materials.at(c.material);
    curves.push_back({{"name", c.name}, {"material", c.material}, {"kind", kind_name(m.kind)}, {"path", c.path.resolved()}});
  }
  for (const auto& [name, m] : d.materials)
    materials[name] = {{"kind", kind_name(m.kind)}, {"mu_r", m.mu_r}, {"eps_r", m.eps_r}, {"tan_delta", m.tan_delta},
                       {"sigma_S_m", m.sigma_S_m}, {"rf_model", m.rf.model}};
  json mesh = {{"order", d.mesh.order}, {"size", d.mesh.size}, {"surface_size", d.mesh.surface_size},
               {"domain_radius", d.mesh.domain_radius}, {"domain_center_z", d.mesh.domain_center_z},
               {"symmetry", d.mesh.symmetry}, {"remove_pec", d.mesh.remove_pec}};
  if (!d.mesh.domain_solid.is_null()) mesh["domain_solid"] = d.mesh.domain_solid;
  json out = {{"name", d.name}, {"symmetry", d.symmetry}, {"parameters", d.parameters}, {"regions", regions},
              {"curves", curves}, {"disabled_regions", d.disabled_regions}, {"materials", materials}, {"mesh", mesh}};
  if (d.pickup) {
    const auto& p = *d.pickup;
    const char* k = p.kind == Pickup::Kind::Azimuthal ? "azimuthal" : (p.kind == Pickup::Kind::Poloidal ? "poloidal" : "port");
    out["pickup"] = {{"type", k}, {"conductor", p.conductor}, {"port", p.port}};
    if (!p.port_boundary.empty()) out["pickup"]["port_boundary"] = p.port_boundary;
  }
  return out;
}

}  // namespace axf::spec
