#include "axionfem/prescribed_source.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace axf {
namespace {

using Point = PrescribedMagneticSource::Point;

[[noreturn]] void bad_source(const std::string& what) {
  throw std::runtime_error("/fullwave/source: " + what);
}

double finite_number(const nlohmann::json& j, const std::string& at) {
  if (!j.is_number()) bad_source(at + " must be a finite SI number");
  const double v = j.get<double>();
  if (!std::isfinite(v)) bad_source(at + " must be a finite SI number");
  return v;
}

Point point(const nlohmann::json& j, const std::string& at) {
  if (!j.is_array() || j.size() != 3) bad_source(at + " must have three Cartesian components");
  return {{finite_number(j[0], at), finite_number(j[1], at), finite_number(j[2], at)}};
}

void compensated_add(Point& sum, Point& compensation, const Point& value) {
  for (int k = 0; k != 3; ++k) {
    const double y = value[k] - compensation[k], t = sum[k] + y;
    compensation[k] = (t - sum[k]) - y;
    sum[k] = t;
  }
}

}  // namespace

PrescribedMagneticSource PrescribedMagneticSource::from_json(const nlohmann::json& input) {
  if (!input.is_object()) bad_source("must be an object; supply kind=uniform, kind=segments or kind=toroidal_box");
  PrescribedMagneticSource source;
  nlohmann::json j = input;
  if (input.contains("file")) {
    if (!input["file"].is_string()) bad_source("file must be the path of a JSON source document");
    for (auto it = input.begin(); it != input.end(); ++it)
      if (it.key() != "file" && it.key() != "azimuthal_average_samples")
        bad_source("a file reference accepts only file and azimuthal_average_samples");
    source.file_ = input["file"].get<std::string>();
    std::ifstream stream(source.file_);
    if (!stream) bad_source("cannot open source file " + source.file_ + "; use an accessible absolute path");
    try { stream >> j; }
    catch (const std::exception& e) { bad_source("invalid source file " + source.file_ + ": " + e.what()); }
    if (!j.is_object() || j.contains("file")) bad_source("source file must be an inline source object, without file recursion");
    if (input.contains("azimuthal_average_samples"))
      j["azimuthal_average_samples"] = input["azimuthal_average_samples"];
  }
  for (auto it = j.begin(); it != j.end(); ++it)
    if (it.key() != "kind" && it.key() != "B_T" && it.key() != "segments_m_A" &&
        it.key() != "B0_T" && it.key() != "r_m" && it.key() != "z_m" && it.key() != "profile" && it.key() != "phi_deg" &&
        it.key() != "azimuthal_average_samples" && it.key() != "metadata")
      bad_source("unknown key " + it.key() + "; allowed keys: kind, B_T, segments_m_A, B0_T, r_m, z_m, profile, phi_deg, "
                 "azimuthal_average_samples, metadata");
  if (!j.contains("kind") || !j["kind"].is_string()) bad_source("kind must be uniform, segments, toroidal_box or axial_box");
  const bool box_keys = j.contains("B0_T") || j.contains("r_m") || j.contains("z_m") || j.contains("profile") ||
                        j.contains("phi_deg");
  if (box_keys && j["kind"] != "toroidal_box" && j["kind"] != "axial_box")
    bad_source("B0_T, r_m, z_m, profile and phi_deg belong to kind=toroidal_box or kind=axial_box");
  source.kind_ = j["kind"].get<std::string>();
  if (j.contains("azimuthal_average_samples")) {
    const auto& count = j["azimuthal_average_samples"];
    if (!count.is_number_integer() || count.get<double>() < 0 || count.get<double>() > 1000000)
      bad_source("azimuthal_average_samples must be 0 (unaveraged) or an integer from 2 to 1000000");
    source.average_samples_ = count.get<int>();
    if (source.average_samples_ == 1)
      bad_source("azimuthal_average_samples=1 is not an average; use 0 or at least 2 samples and check convergence");
  }
  if (j.contains("metadata")) source.metadata_ = j["metadata"];
  if (source.kind_ == "uniform") {
    if (!j.contains("B_T") || j.contains("segments_m_A"))
      bad_source("kind=uniform requires B_T and cannot contain segments_m_A");
    source.uniform_T_ = point(j["B_T"], "B_T");
  } else if (source.kind_ == "segments") {
    if (!j.contains("segments_m_A") || !j["segments_m_A"].is_array() ||
        j["segments_m_A"].empty() || j.contains("B_T"))
      bad_source("kind=segments requires a nonempty segments_m_A array and cannot contain B_T");
    std::size_t index = 0;
    for (const auto& row : j["segments_m_A"]) {
      const std::string at = "segments_m_A[" + std::to_string(index++) + "]";
      if (!row.is_array() || row.size() != 7) bad_source(at + " must be [ax,ay,az,bx,by,bz,current_A]");
      Segment s;
      double length2 = 0;
      for (int k = 0; k < 3; ++k) {
        s.a[k] = finite_number(row[k], at);
        s.b[k] = finite_number(row[k + 3], at);
        length2 += (s.b[k] - s.a[k]) * (s.b[k] - s.a[k]);
      }
      if (!(length2 > 0) || !std::isfinite(length2))
        bad_source(at + " has zero or nonfinite length; provide distinct finite endpoints");
      s.current_A = finite_number(row[6], at);
      source.segments_.push_back(s);
    }
  } else if (source.kind_ == "toroidal_box" || source.kind_ == "axial_box") {
    // axial_box: B = B0 z_hat inside the box (r_m[0] may be 0: a solid cylinder), zero outside.  It reproduces
    // a uniform axial current density Je = B0 [A/m^2 per T] in a cylindrical region, e.g. the bore source of the
    // 2022 COMSOL shielded solenoid of arXiv:2211.00008.
    const bool axial = source.kind_ == "axial_box";
    if (j.contains("B_T") || j.contains("segments_m_A"))
      bad_source("kind=toroidal_box takes B0_T, r_m, z_m and optional profile, not B_T or segments_m_A");
    if (!j.contains("B0_T") || !j.contains("r_m") || !j.contains("z_m"))
      bad_source("kind=toroidal_box requires B0_T (azimuthal field, tesla), r_m [r0, r1] and z_m [z0, z1]");
    source.box_B0_T_ = finite_number(j["B0_T"], "B0_T");
    for (const char* key : {"r_m", "z_m"}) {
      const auto& v = j[key];
      if (!v.is_array() || v.size() != 2) bad_source(std::string(key) + " must be [lower, upper] in metres");
      double* out = std::string(key) == "r_m" ? source.box_r_ : source.box_z_;
      out[0] = finite_number(v[0], key);
      out[1] = finite_number(v[1], key);
      if (!(out[1] > out[0])) bad_source(std::string(key) + " must have upper > lower");
    }
    if (!axial && !(source.box_r_[0] > 0)) bad_source("r_m[0] must be > 0 (the box is an annulus about the z axis)");
    if (axial && !(source.box_r_[0] >= 0)) bad_source("r_m[0] must be >= 0");
    const std::string profile = j.value("profile", std::string("uniform"));
    if (axial && profile != "uniform") bad_source("kind=axial_box supports only profile uniform");
    if (profile != "uniform" && profile != "inverse_r")
      bad_source("profile must be uniform (|B| = B0_T) or inverse_r (|B| = B0_T r_m[0]/rho)");
    source.box_inverse_r_ = profile == "inverse_r";
    if (j.contains("phi_deg")) {
      const auto& v = j["phi_deg"];
      if (!v.is_array() || v.size() != 2) bad_source("phi_deg must be [phi0, phi1] in degrees");
      source.box_phi_deg_[0] = finite_number(v[0], "phi_deg");
      source.box_phi_deg_[1] = finite_number(v[1], "phi_deg");
      const double span = source.box_phi_deg_[1] - source.box_phi_deg_[0];
      if (!(span > 0 && span <= 360)) bad_source("phi_deg must satisfy 0 < phi1 - phi0 <= 360");
      source.box_sector_ = span < 360;
    }
  } else {
    bad_source("unsupported kind " + source.kind_ + "; use uniform, segments, toroidal_box or axial_box");
  }
  return source;
}

PrescribedMagneticSource::Point PrescribedMagneticSource::unaveraged_field_T(const Point& x) const {
  if (kind_ == "uniform") return uniform_T_;
  if (kind_ == "toroidal_box" || kind_ == "axial_box") {
    const double rho = std::hypot(x[0], x[1]);
    if (rho < box_r_[0] || rho > box_r_[1] || x[2] < box_z_[0] || x[2] > box_z_[1]) return {{0., 0., 0.}};
    if (box_sector_) {
      const double deg = std::atan2(x[1], x[0]) * 180 / std::acos(-1.);
      const double rel = std::fmod(std::fmod(deg - box_phi_deg_[0], 360.) + 360., 360.);
      if (rel > box_phi_deg_[1] - box_phi_deg_[0]) return {{0., 0., 0.}};
    }
    if (kind_ == "axial_box") return {{0., 0., box_B0_T_}};
    const double b = box_inverse_r_ ? box_B0_T_ * box_r_[0] / rho : box_B0_T_;
    return {{-b * x[1] / rho, b * x[0] / rho, 0.}};
  }
  Point sum{{0., 0., 0.}}, compensation{{0., 0., 0.}};
  std::size_t index = 0;
  for (const auto& s : segments_) {
    ++index;
    if (s.current_A == 0) continue;
    Point u, v, d;
    double a2 = 0, b2 = 0, uv = 0, dl2 = 0, projection = 0;
    for (int k = 0; k < 3; ++k) {
      u[k] = x[k] - s.a[k];
      v[k] = x[k] - s.b[k];
      d[k] = s.b[k] - s.a[k];
      a2 += u[k] * u[k];
      b2 += v[k] * v[k];
      uv += u[k] * v[k];
      dl2 += d[k] * d[k];
      projection += u[k] * d[k];
    }
    const double a = std::sqrt(a2), b = std::sqrt(b2), ab = a * b;
    // u x v = d x u. The latter avoids subtracting large nearly parallel
    // target vectors for distant observations.
    Point cross{{d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2],
                 d[0] * u[1] - d[1] * u[0]}};
    const double cross2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    const double collinear = 32 * std::numeric_limits<double>::epsilon() *
                             std::sqrt(dl2) * std::max({a, b, std::sqrt(dl2)});
    if (a == 0 || b == 0 || (projection >= 0 && projection <= dl2 && cross2 <= collinear * collinear))
      bad_source("evaluation on filament segment " + std::to_string(index - 1) +
                 "; exclude winding conductors from the RF source domain or refine finite-pack quadrature; no smoothing is applied");
    if (cross2 == 0) continue;  // collinear, strictly outside the finite wire
    double q = ab + uv;
    // Near the wire, avoid cancellation in ab + u.v by the exact identity
    // (ab+u.v)(ab-u.v)=|u x v|^2. No softened denominator is introduced.
    if (q < .25 * ab) q = cross2 / (ab - uv);
    const double factor = 1e-7 * s.current_A * (a + b) / (ab * q);
    Point contribution;
    for (int k = 0; k < 3; ++k) {
      contribution[k] = factor * cross[k];
      if (!std::isfinite(contribution[k]))
        bad_source("nonfinite field at segment " + std::to_string(index - 1) + "; check geometry and source quadrature");
    }
    compensated_add(sum, compensation, contribution);
  }
  return sum;
}

PrescribedMagneticSource::Point PrescribedMagneticSource::field_T(const Point& x) const {
  for (double v : x)
    if (!std::isfinite(v)) bad_source("field evaluation coordinate must be finite");
  if (average_samples_ == 0) return unaveraged_field_T(x);
  Point sum{{0., 0., 0.}}, compensation{{0., 0., 0.}};
  const double twopi = 2 * std::acos(-1.);
  for (int n = 0; n < average_samples_; ++n) {
    const double theta = twopi * n / average_samples_, c = std::cos(theta), s = std::sin(theta);
    const auto B = unaveraged_field_T({{c * x[0] - s * x[1], s * x[0] + c * x[1], x[2]}});
    compensated_add(sum, compensation, {{c * B[0] + s * B[1], -s * B[0] + c * B[1], B[2]}});
  }
  for (double& v : sum) v /= average_samples_;
  return sum;
}

double PrescribedMagneticSource::max_distance_m(const Point& c) const {
  if (kind_ == "uniform") return std::numeric_limits<double>::infinity();
  const double cxy = std::hypot(c[0], c[1]);
  double out = 0;
  for (const auto& s : segments_)
    for (const Point* p : {&s.a, &s.b}) {
      const Point& q = *p;
      const double dz = q[2] - c[2];
      if (average_samples_ == 0) {
        out = std::max(out, std::sqrt((q[0] - c[0]) * (q[0] - c[0]) + (q[1] - c[1]) * (q[1] - c[1]) + dz * dz));
      } else {  // any rotation about z: |q_xy - c_xy| <= |q_xy| + |c_xy|
        const double rxy = std::hypot(q[0], q[1]) + cxy;
        out = std::max(out, std::sqrt(rxy * rxy + dz * dz));
      }
    }
  return out;
}

std::vector<PrescribedMagneticSource::Filament> PrescribedMagneticSource::filaments() const {
  std::vector<Filament> out;
  if (kind_ == "uniform") return out;
  const int copies = std::max(average_samples_, 1);
  const double twopi = 2 * std::acos(-1.);
  for (int r = 0; r < copies; ++r) {
    const double theta = twopi * r / copies, c = std::cos(theta), s = std::sin(theta);
    auto rot = [&](const Point& x) { return Point{{c * x[0] - s * x[1], s * x[0] + c * x[1], x[2]}}; };
    for (const auto& seg : segments_) {
      if (seg.current_A == 0) continue;
      Filament f{rot(seg.a), rot(seg.b), {}, {}};
      for (int k = 0; k < 3; ++k) f.lo[k] = std::min(f.a[k], f.b[k]), f.hi[k] = std::max(f.a[k], f.b[k]);
      out.push_back(f);
    }
  }
  return out;
}

PrescribedMagneticSource::Proximity PrescribedMagneticSource::proximity(const std::vector<Filament>& fils,
                                                                        const std::array<Point, 4>& v,
                                                                        double cutoff) {
  Proximity out;
  Point lo = v[0], hi = v[0], cen{{0., 0., 0.}};
  for (const auto& p : v)
    for (int k = 0; k < 3; ++k)
      lo[k] = std::min(lo[k], p[k]), hi[k] = std::max(hi[k], p[k]), cen[k] += 0.25 * p[k];
  const double size = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]}), tol = 1e-9 * size;
  if (!(size > 0)) return out;
  // inward face normals: face i is opposite vertex i, oriented so that vertex i is on the positive side
  std::array<Point, 4> n;
  std::array<double, 4> off;
  bool degenerate = false;
  for (int i = 0; i < 4 && !degenerate; ++i) {
    const Point& p0 = v[(i + 1) % 4];
    const Point& p1 = v[(i + 2) % 4];
    const Point& p2 = v[(i + 3) % 4];
    const Point e1{{p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}}, e2{{p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]}};
    Point c{{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]}};
    const double norm = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    if (norm == 0) {
      degenerate = true;
      break;
    }
    for (double& x : c) x /= norm;
    double side = 0, o = 0;
    for (int k = 0; k < 3; ++k) side += c[k] * (v[i][k] - p0[k]), o += c[k] * p0[k];
    if (side < 0) {
      for (double& x : c) x = -x;
      o = -o;
    }
    n[i] = c;
    off[i] = o;
  }
  auto meets = [&](const Point& a, const Point& b) {
    double t0 = 0, t1 = 1;
    for (int i = 0; i < 4; ++i) {  // n.(a + t (b - a)) - off >= -tol
      double fa = -off[i], fb = -off[i];
      for (int k = 0; k < 3; ++k) fa += n[i][k] * a[k], fb += n[i][k] * b[k];
      fa += tol, fb += tol;
      if (fa < 0 && fb < 0) return false;
      if (fa < 0) t0 = std::max(t0, fa / (fa - fb));
      else if (fb < 0) t1 = std::min(t1, fa / (fa - fb));
      if (t0 > t1) return false;
    }
    return true;
  };
  const double reach = std::max(cutoff, 1e-9) * size;
  for (const auto& f : fils) {
    bool far = false;
    for (int k = 0; k < 3; ++k)
      if (f.hi[k] < lo[k] - reach || f.lo[k] > hi[k] + reach) far = true;
    if (far) continue;
    // centroid-to-segment distance
    double ab[3], ac[3], L2 = 0, t = 0;
    for (int k = 0; k < 3; ++k) ab[k] = f.b[k] - f.a[k], ac[k] = cen[k] - f.a[k], L2 += ab[k] * ab[k], t += ab[k] * ac[k];
    t = L2 > 0 ? std::clamp(t / L2, 0.0, 1.0) : 0.0;
    double d2 = 0;
    for (int k = 0; k < 3; ++k) d2 += (ac[k] - t * ab[k]) * (ac[k] - t * ab[k]);
    out.distance_over_size = std::min(out.distance_over_size, std::sqrt(d2) / size);
    if (degenerate || out.meets) continue;
    bool box = true;
    for (int k = 0; k < 3; ++k)
      if (f.hi[k] < lo[k] - tol || f.lo[k] > hi[k] + tol) box = false;
    if (box && meets(f.a, f.b)) out.meets = true;
  }
  return out;
}

bool PrescribedMagneticSource::meets_tetrahedron(const std::array<Point, 4>& v) const {
  return proximity(filaments(), v, 0.0).meets;
}

double PrescribedMagneticSource::max_relative_node_imbalance(double tol) const {
  if (kind_ == "uniform" || segments_.empty()) return 0.0;
  struct End {
    Point x;
    double I;  // current leaving the node through this segment end
  };
  std::vector<End> ends;
  ends.reserve(2 * segments_.size());
  double Imax = 0;
  for (const auto& s : segments_) {
    ends.push_back({s.a, s.current_A});
    ends.push_back({s.b, -s.current_A});
    Imax = std::max(Imax, std::fabs(s.current_A));
  }
  if (Imax == 0) return 0.0;
  // cluster endpoints within tol of each other (box test, transitive via union-find) using a hash grid of
  // cell size tol: candidates lie in the 27 neighbouring cells, O(n) for well-separated nodes
  std::vector<std::size_t> parent(ends.size());
  for (std::size_t i = 0; i < ends.size(); ++i) parent[i] = i;
  auto root = [&](std::size_t i) {
    while (parent[i] != i) i = parent[i] = parent[parent[i]];
    return i;
  };
  struct KeyHash {
    std::size_t operator()(const std::array<long long, 3>& k) const {  // unsigned: wrap-around is defined
      const auto u = [](long long x) { return static_cast<unsigned long long>(x); };
      return std::hash<unsigned long long>()(u(k[0]) * 73856093ULL ^ u(k[1]) * 19349663ULL ^ u(k[2]) * 83492791ULL);
    }
  };
  const double cell = tol > 0 ? tol : 1e-300;
  std::unordered_map<std::array<long long, 3>, std::vector<std::size_t>, KeyHash> grid;
  auto key = [&](const Point& x) {
    return std::array<long long, 3>{{static_cast<long long>(std::floor(x[0] / cell)),
                                     static_cast<long long>(std::floor(x[1] / cell)),
                                     static_cast<long long>(std::floor(x[2] / cell))}};
  };
  for (std::size_t i = 0; i < ends.size(); ++i) {
    const auto k = key(ends[i].x);
    for (long long dx = -1; dx <= 1; ++dx)
      for (long long dy = -1; dy <= 1; ++dy)
        for (long long dz = -1; dz <= 1; ++dz) {
          const auto it = grid.find({{k[0] + dx, k[1] + dy, k[2] + dz}});
          if (it == grid.end()) continue;
          for (std::size_t j : it->second)
            if (std::fabs(ends[j].x[0] - ends[i].x[0]) <= tol && std::fabs(ends[j].x[1] - ends[i].x[1]) <= tol &&
                std::fabs(ends[j].x[2] - ends[i].x[2]) <= tol)
              parent[root(j)] = root(i);
        }
    grid[k].push_back(i);
  }
  std::vector<double> net(ends.size(), 0.0);
  for (std::size_t i = 0; i < ends.size(); ++i) net[root(i)] += ends[i].I;
  double worst = 0;
  for (double v : net) worst = std::max(worst, std::fabs(v));
  return worst / Imax;
}

nlohmann::json PrescribedMagneticSource::summary() const {
  nlohmann::json out = {{"kind", kind_}, {"segments", segments_.size()},
                        {"azimuthal_average_samples", average_samples_}, {"regularization_m", 0.0},
                        {"static_response", "prescribed vacuum field; no static superconducting or magnetic response"}};
  if (!file_.empty()) out["file"] = file_;
  if (kind_ == "uniform") out["B_T"] = uniform_T_;
  if (kind_ == "toroidal_box" || kind_ == "axial_box") {
    out["B0_T"] = box_B0_T_;
    out["r_m"] = {box_r_[0], box_r_[1]};
    out["z_m"] = {box_z_[0], box_z_[1]};
    out["profile"] = box_inverse_r_ ? "inverse_r" : "uniform";
    out["phi_deg"] = {box_phi_deg_[0], box_phi_deg_[1]};
  }
  if (!metadata_.is_null()) out["metadata"] = metadata_;
  return out;
}

}  // namespace axf
