#include "axionfem/prescribed_source.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using Source = axf::PrescribedMagneticSource;
using Point = Source::Point;
using json = nlohmann::json;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double norm(const Point& a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

double rel(const Point& a, const Point& b) {
  return norm({{a[0] - b[0], a[1] - b[1], a[2] - b[2]}}) / norm(b);
}

// Independent composite Simpson integration of dB = mu0 I dl x r/(4 pi |r|^3).
// It uses neither the segment formula nor the implementation's cancellation identity.
Point numerical_wire(const Point& a, const Point& b, double current, const Point& x) {
  constexpr int n = 4000;
  Point sum{{0., 0., 0.}}, d{{b[0] - a[0], b[1] - a[1], b[2] - a[2]}};
  for (int i = 0; i <= n; ++i) {
    const double t = double(i) / n;
    Point r{{x[0] - a[0] - t * d[0], x[1] - a[1] - t * d[1], x[2] - a[2] - t * d[2]}};
    const double weight = (i == 0 || i == n ? 1 : i % 2 ? 4 : 2) * 1e-7 * current /
                          (3 * n * std::pow(norm(r), 3));
    sum[0] += weight * (d[1] * r[2] - d[2] * r[1]);
    sum[1] += weight * (d[2] * r[0] - d[0] * r[2]);
    sum[2] += weight * (d[0] * r[1] - d[1] * r[0]);
  }
  return sum;
}

template <typename Callable>
void rejects(Callable call) {
  bool failed = false;
  try { call(); }
  catch (const std::exception&) { failed = true; }
  require(failed, "invalid prescribed source was accepted");
}

}  // namespace

int main() {
  try {
    const Source wire = Source::from_json({{"kind", "segments"},
                                           {"segments_m_A", {{0., 0., -1., 0., 0., 1., 3.}}}});
    const Point observation{{.4, 0., 0.}};
    const Point analytic{{0., 2e-7 * 3. / (.4 * std::sqrt(1. + .4 * .4)), 0.}};
    const double straight_error = rel(wire.field_T(observation), analytic);
    require(straight_error < 1e-14, "finite straight wire analytic normalization/sign failed");
    require(norm(wire.field_T({{0., 0., 2.}})) == 0., "collinear exterior point must have zero field");
    rejects([&] { wire.field_T({{0., 0., 0.}}); });
    rejects([&] { wire.field_T({{0., 0., 1.}}); });
    const auto near = wire.field_T({{1e-10, 0., 0.}});
    require(std::abs(near[1] / 6000. - 1.) < 1e-13, "near-filament cancellation safeguard failed");

    const Point a{{-.2, .13, .7}}, b{{.4, .6, -.2}}, x{{.7, -.2, .1}};
    const Source rotated = Source::from_json({{"kind", "segments"},
                                              {"segments_m_A", {{a[0], a[1], a[2], b[0], b[1], b[2], -2.3}}}});
    const double integral_error = rel(rotated.field_T(x), numerical_wire(a, b, -2.3, x));
    require(integral_error < 2e-12, "independent Biot--Savart quadrature failed");

    // A polygon approaches the analytic on-axis field of a circular loop. This
    // deliberately resolves arc error rather than calling the polygon exact.
    const double pi = std::acos(-1.), radius = .8, z = .3, current = 2.1;
    const Point circle{{0., 0., 2e-7 * pi * current * radius * radius /
                                    std::pow(radius * radius + z * z, 1.5)}};
    double polygon_errors[2];
    int iteration = 0;
    for (int n : {64, 256}) {
      json segments = json::array();
      for (int i = 0; i < n; ++i) {
        const double t0 = 2 * pi * i / n, t1 = 2 * pi * (i + 1) / n;
        segments.push_back({radius * std::cos(t0), radius * std::sin(t0), 0.,
                            radius * std::cos(t1), radius * std::sin(t1), 0., current});
      }
      polygon_errors[iteration++] = rel(Source::from_json({{"kind", "segments"}, {"segments_m_A", segments}})
                                            .field_T({{0., 0., z}}), circle);
    }
    require(polygon_errors[1] < 4e-5 && polygon_errors[0] / polygon_errors[1] > 15.,
            "polygonal loop did not converge toward analytic circular loop");

    // radiation-boundary support checks: closure (Kirchhoff) and the largest current radius
    {
      json segments = json::array();
      for (int i = 0; i < 64; ++i) {
        const double t0 = 2 * pi * i / 64, t1 = 2 * pi * (i + 1) / 64;
        segments.push_back({radius * std::cos(t0), radius * std::sin(t0), z,
                            radius * std::cos(t1), radius * std::sin(t1), z, current});
      }
      const auto loop = Source::from_json({{"kind", "segments"}, {"segments_m_A", segments}});
      require(loop.max_relative_node_imbalance(1e-9) == 0., "closed polygon reported a Kirchhoff imbalance");
      require(std::abs(loop.max_distance_m({{0., 0., 0.}}) - std::hypot(radius, z)) < 1e-15,
              "closed polygon maximum radius");
      require(wire.max_relative_node_imbalance(1e-9) == 1., "open wire must report a full node imbalance");
      segments[5][6] = 2 * current;
      const auto unbalanced = Source::from_json({{"kind", "segments"}, {"segments_m_A", segments}});
      require(std::abs(unbalanced.max_relative_node_imbalance(1e-9) - .5) < 1e-15, "unbalanced node current");
      // planar loop at x = const (all endpoints share x), with an endpoint perturbed inside / outside tolerance
      json planar = json::array();
      for (int i = 0; i < 400; ++i) {
        const double t0 = 2 * pi * i / 400, t1 = 2 * pi * (i + 1) / 400;
        planar.push_back({.2, .5 * std::cos(t0), .5 * std::sin(t0), .2, .5 * std::cos(t1), .5 * std::sin(t1), 1.});
      }
      planar[7][4] = planar[7][4].get<double>() + 1e-12;
      require(Source::from_json({{"kind", "segments"}, {"segments_m_A", planar}}).max_relative_node_imbalance(1e-9) == 0.,
              "endpoint within tolerance must close");
      planar[7][4] = planar[7][4].get<double>() + 1e-6;
      require(Source::from_json({{"kind", "segments"}, {"segments_m_A", planar}}).max_relative_node_imbalance(1e-9) == 1.,
              "endpoint outside tolerance must open the circuit");
      json averaged = {{"kind", "segments"}, {"segments_m_A", {{.3, 0., 0., .3, .1, 0., 1.}, {.3, .1, 0., .3, 0., 0., 1.}}},
                       {"azimuthal_average_samples", 8}};
      // rotated copies about z reach |q_xy| + |c_xy| = sqrt(.1) + .1 from a centre at (.1, 0, 0)
      require(std::abs(Source::from_json(averaged).max_distance_m({{.1, 0., 0.}}) - (std::hypot(.3, .1) + .1)) < 1e-15,
              "azimuthal average must bound every rotated copy");
      require(std::isinf(Source::from_json({{"kind", "uniform"}, {"B_T", {0., 0., 1.}}}).max_distance_m({{0., 0., 0.}})),
              "a uniform field is not compact");
      // filament / tetrahedron intersection (counts elements where the 1/d field defeats Gaussian quadrature)
      const std::array<Source::Point, 4> tet{{{{0., 0., 0.}}, {{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
      auto seg = [](double ax, double ay, double az, double bx, double by, double bz) {
        return Source::from_json({{"kind", "segments"}, {"segments_m_A", {{ax, ay, az, bx, by, bz, 1.}}}});
      };
      require(seg(.1, .1, -1., .1, .1, 1.).meets_tetrahedron(tet), "segment through the tetrahedron");
      require(seg(.2, .2, .2, .25, .2, .2).meets_tetrahedron(tet), "segment inside the tetrahedron");
      require(!seg(.6, .6, -1., .6, .6, 1.).meets_tetrahedron(tet), "segment beside the slanted face");
      require(!seg(.1, .1, .9, .1, .1, 2.).meets_tetrahedron(tet), "segment ending outside the slanted face");
      require(seg(.1, .1, .79, .1, .1, 2.).meets_tetrahedron(tet), "segment entering through the slanted face");
      require(!Source::from_json({{"kind", "uniform"}, {"B_T", {0., 0., 1.}}}).meets_tetrahedron(tet),
              "a uniform field has no filaments");
      // a rotated copy (half turn about z) of a filament at x = -.1..-.2 enters the tetrahedron
      require(!seg(-.1, -.1, .1, -.2, -.1, .1).meets_tetrahedron(tet), "unrotated copy is outside");
      require(Source::from_json({{"kind", "segments"}, {"segments_m_A", {{-.1, -.1, .1, -.2, -.1, .1, 1.}}},
                                 {"azimuthal_average_samples", 2}}).meets_tetrahedron(tet),
              "rotated copies of an azimuthal average must be tested");
      {  // closest approach in element sizes (centroid (.25,.25,.25), size 1); far filaments are skipped
        const auto f = seg(.25, .25, 2., .25, 1.25, 2.).filaments();
        require(f.size() == 1, "one filament");
        const auto pr = Source::proximity(f, tet, 3.0);
        require(!pr.meets && std::fabs(pr.distance_over_size - 1.75) < 1e-12, "centroid-to-segment distance");
        require(Source::proximity(f, tet, 0.5).distance_over_size > 1e299, "beyond the cutoff: not evaluated");
        require(Source::from_json({{"kind", "segments"}, {"segments_m_A", {{0., 0., 0., 1., 0., 0., 1.}}},
                                   {"azimuthal_average_samples", 4}}).filaments().size() == 4,
                "every rotated copy is a filament");
      }
    }

    const auto average = Source::from_json({{"kind", "uniform"}, {"B_T", {2., -3., 4.}},
                                            {"azimuthal_average_samples", 16}});
    require(rel(average.field_T({{.4, -.2, .7}}), {{0., 0., 4.}}) < 2e-15,
            "azimuthal average did not rotate vector components back to observation frame");
    require(average.summary()["regularization_m"] == 0., "source must not silently soften singularities");
    const char* filename = "test_prescribed_source_input.json";
    { std::ofstream stream(filename); stream << json{{"kind", "uniform"}, {"B_T", {1., 2., 3.}}}; }
    const auto imported = Source::from_json({{"file", filename}});
    std::remove(filename);
    require(rel(imported.field_T(x), {{1., 2., 3.}}) == 0., "source-file import failed");
    rejects([] { Source::from_json({{"kind", "segments"}, {"segments_m_A", {{0, 0, 0, 0, 0, 0, 1}}}}); });
    rejects([] { Source::from_json({{"kind", "uniform"}, {"B_T", {1, 2}}}); });
    rejects([] { Source::from_json({{"kind", "uniform"}, {"B_T", {0, 0, 1}}, {"soften", .1}}); });
    rejects([] { Source::from_json({{"kind", "uniform"}, {"B_T", {0, 0, 1}}, {"azimuthal_average_samples", 1}}); });
    rejects([&] { wire.field_T({{std::numeric_limits<double>::quiet_NaN(), 0, 0}}); });
    // toroidal_box: B0 phi_hat inside the annular box, zero outside; inverse_r scales as r0/rho
    const json box{{"kind", "toroidal_box"}, {"B0_T", 2.}, {"r_m", {.5, 1.}}, {"z_m", {-.25, .25}}};
    const Source tor = Source::from_json(box);
    require(rel(tor.field_T({{0., .75, .1}}), {{-2., 0., 0.}}) < 1e-15, "toroidal_box: B must be B0 phi_hat");
    require(rel(tor.field_T({{.6 / std::sqrt(2.), .6 / std::sqrt(2.), 0.}}), {{-std::sqrt(2.), std::sqrt(2.), 0.}}) < 1e-15,
            "toroidal_box: wrong direction off the axes");
    for (const auto& outside : {Source::Point{{.4, 0., 0.}}, Source::Point{{1.1, 0., 0.}}, Source::Point{{.75, 0., .3}}})
      require(norm(tor.field_T(outside)) == 0., "toroidal_box: field outside the box must vanish");
    json inv = box;
    inv["profile"] = "inverse_r";
    require(rel(Source::from_json(inv).field_T({{-1., 0., 0.}}), {{0., -1., 0.}}) < 1e-15, "toroidal_box: inverse_r profile");
    require(tor.summary()["r_m"][1] == 1. && tor.summary()["profile"] == "uniform", "toroidal_box: summary");
    rejects([] { Source::from_json({{"kind", "toroidal_box"}, {"B0_T", 1.}, {"r_m", {0., 1.}}, {"z_m", {0., 1.}}}); });
    rejects([] { Source::from_json({{"kind", "toroidal_box"}, {"B0_T", 1.}, {"r_m", {1., .5}}, {"z_m", {0., 1.}}}); });
    rejects([] { Source::from_json({{"kind", "toroidal_box"}, {"B0_T", 1.}, {"r_m", {.5, 1.}}}); });
    rejects([&] { json bad = box; bad["profile"] = "1/r"; Source::from_json(bad); });
    rejects([] { Source::from_json({{"kind", "uniform"}, {"B_T", {0, 0, 1}}, {"B0_T", 1.}}); });
    // axial_box: B0 z_hat inside a solid (r_m[0] = 0) or annular cylinder, zero outside (COMSOL solenoid bore Je)
    const Source ax = Source::from_json({{"kind", "axial_box"}, {"B0_T", 1.}, {"r_m", {0., .5}}, {"z_m", {0., 1.5}}});
    require(rel(ax.field_T({{0., 0., .7}}), {{0., 0., 1.}}) < 1e-15, "axial_box: B0 z_hat on the axis");
    require(rel(ax.field_T({{.3, -.2, 1.4}}), {{0., 0., 1.}}) < 1e-15, "axial_box: B0 z_hat inside");
    for (const auto& outside : {Source::Point{{.6, 0., .5}}, Source::Point{{0., 0., 1.6}}, Source::Point{{.1, 0., -.1}}})
      require(norm(ax.field_T(outside)) == 0., "axial_box: field outside the box must vanish");
    rejects([] { Source::from_json({{"kind", "axial_box"}, {"B0_T", 1.}, {"r_m", {0., .5}}, {"z_m", {0., 1.}},
                                    {"profile", "inverse_r"}}); });
    rejects([] { Source::from_json({{"kind", "axial_box"}, {"B0_T", 1.}, {"r_m", {-.1, .5}}, {"z_m", {0., 1.}}}); });
    // sector: phi in [0, 355] deg (a 5 deg gap at 355..360), as the 2211.00008 COMSOL toroid
    json sec = box;
    sec["phi_deg"] = {0., 355.};
    const Source ts = Source::from_json(sec);
    const double g = 357.5 * std::acos(-1.) / 180, in = 300 * std::acos(-1.) / 180;
    require(norm(ts.field_T({{.75 * std::cos(g), .75 * std::sin(g), 0.}})) == 0., "toroidal_box: field in the excluded sector");
    require(rel(ts.field_T({{.75 * std::cos(in), .75 * std::sin(in), 0.}}), {{-2. * std::sin(in), 2. * std::cos(in), 0.}}) < 1e-14,
            "toroidal_box: field inside the sector");
    rejects([&] { json bad = box; bad["phi_deg"] = {10., 5.}; Source::from_json(bad); });
    std::cout << json{{"test", "prescribed_source"}, {"straight_wire_relative_error", straight_error},
                      {"independent_integral_relative_error", integral_error},
                      {"polygon64_relative_error", polygon_errors[0]}, {"polygon256_relative_error", polygon_errors[1]},
                      {"passed", true}}.dump() << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
