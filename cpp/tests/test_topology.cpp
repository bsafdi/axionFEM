// First cohomology of element subsets of a Cartesian tetrahedral mesh.
#include <cmath>
#include <cstdio>
#include <functional>

#include "axionfem/topology.hpp"

static int failures = 0;
#define CHECK(cond, ...)                         \
  do {                                           \
    if (!(cond)) {                               \
      std::printf("FAIL: " __VA_ARGS__);         \
      std::printf("\n");                         \
      ++failures;                                \
    }                                            \
  } while (0)

int main() {
  const int n = 12;
  mfem::Mesh m = mfem::Mesh::MakeCartesian3D(n, n, n, mfem::Element::TETRAHEDRON, 1.0, 1.0, 1.0);
  auto subset = [&](const std::function<bool(double, double, double)>& in) {
    std::vector<char> s(m.GetNE(), 0);
    mfem::Vector c(3);
    for (int e = 0; e < m.GetNE(); ++e) {
      // select whole Cartesian cells (all tetrahedra of a cube) to get manifold subsets
      m.GetElementCenter(e, c);
      double cx = (std::floor(c(0) * n) + 0.5) / n, cy = (std::floor(c(1) * n) + 0.5) / n,
             cz = (std::floor(c(2) * n) + 0.5) / n;
      s[e] = in(cx, cy, cz);
    }
    return s;
  };
  auto closed = [&](const std::vector<double>& h, const std::vector<char>& s) {
    mfem::Array<int> f, o, ed, eo;
    double worst = 0, scale = 0;
    for (double v : h) scale = std::max(scale, std::fabs(v));
    for (int e = 0; e < m.GetNE(); ++e) {
      if (!s[e]) continue;
      m.GetElementFaces(e, f, o);
      for (int fc : f) {
        m.GetFaceEdges(fc, ed, eo);
        double sum = 0;
        for (int k = 0; k < ed.Size(); ++k) sum += eo[k] * h[ed[k]];
        worst = std::max(worst, std::fabs(sum));
      }
    }
    return worst <= 1e-9 * std::max(scale, 1.0);
  };
  struct Case {
    const char* name;
    std::function<bool(double, double, double)> in;
    int b1;
  };
  auto ring = [](double x, double y, double lo, double hi) {
    bool outer = x > lo && x < 1 - lo && y > lo && y < 1 - lo;
    bool hole = x > hi && x < 1 - hi && y > hi && y < 1 - hi;
    return outer && !hole;
  };
  Case cases[] = {
      {"solid block", [](double x, double y, double z) { return x > 0.2 && x < 0.8 && y > 0.2 && y < 0.8 && z > 0.2 && z < 0.8; }, 0},
      {"square ring (solid torus)", [&](double x, double y, double z) { return ring(x, y, 0.1, 0.4) && z > 0.3 && z < 0.7; }, 1},
      {"hollow block shell", [](double x, double y, double z) {
         bool o = x > 0.1 && x < 0.9 && y > 0.1 && y < 0.9 && z > 0.1 && z < 0.9;
         bool i = x > 0.3 && x < 0.7 && y > 0.3 && y < 0.7 && z > 0.3 && z < 0.7;
         return o && !i; }, 0},
      {"hollow square ring (torus shell)", [&](double x, double y, double z) {
         bool solid = ring(x, y, 0.05, 0.42) && z > 0.2 && z < 0.8;
         bool cavity = ring(x, y, 0.15, 0.34) && z > 0.3 && z < 0.7;
         return solid && !cavity; }, 2},
      {"slit torus shell", [&](double x, double y, double z) {
         bool solid = ring(x, y, 0.05, 0.42) && z > 0.2 && z < 0.8;
         bool cavity = ring(x, y, 0.15, 0.34) && z > 0.3 && z < 0.7;
         bool slit = y > 0.45 && y < 0.55 && x < 0.5;
         return solid && !cavity && !slit; }, 1},
  };
  for (auto& c : cases) {
    auto s = subset(c.in);
    auto h = axf::topo::cohomology_h1(m, s);
    long chi = axf::topo::euler_characteristic(m, s);
    std::printf("%-34s b1 = %d (expected %d), parameters %d, euler %ld\n", c.name, h.betti1, c.b1, h.parameters, chi);
    CHECK(h.betti1 == c.b1, "%s: b1 = %d, expected %d", c.name, h.betti1, c.b1);
    for (auto& v : h.cochains) CHECK(closed(v, s), "%s: cochain not closed", c.name);
  }
  std::printf(failures ? "%d failure(s)\n" : "all topology checks passed\n", failures);
  return failures ? 1 : 0;
}
