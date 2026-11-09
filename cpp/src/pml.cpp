#include "axionfem/pml.hpp"

#include <cmath>

#include "axionfem/core.hpp"
#include "axionfem/units.hpp"

namespace axf::pml {

namespace {

// stretch of one direction at local coordinate q: s = dq~/dq and q~
void stretch(const Direction& D, double q, double lambda, double F, double p, cplx& s, cplx& qt) {
  s = 1.0;
  qt = q;
  if (!D.on) return;
  const double xi = D.sign * (q - D.q0) / D.d;
  if (!(xi > 0)) return;
  const cplx a(F * lambda, F * lambda);  // (1 + i) F lambda
  s = 1.0 + a / D.d * p * std::pow(xi, p - 1.0);
  qt = q + static_cast<double>(D.sign) * a * std::pow(xi, p);
}

void cross(const double a[3], const double b[3], double c[3]) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

}  // namespace

const Direction& Layer::pick(int i, double q) const {
  static const Direction none;
  if (dir[i].on && q > dir[i].q0) return dir[i];
  if (dirn[i].on && q < dirn[i].q0) return dirn[i];
  return none;
}

void Layer::local(const double* x, double q[3]) const {
  q[0] = q[1] = q[2] = 0.0;
  if (space == 1) {  // axisymmetric (r, z): cylindrical about the axis, or spherical about (0, center[1])
    if (type == "spherical") {
      q[0] = std::hypot(x[0], x[1] - center[1]);
    } else {
      q[0] = x[0];
      q[2] = x[1];
    }
    return;
  }
  const int n = space == 3 ? 3 : 2;
  double d[3] = {0, 0, 0};
  for (int i = 0; i < n; ++i) d[i] = x[i] - (type == "cartesian" ? 0.0 : center[i]);
  if (type == "cartesian") {
    for (int i = 0; i < n; ++i) q[i] = x[i];
  } else if (type == "cylindrical") {
    if (space == 2) {
      q[0] = std::hypot(d[0], d[1]);
    } else {
      const double t = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
      double rv[3];
      for (int i = 0; i < 3; ++i) rv[i] = d[i] - t * axis[i];
      q[0] = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
      q[2] = t;
    }
  } else {  // spherical
    q[0] = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  }
}

void Layer::tensors(const double* x, double omega, cplx Lt[9], cplx Li[9], double* La, double* Lia, cplx* dLt,
                    cplx* dLi) const {
  const double lambda = wavelength > 0 ? wavelength : 2 * units::pi * units::c / omega;
  double q[3];
  local(x, q);
  // local frame e[i] (components of `space`) and Jacobian entries J[i]
  double e[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  cplx J[3] = {1.0, 1.0, 1.0};
  cplx s, qt;
  const bool cart = type == "cartesian" && space != 1;
  if (cart) {
    for (int i = 0; i < 3; ++i) {
      stretch(pick(i, q[i]), q[i], lambda, scaling, curvature, s, qt);
      J[i] = s;
    }
  } else {
    const double rad = q[0];
    stretch(pick(0, rad), rad, lambda, scaling, curvature, s, qt);
    J[0] = s;
    const cplx ratio = rad > 0 ? qt / rad : cplx(1.0);
    if (type == "spherical") {
      J[1] = J[2] = ratio;
      if (space == 1) {  // (r, z, phi): e_R = (r, z - zc) / R, e_theta = (z - zc, -r) / R, e_phi
        const double R = rad > 0 ? rad : 1.0, ar = x[0] / R, az = (x[1] - center[1]) / R;
        const double E[3][3] = {{ar, az, 0}, {az, -ar, 0}, {0, 0, 1}};
        for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) e[i][k] = E[i][k];
      } else {
        double d[3];
        for (int i = 0; i < 3; ++i) d[i] = x[i] - center[i];
        const double R = rad > 0 ? rad : 1.0;
        for (int k = 0; k < 3; ++k) e[0][k] = d[k] / R;
        // any orthonormal completion (J[1] = J[2])
        double t[3] = {1, 0, 0};
        if (std::fabs(e[0][0]) > 0.9) { t[0] = 0; t[1] = 1; }
        cross(e[0], t, e[1]);
        const double n1 = std::sqrt(e[1][0] * e[1][0] + e[1][1] * e[1][1] + e[1][2] * e[1][2]);
        for (int k = 0; k < 3; ++k) e[1][k] /= n1;
        cross(e[0], e[1], e[2]);
      }
    } else {  // cylindrical: (rho, phi, axial)
      J[1] = ratio;
      stretch(pick(2, q[2]), q[2], lambda, scaling, curvature, s, qt);
      J[2] = s;
      if (space == 1) {  // (r, z, phi) components: e_rho = r, e_phi = phi, e_axial = z
        const double E[3][3] = {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}};
        for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) e[i][k] = E[i][k];
      } else if (space == 2) {  // (x, y, out of plane): about the point center
        const double dx = x[0] - center[0], dy = x[1] - center[1], R = rad > 0 ? rad : 1.0;
        const double E[3][3] = {{dx / R, dy / R, 0}, {-dy / R, dx / R, 0}, {0, 0, 1}};
        for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) e[i][k] = E[i][k];
      } else {
        double d[3], rv[3];
        for (int i = 0; i < 3; ++i) d[i] = x[i] - center[i];
        const double t = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
        for (int i = 0; i < 3; ++i) rv[i] = d[i] - t * axis[i];
        const double R = rad > 0 ? rad : 1.0;
        if (rad > 0) {
          for (int k = 0; k < 3; ++k) e[0][k] = rv[k] / R;
        } else {  // on the axis: any radial direction (J[0] = J[1] there for a radial layer away from the axis)
          double t2[3] = {1, 0, 0};
          if (std::fabs(axis[0]) > 0.9) { t2[0] = 0; t2[1] = 1; }
          cross(axis, t2, e[0]);
          const double n0 = std::sqrt(e[0][0] * e[0][0] + e[0][1] * e[0][1] + e[0][2] * e[0][2]);
          for (int k = 0; k < 3; ++k) e[0][k] /= n0;
        }
        for (int k = 0; k < 3; ++k) e[2][k] = axis[k];
        cross(e[2], e[0], e[1]);
      }
    }
  }
  const cplx det = J[0] * J[1] * J[2];
  for (int k = 0; k < 9; ++k) {
    Lt[k] = Li[k] = 0.0;
    if (La) La[k] = 0.0;
    if (Lia) Lia[k] = 0.0;
    if (dLt) dLt[k] = 0.0;
    if (dLi) dLi[k] = 0.0;
  }
  // d ln J_i / domega: J_i - 1 is proportional to lambda (stretches and the radial ratio q~/q alike), and lambda =
  // 2 pi c / omega unless a wavelength is fixed
  cplx dlogJ[3] = {0.0, 0.0, 0.0};
  if (!(wavelength > 0))
    for (int i = 0; i < 3; ++i) dlogJ[i] = -(J[i] - 1.0) / (omega * J[i]);
  const cplx dlogdet = dlogJ[0] + dlogJ[1] + dlogJ[2];
  for (int i = 0; i < 3; ++i) {
    const cplx lam = det / (J[i] * J[i]);
    const cplx dlam = lam * (dlogdet - 2.0 * dlogJ[i]);   // d lam / domega
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const double ee = e[i][a] * e[i][b];
        Lt[3 * a + b] += lam * ee;
        Li[3 * a + b] += 1.0 / lam * ee;
        if (La) La[3 * a + b] += std::abs(lam) * ee;
        if (Lia) Lia[3 * a + b] += ee / std::abs(lam);
        if (dLt) dLt[3 * a + b] += dlam * ee;
        if (dLi) dLi[3 * a + b] += -dlam / (lam * lam) * ee;
      }
  }
}

nlohmann::json Layer::summary() const {
  nlohmann::json dirs = nlohmann::json::array();
  const char* names_c[3] = {"x", "y", "z"};
  const char* names_cyl[3] = {"rho", "-", "axial"};
  const char* names_sph[3] = {"r", "-", "-"};
  const bool cart = type == "cartesian" && space != 1;
  for (int i = 0; i < 3; ++i)
    for (const Direction* D : {&dir[i], &dirn[i]})
      if (D->on)
        dirs.push_back({{"coordinate", cart ? names_c[i] : type == "spherical" ? names_sph[i] : names_cyl[i]},
                        {"start_m", D->q0}, {"thickness_m", D->d}, {"towards", D->sign > 0 ? "+" : "-"}});
  nlohmann::json j = {{"regions", name}, {"type", space == 1 && type == "cartesian" ? "cylindrical" : type},
                      {"scaling_factor", scaling}, {"curvature", curvature},
                      {"wavelength_m", wavelength > 0 ? nlohmann::json(wavelength) : nlohmann::json("c/f (vacuum)")},
                      {"stretched", dirs},
                      {"stretch", "q~ = q0 + sign (d xi + (1+i) F lambda xi^p), xi = sign (q - q0)/d (e^{-i omega t}); "
                                  "per point, only beyond the physical domain on that side"}};
  if (type != "cartesian") j["center"] = {center[0], center[1], center[2]};
  if (type == "cylindrical" && space == 3) j["axis"] = {axis[0], axis[1], axis[2]};
  return j;
}

void configure(Layer& L, const Extents& phys, const Extents& reg, double tol) {
  const bool cart = L.type == "cartesian" && L.space != 1;
  auto set = [&](int i, bool radial) {
    Direction& P = L.dir[i];
    Direction& N = L.dirn[i];
    P = Direction{};
    N = Direction{};
    if (reg.hi[i] > phys.hi[i] + tol) {
      P.on = true;
      P.sign = 1;
      P.q0 = std::max(phys.hi[i], reg.lo[i]);
      P.d = reg.hi[i] - P.q0;
    }
    if (!radial && reg.lo[i] < phys.lo[i] - tol) {
      N.on = true;
      N.sign = -1;
      N.q0 = std::min(phys.lo[i], reg.hi[i]);
      N.d = N.q0 - reg.lo[i];
    }
    if (P.on && !(P.d > tol)) P.on = false;
    if (N.on && !(N.d > tol)) N.on = false;
  };
  if (cart) {
    for (int i = 0; i < (L.space == 3 ? 3 : 2); ++i) set(i, false);
  } else if (L.type == "cylindrical" || (L.space == 1 && L.type == "cartesian")) {  // axisymmetric cartesian = cylindrical
    set(0, true);
    if (L.space != 2) set(2, false);
  } else {
    set(0, true);
    AXF_REQUIRE(L.dir[0].on && reg.lo[0] >= phys.hi[0] - tol,
                strcat("spherical PML '", L.name, "': the layer (radius ", reg.lo[0], " to ", reg.hi[0],
                       " m) is not outside the field domain (which reaches radius ", phys.hi[0], " m)"),
                "/fullwave/pml", "Check the PML center, or use a cartesian/cylindrical layer.");
  }
  bool any = false;
  for (int i = 0; i < 3; ++i) any = any || L.dir[i].on || L.dirn[i].on;
  AXF_REQUIRE(any, strcat("PML '", L.name, "' (", L.type, "): the layer does not reach beyond the non-PML field domain "
                          "in any stretched coordinate"),
              "/fullwave/pml",
              "A PML region must lie outside the physical domain (a slab, a frame or box shell, a cylindrical or "
              "spherical shell); check its type and center/axis.");
}

Layer from_json(const nlohmann::json& j, int space, const std::string& where) {
  Layer L;
  L.space = space;
  L.type = j.value("type", std::string("cartesian"));
  AXF_REQUIRE(L.type == "cartesian" || L.type == "cylindrical" || L.type == "spherical",
              "PML type must be cartesian, cylindrical or spherical", where + "/type");
  AXF_REQUIRE(!(space == 2 && L.type == "spherical"), "a planar 2D design has no spherical PML (use cylindrical)",
              where + "/type");
  if (j.contains("center")) {
    const auto& c = j.at("center");
    AXF_REQUIRE(c.is_array() && (c.size() == 3 || (space != 3 && c.size() == 2)),
                "PML center must be [x, y, z] (3D) or a 2D point", where + "/center");
    for (size_t i = 0; i < c.size(); ++i) L.center[i] = c.at(i).get<double>();
  }
  if (space == 1 && j.contains("center"))
    AXF_REQUIRE(std::fabs(L.center[0]) == 0.0, "an axisymmetric PML center must lie on the axis: [0, z]",
                where + "/center");
  if (j.contains("axis")) {
    const auto& a = j.at("axis");
    AXF_REQUIRE(space == 3 && a.is_array() && a.size() == 3, "PML axis is a 3D unit vector [ax, ay, az] (3D only)",
                where + "/axis");
    double n = 0;
    for (int i = 0; i < 3; ++i) { L.axis[i] = a.at(i).get<double>(); n += L.axis[i] * L.axis[i]; }
    AXF_REQUIRE(n > 0, "PML axis must be nonzero", where + "/axis");
    for (int i = 0; i < 3; ++i) L.axis[i] /= std::sqrt(n);
  }
  L.scaling = j.value("scaling_factor", 1.0);
  L.curvature = j.value("curvature", 1.0);
  L.wavelength = j.value("wavelength_m", 0.0);
  AXF_REQUIRE(std::isfinite(L.scaling) && L.scaling > 0, "PML scaling_factor must be positive", where);
  AXF_REQUIRE(std::isfinite(L.curvature) && L.curvature >= 1, "PML curvature must be >= 1", where);
  AXF_REQUIRE(std::isfinite(L.wavelength) && L.wavelength >= 0, "PML wavelength_m must be >= 0 (0: c/f)", where);
  return L;
}

}  // namespace axf::pml
