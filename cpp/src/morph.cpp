#include "axionfem/morph.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>

#include "axionfem/core.hpp"
#include "axionfem/units.hpp"

namespace axf::morph {

using units::pi;

namespace {

using Vec2 = spec::Vec2;

double wrap_pm_pi(double a) {
  while (a > pi) a -= 2 * pi;
  while (a < -pi) a += 2 * pi;
  return a;
}

void push_segment(std::vector<Prim>& out, double r0, double z0, double r1, double z1) {
  Prim p;
  p.r0 = r0;
  p.z0 = z0;
  p.r1 = r1;
  p.z1 = z1;
  out.push_back(p);
}

// Arc from (r0, z0) to (r1, z1) about (cr, cz), the short way round (as OCC's addCircleArc).
void push_arc(std::vector<Prim>& out, double cr, double cz, double r0, double z0, double r1, double z1) {
  Prim p;
  p.arc = true;
  p.cr = cr;
  p.cz = cz;
  p.rad = std::hypot(r0 - cr, z0 - cz);
  p.a0 = std::atan2(z0 - cz, r0 - cr);
  p.a1 = p.a0 + wrap_pm_pi(std::atan2(z1 - cz, r1 - cr) - p.a0);
  p.r0 = r0;
  p.z0 = z0;
  p.r1 = r1;
  p.z1 = z1;
  out.push_back(p);
}

// Rounded rectangle, exactly as occ::addRectangle(..., fillet): four edges shortened by f
// and four quarter arcs, counter-clockwise from the bottom edge.  Zero-radius arcs are
// emitted as degenerate primitives so that the list structure is independent of f.
void rect_prims(double a, double b, double c, double d, double f, std::vector<Prim>& out) {
  f = std::max(f, 0.0);
  f = std::min(f, 0.5 * std::min(std::fabs(b - a), std::fabs(d - c)));
  push_segment(out, a + f, c, b - f, c);
  push_arc(out, b - f, c + f, b - f, c, b, c + f);
  push_segment(out, b, c + f, b, d - f);
  push_arc(out, b - f, d - f, b, d - f, b - f, d);
  push_segment(out, b - f, d, a + f, d);
  push_arc(out, a + f, d - f, a + f, d, a, d - f);
  push_segment(out, a, d - f, a, c + f);
  push_arc(out, a + f, c + f, a, c + f, a + f, c);
}

// Closed polygon with a fillet at every vertex, as axf-mesher's polygon_wire: per vertex an
// arc (degenerate when the corner is straight or the fillet vanishes) and then the edge to the
// next vertex.
void polygon_prims(const std::vector<Vec2>& P, double fillet, std::vector<Prim>& out) {
  const size_t n = P.size();
  if (n < 3) return;
  struct Corner { double ax, ay, bx, by, cx, cy; bool round; };
  std::vector<Corner> C(n);
  for (size_t i = 0; i < n; ++i) {
    const Vec2& prev = P[(i + n - 1) % n];
    const Vec2& v = P[i];
    const Vec2& next = P[(i + 1) % n];
    double ux = prev.r - v.r, uy = prev.z - v.z, wx = next.r - v.r, wy = next.z - v.z;
    const double lu = std::hypot(ux, uy), lw = std::hypot(wx, wy);
    if (lu <= 0 || lw <= 0 || fillet <= 0) {
      C[i] = {v.r, v.z, v.r, v.z, 0, 0, false};
      continue;
    }
    ux /= lu; uy /= lu; wx /= lw; wy /= lw;
    const double ang = std::acos(std::clamp(ux * wx + uy * wy, -1.0, 1.0));
    if (ang < 1e-6 || std::fabs(pi - ang) < 1e-6) {
      C[i] = {v.r, v.z, v.r, v.z, 0, 0, false};
      continue;
    }
    const double dd = fillet / std::tan(ang / 2), dc = fillet / std::sin(ang / 2);
    const double bxv = ux + wx, byv = uy + wy, lb = std::hypot(bxv, byv);
    C[i] = {v.r + dd * ux, v.z + dd * uy, v.r + dd * wx, v.z + dd * wy,
            v.r + dc * bxv / lb, v.z + dc * byv / lb, true};
  }
  for (size_t i = 0; i < n; ++i) {
    if (C[i].round) push_arc(out, C[i].cx, C[i].cy, C[i].ax, C[i].ay, C[i].bx, C[i].by);
    else push_segment(out, C[i].ax, C[i].ay, C[i].ax, C[i].ay);  // degenerate placeholder
    push_segment(out, C[i].bx, C[i].by, C[(i + 1) % n].ax, C[(i + 1) % n].ay);
  }
}

void shape_prims(const spec::Shape& s, std::vector<Prim>& out) {
  using K = spec::Shape::Kind;
  switch (s.kind) {
    case K::Imported:
      throw Error("an imported region has no shape primitives (shape sensitivities need parametric shapes)");
    case K::Rect:
      rect_prims(s.r0, s.r1, s.z0, s.z1, s.fillet, out);
      break;
    case K::Frame:
      rect_prims(s.r0, s.r1, s.z0, s.z1, s.fillet, out);
      rect_prims(s.r0 + s.thickness, s.r1 - s.thickness, s.z0 + s.thickness, s.z1 - s.thickness,
                 std::max(s.fillet - s.thickness, 0.0), out);
      break;
    case K::Polygon:
      polygon_prims(s.points, s.fillet, out);
      break;
    case K::Disk: {
      Prim p;
      p.arc = true;
      p.cr = s.center.r;
      p.cz = s.center.z;
      p.rad = s.radius;
      p.a0 = 0;
      p.a1 = 2 * pi;
      p.r0 = p.r1 = s.center.r + s.radius;
      p.z0 = p.z1 = s.center.z;
      out.push_back(p);
      break;
    }
    case K::Union:
    case K::Difference:
      // The boundary of a boolean lies in the boundaries of its children.
      for (auto& c : s.children) shape_prims(c, out);
      break;
  }
}

void path_prims(const spec::Path& p, std::vector<Prim>& out) {
  using K = spec::Path::Kind;
  switch (p.kind) {
    case K::Segment:
    case K::Polyline: {
      const size_t n = p.points.size();
      for (size_t i = 0; i + 1 < n; ++i)
        push_segment(out, p.points[i].r, p.points[i].z, p.points[i + 1].r, p.points[i + 1].z);
      if (p.closed && n > 2) push_segment(out, p.points[n - 1].r, p.points[n - 1].z, p.points[0].r, p.points[0].z);
      break;
    }
    case K::Rect:
      rect_prims(p.r0, p.r1, p.z0, p.z1, p.fillet, out);
      break;
    case K::Arc: {
      Prim a;
      a.arc = true;
      a.cr = p.center.r;
      a.cz = p.center.z;
      a.rad = p.radius;
      a.a0 = p.angle0;
      a.a1 = p.angle1;
      a.point(0, a.r0, a.z0);
      a.point(1, a.r1, a.z1);
      out.push_back(a);
      break;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------- Prim
double Prim::length() const {
  if (arc) return std::fabs(rad * (a1 - a0));
  return std::hypot(r1 - r0, z1 - z0);
}

void Prim::point(double t, double& r, double& z) const {
  if (arc) {
    const double a = a0 + t * (a1 - a0);
    r = cr + rad * std::cos(a);
    z = cz + rad * std::sin(a);
  } else {
    r = r0 + t * (r1 - r0);
    z = z0 + t * (z1 - z0);
  }
}

double Prim::project(double r, double z, double& dist) const {
  double t = 0;
  if (arc) {
    const double sweep = a1 - a0;
    double a = std::atan2(z - cz, r - cr);
    if (sweep >= 0) {
      while (a < a0) a += 2 * pi;
      while (a >= a0 + 2 * pi) a -= 2 * pi;
    } else {
      while (a > a0) a -= 2 * pi;
      while (a <= a0 - 2 * pi) a += 2 * pi;
    }
    // u >= 0 grows along the arc; u > 1 means the point's angle is outside the sweep, and the
    // closer end (in angle) wins.
    const double u = sweep != 0 ? (a - a0) / sweep : 0.0;
    const double full = sweep != 0 ? 2 * pi / std::fabs(sweep) : 1.0;
    t = u <= 1 ? u : ((u - 1 < full - u) ? 1.0 : 0.0);
  } else {
    const double dr = r1 - r0, dz = z1 - z0, l2 = dr * dr + dz * dz;
    t = l2 > 0 ? std::clamp(((r - r0) * dr + (z - z0) * dz) / l2, 0.0, 1.0) : 0.0;
  }
  double pr, pz;
  point(t, pr, pz);
  dist = std::hypot(r - pr, z - pz);
  return t;
}

void Prim::frame(double t, double& tr, double& tz, double& nr, double& nz) const {
  if (arc) {
    const double a = a0 + t * (a1 - a0);
    const double s = (a1 >= a0) ? 1.0 : -1.0;
    tr = -s * std::sin(a);
    tz = s * std::cos(a);
  } else {
    const double dr = r1 - r0, dz = z1 - z0, l = std::hypot(dr, dz);
    tr = l > 0 ? dr / l : 1.0;
    tz = l > 0 ? dz / l : 0.0;
  }
  nr = -tz;
  nz = tr;
}

std::vector<Prim> boundary_primitives(const spec::Design& d) {
  std::vector<Prim> out;
  for (auto& g : d.regions) shape_prims(g.shape, out);
  for (auto& c : d.curves) path_prims(c.path, out);
  return out;
}

void mesh_extent(const spec::Design& d, double& device_radius, double& domain_radius, double& extent) {
  double rmin_all = 1e300, rmax_all = 0, zmin_all = 1e300, zmax_all = -1e300;
  auto grow = [&](double a, double b, double c, double e) {
    rmin_all = std::min(rmin_all, a);
    rmax_all = std::max(rmax_all, b);
    zmin_all = std::min(zmin_all, c);
    zmax_all = std::max(zmax_all, e);
  };
  for (auto& g : d.regions) {
    double a, b, c, e;
    g.shape.bbox(a, b, c, e);
    grow(a, b, c, e);
  }
  for (auto& cv : d.curves) {
    double a, b, c, e;
    cv.path.bbox(a, b, c, e);
    grow(a, b, c, e);
  }
  // radii are measured from the centre of the outer semicircle, (r, z) = (0, domain_center_z)
  const double z0 = d.mesh.domain_center_z;
  const double zabs = std::max(std::fabs(zmin_all - z0), std::fabs(zmax_all - z0));
  device_radius = std::sqrt(rmax_all * rmax_all + zabs * zabs);
  extent = std::max(rmax_all - std::max(0.0, rmin_all), zmax_all - zmin_all);
  domain_radius = d.mesh.domain_radius > 0 ? d.mesh.domain_radius : 2.5 * device_radius;
}

json Report::to_json() const {
  return {{"nodes", nodes},
          {"on_geometry", matched},
          {"on_outer_boundary", outer},
          {"axis_r_fixed", axis_fixed},
          {"interior_harmonic", interior},
          {"crossings", crossings},
          {"degenerate_primitives", skipped_prims},
          {"max_displacement_m", max_displacement},
          {"max_boundary_displacement_m", max_boundary_displacement},
          {"domain_radius_m", {domain_radius_from, domain_radius_to}}};
}

// ---------------------------------------------------------------- Morpher
// Everything that depends on the nominal mesh alone is built once: the primitive a node sits on
// with its parameter and local frame, the 2x2 least-squares system that places the node, and the
// factorized Laplace extension for each coordinate.
struct Morpher::Impl {
  const axi::MeshBundle* mb = nullptr;
  spec::Design from;
  std::vector<Prim> p0;
  double Rb0 = 0, Rdev0 = 0, L0 = 0;
  int nn = 0;
  std::vector<double> X;  // nominal node coordinates, (r, z) pairs
  struct Claim {
    int prim;
    double t, tr, tz, nr, nz, wt;  // parameter, tangent, normal, tangential weight
  };
  struct Node {
    int node;
    std::vector<Claim> claims;
    double Minv[4];  // inverse of the 2x2 normal-equation matrix
  };
  std::vector<Node> matched;
  std::vector<char> ess_r, ess_z, is_outer;
  Report rep0;
  // reduced Laplace system per coordinate (the essential sets differ: r is fixed on the axis)
  struct Sys {
    std::vector<int> idx;  // node -> free index (-1 if essential)
    int nf = 0;
    Eigen::SparseMatrix<double> A;  // full Laplacian rows of the free nodes (columns: all nodes)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol;
  };
  Sys sys[2];
};

Morpher::~Morpher() = default;
Morpher::Morpher(Morpher&&) = default;

Morpher::Morpher(const axi::MeshBundle& mb, const spec::Design& from) : impl_(std::make_unique<Impl>()) {
  Impl& I = *impl_;
  I.mb = &mb;
  I.from = from;
  I.p0 = boundary_primitives(from);
  mesh_extent(from, I.Rdev0, I.Rb0, I.L0);

  mfem::Mesh& m = *mb.mesh;
  const mfem::GridFunction* nodes = m.GetNodes();
  AXF_REQUIRE(nodes != nullptr,
              "mesh morphing needs a mesh with nodes (curved geometry); the mesher writes order-2 geometry");
  const mfem::FiniteElementSpace* nfes = nodes->FESpace();
  mfem::FiniteElementSpace sfes(&m, nfes->FEColl());
  I.nn = sfes.GetNDofs();
  AXF_REQUIRE(I.nn == nfes->GetNDofs(), "internal: node space size mismatch");
  I.X.resize(2 * static_cast<size_t>(I.nn));
  for (int i = 0; i < I.nn; ++i) {
    I.X[2 * i] = (*nodes)(nfes->DofToVDof(i, 0));
    I.X[2 * i + 1] = (*nodes)(nfes->DofToVDof(i, 1));
  }
  Report& rep = I.rep0;
  rep.nodes = I.nn;
  rep.domain_radius_from = I.Rb0;
  rep.domain_radius_to = I.Rb0;
  for (auto& p : I.p0) if (p.degenerate()) ++rep.skipped_prims;

  // ---------------------------------------------------------- which primitives each node sits on
  // A node on several primitives (a corner, or a crossing of overlapping regions) must satisfy all
  // of them: match their normal displacements exactly (least squares) and, when they are parallel
  // and therefore leave the tangential direction free, follow the parametric correspondence.
  const double tol = 1e-7 * std::max(I.L0, 1e-30);
  I.ess_r.assign(I.nn, 0);
  I.ess_z.assign(I.nn, 0);
  I.is_outer.assign(I.nn, 0);
  for (int i = 0; i < I.nn; ++i) {
    const double r = I.X[2 * i], z = I.X[2 * i + 1];
    Impl::Node nd;
    nd.node = i;
    double n0r = 0, n0z = 0, span = 0;
    for (size_t k = 0; k < I.p0.size(); ++k) {
      if (I.p0[k].degenerate()) continue;
      double dist = 0;
      const double t = I.p0[k].project(r, z, dist);
      if (dist > tol) continue;
      Impl::Claim c;
      c.prim = static_cast<int>(k);
      c.t = t;
      I.p0[k].frame(t, c.tr, c.tz, c.nr, c.nz);
      c.wt = 0;
      if (nd.claims.empty()) { n0r = c.nr; n0z = c.nz; }
      else span = std::max(span, std::fabs(n0r * c.nz - n0z * c.nr));
      nd.claims.push_back(c);
    }
    if (nd.claims.empty()) continue;
    const bool parallel = span < 0.1;  // no crossing: the tangential motion is not determined
    if (!parallel) ++rep.crossings;
    double M00 = 0, M01 = 0, M11 = 0;
    for (auto& c : nd.claims) {
      c.wt = parallel ? 0.3 : 0.0;
      const double w2t = c.wt * c.wt;
      M00 += c.nr * c.nr + w2t * c.tr * c.tr;
      M01 += c.nr * c.nz + w2t * c.tr * c.tz;
      M11 += c.nz * c.nz + w2t * c.tz * c.tz;
    }
    const double reg = 1e-12 * std::max(M00 + M11, 1e-300);
    M00 += reg;
    M11 += reg;
    const double det = M00 * M11 - M01 * M01;
    if (det <= 0) continue;
    nd.Minv[0] = M11 / det;
    nd.Minv[1] = -M01 / det;
    nd.Minv[2] = -M01 / det;
    nd.Minv[3] = M00 / det;
    I.matched.push_back(std::move(nd));
    I.ess_r[i] = I.ess_z[i] = 1;
    ++rep.matched;
  }
  // outer semicircle (scales with the domain radius) and axis (r stays zero)
  mfem::Array<int> bdofs;
  for (int be = 0; be < m.GetNBE(); ++be) {
    const int a = m.GetBdrAttribute(be);
    if (a != axi::kOuterAttr && a != axi::kAxisAttr) continue;
    sfes.GetBdrElementDofs(be, bdofs);
    for (int dd : bdofs) {
      if (a == axi::kOuterAttr) {
        if (!I.is_outer[dd]) ++rep.outer;
        I.is_outer[dd] = 1;
        I.ess_r[dd] = I.ess_z[dd] = 1;
      } else {
        if (!I.ess_r[dd]) ++rep.axis_fixed;
        I.ess_r[dd] = 1;
      }
    }
  }

  // ---------------------------------------------------------- harmonic extension, factorized once
  mfem::BilinearForm a(&sfes);
  a.AddDomainIntegrator(new mfem::DiffusionIntegrator);
  a.Assemble();
  a.Finalize();
  const mfem::SparseMatrix& A = a.SpMat();
  const int* Ai = A.GetI();
  const int* Aj = A.GetJ();
  const double* Av = A.GetData();
  for (int comp = 0; comp < 2; ++comp) {
    const std::vector<char>& ess = comp == 0 ? I.ess_r : I.ess_z;
    Impl::Sys& S = I.sys[comp];
    S.idx.assign(I.nn, -1);
    S.nf = 0;
    for (int i = 0; i < I.nn; ++i) if (!ess[i]) S.idx[i] = S.nf++;
    if (S.nf == 0) continue;
    std::vector<Eigen::Triplet<double>> free_trip, bnd_trip;
    for (int i = 0; i < I.nn; ++i) {
      if (ess[i]) continue;
      for (int k = Ai[i]; k < Ai[i + 1]; ++k) {
        if (ess[Aj[k]]) bnd_trip.emplace_back(S.idx[i], Aj[k], Av[k]);
        else free_trip.emplace_back(S.idx[i], S.idx[Aj[k]], Av[k]);
      }
    }
    Eigen::SparseMatrix<double> K(S.nf, S.nf);
    K.setFromTriplets(free_trip.begin(), free_trip.end());
    S.A.resize(S.nf, I.nn);
    S.A.setFromTriplets(bnd_trip.begin(), bnd_trip.end());
    S.chol.compute(K);
    AXF_REQUIRE(S.chol.info() == Eigen::Success, "the mesh-morphing Laplace extension could not be factorized");
    if (comp == 0) rep.interior = S.nf;
  }
}

const Report& Morpher::nominal_report() const { return impl_->rep0; }

axi::MeshBundle Morpher::to(const spec::Design& d1, Report* report) const {
  const Impl& I = *impl_;
  const spec::Design& d0 = I.from;
  AXF_REQUIRE(d0.regions.size() == d1.regions.size() && d0.curves.size() == d1.curves.size(),
              "the perturbed design has a different number of regions or curves, so the mesh cannot be morphed",
              "/parameters",
              "A parameter that switches regions on or off (\"enabled\") changes the mesh topology: use "
              "method='fd_remesh' for it.");
  for (size_t i = 0; i < d0.regions.size(); ++i)
    AXF_REQUIRE(d0.regions[i].name == d1.regions[i].name, "region order changed; the mesh cannot be morphed");
  const std::vector<Prim> p1 = boundary_primitives(d1);
  AXF_REQUIRE(p1.size() == I.p0.size(),
              "the perturbed geometry decomposes into a different number of boundary primitives", "/parameters",
              "A parameter that changes the number of polygon points cannot be differentiated by morphing.");
  double Rdev1 = 0, Rb1 = 0, L1 = 0;
  mesh_extent(d1, Rdev1, Rb1, L1);

  Report rep = I.rep0;
  rep.domain_radius_to = Rb1;

  std::vector<double> ur(I.nn, 0.0), uz(I.nn, 0.0);
  for (const Impl::Node& nd : I.matched) {
    const double r = I.X[2 * nd.node], z = I.X[2 * nd.node + 1];
    double b0 = 0, b1 = 0;
    for (const Impl::Claim& c : nd.claims) {
      double R = 0, Z = 0;
      p1[c.prim].point(c.t, R, Z);
      const double dr = R - r, dz = Z - z;
      const double bn = c.nr * dr + c.nz * dz, bt = c.tr * dr + c.tz * dz;
      const double w2t = c.wt * c.wt;
      b0 += c.nr * bn + w2t * c.tr * bt;
      b1 += c.nz * bn + w2t * c.tz * bt;
    }
    ur[nd.node] = nd.Minv[0] * b0 + nd.Minv[1] * b1;
    uz[nd.node] = nd.Minv[2] * b0 + nd.Minv[3] * b1;
    rep.max_boundary_displacement = std::max(rep.max_boundary_displacement, std::hypot(ur[nd.node], uz[nd.node]));
  }
  const double scale = I.Rb0 > 0 ? Rb1 / I.Rb0 - 1.0 : 0.0;
  const double z0 = d1.mesh.domain_center_z;  // the semicircle is centred at (0, z0), not at the origin
  for (int i = 0; i < I.nn; ++i) {
    if (!I.is_outer[i]) continue;
    ur[i] = scale * I.X[2 * i];
    uz[i] = z0 - d0.mesh.domain_center_z + scale * (I.X[2 * i + 1] - d0.mesh.domain_center_z);
    rep.max_boundary_displacement = std::max(rep.max_boundary_displacement, std::hypot(ur[i], uz[i]));
  }
  for (int i = 0; i < I.nn; ++i) if (I.ess_r[i] && !I.ess_z[i]) ur[i] = 0.0;  // axis

  for (int comp = 0; comp < 2; ++comp) {
    const Impl::Sys& S = I.sys[comp];
    if (S.nf == 0) continue;
    std::vector<double>& u = comp == 0 ? ur : uz;
    const std::vector<char>& ess = comp == 0 ? I.ess_r : I.ess_z;
    Eigen::VectorXd ud = Eigen::VectorXd::Zero(I.nn);
    for (int i = 0; i < I.nn; ++i) if (ess[i]) ud(i) = u[i];
    Eigen::VectorXd x = S.chol.solve(Eigen::VectorXd(-(S.A * ud)));
    for (int i = 0; i < I.nn; ++i) if (!ess[i]) u[i] = x(S.idx[i]);
  }

  axi::MeshBundle out;
  out.mesh = std::make_unique<mfem::Mesh>(*I.mb->mesh);
  out.region_of_attr = I.mb->region_of_attr;
  out.attr_of_region = I.mb->attr_of_region;
  out.attr_of_curve = I.mb->attr_of_curve;
  out.domain_radius = Rb1;
  out.domain_center_z = d1.mesh.domain_center_z;
  out.device_radius = Rdev1;
  out.info = I.mb->info;
  out.info["morphed"] = true;
  mfem::GridFunction* nodes = out.mesh->GetNodes();
  const mfem::FiniteElementSpace* nfes = nodes->FESpace();
  for (int i = 0; i < I.nn; ++i) {
    rep.max_displacement = std::max(rep.max_displacement, std::hypot(ur[i], uz[i]));
    (*nodes)(nfes->DofToVDof(i, 0)) = I.X[2 * i] + ur[i];
    (*nodes)(nfes->DofToVDof(i, 1)) = I.X[2 * i + 1] + uz[i];
  }
  out.mesh->NodesUpdated();
  if (report) *report = rep;
  return out;
}

axi::MeshBundle morph(const axi::MeshBundle& mb, const spec::Design& from, const spec::Design& to, Report* report) {
  return Morpher(mb, from).to(to, report);
}

}  // namespace axf::morph
