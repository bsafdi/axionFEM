// Mesh morphing: carry one axisymmetric mesh over to a design with slightly
// different parameters, keeping the topology (same elements, same DOF numbering).
//
// Shape sensitivities need the derivative of the discrete problem with respect to
// the node positions, which means a mesh velocity field.  The mesh is built by
// gmsh from analytic shapes, so the correspondence is analytic too:
//
//  * every shape and path decomposes into boundary primitives (straight segments
//    and circular arcs) in a fixed order that does not depend on the numbers, so
//    primitive k of the nominal geometry corresponds to primitive k of the
//    perturbed one ("parametric correspondence").  Booleans contribute the
//    primitives of their children, which contain the boundary of the result;
//  * a mesh node lying on one or more primitives moves with them (least squares
//    over the primitives' normal displacements, so crossings and corners land on
//    the intersection of the perturbed primitives);
//  * nodes on the outer semicircle scale with the domain radius, nodes on the
//    axis keep r = 0;
//  * the interior follows a harmonic (component-wise Laplace) extension.
//
// A uniform dilation of the design is reproduced exactly (r and z are harmonic).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionfem/axi.hpp"
#include "axionfem/spec.hpp"

namespace axf::morph {

using json = nlohmann::json;

// One boundary primitive of a resolved shape in the (r, z) half-plane.
struct Prim {
  bool arc = false;
  double r0 = 0, z0 = 0, r1 = 0, z1 = 0;          // endpoints (segment and arc)
  double cr = 0, cz = 0, rad = 0, a0 = 0, a1 = 0;  // arc centre, radius, angles [rad]
  double length() const;
  bool degenerate() const { return length() <= 0; }
  void point(double t, double& r, double& z) const;
  // Closest point: returns its parameter t in [0, 1] and sets the distance.
  double project(double r, double z, double& dist) const;
  // Unit tangent and normal at parameter t.
  void frame(double t, double& tr, double& tz, double& nr, double& nz) const;
};

// Boundary primitives of every region shape and curve path, in a deterministic order.
std::vector<Prim> boundary_primitives(const spec::Design& d);

// Device radius, domain radius and extent of a design, exactly as the mesher computes them.
void mesh_extent(const spec::Design& d, double& device_radius, double& domain_radius, double& extent);

struct Report {
  int nodes = 0, matched = 0, outer = 0, axis_fixed = 0, interior = 0, crossings = 0, skipped_prims = 0;
  double max_displacement = 0, max_boundary_displacement = 0;
  double domain_radius_from = 0, domain_radius_to = 0;
  json to_json() const;
};

// Repeated morphing of one mesh (one nominal design, many perturbed ones, as the finite
// differences of the sensitivity engine need).  Everything that depends on the nominal mesh
// alone — which node sits on which primitive, and the factorized Laplace extension — is built
// once in the constructor.  The nominal mesh bundle is referenced, not copied: it must outlive
// the Morpher.
class Morpher {
 public:
  Morpher(const axi::MeshBundle& mb, const spec::Design& from);
  ~Morpher();
  Morpher(Morpher&&);
  // Mesh of design `to`, obtained by moving the nodes of the nominal mesh (same topology).
  // Throws if `to` is not structurally identical to the nominal design.
  axi::MeshBundle to(const spec::Design& to, Report* report = nullptr) const;
  const Report& nominal_report() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// One-off morph (builds a Morpher and uses it once).
axi::MeshBundle morph(const axi::MeshBundle& mb, const spec::Design& from, const spec::Design& to,
                      Report* report = nullptr);

}  // namespace axf::morph
