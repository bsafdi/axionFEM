// Error indicator of the 3D full-wave port fields (fw_error_indicator.hpp; docs/physics/fullwave_error_indicator.md):
// recovery (Zienkiewicz-Zhu) at order 1; at order 2 and above element residual problems for the magnetic part and a
// global gradient projection for the electric part. Every rank evaluates its own elements; the nodal averages are
// summed over the ranks that share a node (the H1 space's group communicator; on cuts, one sum per side over all
// ranks), the residual problems read the face neighbours of other ranks, and the gradient projection is one parallel
// solve, so the result does not depend on the partition beyond the solvers' tolerances.
#include "fw_error_indicator.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>

#include "axionfem/axi.hpp"
#include "axionfem/core.hpp"
#include "axionfem/units.hpp"

namespace axf {

namespace {

using cplx = std::complex<double>;

// per node: the real and imaginary parts of E and of curl E (12), and the number of contributions
constexpr int kComponents = 13;

bool perfect_conductor(spec::MaterialKind kind) {
  return kind == spec::MaterialKind::Superconductor || kind == spec::MaterialKind::Conductor;
}

// a / b, NaN (JSON null) when b is not a finite positive number or a is not finite: a failed solve's norms must not
// read as an estimated error of 0
double ratio(double a, double b) {
  if (!(b > 0) || !std::isfinite(b) || !std::isfinite(a)) return std::numeric_limits<double>::quiet_NaN();
  return a / b;
}

// the node of an H1 element: its support (the element's vertices with a positive barycentric coordinate), as sorted
// (serial vertex, rounded coordinate) pairs, which name the node the same way from every element that contains it
using NodeKey = std::vector<std::pair<int, long long>>;

NodeKey node_key(const mfem::Array<int>& vertices, const mfem::IntegrationPoint& ip) {
  const double lambda[4] = {1 - ip.x - ip.y - ip.z, ip.x, ip.y, ip.z};
  NodeKey key;
  for (int i = 0; i < 4; ++i)
    if (lambda[i] > 1e-9) key.emplace_back(vertices[i], std::llround(lambda[i] * 1e9));
  std::sort(key.begin(), key.end());
  return key;
}

// how the element residual estimator treats a face of a field element: the average of the two sides' tractions (an
// interior face), the element's own traction (no residual: a port or sheet face, a Robin boundary, the radiating
// shell, a PML layer), zero traction (a natural n x H = 0 boundary: a PMC or unmarked outer face; an interior pmc face
// is an ordinary interior face, since the solver imposes nothing there) or a perfect conductor (the local error's
// tangential trace is zero)
enum : signed char { kAverage = 0, kOwn = 1, kNatural = 2, kPec = 3 };

// the effectivities measured (docs/physics/fullwave_error_indicator.md, Validation), for Describe()
const char* const RELIABILITY_ORDER_1 =
    "an indicator of where the field is under-resolved and roughly how much, not a bound. Order 1 (recovery): the L "
    "estimate 1.1-1.4 times the exact error of a TEM line where the magnetic part dominates, 3-9 where the parts "
    "partly cancel. The signed estimate: where the magnetic part dominates 0.9-1.05 on the TEM line and 1.1-1.3 next "
    "to a crack and a 270-degree edge; where the parts partly cancel 0.27-1.0 on the TEM line and 1.4-3.1 next to the "
    "crack and the edge; 0.85-1.18 on the two-port copper frame";
const char* const RELIABILITY_ORDER_2 =
    "an estimate of the error's size and, where its two parts differ by more than a factor 8, of its sign; not a "
    "bound in either direction. Order 2 (element residual problems for the magnetic part, a global gradient "
    "projection for the electric part): the signed L estimate was 0.39-0.94 of the error of L where measured (a TEM "
    "line with an exact Z, the two-port copper frame, a small CORE-type slotted box, a revolved coax, a crack and a "
    "270-degree edge). Next to cracks and re-entrant edges it is lowest and falls under refinement (0.39 and 0.48 on 1.3M DOFs, "
    "tending to about 0.37 and 0.46; the two-port frame's fine mesh 0.51), so the error can be up to about 2.7 times "
    "the estimate; on smooth fields 0.82-0.94. With the parts' measured effectivities (magnetic 0.37-1, electric "
    "0.45-1) the factor-8 rule keeps the signed estimate within 0.34-1.32 of the error in models without loss-phase "
    "contrast; a fit on the CORE-like box suggests a magnetic part 2-3 times over, outside those ranges, and where "
    "such a part dominates the signed estimate would be as far over. Where the field materials' loss phases differ "
    "the electric part can exceed the error's electric energy by up to gradient.electric_bound_factor (up to 2), and "
    "a conductor meshed as a field region needs its skin depth resolved (otherwise use an impedance boundary). The "
    "magnetic range and the conductor statement assume |k| h of about 1 or less in every direction: with cells that "
    "resolve the skin depth normal to a conductor's surface but are 4 times longer along it (|k| h of 1.3-5.4 there), "
    "the magnetic part was "
    "up to 1.23 of the error. residual_kh_warning counts elements past |k| h_max = 1 (sliver-like ones past 0.2), a "
    "heuristic: a near-resonant local problem inside those bounds is not detected. The "
    "electric part reproduced a smooth TEM line's exact electric error to 0.997-1.000 but about half of it at a "
    "crack. Order 3 is not validated. surface_relative_error is the recovery's and unvalidated at order 2 (it sees "
    "only the jumps of curl E)";

// the gradient projection's solve: relative residual and iterations. The projection's energy is quadratic in the PCG
// error: against 1e-6, 1e-4 moved the estimates by at most 6e-9 with PCG and 4e-5 with GMRES (the lossy line), with
// 35-45 % fewer iterations
constexpr double kGradientTolerance = 1e-4;   // AXF_FW_INDICATOR_TOL overrides it (fw_indicator_gradient_tolerance)
constexpr int kGradientIterations = 500;
// below this share of the norm the electric part is not computed (deep MQS: there the gradient projection cannot move
// the estimate; the largest electric-to-magnetic ratio of relative errors in the validation, 44, gives at most 0.2 %
// of eta^2 at this share)
constexpr double kElectricShareSkip = 1e-6;
// the signed estimate is withheld where the two parts are within this factor of each other (energies)
constexpr double kSignedResolution = 8.0;

using cvec = Eigen::VectorXcd;
using cmat = Eigen::MatrixXcd;
using rmat = Eigen::MatrixXd;
using rvec = Eigen::VectorXd;

}  // namespace

// The magnetic part at order p >= 2: element residual problems. Per field element K the error e = E - E_h obeys
//   a_K(e, v) = i omega (J, v)_K - a_K(E_h, v) - <n x nu curl E, v>_dK      for all v,
// a_K(u, v) = int_K nu curl u . curl v - omega^2 eps u . v. This is solved in ND_q(K), q = p + 1, on the complement of
// the local gradients, with the unknown traction n x nu curl E replaced by the average of the two sides' discrete
// tractions (an interior face), the element's own (no residual: a port or sheet face, a Robin group, the radiating
// shell, a PML layer) or zero (a natural n x H = 0 face); on conductor faces and edges the local error's tangential
// trace is zero. The port current J is locally a gradient (J = scale grad V), so it does not enter the complement.
// The gradient part is not solved locally: for gradients the averaged normal flux replaces the exact one at the error's
// own order, and the local estimate came out 9-16 times high on a TEM line; the global projection below needs no
// flux (it is the exact projection of the error onto the gradients of H1_{p+1}).
struct FwErrorIndicator::Residual {
  int q = 3;   // the local order p + 1
  int n = 0;   // ND_q dofs
  int m = 0;   // H1_q dofs
  std::unique_ptr<mfem::ND_FECollection> nd_fec;
  std::unique_ptr<mfem::H1_FECollection> h1_fec;
  const mfem::FiniteElement* nd = nullptr;
  const mfem::FiniteElement* h1 = nullptr;
  rmat Pi;   // n x n_p: an ND_p field in the ND_q basis (reference elements, raw bases)
  rmat G;    // n x m: the gradients of the H1_q basis in the ND_q basis
  const mfem::IntegrationRule* vol = nullptr;
  const mfem::IntegrationRule* face = nullptr;
  std::vector<rmat> vshape, cshape;   // per volume point: n x 3 reference shapes and curls
  // the reference moments int phi_i,a phi_j,b and int curl phi_i,a curl phi_j,b (a, b = x, y, z; index 3 a + b): on an
  // affine element MM = |det J| sum_ab (J^-1 J^-T)_ab M_ab and CC = sum_ab (J^T J)_ab C_ab / |det J|
  std::array<rmat, 9> mref, cref;
  bool affine_test = true;   // geometry of order <= 2 (the test is not sufficient above)
  // the reference shapes and curls at face points, keyed by the point's reference coordinates (a finite set: the face
  // rule's points on the four faces in their orientations)
  std::map<std::array<long long, 3>, int> point_index;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> fvshape, fcshape;
  // per reference face and edge: the ND_q dofs with a tangential trace there, and the H1_q nodes on it
  std::array<std::vector<char>, 4> face_nd, face_h1;
  std::array<std::vector<char>, 6> edge_nd, edge_h1;
  // per local element: the kind of each face, and the faces and edges on perfect conductors (bit masks)
  std::vector<std::array<signed char, 4>> kind;
  std::vector<unsigned char> pec_faces, pec_edges;
  // per pattern of perfect-conductor faces and edges: the free ND_q dofs, and the H1_q basis of the local gradients
  // in the constrained space (the nodes on one connected conductor part share one value; the first part, or without
  // one vertex 0, is the reference)
  struct Reduced {
    std::vector<int> free;
    rmat T;    // m x m': the reduced H1_q dofs as nodal coefficients
    rmat Gr;   // |free| x m': their gradients in the free ND_q dofs
    // the free dofs other than m' rows of Gr that form an invertible block (positions in `free`): the basis Q C of the
    // complement range(Q) on which the local problem is posed (fw_local_residual). Which rows round-off picks does not
    // matter
    std::vector<int> complement;
    std::array<rmat, 9> S;   // G_r^T M_ab G_r (free rows): on an affine element G_r^T MM G_r = sum_ab g_ab S_ab
    std::array<rmat, 9> MG;  // M_ab G_r (free rows): on an affine element MM G_r = sum_ab g_ab MG_ab
  };
  std::map<int, Reduced> reduced;
  // eps (re, im) and nu (re, im) per element, exchanged for the face neighbours of other ranks
  std::unique_ptr<mfem::L2_FECollection> l2_fec;
  std::unique_ptr<mfem::ParFiniteElementSpace> l2_space;
  std::unique_ptr<mfem::ParGridFunction> coef;

  int ShapeAt(const mfem::IntegrationPoint& ip) {
    const std::array<long long, 3> key = {std::llround(ip.x * 1e12), std::llround(ip.y * 1e12),
                                          std::llround(ip.z * 1e12)};
    auto it = point_index.find(key);
    if (it != point_index.end()) return it->second;
    mfem::DenseMatrix vs(n, 3), cs(n, 3);
    nd->CalcVShape(ip, vs);
    nd->CalcCurlShape(ip, cs);
    fvshape.push_back(Eigen::Map<const rmat>(vs.Data(), n, 3));
    fcshape.push_back(Eigen::Map<const rmat>(cs.Data(), n, 3));
    const int index = static_cast<int>(fvshape.size()) - 1;
    point_index[key] = index;
    return index;
  }

  const Reduced& Reduce(int faces, int edges) {
    const int key = faces | (edges << 4);
    auto it = reduced.find(key);
    if (it != reduced.end()) return it->second;
    Reduced r;
    std::vector<char> essential(n, 0);
    for (int j = 0; j < 4; ++j)
      if (faces & (1 << j))
        for (int i = 0; i < n; ++i) essential[i] |= face_nd[j][i];
    for (int k = 0; k < 6; ++k)
      if (edges & (1 << k))
        for (int i = 0; i < n; ++i) essential[i] |= edge_nd[k][i];
    for (int i = 0; i < n; ++i)
      if (!essential[i]) r.free.push_back(i);
    // the conductor entities (faces, then edges), joined into parts through shared vertices
    using Tet = mfem::Geometry::Constants<mfem::Geometry::TETRAHEDRON>;
    std::vector<std::vector<int>> vertices;
    std::vector<const std::vector<char>*> nodes;
    for (int j = 0; j < 4; ++j)
      if (faces & (1 << j)) {
        vertices.push_back({Tet::FaceVert[j][0], Tet::FaceVert[j][1], Tet::FaceVert[j][2]});
        nodes.push_back(&face_h1[j]);
      }
    for (int k = 0; k < 6; ++k)
      if (edges & (1 << k)) {
        vertices.push_back({Tet::Edges[k][0], Tet::Edges[k][1]});
        nodes.push_back(&edge_h1[k]);
      }
    const int entities = static_cast<int>(vertices.size());
    std::vector<int> parent(entities);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int i) {
      while (parent[i] != i) i = parent[i] = parent[parent[i]];
      return i;
    };
    for (int a = 0; a < entities; ++a)
      for (int b = a + 1; b < entities; ++b)
        for (int va : vertices[a])
          for (int vb : vertices[b])
            if (va == vb) parent[root(a)] = root(b);
    std::map<int, std::vector<char>> part_nodes;
    for (int a = 0; a < entities; ++a) {
      std::vector<char>& on = part_nodes[root(a)];
      on.resize(m, 0);
      for (int l = 0; l < m; ++l) on[l] |= (*nodes[a])[l];
    }
    std::vector<char> on_conductor(m, 0);
    for (const auto& [part, on] : part_nodes)
      for (int l = 0; l < m; ++l) on_conductor[l] |= on[l];
    std::vector<rvec> columns;
    for (int l = 0; l < m; ++l) {
      if (on_conductor[l]) continue;
      if (part_nodes.empty() && l == 0) continue;   // without a conductor: vertex 0 fixes the constant
      rvec c = rvec::Zero(m);
      c(l) = 1.0;
      columns.push_back(c);
    }
    bool first = true;
    for (const auto& [part, on] : part_nodes) {
      if (first) {   // the first conductor part is the reference (psi = 0 there)
        first = false;
        continue;
      }
      rvec c = rvec::Zero(m);
      for (int l = 0; l < m; ++l)
        if (on[l]) c(l) = 1.0;
      columns.push_back(c);
    }
    r.T.resize(m, static_cast<int>(columns.size()));
    for (size_t c = 0; c < columns.size(); ++c) r.T.col(static_cast<int>(c)) = columns[c];
    const int nf = static_cast<int>(r.free.size());
    rmat Gf(nf, m);
    for (int i = 0; i < nf; ++i) Gf.row(i) = G.row(r.free[i]);
    r.Gr = Gf * r.T;
    r.complement = fw_gradient_complement(r.Gr);
    for (int ab = 0; ab < 9; ++ab) {
      rmat Mf(nf, nf);
      for (int i = 0; i < nf; ++i)
        for (int j = 0; j < nf; ++j) Mf(i, j) = mref[ab](r.free[i], r.free[j]);
      r.MG[ab] = Mf * r.Gr;
      r.S[ab] = r.Gr.transpose() * r.MG[ab];
    }
    return reduced.emplace(key, std::move(r)).first->second;
  }
};

FwErrorIndicator::~FwErrorIndicator() = default;

FwErrorIndicator::FwErrorIndicator(const Model3D::Impl& P, const spec::Design& d, const std::vector<char>& in_pml,
                                   const Terminations& terminations)
    : pm_(*P.pmesh), fes_(*P.fes), order_(P.order), ne_(P.pmesh->GetNE()), terminations_(terminations) {
  Timer timer;
  const MPI_Comm comm = pm_.GetComm();
  static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
  auto material_of = [&](const std::string& region) {
    const spec::Region* r = d.find_region(region);
    return r ? r->material : std::string("vacuum");
  };
  auto material = [&](const std::string& name) -> const spec::Material& {
    auto it = d.materials.find(name);
    return it == d.materials.end() ? vacuum : it->second;
  };
  // the region and group lists come from the attribute map, which every rank holds whole
  std::set<std::string> region_set{"air"};
  for (const auto& [attr, name] : P.region_of_attr) region_set.insert(name);
  regions_.assign(region_set.begin(), region_set.end());
  std::map<std::string, int> region_index;
  for (size_t r = 0; r < regions_.size(); ++r) region_index[regions_[r]] = static_cast<int>(r);
  std::set<std::string> group_set;
  for (const std::string& r : regions_) group_set.insert(material_of(r));
  groups_.assign(group_set.begin(), group_set.end());
  std::map<std::string, int> group_index;
  for (size_t g = 0; g < groups_.size(); ++g) group_index[groups_[g]] = static_cast<int>(g);

  // field elements: outside perfect conductors, the infinity shell and PML layers; recovered per material
  group_of_.assign(ne_, -1);
  region_of_.assign(ne_, region_index.at("air"));
  members_.assign(groups_.size(), std::vector<int>());
  long excluded[3] = {0, 0, 0};
  int max_attr = 0;
  for (int e = 0; e < ne_; ++e) max_attr = std::max(max_attr, pm_.GetAttribute(e));
  MPI_Allreduce(MPI_IN_PLACE, &max_attr, 1, MPI_INT, MPI_MAX, comm);
  std::vector<int> pml_attr(max_attr + 1, 0);   // PML layers are whole regions: their attributes, over all ranks
  for (int e = 0; e < ne_; ++e) {
    auto it = region_index.find(P.elem_region[e]);
    if (it != region_index.end()) region_of_[e] = it->second;
    const bool conductor = perfect_conductor(P.elem_mat[e]->kind);
    const bool shell = P.sh_attr > 0 && pm_.GetAttribute(e) == P.sh_attr;
    const bool pml = !in_pml.empty() && in_pml[e];
    if (pml) pml_attr[pm_.GetAttribute(e)] = 1;
    if (conductor || shell || pml) {
      excluded[conductor ? 0 : shell ? 1 : 2] += 1;
      continue;
    }
    group_of_[e] = group_index.at(material_of(P.elem_region[e]));
    members_[group_of_[e]].push_back(e);
  }
  MPI_Allreduce(MPI_IN_PLACE, pml_attr.data(), max_attr + 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce(MPI_IN_PLACE, excluded, 3, MPI_LONG, MPI_SUM, comm);
  excluded_conductor_ = excluded[0];
  excluded_shell_ = excluded[1];
  excluded_pml_ = excluded[2];
  group_size_.assign(groups_.size(), 0);
  for (size_t g = 0; g < groups_.size(); ++g) group_size_[g] = static_cast<long>(members_[g].size());
  MPI_Allreduce(MPI_IN_PLACE, group_size_.data(), static_cast<int>(group_size_.size()), MPI_LONG, MPI_SUM, comm);
  for (long n : group_size_) field_elements_ += n;

  // conductor faces: between a field element and a perfect conductor; a face shared by two ranks is taken by the rank
  // holding its field side
  pm_.ExchangeFaceNbrData();
  std::map<int, bool> conductor_attr;
  for (const auto& [attr, name] : P.region_of_attr)
    conductor_attr[attr] = perfect_conductor(material(material_of(name)).kind);
  auto conductor = [&](int attr) {
    auto it = conductor_attr.find(attr);
    return it != conductor_attr.end() && it->second;
  };
  faces_.assign(groups_.size(), std::vector<Face>());
  for (int f = 0; f < pm_.GetNumFaces(); ++f) {
    const mfem::Mesh::FaceInformation info = pm_.GetFaceInformation(f);
    if (!info.IsInterior() || !info.IsConforming()) continue;
    const int e0 = info.element[0].index;
    const bool field0 = group_of_[e0] >= 0;
    const bool conductor0 = perfect_conductor(P.elem_mat[e0]->kind);
    if (info.IsLocal()) {
      const int e1 = info.element[1].index;
      const bool field1 = group_of_[e1] >= 0;
      const bool conductor1 = perfect_conductor(P.elem_mat[e1]->kind);
      if (field0 && conductor1) faces_[group_of_[e0]].push_back({f, 0, e0});
      else if (field1 && conductor0) faces_[group_of_[e1]].push_back({f, 1, e1});
    } else if (info.IsShared()) {
      const int neighbor_attr = pm_.face_nbr_elements[info.element[1].index]->GetAttribute();
      if (field0 && conductor(neighbor_attr)) faces_[group_of_[e0]].push_back({f, 0, e0});
    }
  }
  long faces = 0;
  for (const auto& list : faces_) faces += static_cast<long>(list.size());
  MPI_Allreduce(&faces, &conductor_faces_, 1, MPI_LONG, MPI_SUM, comm);

  // the recovery space: H1 of the field's order for E and curl E alike
  fec_ = std::make_unique<mfem::H1_FECollection>(order_, 3);
  space_ = std::make_unique<mfem::ParFiniteElementSpace>(&pm_, fec_.get(), kComponents, mfem::Ordering::byNODES);
  sums_.SetSize(space_->GetVSize());
  e_re_ = std::make_unique<mfem::ParGridFunction>(P.fes.get());
  e_im_ = std::make_unique<mfem::ParGridFunction>(P.fes.get());
  c_re_ = std::make_unique<mfem::ParGridFunction>(P.fes.get());
  c_im_ = std::make_unique<mfem::ParGridFunction>(P.fes.get());
  out3_.SetSize(3);

  // the serial elements' recovery groups (every rank holds the serial mesh), for the cuts
  const int nser = P.serial->GetNE();
  std::vector<int> serial_group(nser, -1);
  // the solver's closed elements: perfect conductors, and the shell when the sphere is closed (all their dofs are
  // essential)
  std::vector<char> serial_closed(nser, 0);
  for (int s = 0; s < nser; ++s) {
    const int a = P.ser_attr[s];
    auto it = P.region_of_attr.find(a);
    const std::string region = it == P.region_of_attr.end() ? std::string("air") : it->second;
    const bool shell = P.sh_attr > 0 && a == P.sh_attr;
    const bool pml = a >= 0 && a <= max_attr && pml_attr[a];
    const bool conductor = perfect_conductor(material(material_of(region)).kind);
    serial_closed[s] = conductor || (shell && !terminations_.radiating_shell);
    if (conductor || shell || pml) continue;
    serial_group[s] = group_index.at(material_of(region));
  }
  FindCuts(P, serial_group);
  if (order_ >= 2) {   // the magnetic part's element problems and the electric part's gradient projection
    nd_fec_ = P.fec.get();
    SetupResidual(P, serial_group, serial_closed);
    SetupGradient();
  }
  setup_seconds_ = timer.seconds();
  MPI_Allreduce(MPI_IN_PLACE, &setup_seconds_, 1, MPI_DOUBLE, MPI_MAX, comm);
}

void FwErrorIndicator::FindCuts(const Model3D::Impl& P, const std::vector<int>& serial_group) {
  const mfem::Mesh& S = *P.serial;
  // the boundary groups of the mesh info (imported meshes: ports, sheets, PEC, PMC, impedance, scattering)
  std::map<int, std::string> group_name;
  for (const auto& g : P.mesh_info.value("boundary_groups", json::array())) {
    const int attr = g.at("attr").get<int>();
    group_name[attr] = g.value("name", strcat("attr_", attr)) + " (" + g.value("type", std::string("pec")) + ")";
  }
  // the cut faces: a boundary group's faces with field elements of one group on both sides
  std::set<std::array<int, 3>> cut_faces;
  std::set<std::string> cut_names;
  mfem::Array<int> fv;
  for (int be = 0; be < S.GetNBE(); ++be) {
    auto it = group_name.find(S.GetBdrAttribute(be));
    if (it == group_name.end()) continue;
    const int face = S.GetBdrElementFaceIndex(be);
    int s1 = -1;
    int s2 = -1;
    S.GetFaceElements(face, &s1, &s2);
    if (s1 < 0 || s2 < 0 || serial_group[s1] < 0 || serial_group[s1] != serial_group[s2]) continue;
    S.GetFaceVertices(face, fv);
    std::array<int, 3> t = {fv[0], fv[1], fv[2]};
    std::sort(t.begin(), t.end());
    cut_faces.insert(t);
    cut_names.insert(it->second);
  }
  cut_faces_ = static_cast<long>(cut_faces.size());
  cut_groups_.assign(cut_names.begin(), cut_names.end());
  cut_slot_.assign(ne_, std::vector<int>());
  if (cut_faces.empty()) return;
  std::set<std::array<int, 2>> cut_edges;
  std::set<int> cut_vertices;
  for (const auto& t : cut_faces) {
    cut_edges.insert({t[0], t[1]});
    cut_edges.insert({t[0], t[2]});
    cut_edges.insert({t[1], t[2]});
    for (int v : t) cut_vertices.insert(v);
  }
  auto on_cut = [&](const NodeKey& key) {
    if (key.size() == 1) return cut_vertices.count(key[0].first) > 0;
    if (key.size() == 2) return cut_edges.count({key[0].first, key[1].first}) > 0;
    if (key.size() == 3) return cut_faces.count({key[0].first, key[1].first, key[2].first}) > 0;
    return false;
  };
  // the field elements around the cuts, and every one of their H1 nodes that lies on a cut (the reference element's
  // nodes, in the local dof order of every element)
  const mfem::IntegrationRule& nodes = fec_->FiniteElementForGeometry(mfem::Geometry::TETRAHEDRON)->GetNodes();
  std::set<int> around;
  mfem::Array<int> vv;
  for (int s = 0; s < S.GetNE(); ++s) {
    if (serial_group[s] < 0) continue;
    S.GetElementVertices(s, vv);
    for (int v : vv)
      if (cut_vertices.count(v)) around.insert(s);
  }
  std::map<NodeKey, std::vector<int>> patch;   // node on a cut -> the field elements that contain it
  std::map<int, std::array<int, 4>> sorted_vertices;
  for (int s : around) {
    S.GetElementVertices(s, vv);
    std::array<int, 4> sv = {vv[0], vv[1], vv[2], vv[3]};
    std::sort(sv.begin(), sv.end());
    sorted_vertices[s] = sv;
    for (int k = 0; k < nodes.GetNPoints(); ++k) {
      NodeKey key = node_key(vv, nodes.IntPoint(k));
      if (on_cut(key)) patch[key].push_back(s);
    }
  }
  // per node, the sides: elements of one group joined through a shared face that is not a cut face (around a cut's
  // free edge the two sides join, and the node keeps one average)
  std::map<std::pair<const NodeKey*, int>, int> slot_of;   // (node, serial element) -> slot
  for (const auto& [key, elements] : patch) {
    const int m = static_cast<int>(elements.size());
    std::vector<int> parent(m);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](int i) {
      while (parent[i] != i) i = parent[i] = parent[parent[i]];
      return i;
    };
    for (int i = 0; i < m; ++i)
      for (int j = i + 1; j < m; ++j) {
        if (serial_group[elements[i]] != serial_group[elements[j]]) continue;
        const auto& a = sorted_vertices[elements[i]];
        const auto& b = sorted_vertices[elements[j]];
        std::vector<int> common;
        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(common));
        if (common.size() != 3 || cut_faces.count({common[0], common[1], common[2]})) continue;
        parent[root(i)] = root(j);
      }
    std::map<int, int> slot_of_root;
    for (int i = 0; i < m; ++i) {
      const int r = root(i);
      if (!slot_of_root.count(r)) slot_of_root[r] = slots_++;
      slot_of[{&key, elements[i]}] = slot_of_root[r];
    }
  }
  slot_sums_.assign(static_cast<size_t>(kComponents) * slots_, 0.0);
  // the local elements' slots, through the serial element each one is
  for (int e = 0; e < ne_; ++e) {
    const int s = P.loc2ser[e];
    if (group_of_[e] < 0 || !around.count(s)) continue;
    S.GetElementVertices(s, vv);
    std::vector<int> slot(nodes.GetNPoints(), -1);
    bool any = false;
    for (int k = 0; k < nodes.GetNPoints(); ++k) {
      auto it = patch.find(node_key(vv, nodes.IntPoint(k)));
      if (it == patch.end()) continue;
      slot[k] = slot_of.at({&it->first, s});
      any = true;
    }
    if (any) cut_slot_[e] = slot;
  }
}

void FwErrorIndicator::SetupResidual(const Model3D::Impl& P, const std::vector<int>& serial_group,
                                     const std::vector<char>& closed) {
  res_ = std::make_unique<Residual>();
  Residual& R = *res_;
  using Tet = mfem::Geometry::Constants<mfem::Geometry::TETRAHEDRON>;
  R.q = order_ + 1;
  R.nd_fec = std::make_unique<mfem::ND_FECollection>(R.q, 3);
  R.h1_fec = std::make_unique<mfem::H1_FECollection>(R.q, 3);
  R.nd = R.nd_fec->FiniteElementForGeometry(mfem::Geometry::TETRAHEDRON);
  R.h1 = R.h1_fec->FiniteElementForGeometry(mfem::Geometry::TETRAHEDRON);
  R.n = R.nd->GetDof();
  R.m = R.h1->GetDof();
  // reference-element operators: ND_p in ND_q, and the gradients of H1_q
  mfem::IsoparametricTransformation reference;
  reference.SetIdentityTransformation(mfem::Geometry::TETRAHEDRON);
  mfem::DenseMatrix Pi, G;
  R.nd->Project(*P.fec->FiniteElementForGeometry(mfem::Geometry::TETRAHEDRON), reference, Pi);
  R.nd->ProjectGrad(*R.h1, reference, G);
  R.Pi = Eigen::Map<const rmat>(Pi.Data(), Pi.Height(), Pi.Width());
  R.G = Eigen::Map<const rmat>(G.Data(), G.Height(), G.Width());
  R.vol = &mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * R.q);
  R.face = &mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * R.q);
  mfem::DenseMatrix vs(R.n, 3), cs(R.n, 3);
  mfem::Vector hs(R.m);
  for (int k = 0; k < R.vol->GetNPoints(); ++k) {
    R.nd->CalcVShape(R.vol->IntPoint(k), vs);
    R.nd->CalcCurlShape(R.vol->IntPoint(k), cs);
    R.vshape.push_back(Eigen::Map<const rmat>(vs.Data(), R.n, 3));
    R.cshape.push_back(Eigen::Map<const rmat>(cs.Data(), R.n, 3));
  }
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      R.mref[3 * a + b] = rmat::Zero(R.n, R.n);
      R.cref[3 * a + b] = rmat::Zero(R.n, R.n);
      for (int k = 0; k < R.vol->GetNPoints(); ++k) {
        const double wk = R.vol->IntPoint(k).weight;
        R.mref[3 * a + b] += wk * R.vshape[k].col(a) * R.vshape[k].col(b).transpose();
        R.cref[3 * a + b] += wk * R.cshape[k].col(a) * R.cshape[k].col(b).transpose();
      }
    }
  // which dofs live on a reference face or edge: a nonzero tangential trace (ND) or value (H1) at points there
  const double corner[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  auto at = [&](const double* a, const double* b, const double* c, double s, double t) {
    mfem::IntegrationPoint ip;
    const double r = 1 - s - t;
    ip.Set3(r * a[0] + s * b[0] + t * c[0], r * a[1] + s * b[1] + t * c[1], r * a[2] + s * b[2] + t * c[2]);
    return ip;
  };
  const double samples[4][2] = {{0.2, 0.3}, {0.55, 0.15}, {0.1, 0.7}, {0.33, 0.33}};
  for (int j = 0; j < 4; ++j) {
    const double* a = corner[Tet::FaceVert[j][0]];
    const double* b = corner[Tet::FaceVert[j][1]];
    const double* c = corner[Tet::FaceVert[j][2]];
    const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const double v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const double normal[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    R.face_nd[j].assign(R.n, 0);
    R.face_h1[j].assign(R.m, 0);
    for (const auto& st : samples) {
      const mfem::IntegrationPoint ip = at(a, b, c, st[0], st[1]);
      R.nd->CalcVShape(ip, vs);
      R.h1->CalcShape(ip, hs);
      for (int i = 0; i < R.n; ++i) {
        const double x[3] = {vs(i, 0), vs(i, 1), vs(i, 2)};
        const double t2 = std::pow(x[1] * normal[2] - x[2] * normal[1], 2) +
                          std::pow(x[2] * normal[0] - x[0] * normal[2], 2) +
                          std::pow(x[0] * normal[1] - x[1] * normal[0], 2);
        if (t2 > 1e-20) R.face_nd[j][i] = 1;
      }
      for (int l = 0; l < R.m; ++l)
        if (std::fabs(hs(l)) > 1e-10) R.face_h1[j][l] = 1;
    }
  }
  for (int k = 0; k < 6; ++k) {
    const double* a = corner[Tet::Edges[k][0]];
    const double* b = corner[Tet::Edges[k][1]];
    const double t[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    R.edge_nd[k].assign(R.n, 0);
    R.edge_h1[k].assign(R.m, 0);
    for (double s : {0.15, 0.4, 0.65, 0.9}) {
      const mfem::IntegrationPoint ip = at(a, b, b, s, 0.0);
      R.nd->CalcVShape(ip, vs);
      R.h1->CalcShape(ip, hs);
      for (int i = 0; i < R.n; ++i)
        if (std::fabs(vs(i, 0) * t[0] + vs(i, 1) * t[1] + vs(i, 2) * t[2]) > 1e-10) R.edge_nd[k][i] = 1;
      for (int l = 0; l < R.m; ++l)
        if (std::fabs(hs(l)) > 1e-10) R.edge_h1[k][l] = 1;
    }
  }

  // the faces' kinds, from the serial mesh (the same on every rank): perfect conductors are the solver's essential
  // set (the closed elements, the outer attribute and pec groups)
  const mfem::Mesh& S = *P.serial;
  const int nser = S.GetNE();
  std::map<int, std::string> group_type;
  for (const auto& g : P.mesh_info.value("boundary_groups", json::array()))
    group_type[g.at("attr").get<int>()] = g.value("type", std::string("pec"));
  std::vector<signed char> boundary_kind(S.GetNFaces(), -1);
  for (int be = 0; be < S.GetNBE(); ++be) {
    const int attr = S.GetBdrAttribute(be);
    const int f = S.GetBdrElementFaceIndex(be);
    auto it = group_type.find(attr);
    if (attr == axi::kOuterAttr || (it != group_type.end() && it->second == "pec")) boundary_kind[f] = kPec;
    else if (it != group_type.end()) boundary_kind[f] = it->second == "pmc" ? kNatural : kOwn;
  }
  std::vector<char> pec_face(S.GetNFaces(), 0);
  std::vector<char> pec_edge(S.GetNEdges(), 0);
  mfem::Array<int> faces, edges, orientation;
  for (int s = 0; s < nser; ++s) {
    if (!closed[s]) continue;
    S.GetElementFaces(s, faces, orientation);
    for (int f : faces) pec_face[f] = 1;
    S.GetElementEdges(s, edges, orientation);
    for (int g : edges) pec_edge[g] = 1;
  }
  for (int f = 0; f < S.GetNFaces(); ++f) {
    if (boundary_kind[f] != kPec) continue;
    pec_face[f] = 1;
    S.GetFaceEdges(f, edges, orientation);
    for (int g : edges) pec_edge[g] = 1;
  }
  R.kind.assign(ne_, {kOwn, kOwn, kOwn, kOwn});
  R.pec_faces.assign(ne_, 0);
  R.pec_edges.assign(ne_, 0);
  for (int e = 0; e < ne_; ++e) {
    if (group_of_[e] < 0) continue;
    const int s = P.loc2ser[e];
    S.GetElementFaces(s, faces, orientation);
    for (int j = 0; j < 4; ++j) {
      const int f = faces[j];
      if (pec_face[f]) {
        R.kind[e][j] = kPec;
        R.pec_faces[e] |= static_cast<unsigned char>(1 << j);
        continue;
      }
      int s1 = -1;
      int s2 = -1;
      S.GetFaceElements(f, &s1, &s2);
      const int other = s1 == s ? s2 : s1;
      // an interior face of a pmc group is an ordinary interior face: the solver imposes nothing there (a natural
      // condition between two elements is no condition), so the field and its traction pass through it
      const bool interior_pmc = other >= 0 && boundary_kind[f] == kNatural;
      if (boundary_kind[f] >= 0 && !interior_pmc)
        R.kind[e][j] = other < 0 ? boundary_kind[f] : static_cast<signed char>(kOwn);
      else if (other < 0) R.kind[e][j] = kNatural;
      else R.kind[e][j] = serial_group[other] >= 0 ? kAverage : kOwn;
    }
    S.GetElementEdges(s, edges, orientation);
    for (int k = 0; k < 6; ++k)
      if (pec_edge[edges[k]]) R.pec_edges[e] |= static_cast<unsigned char>(1 << k);
  }
  // the affine test compares the Jacobian at the four vertices, which proves affinity for geometry of order <= 2 only
  const mfem::FiniteElementSpace* nodes = pm_.GetNodalFESpace();
  int geometry_order = nodes ? nodes->GetMaxElementOrder() : 1;
  MPI_Allreduce(MPI_IN_PLACE, &geometry_order, 1, MPI_INT, MPI_MAX, pm_.GetComm());
  R.affine_test = geometry_order <= 2;
  R.l2_fec = std::make_unique<mfem::L2_FECollection>(0, 3);
  R.l2_space = std::make_unique<mfem::ParFiniteElementSpace>(&pm_, R.l2_fec.get(), 4);
  R.coef = std::make_unique<mfem::ParGridFunction>(R.l2_space.get());
}

namespace {

// a real matrix times a complex vector
cvec times(const rmat& A, const cvec& x) {
  cvec y(A.rows());
  y.real() = A * x.real();
  y.imag() = A * x.imag();
  return y;
}

// the solution of a real SPD system for a complex right-hand side
cvec solve(const Eigen::LLT<rmat>& llt, const cvec& b) {
  cvec x(b.size());
  x.real() = llt.solve(b.real());
  x.imag() = llt.solve(b.imag());
  return x;
}

// PRESB (Axelsson): for [[A, -B], [B, A]] with A and B symmetric positive semidefinite, P = [[A, -B], [B, A + 2B]] is
// inverted by two solves with A + B (here one AMG cycle each): s = (A + B)^-1 (f + g), x = (A + B)^-1 (f + B s),
// y = s - x. With exact solves the preconditioned eigenvalues lie in [1/2, 1], independently of h and of the phases.
class Presb : public mfem::Solver {
 public:
  Presb(const mfem::HypreParMatrix& B, mfem::Solver& sum, int n) : mfem::Solver(2 * n), B_(B), sum_(sum), n_(n),
                                                                    t_(n), s_(n) {}
  void SetOperator(const mfem::Operator&) override {}
  void Mult(const mfem::Vector& in, mfem::Vector& out) const override {
    const mfem::Vector f(const_cast<mfem::real_t*>(in.GetData()), n_);
    const mfem::Vector g(const_cast<mfem::real_t*>(in.GetData()) + n_, n_);
    mfem::Vector x(out.GetData(), n_);
    mfem::Vector y(out.GetData() + n_, n_);
    add(f, g, t_);
    sum_.Mult(t_, s_);
    B_.Mult(s_, t_);
    t_ += f;
    sum_.Mult(t_, x);
    subtract(s_, x, y);
  }

 private:
  const mfem::HypreParMatrix& B_;
  mfem::Solver& sum_;
  int n_;
  mutable mfem::Vector t_, s_;
};

// the system and its preconditioner scaled symmetrically by d = diag(A + B)^-1/2 on both blocks: d M d and
// d^-1 P^-1 d^-1. Left-preconditioned GMRES then stops on d^-1 P^-1 r, the error's estimate weighted with the diagonal
// of the |eps|-scaled operator, so its Euclidean norm follows the projection's energy. Unscaled it weighs every
// coefficient alike, and at a contrast of 1e10 (a conductor meshed as a field region) the conductor's coefficients,
// small but heavily weighted in the energy, go unresolved
class Scaled : public mfem::Solver {
 public:
  Scaled(const mfem::Operator& op, const mfem::Vector& d, bool inverse) : mfem::Solver(op.Height()), op_(op), d_(d),
                                                                           inverse_(inverse), t_(op.Height()) {}
  void SetOperator(const mfem::Operator&) override {}
  void Mult(const mfem::Vector& in, mfem::Vector& out) const override {
    for (int i = 0; i < t_.Size(); ++i) t_(i) = inverse_ ? in(i) / d_(i) : in(i) * d_(i);
    op_.Mult(t_, out);
    for (int i = 0; i < out.Size(); ++i) out(i) = inverse_ ? out(i) / d_(i) : out(i) * d_(i);
  }

 private:
  const mfem::Operator& op_;
  const mfem::Vector& d_;
  bool inverse_;
  mutable mfem::Vector t_;
};

// a coefficient per local element (ElementTransformation::ElementNo)
class PerElement : public mfem::Coefficient {
 public:
  explicit PerElement(const std::vector<double>& values) : values_(values) {}
  mfem::real_t Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint&) override {
    return T.ElementNo < static_cast<int>(values_.size()) ? values_[T.ElementNo] : 0.0;
  }

 private:
  const std::vector<double>& values_;
};

}  // namespace

double fw_indicator_gradient_tolerance() {
  const char* v = std::getenv("AXF_FW_INDICATOR_TOL");
  if (!v || !*v) return kGradientTolerance;
  char* end = nullptr;
  const double tolerance = std::strtod(v, &end);
  // only tighter than the default: a looser projection would end as "converged" and silently lower the electric part
  // (a CG iterate underestimates the projection's energy)
  AXF_REQUIRE(end != v && *end == '\0' && std::isfinite(tolerance) && tolerance > 0.0 &&
                  tolerance <= kGradientTolerance,
              strcat("AXF_FW_INDICATOR_TOL must be a number in (0, 1e-4], got '", v, "'"), "$AXF_FW_INDICATOR_TOL",
              "it can only tighten the order-2 error indicator's gradient projection; unset it for the default 1e-4");
  return tolerance;
}

void FwErrorIndicator::SetupGradient() {
  const Residual& R = *res_;
  gfec_ = std::make_unique<mfem::H1_FECollection>(order_ + 1, 3);
  gspace_ = std::make_unique<mfem::ParFiniteElementSpace>(&pm_, gfec_.get());
  // psi = 0 off the field elements (conductors, the shell, PML layers), on conductor faces and edges, and on the faces
  // whose traction is the element's own (ports, sheets, Robin groups, the radiating shell's and PML layers'
  // interfaces), where the exact field's normal flux is not zero or not known; natural on PMC and unmarked outer faces
  mfem::Array<int> marker(gspace_->GetVSize());
  marker = 0;
  mfem::Array<int> dofs, faces, edges, orientation;
  for (int e = 0; e < ne_; ++e) {
    if (group_of_[e] < 0) {
      gspace_->GetElementDofs(e, dofs);
      for (int d : dofs) marker[d] = 1;
      continue;
    }
    pm_.GetElementFaces(e, faces, orientation);
    for (int j = 0; j < 4; ++j) {
      if (R.kind[e][j] != kPec && R.kind[e][j] != kOwn) continue;
      gspace_->GetFaceDofs(faces[j], dofs);
      for (int d : dofs) marker[d] = 1;
    }
    if (R.pec_edges[e] == 0) continue;
    pm_.GetElementEdges(e, edges, orientation);
    for (int k = 0; k < 6; ++k) {
      if (!(R.pec_edges[e] & (1 << k))) continue;
      gspace_->GetEdgeDofs(edges[k], dofs);
      for (int d : dofs) marker[d] = 1;
    }
  }
  gspace_->Synchronize(marker);
  for (int d = 0; d < marker.Size(); ++d) {
    if (!marker[d]) continue;
    const int t = gspace_->GetLocalTDofNumber(d);
    if (t >= 0) gess_.Append(t);
  }
  gess_.Sort();
  gess_.Unique();
  // reference shapes on the gradient's rule: E (ND_p) and the H1_{p+1} gradients
  const mfem::FiniteElement* nd = nd_fec_->FiniteElementForGeometry(mfem::Geometry::TETRAHEDRON);
  const mfem::FiniteElement* h1 = gfec_->FiniteElementForGeometry(mfem::Geometry::TETRAHEDRON);
  grule_ = &mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * order_ + 1);
  mfem::DenseMatrix vs(nd->GetDof(), 3), ds(h1->GetDof(), 3);
  for (int k = 0; k < grule_->GetNPoints(); ++k) {
    nd->CalcVShape(grule_->IntPoint(k), vs);
    h1->CalcDShape(grule_->IntPoint(k), ds);
    gvshape_.push_back(Eigen::Map<const rmat>(vs.Data(), vs.Height(), 3));
    gdshape_.push_back(Eigen::Map<const rmat>(ds.Data(), ds.Height(), 3));
  }
}

json FwErrorIndicator::EvaluateGradient(double w, const std::vector<cplx>& eps, std::vector<double>& energy,
                                        std::vector<cplx>& form) {
  Timer timer;
  const MPI_Comm comm = pm_.GetComm();
  // the weights: |eps| (the operator and its preconditioner) and, where the field materials' loss phases differ,
  // eps itself (the projection is then complex-symmetric)
  std::vector<double> weight(ne_, 0.0);
  std::vector<double> weight_re(ne_, 0.0);
  std::vector<double> weight_im(ne_, 0.0);
  double phase[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
  for (int e = 0; e < ne_; ++e) {
    if (group_of_[e] < 0) continue;
    weight[e] = std::abs(eps[e]);
    weight_re[e] = eps[e].real();
    weight_im[e] = eps[e].imag();
    phase[0] = std::min(phase[0], std::arg(eps[e]));
    phase[1] = std::min(phase[1], -std::arg(eps[e]));
  }
  MPI_Allreduce(MPI_IN_PLACE, phase, 2, MPI_DOUBLE, MPI_MIN, comm);
  // one loss phase everywhere (lossless models in particular): (eps grad psi, grad xi) = e^{i phi} (|eps| grad psi,
  // grad xi), so the real |eps| operator gives the same psi
  const bool complex_eps = std::isfinite(phase[0]) && -phase[1] - phase[0] > 1e-9;
  int changed = weight != geps_ ? 1 : 0;
  if (complex_eps && (weight_re != geps_re_ || weight_im != geps_im_)) changed = 1;
  MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, comm);
  auto assemble = [&](const std::vector<double>& values) {
    PerElement coef(values);
    mfem::ParBilinearForm a(gspace_.get());
    a.AddDomainIntegrator(new mfem::DiffusionIntegrator(coef));
    a.Assemble();
    a.Finalize();
    std::unique_ptr<mfem::HypreParMatrix> A(a.ParallelAssemble());
    // the weight vanishes off the field elements, whose rows hold only zeros: hypre on a GPU drops them, and the
    // elimination could not set their unit diagonal (docs/validation/gpu.md). A no-op on the host, where every
    // row keeps its diagonal (as the heat solve, model3d_mech.cpp)
    ensure_diagonal(A, gess_);
    std::unique_ptr<mfem::HypreParMatrix> eliminated(A->EliminateRowsCols(gess_));
    return A;
  };
  if (changed) {   // eps changes with the frequency only in conducting media
    gamg_.reset();   // before the matrices it was set up with
    gA_.reset();
    gAr_.reset();
    gAi_.reset();
    geps_ = weight;
    geps_re_.clear();
    geps_im_.clear();
    if (!complex_eps) {   // the |eps| operator and its AMG
      gA_ = assemble(geps_);
      gamg_ = std::make_unique<mfem::HypreBoomerAMG>(*gA_);
    } else {
      // the real and imaginary parts, and the AMG of their sum for PRESB. At the essential dofs every block keeps a
      // unit diagonal (on a GPU too: ensure_diagonal in assemble, also where A_im is zero), so the 2 x 2 block there,
      // [[1, -1], [1, 1]], is nonsingular and psi = 0
      geps_re_ = weight_re;
      geps_im_ = weight_im;
      std::vector<double> weight_sum(ne_, 0.0);
      for (int e = 0; e < ne_; ++e) weight_sum[e] = weight_re[e] + weight_im[e];
      gAr_ = assemble(geps_re_);
      gAi_ = assemble(geps_im_);
      gA_ = assemble(weight_sum);
      gamg_ = std::make_unique<mfem::HypreBoomerAMG>(*gA_);
    }
    gamg_->SetPrintLevel(0);
  }
  // (eps grad psi, grad xi) = -(eps E_h, grad xi) for every xi of H1_{p+1}: the error's projection onto the gradients
  // (the exact field's term, int_dK xi n . eps E, cancels between elements, where n . eps E is continuous, and
  // vanishes on the boundary). With one loss phase |eps| stands for eps.
  const int nh = static_cast<int>(gdshape_[0].rows());
  mfem::Vector b_re(gspace_->GetVSize()), b_im(gspace_->GetVSize());
  b_re = 0.0;
  b_im = 0.0;
  mfem::Array<int> dofs;
  rvec lr(nh), li(nh);
  mfem::IsoparametricTransformation T;
  for (int e = 0; e < ne_; ++e) {
    if (group_of_[e] < 0) continue;
    Load(e, *e_re_, *e_im_);
    const Eigen::Map<const rvec> er(le_re_.GetData(), le_re_.Size());
    const Eigen::Map<const rvec> ei(le_im_.GetData(), le_im_.Size());
    const cplx factor = complex_eps ? eps[e] : cplx(weight[e], 0.0);
    pm_.GetElementTransformation(e, &T);
    gspace_->GetElementDofs(e, dofs);
    lr.setZero();
    li.setZero();
    for (int k = 0; k < grule_->GetNPoints(); ++k) {
      const mfem::IntegrationPoint& ip = grule_->IntPoint(k);
      T.SetIntPoint(&ip);
      const double dv = ip.weight * T.Weight();
      const Eigen::Map<const Eigen::Matrix3d> invJ(T.InverseJacobian().Data());
      // E = J^-T (v_ref^T e) and grad xi = d_ref J^-1, so grad xi . E = d_ref (J^-1 J^-T v_ref^T e)
      const Eigen::Matrix3d metric = invJ * invJ.transpose();
      const Eigen::Vector3d tr = metric * (gvshape_[k].transpose() * er);
      const Eigen::Vector3d ti = metric * (gvshape_[k].transpose() * ei);
      const Eigen::Vector3d fr = factor.real() * tr - factor.imag() * ti;
      const Eigen::Vector3d fi = factor.real() * ti + factor.imag() * tr;
      lr.noalias() -= dv * (gdshape_[k] * fr);
      li.noalias() -= dv * (gdshape_[k] * fi);
    }
    for (int l = 0; l < nh; ++l) {
      b_re(dofs[l]) += lr(l);
      b_im(dofs[l]) += li(l);
    }
  }
  mfem::ParGridFunction psi_re(gspace_.get()), psi_im(gspace_.get());
  const int nt = gspace_->GetTrueVSize();
  int iterations = 0;
  double residual = 0.0;
  int converged = 1;
  // AXF_FW_INDICATOR_MAX_ITER caps the iterations (tests drive the withholding of a projection that did not converge)
  int max_iterations = kGradientIterations;
  if (const char* v = std::getenv("AXF_FW_INDICATOR_MAX_ITER"); v && *v) max_iterations = std::max(1, std::atoi(v));
  const double tolerance = fw_indicator_gradient_tolerance();
  if (!complex_eps) {   // two real SPD solves; hypre's PCG stops on <C r, r>, an energy-like norm
    for (int part = 0; part < 2; ++part) {
      mfem::Vector B(nt), X(nt);
      gspace_->GetProlongationMatrix()->MultTranspose(part == 0 ? b_re : b_im, B);
      for (int i : gess_) B(i) = 0.0;
      X = 0.0;
      mfem::HyprePCG pcg(*gA_);
      pcg.SetTol(tolerance);
      pcg.SetMaxIter(max_iterations);
      pcg.SetPrintLevel(0);
      pcg.SetPreconditioner(*gamg_);
      pcg.Mult(B, X);
      int its = 0;
      pcg.GetNumIterations(its);
      double r = 0.0;
      pcg.GetFinalResidualNorm(r);
      // a zero right-hand side (no field in a part) needs no iteration; the norm is global, so that every rank decides
      // alike (a rank's own part can be zero where all its dofs are essential)
      const bool loaded = mfem::InnerProduct(comm, B, B) > 0.0;
      if (loaded && !(r <= tolerance)) converged = 0;
      iterations = std::max(iterations, its);
      residual = std::max(residual, loaded ? r : 0.0);
      (part == 0 ? psi_re : psi_im).SetFromTrueDofs(X);
    }
  } else {   // the complex-symmetric system [[A_re, -A_im], [A_im, A_re]] by GMRES, preconditioned by PRESB
    mfem::ComplexHypreParMatrix A(gAr_.get(), gAi_.get(), false, false);
    Presb preconditioner(*gAi_, *gamg_, nt);
    mfem::Vector d(2 * nt);
    // the diagonal of A_re + A_im, found by column (not GetDiag, which reads each row's first entry and, with hypre
    // on a GPU, a stale host copy; model3d_impl.hpp, docs/validation/gpu.md)
    const mfem::Vector diagonal = stored_diagonal(*gA_);
    for (int i = 0; i < nt; ++i) {
      d(i) = 1.0 / std::sqrt(diagonal(i));
      d(nt + i) = d(i);
    }
    Scaled scaled_operator(A, d, false);
    Scaled scaled_preconditioner(preconditioner, d, true);
    mfem::Vector B(2 * nt), Y(2 * nt);
    mfem::Vector B_re(B, 0, nt), B_im(B, nt, nt);
    gspace_->GetProlongationMatrix()->MultTranspose(b_re, B_re);
    gspace_->GetProlongationMatrix()->MultTranspose(b_im, B_im);
    for (int i : gess_) {
      B_re(i) = 0.0;
      B_im(i) = 0.0;
    }
    for (int i = 0; i < 2 * nt; ++i) B(i) *= d(i);
    const bool loaded = mfem::InnerProduct(comm, B, B) > 0.0;   // global, as in the real path
    Y = 0.0;
    mfem::GMRESSolver gmres(comm);
    gmres.SetOperator(scaled_operator);
    gmres.SetPreconditioner(scaled_preconditioner);
    gmres.SetRelTol(tolerance);
    gmres.SetAbsTol(0.0);
    gmres.SetMaxIter(max_iterations);
    gmres.SetKDim(100);
    gmres.SetPrintLevel(0);
    gmres.Mult(B, Y);
    iterations = gmres.GetNumIterations();
    residual = gmres.GetFinalRelNorm();
    if (loaded && !gmres.GetConverged()) converged = 0;
    for (int i = 0; i < 2 * nt; ++i) Y(i) *= d(i);
    mfem::Vector X_re(Y, 0, nt), X_im(Y, nt, nt);
    psi_re.SetFromTrueDofs(X_re);
    psi_im.SetFromTrueDofs(X_im);
  }
  // the solvers' norms are global already; reduced again so that every rank takes the same branch in Evaluate
  MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &residual, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Allreduce(MPI_IN_PLACE, &iterations, 1, MPI_INT, MPI_MAX, comm);
  // per element omega^2 |eps| |grad psi|^2 and the form -omega^2 eps grad psi . grad psi (not conjugated)
  energy.assign(ne_, 0.0);
  form.assign(ne_, cplx(0.0, 0.0));
  mfem::Vector pr(nh), pi(nh);
  for (int e = 0; e < ne_; ++e) {
    if (group_of_[e] < 0) continue;
    pm_.GetElementTransformation(e, &T);
    gspace_->GetElementDofs(e, dofs);
    psi_re.GetSubVector(dofs, pr);
    psi_im.GetSubVector(dofs, pi);
    const Eigen::Map<const rvec> qr(pr.GetData(), nh);
    const Eigen::Map<const rvec> qi(pi.GetData(), nh);
    for (int k = 0; k < grule_->GetNPoints(); ++k) {
      const mfem::IntegrationPoint& ip = grule_->IntPoint(k);
      T.SetIntPoint(&ip);
      const double dv = ip.weight * T.Weight();
      const Eigen::Map<const Eigen::Matrix3d> invJ(T.InverseJacobian().Data());
      const Eigen::Vector3d gr = invJ.transpose() * (gdshape_[k].transpose() * qr);
      const Eigen::Vector3d gi = invJ.transpose() * (gdshape_[k].transpose() * qi);
      energy[e] += w * w * std::abs(eps[e]) * (gr.squaredNorm() + gi.squaredNorm()) * dv;
      form[e] += -w * w * eps[e] * cplx(gr.squaredNorm() - gi.squaredNorm(), 2 * gr.dot(gi)) * dv;
    }
  }
  double seconds = timer.seconds();
  MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
  json out = {{"seconds", seconds},
              {"iterations", iterations},
              {"relative_residual", residual},
              {"tolerance", tolerance},
              {"converged", converged == 1},
              {"solver", complex_eps ? "GMRES with PRESB, complex eps (field materials with differing loss phases); "
                                       "relative residual in the diagonally scaled energy"
                                     : "PCG, |eps|; relative residual in hypre's preconditioned norm"}};
  // the spread of arg eps over the field materials: with phase contrast the projection is oblique, and its energy
  // can exceed the error's electric energy by up to 1 / cos^2(spread / 2)
  if (complex_eps) {
    const double spread = -phase[1] - phase[0];
    out["loss_phase_spread_deg"] = spread * 180.0 / units::pi;
    out["electric_bound_factor"] = 1.0 / std::pow(std::cos(spread / 2.0), 2);
  }
  return out;
}

json FwErrorIndicator::EvaluateResidual(double w, const std::vector<cplx>& eps, const std::vector<cplx>& nu,
                                        const mfem::ParGridFunction& c_re, const mfem::ParGridFunction& c_im,
                                        std::vector<double>& magnetic, std::vector<double>& electric,
                                        std::vector<cplx>& form, std::vector<double>& norm_m,
                                        std::vector<double>& norm_e) {
  Timer timer;
  Residual& R = *res_;
  int kh_outside = 0;   // elements past |k| h_max = 1, and sliver-like elements past 0.2 (fw_local_kh_warning)
  int kh_sliver = 0;
  double vertices[4][3];
  mfem::Array<int> element_vertices;
  const bool representative = &c_re != e_re_.get();
  // the face neighbours' fields and material coefficients (collective)
  e_re_->ExchangeFaceNbrData();
  e_im_->ExchangeFaceNbrData();
  if (representative) {
    c_re_->ExchangeFaceNbrData();
    c_im_->ExchangeFaceNbrData();
  }
  mfem::Array<int> vd;
  for (int e = 0; e < ne_; ++e) {
    R.l2_space->GetElementVDofs(e, vd);
    (*R.coef)(vd[0]) = eps[e].real();
    (*R.coef)(vd[1]) = eps[e].imag();
    (*R.coef)(vd[2]) = nu[e].real();
    (*R.coef)(vd[3]) = nu[e].imag();
  }
  R.coef->ExchangeFaceNbrData();
  magnetic.assign(ne_, 0.0);
  electric.assign(ne_, 0.0);
  form.assign(ne_, cplx(0.0, 0.0));
  norm_m.assign(ne_, 0.0);
  norm_e.assign(ne_, 0.0);
  const int n = R.n;
  const double w2 = w * w;
  rmat MM(n, n), CC(n, n), vp(n, 3), cp(n, 3);
  cvec EP(n), CP(n), EN(n), CN(n), b(n), fcurl(n);
  rvec CPr(n), CPi(n), CNr(n), CNi(n), fr(n), fi(n);
  mfem::IsoparametricTransformation T, T1, T2;
  mfem::FaceElementTransformations FT;
  mfem::Array<int> faces, orientation;
  mfem::Vector normal(3);
  // E and curl E of the loaded element in the ND_q basis
  auto lift = [&](cvec& E, cvec& C) {
    const Eigen::Map<const rvec> er(le_re_.GetData(), le_re_.Size());
    const Eigen::Map<const rvec> ei(le_im_.GetData(), le_im_.Size());
    const Eigen::Map<const rvec> cr(lc_re_.GetData(), lc_re_.Size());
    const Eigen::Map<const rvec> ci(lc_im_.GetData(), lc_im_.Size());
    E.real() = R.Pi * er;
    E.imag() = R.Pi * ei;
    C.real() = R.Pi * cr;
    C.imag() = R.Pi * ci;
  };
  // the physical ND_q shapes and curls at the transformation's point, from the reference ones
  auto physical = [&](mfem::ElementTransformation& Tr, const rmat& vref, const rmat& cref) {
    const Eigen::Map<const Eigen::Matrix3d> J(Tr.Jacobian().Data());
    const Eigen::Map<const Eigen::Matrix3d> invJ(Tr.InverseJacobian().Data());
    vp.noalias() = vref * invJ;
    cp.noalias() = cref * J.transpose() / Tr.Weight();
  };
  for (int e = 0; e < ne_; ++e) {
    if (group_of_[e] < 0) continue;
    // the local matrices over ND_q: int_K curl phi_i . curl phi_j and int_K phi_i . phi_j; from the reference moments
    // where the Jacobian is the same at the four vertices (straight elements, also inside curved meshes)
    pm_.GetElementTransformation(e, &T);
    MM.setZero();
    CC.setZero();
    Eigen::Matrix3d J0;
    std::array<double, 9> gm_ab{};
    bool affine = R.affine_test;
    for (int k = 0; k < 4 && affine; ++k) {
      T.SetIntPoint(&mfem::Geometries.GetVertices(mfem::Geometry::TETRAHEDRON)->IntPoint(k));
      const Eigen::Map<const Eigen::Matrix3d> Jk(T.Jacobian().Data());
      if (k == 0) J0 = Jk;
      else affine = (Jk - J0).norm() <= 1e-12 * J0.norm();
    }
    if (affine) {
      const double det = std::fabs(J0.determinant());
      const Eigen::Matrix3d invJ = J0.inverse();
      const Eigen::Matrix3d gm = det * invJ * invJ.transpose();
      const Eigen::Matrix3d gc = J0.transpose() * J0 / det;
      for (int ab = 0; ab < 9; ++ab) gm_ab[ab] = gm(ab / 3, ab % 3);
      for (int ab = 0; ab < 9; ++ab) {
        MM.noalias() += gm(ab / 3, ab % 3) * R.mref[ab];
        CC.noalias() += gc(ab / 3, ab % 3) * R.cref[ab];
      }
    } else {
      for (int k = 0; k < R.vol->GetNPoints(); ++k) {
        const mfem::IntegrationPoint& ip = R.vol->IntPoint(k);
        T.SetIntPoint(&ip);
        const double dv = ip.weight * T.Weight();
        physical(T, R.vshape[k], R.cshape[k]);
        MM.selfadjointView<Eigen::Lower>().rankUpdate(vp, dv);
        CC.selfadjointView<Eigen::Lower>().rankUpdate(cp, dv);
      }
      MM.triangularView<Eigen::StrictlyUpper>() = MM.transpose();
      CC.triangularView<Eigen::StrictlyUpper>() = CC.transpose();
    }
    Load(e, c_re, c_im);
    lift(EP, CP);
    const cplx nu_e = nu[e];
    const cplx eps_e = eps[e];
    // -a_K(E_h, v): the curl from the curl representative, the mass term from E itself
    const cvec CC_CP = times(CC, CP);
    const cvec MM_EP = times(MM, EP);
    b = -nu_e * CC_CP + w2 * eps_e * MM_EP;
    // the norm's parts: nu |curl E|^2 and omega^2 |eps| |E|^2 over the element (exact on straight elements)
    norm_m[e] = std::abs(nu_e) * CP.dot(CC_CP).real();
    norm_e[e] = w2 * std::abs(eps_e) * EP.dot(MM_EP).real();
    // the faces' tractions against the ND_q shapes
    CPr = CP.real();
    CPi = CP.imag();
    fr.setZero();
    fi.setZero();
    pm_.GetElementFaces(e, faces, orientation);
    for (int j = 0; j < 4; ++j) {
      const signed char kind = R.kind[e][j];
      if (kind == kNatural || kind == kPec) continue;   // zero traction; no tangential trace
      const int f = faces[j];
      const bool average = kind == kAverage;
      if (pm_.GetFaceInformation(f).IsShared()) pm_.GetSharedFaceTransformationsByLocalIndex(f, FT, T1, T2, average);
      else pm_.GetFaceElementTransformations(f, FT, T1, T2);
      const int side = FT.Elem1No == e ? 0 : 1;
      cplx nu_n = nu_e;
      if (average) {
        const int other = side == 0 ? FT.Elem2No : FT.Elem1No;
        if (other < ne_) {
          Load(other, c_re, c_im);
          nu_n = nu[other];
        } else {
          LoadNbr(other - ne_, c_re, c_im);
          R.l2_space->GetFaceNbrElementVDofs(other - ne_, vd);
          const mfem::Vector& data = R.coef->FaceNbrData();
          nu_n = cplx(data(vd[2]), data(vd[3]));
        }
        lift(EN, CN);
        CNr = CN.real();
        CNi = CN.imag();
      }
      const cplx a_n = (average ? 0.5 : 0.0) * nu_n;
      const cplx a_k = (average ? 0.5 : 1.0) * nu_e;
      for (int p = 0; p < R.face->GetNPoints(); ++p) {
        const mfem::IntegrationPoint& ip = R.face->IntPoint(p);
        FT.SetAllIntPoints(&ip);
        mfem::CalcOrtho(FT.Jacobian(), normal);   // from element 1 to element 2, |normal| the area element
        const double area = normal.Norml2();
        const double sign = side == 0 ? 1.0 : -1.0;
        const Eigen::Vector3d nhat(sign * normal(0) / area, sign * normal(1) / area, sign * normal(2) / area);
        // nu curl E on the face (the average of the two sides, or the element's own), from the reference curls:
        // curl = J curl_ref / det J
        Eigen::Vector3d hr = Eigen::Vector3d::Zero();
        Eigen::Vector3d hi = Eigen::Vector3d::Zero();
        if (average) {
          mfem::ElementTransformation& Tn = side == 0 ? *FT.Elem2 : *FT.Elem1;
          const int sn = R.ShapeAt(side == 0 ? FT.GetElement2IntPoint() : FT.GetElement1IntPoint());
          const Eigen::Map<const Eigen::Matrix3d> Jn(Tn.Jacobian().Data());
          const Eigen::Vector3d cr = Jn * (R.fcshape[sn].transpose() * CNr) / Tn.Weight();
          const Eigen::Vector3d ci = Jn * (R.fcshape[sn].transpose() * CNi) / Tn.Weight();
          hr += a_n.real() * cr - a_n.imag() * ci;
          hi += a_n.real() * ci + a_n.imag() * cr;
        }
        mfem::ElementTransformation& Tk = side == 0 ? *FT.Elem1 : *FT.Elem2;
        const int sk = R.ShapeAt(side == 0 ? FT.GetElement1IntPoint() : FT.GetElement2IntPoint());
        const Eigen::Map<const Eigen::Matrix3d> Jk(Tk.Jacobian().Data());
        const Eigen::Vector3d cr = Jk * (R.fcshape[sk].transpose() * CPr) / Tk.Weight();
        const Eigen::Vector3d ci = Jk * (R.fcshape[sk].transpose() * CPi) / Tk.Weight();
        hr += a_k.real() * cr - a_k.imag() * ci;
        hi += a_k.real() * ci + a_k.imag() * cr;
        // (n x nu curl E) . phi_i with phi = phi_ref J^-1: phi_ref (J^-1 (n x H))
        const Eigen::Map<const Eigen::Matrix3d> invJk(Tk.InverseJacobian().Data());
        const double dA = ip.weight * area;
        const Eigen::Vector3d tr = invJk * nhat.cross(hr);
        const Eigen::Vector3d ti = invJk * nhat.cross(hi);
        fr.noalias() += dA * (R.fvshape[sk] * tr);
        fi.noalias() += dA * (R.fvshape[sk] * ti);
      }
    }
    fcurl.real() = fr;
    fcurl.imag() = fi;
    b -= fcurl;
    // the free ND_q dofs, the local gradients of the constrained space and the complement of the gradients
    const Residual::Reduced& red = R.Reduce(R.pec_faces[e], R.pec_edges[e]);
    const int nf = static_cast<int>(red.free.size());
    const int nc = static_cast<int>(red.complement.size());
    if (nc == 0) continue;
    rmat MMf(nf, nf), CCf(nf, nf);
    cvec bf(nf);
    for (int r = 0; r < nf; ++r) {
      for (int c = 0; c < nf; ++c) {
        MMf(r, c) = MM(red.free[r], red.free[c]);
        CCf(r, c) = CC(red.free[r], red.free[c]);
      }
      bf(r) = b(red.free[r]);
    }
    // MG = MM G_r and S = G_r^T MM G_r: on an affine element from the pattern's per-component products (9 nf m'
    // flops instead of nf^2 m')
    Eigen::LLT<rmat> llt;
    rmat MG(nf, red.Gr.cols());
    if (red.Gr.cols() > 0) {
      rmat S = rmat::Zero(red.Gr.cols(), red.Gr.cols());
      MG.setZero();
      if (affine) {
        for (int ab = 0; ab < 9; ++ab) {
          S.noalias() += gm_ab[ab] * red.S[ab];
          MG.noalias() += gm_ab[ab] * red.MG[ab];
        }
      } else {
        MG.noalias() = MMf * red.Gr;
        S.noalias() = red.Gr.transpose() * MG;
      }
      llt.compute(S);
    }
    const FwLocalResidual local = fw_local_residual(MMf, CCf, red.Gr, MG, llt, red.complement, bf, nu_e, eps_e, w2);
    pm_.GetElementVertices(e, element_vertices);
    for (int i = 0; i < 4; ++i) {
      const mfem::real_t* x = pm_.GetVertex(element_vertices[i]);
      for (int c = 0; c < 3; ++c) vertices[i][c] = x[c];
    }
    const int flags = fw_local_kh_warning(fw_local_shape(vertices), std::sqrt(w2 * std::abs(eps_e) / std::abs(nu_e)));
    if (flags & 1) ++kh_outside;
    if (flags & 2) ++kh_sliver;
    magnetic[e] = local.magnetic;
    electric[e] = local.electric;
    form[e] = local.form;
  }
  double seconds = timer.seconds();
  MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, pm_.GetComm());
  int counts[2] = {kh_outside, kh_sliver};
  MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT, MPI_SUM, pm_.GetComm());
  return {{"seconds", seconds}, {"kh_outside", counts[0]}, {"kh_sliver", counts[1]}};
}

void FwErrorIndicator::Load(int e, const mfem::ParGridFunction& c_re, const mfem::ParGridFunction& c_im) {
  mfem::DofTransformation doftrans;
  fes_.GetElementVDofs(e, vdofs_, doftrans);
  e_re_->GetSubVector(vdofs_, le_re_);
  e_im_->GetSubVector(vdofs_, le_im_);
  c_re.GetSubVector(vdofs_, lc_re_);
  c_im.GetSubVector(vdofs_, lc_im_);
  doftrans.InvTransformPrimal(le_re_);
  doftrans.InvTransformPrimal(le_im_);
  doftrans.InvTransformPrimal(lc_re_);
  doftrans.InvTransformPrimal(lc_im_);
  loaded_fe_ = fes_.GetFE(e);
  const int nd = loaded_fe_->GetDof();
  vshape_.SetSize(nd, 3);
  cshape_.SetSize(nd, 3);
  loaded_ = e;
}

void FwErrorIndicator::LoadNbr(int nbr, const mfem::ParGridFunction& c_re, const mfem::ParGridFunction& c_im) {
  mfem::DofTransformation doftrans;
  fes_.GetFaceNbrElementVDofs(nbr, vdofs_, doftrans);
  e_re_->FaceNbrData().GetSubVector(vdofs_, le_re_);
  e_im_->FaceNbrData().GetSubVector(vdofs_, le_im_);
  c_re.FaceNbrData().GetSubVector(vdofs_, lc_re_);
  c_im.FaceNbrData().GetSubVector(vdofs_, lc_im_);
  doftrans.InvTransformPrimal(le_re_);
  doftrans.InvTransformPrimal(le_im_);
  doftrans.InvTransformPrimal(lc_re_);
  doftrans.InvTransformPrimal(lc_im_);
  loaded_fe_ = fes_.GetFaceNbrFE(nbr);
  const int nd = loaded_fe_->GetDof();
  vshape_.SetSize(nd, 3);
  cshape_.SetSize(nd, 3);
  loaded_ = ne_ + nbr;
}

void FwErrorIndicator::Values(mfem::ElementTransformation& T, double out[12]) {
  const mfem::FiniteElement* fe = loaded_fe_;
  fe->CalcVShape(T, vshape_);
  fe->CalcPhysCurlShape(T, cshape_);
  const mfem::Vector* local[4] = {&le_re_, &le_im_, &lc_re_, &lc_im_};
  for (int f = 0; f < 4; ++f) {
    (f < 2 ? vshape_ : cshape_).MultTranspose(*local[f], out3_);
    for (int i = 0; i < 3; ++i) out[3 * f + i] = out3_(i);
  }
}

void FwErrorIndicator::Recover(const std::vector<int>& elements, const mfem::ParGridFunction& c_re,
                               const mfem::ParGridFunction& c_im) {
  sums_ = 0.0;
  std::fill(slot_sums_.begin(), slot_sums_.end(), 0.0);
  const int n = space_->GetNDofs();   // byNODES: component c of node k is entry k + c n
  mfem::Array<int> dofs;
  double v[12];
  for (int e : elements) {
    Load(e, c_re, c_im);
    space_->GetElementDofs(e, dofs);
    const mfem::IntegrationRule& nodes = space_->GetFE(e)->GetNodes();
    mfem::ElementTransformation* T = pm_.GetElementTransformation(e);
    const std::vector<int>& slot = cut_slot_[e];
    for (int k = 0; k < nodes.GetNPoints(); ++k) {
      T->SetIntPoint(&nodes.IntPoint(k));
      Values(*T, v);
      for (int c = 0; c < 12; ++c) sums_(dofs[k] + c * n) += v[c];
      sums_(dofs[k] + 12 * n) += 1.0;
      if (slot.empty() || slot[k] < 0) continue;
      double* s = &slot_sums_[static_cast<size_t>(kComponents) * slot[k]];
      for (int c = 0; c < 12; ++c) s[c] += v[c];
      s[12] += 1.0;
    }
  }
  // sums over the ranks sharing a node, and on cuts over all ranks per side
  mfem::GroupCommunicator& comm = space_->GroupComm();
  comm.Reduce<mfem::real_t>(sums_.GetData(), mfem::GroupCommunicator::Sum);
  comm.Bcast<mfem::real_t>(sums_.GetData());
  for (int k = 0; k < n; ++k) {
    const double count = sums_(k + 12 * n);
    if (count <= 0) continue;
    for (int c = 0; c < 12; ++c) sums_(k + c * n) /= count;
  }
  if (slots_ == 0) return;
  MPI_Allreduce(MPI_IN_PLACE, slot_sums_.data(), static_cast<int>(slot_sums_.size()), MPI_DOUBLE, MPI_SUM,
                pm_.GetComm());
  for (int s = 0; s < slots_; ++s) {
    double* x = &slot_sums_[static_cast<size_t>(kComponents) * s];
    if (x[12] <= 0) continue;
    for (int c = 0; c < 12; ++c) x[c] /= x[12];
  }
}

void FwErrorIndicator::Nodal(int e, const mfem::Array<int>& dofs) {
  const int n = space_->GetNDofs();
  const int nd = dofs.Size();
  nodal_.assign(static_cast<size_t>(12) * nd, 0.0);
  const std::vector<int>& slot = cut_slot_[e];
  for (int k = 0; k < nd; ++k) {
    const bool cut = !slot.empty() && slot[k] >= 0;
    for (int c = 0; c < 12; ++c)
      nodal_[c * nd + k] = cut ? slot_sums_[static_cast<size_t>(kComponents) * slot[k] + c] : sums_(dofs[k] + c * n);
  }
}

json FwErrorIndicator::Evaluate(const mfem::Vector& Er, const mfem::Vector& Ei, const mfem::Vector* Cr,
                                const mfem::Vector* Ci, double w, const std::vector<cplx>& eps,
                                const std::vector<cplx>& nu, cplx Z, bool accepted) {
  Timer timer;
  const MPI_Comm comm = pm_.GetComm();
  local_.assign(ne_, 0.0);
  if (!accepted)   // the same decision on every rank: no collective is skipped on one rank only
    return {{"withheld", "the solve did not pass its residual gate (solver.accepted is false): its field is not an "
                         "approximation the indicator can assess"},
            {"seconds", timer.seconds()}};
  e_re_->SetFromTrueDofs(Er);
  e_im_->SetFromTrueDofs(Ei);
  const bool representative = Cr && Ci;
  if (representative) {
    c_re_->SetFromTrueDofs(*Cr);
    c_im_->SetFromTrueDofs(*Ci);
  }
  // the curl from the curl representative when there is one (no cancellation against a large gradient part)
  const mfem::ParGridFunction& c_re = representative ? *c_re_ : *e_re_;
  const mfem::ParGridFunction& c_im = representative ? *c_im_ : *e_im_;
  // per local element: the magnetic and electric parts of eta_e^2 and of the squared norm
  std::vector<double> eta_m(ne_, 0.0), eta_e(ne_, 0.0), norm_m(ne_, 0.0), norm_e(ne_, 0.0);
  double surface_eta = 0.0;
  double surface_norm = 0.0;
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * order_ + 1);
  const mfem::IntegrationRule& irs = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * order_ + 2);
  mfem::Array<int> dofs;
  mfem::Vector shape, normal(3);
  double v[12];
  double star[12];
  // the recovered fields at a point (E re, im, curl E re, im): the element's nodal values weighted by the H1 shapes
  auto recovered = [&]() {
    const int nd = shape.Size();
    for (int c = 0; c < 12; ++c) {
      star[c] = 0.0;
      for (int k = 0; k < nd; ++k) star[c] += shape(k) * nodal_[c * nd + k];
    }
  };
  for (size_t g = 0; g < groups_.size(); ++g) {
    if (group_size_[g] == 0) continue;   // the same on every rank: the recovery is collective
    Recover(members_[g], c_re, c_im);
    // the volume parts (the estimate at order 1; at order 2 and above the magnetic part is reported as a diagnostic)
    for (int e : members_[g]) {
      const mfem::FiniteElement* fe = space_->GetFE(e);
      space_->GetElementDofs(e, dofs);
      shape.SetSize(fe->GetDof());
      Nodal(e, dofs);
      Load(e, c_re, c_im);
      const double nu_weight = std::abs(nu[e]);
      const double eps_weight = w * w * std::abs(eps[e]);
      mfem::ElementTransformation* T = pm_.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        const double dv = ip.weight * T->Weight();
        Values(*T, v);
        fe->CalcShape(ip, shape);
        recovered();
        for (int i = 0; i < 3; ++i) {
          const double der = v[i] - star[i];
          const double dei = v[3 + i] - star[3 + i];
          const double dcr = v[6 + i] - star[6 + i];
          const double dci = v[9 + i] - star[9 + i];
          eta_m[e] += nu_weight * (dcr * dcr + dci * dci) * dv;
          norm_m[e] += nu_weight * (v[6 + i] * v[6 + i] + v[9 + i] * v[9 + i]) * dv;
          eta_e[e] += eps_weight * (der * der + dei * dei) * dv;
          norm_e[e] += eps_weight * (v[i] * v[i] + v[3 + i] * v[3 + i]) * dv;
        }
      }
    }
    // conductor faces: the recovery error of curl E (B, so n x H) weighted by 1/mu_r^2 of the field side, as the
    // surface losses weigh |H|^2
    for (const Face& face : faces_[g]) {
      const int e = face.element;
      const mfem::FiniteElement* fe = space_->GetFE(e);
      space_->GetElementDofs(e, dofs);
      shape.SetSize(fe->GetDof());
      Nodal(e, dofs);
      Load(e, c_re, c_im);
      const double inverse_mu_r = units::mu0 * std::abs(nu[e]);
      const double weight = inverse_mu_r * inverse_mu_r;
      mfem::FaceElementTransformations* FT = pm_.GetFaceElementTransformations(face.face);
      for (int q = 0; q < irs.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = irs.IntPoint(q);
        FT->SetAllIntPoints(&ip);
        mfem::ElementTransformation& T = face.side == 0 ? *FT->Elem1 : *FT->Elem2;
        Values(T, v);
        fe->CalcShape(T.GetIntPoint(), shape);
        mfem::CalcOrtho(FT->Jacobian(), normal);
        const double da = ip.weight * normal.Norml2() * weight;
        recovered();
        for (int i = 0; i < 3; ++i) {
          const double dcr = v[6 + i] - star[6 + i];
          const double dci = v[9 + i] - star[9 + i];
          surface_eta += (dcr * dcr + dci * dci) * da;
          surface_norm += (v[6 + i] * v[6 + i] + v[9 + i] * v[9 + i]) * da;
        }
      }
    }
  }

  // order 2 and above: the element residual problems (magnetic part) and the global gradient projection (electric
  // part), and a(e, e) per element; the recovery's magnetic part is kept as a diagnostic
  const int nr = static_cast<int>(regions_.size());
  std::vector<cplx> form;
  json gradient;
  double residual_seconds = 0.0;
  int residual_kh_outside = 0;
  int residual_kh_sliver = 0;
  double recovery_magnetic = 0.0;
  if (order_ >= 2) {
    for (int e = 0; e < ne_; ++e)
      if (group_of_[e] >= 0) recovery_magnetic += eta_m[e];
    std::vector<double> electric, gradient_energy;
    std::vector<cplx> curl_form, gradient_form;
    const json residual = EvaluateResidual(w, eps, nu, c_re, c_im, eta_m, electric, curl_form, norm_m, norm_e);
    residual_seconds = residual["seconds"].get<double>();
    residual_kh_outside = residual["kh_outside"].get<int>();
    residual_kh_sliver = residual["kh_sliver"].get<int>();
    // the electric share of the norm: below kElectricShareSkip the gradient projection is not computed
    double norms[2] = {0.0, 0.0};
    for (int e = 0; e < ne_; ++e) {
      if (group_of_[e] < 0) continue;
      norms[0] += norm_m[e];
      norms[1] += norm_e[e];
    }
    MPI_Allreduce(MPI_IN_PLACE, norms, 2, MPI_DOUBLE, MPI_SUM, comm);
    const double share = ratio(norms[1], norms[0] + norms[1]);
    if (share < kElectricShareSkip) {
      gradient_energy.assign(ne_, 0.0);
      gradient_form.assign(ne_, cplx(0.0, 0.0));
      gradient = {{"skipped", "the electric share of the norm is below 1e-6"}};
    } else {
      gradient = EvaluateGradient(w, eps, gradient_energy, gradient_form);
    }
    if (gradient.contains("converged") && !gradient["converged"].get<bool>()) {   // the same on every rank
      local_.assign(ne_, 0.0);
      return {{"withheld", strcat("the gradient projection did not converge (relative residual ",
                                  gradient["relative_residual"].get<double>(), " after ",
                                  gradient["iterations"].get<int>(), " iterations, ",
                                  gradient["solver"].get<std::string>(),
                                  "): its electric part would be an underestimate")},
              {"gradient", gradient},
              {"seconds", timer.seconds()}};
    }
    form.assign(ne_, cplx(0.0, 0.0));
    for (int e = 0; e < ne_; ++e) {
      if (group_of_[e] < 0) continue;
      eta_e[e] = electric[e] + gradient_energy[e];
      form[e] = curl_form[e] + gradient_form[e];
    }
    double totals[2] = {recovery_magnetic, norms[0]};
    MPI_Allreduce(MPI_IN_PLACE, totals, 1, MPI_DOUBLE, MPI_SUM, comm);
    recovery_magnetic = std::sqrt(ratio(totals[0], totals[1]));
  }

  // totals, per region [eta_m, eta_e, norm_m, norm_e], then the surface pair and (order 2 and above) a(e, e) (re, im)
  std::vector<double> sums(4 * nr + (order_ >= 2 ? 4 : 2), 0.0);
  for (int e = 0; e < ne_; ++e) {
    if (group_of_[e] < 0) continue;
    double* s = &sums[4 * region_of_[e]];
    s[0] += eta_m[e];
    s[1] += eta_e[e];
    s[2] += norm_m[e];
    s[3] += norm_e[e];
    if (!form.empty()) {
      sums[4 * nr + 2] += form[e].real();
      sums[4 * nr + 3] += form[e].imag();
    }
  }
  sums[4 * nr] = surface_eta;
  sums[4 * nr + 1] = surface_norm;
  MPI_Allreduce(MPI_IN_PLACE, sums.data(), static_cast<int>(sums.size()), MPI_DOUBLE, MPI_SUM, comm);
  double total_eta_m = 0.0;
  double total_eta_e = 0.0;
  double total_norm_m = 0.0;
  double total_norm_e = 0.0;
  for (int r = 0; r < nr; ++r) {
    total_eta_m += sums[4 * r];
    total_eta_e += sums[4 * r + 1];
    total_norm_m += sums[4 * r + 2];
    total_norm_e += sums[4 * r + 3];
  }
  const double total_eta = total_eta_m + total_eta_e;
  const double total_norm = total_norm_m + total_norm_e;
  const cplx total_form = order_ >= 2 ? cplx(sums[4 * nr + 2], sums[4 * nr + 3]) : cplx(0.0, 0.0);
  for (int e = 0; e < ne_; ++e) local_[e] = ratio(eta_m[e] + eta_e[e], total_norm);

  // the element with the largest indicator: its centre and region, from the rank that holds it
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  struct {
    double value;
    int rank;
  } mine{-1.0, rank}, top{-1.0, rank};
  int worst = -1;
  for (int e = 0; e < ne_; ++e) {
    if (group_of_[e] < 0 || !(eta_m[e] + eta_e[e] > mine.value)) continue;
    mine.value = eta_m[e] + eta_e[e];
    worst = e;
  }
  MPI_Allreduce(&mine, &top, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
  double where[4] = {0.0, 0.0, 0.0, -1.0};
  if (rank == top.rank && worst >= 0) {
    mfem::Vector center(3);
    pm_.GetElementCenter(worst, center);
    for (int i = 0; i < 3; ++i) where[i] = center(i);
    where[3] = region_of_[worst];
  }
  MPI_Bcast(where, 4, MPI_DOUBLE, top.rank, comm);

  json by_region = json::object();
  for (int r = 0; r < nr; ++r) {
    const double eta = sums[4 * r] + sums[4 * r + 1];
    const double norm = sums[4 * r + 2] + sums[4 * r + 3];
    if (!(eta > 0) && !(norm > 0)) continue;
    by_region[regions_[r]] = {{"share", ratio(eta, total_eta)}, {"norm_share", ratio(norm, total_norm)}};
  }
  json out = {{"estimated_relative_error", std::sqrt(ratio(total_eta, total_norm))},
              {"magnetic_relative_error", std::sqrt(ratio(total_eta_m, total_norm_m))},
              {"electric_relative_error", std::sqrt(ratio(total_eta_e, total_norm_e))},
              {"electric_norm_share", ratio(total_norm_e, total_norm)},
              {"by_region", by_region},
              {"curl_from", representative ? "curl representative" : "E"}};
  // without the projection (deep MQS) the electric part is only the residual problems' (kh)^2 by-product: kept in eta
  // (it cannot move it), but not reported as the electric error, which it is not
  if (gradient.contains("skipped")) {
    out["electric_relative_error"] = nullptr;
    out["electric_relative_error_withheld"] = "the gradient projection was skipped (electric share of the norm below "
                                              "1e-6): the electric error is not estimated";
  }
  // Z and L: from a's volume terms over the field elements only; with PML layers or Robin groups the terminations'
  // own error (part of a, not of the norm) is left out and can dominate, so the estimates are withheld
  std::string left_out;
  if (terminations_.pml) left_out = "PML layers";
  for (const std::string& g : terminations_.robin_groups)
    left_out += (left_out.empty() ? "Robin groups " : ", ") + g;
  if (!left_out.empty()) {
    out["Z_L_withheld"] = "the operator's " + left_out + " are not in the estimate's norm, and their discretization "
                          "error can dominate that of Z (2e-5 of it on a PML-terminated line): check the termination "
                          "separately";
  } else {
    out["Z_error_estimate_ohm"] = std::isfinite(total_eta) ? total_eta / w : std::numeric_limits<double>::quiet_NaN();
    out["Z_relative_error_estimate"] = ratio(total_eta / w, std::abs(Z));
    const double L = -Z.imag() / w;
    // the two parts move L in opposite directions. At order 2 and above, where they are within a factor 8 of each
    // other (energies), the sign and size of their difference are not resolved: with the measured effectivities of
    // the parts (magnetic 0.37-1, electric 0.45-1), a factor 8 leaves the signed estimate within 0.34-1.32 of the
    // error of L (a factor 4 would leave 0.31-2.3). At order 1 the recovery's signed estimate is kept (where the
    // parts partly cancel 0.27-1.0 of the TEM line's error, 1.4-3.1 next to a crack and a 270-degree edge)
    const bool resolved = order_ < 2 || std::max(total_eta_m, total_eta_e) >= kSignedResolution *
                                                                              std::min(total_eta_m, total_eta_e);
    if (order_ >= 2) {
      // Z_h - Z = a(e, e) / (i omega) per ampere, a(e, e) estimated element by element (no lossless assumption)
      const cplx dZ = total_form / cplx(0.0, w);
      if (resolved) out["Z_signed_error_estimate_ohm"] = {{"re", dZ.real()}, {"im", dZ.imag()}};
      else out["Z_signed_error_estimate_ohm"] = nullptr;
    }
    if (L > 0) {
      out["L_relative_error_estimate"] = ratio(total_eta, w * w * L);
      // with sign: L_h / L - 1 = Re a(e, e) / (omega^2 L); at order 1 for a lossless model (E = i e, e real, so
      // a(e, e) = -(||curl e||^2 - omega^2 ||e||^2)): -(eta_m^2 - eta_e^2) / (omega^2 L)
      if (resolved) {
        out["L_signed_relative_error_estimate"] = order_ >= 2 ? ratio(total_form.real(), w * w * L)
                                                              : ratio(-(total_eta_m - total_eta_e), w * w * L);
      } else {
        out["L_signed_relative_error_estimate"] = nullptr;
        out["L_signed_withheld"] = "order 2 or above, with the magnetic and electric parts within a factor 8 of each "
                                   "other: the sign and size of their difference are not resolved";
      }
    }
    if (terminations_.radiating_shell)
      out["Z_L_scope"] = "the field region's volume error only: the radiating shell's and the exterior expansion's "
                         "discretization error are not included";
  }
  if (top.value > 0 && where[3] >= 0)
    out["largest_element"] = {{"center_m", {where[0], where[1], where[2]}},
                              {"region", regions_[static_cast<int>(where[3])]},
                              {"share", ratio(top.value, total_eta)}};
  if (conductor_faces_ > 0) out["surface_relative_error"] = std::sqrt(ratio(sums[4 * nr], sums[4 * nr + 1]));
  if (order_ >= 2) {
    out["recovery_magnetic_relative_error"] = recovery_magnetic;
    out["residual_seconds"] = residual_seconds;
    if (residual_kh_outside + residual_kh_sliver > 0)   // only where it happens: other models' results are unchanged
      out["residual_kh_warning"] = {
          {"elements_kh_max_above_1", residual_kh_outside},
          {"sliver_elements_kh_max_above_0.2", residual_kh_sliver},
          {"note", "a heuristic, not a resonance detector: elements where |k| h_max (k the local wave number, h_max "
                   "the longest edge) exceeds 1, the validated range of the local problems (the first local resonance "
                   "lies at 1.0-1.7 on reasonable shapes; in a conductor the magnetic part was validated only for "
                   "|k| h <~ 1 in every direction), and sliver-like elements (smallest height below 0.2 of the "
                   "shortest edge) past 0.2, where a flat sliver resonated from 0.27. Their magnetic part may be "
                   "inflated; refine there. Near-resonant elements inside these bounds are not detected"}};
    out["gradient"] = gradient;
  }
  out["seconds"] = timer.seconds();
  return out;
}

void FwErrorIndicator::Save(const std::string& path) const {
  mfem::L2_FECollection fec(0, 3);
  mfem::FiniteElementSpace space(&pm_, &fec);
  mfem::GridFunction values(&space);
  values = 0.0;
  mfem::Array<int> dofs;
  for (int e = 0; e < ne_ && e < static_cast<int>(local_.size()); ++e) {
    space.GetElementDofs(e, dofs);
    values(dofs[0]) = std::isfinite(local_[e]) ? local_[e] : 0.0;
  }
  std::ofstream stream(path);
  if (stream) {
    stream << std::setprecision(17);
    values.Save(stream);
    stream.flush();
  }
  // a failure on one rank must not leave the others waiting in the next collective
  int failed = stream ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_SUM, pm_.GetComm());
  AXF_REQUIRE(failed == 0, strcat("cannot write the full-wave error indicator files on ", failed,
                                  " rank(s) (this rank's file: '", path, "')"),
              "/fullwave/field_export_directory");
}

json FwErrorIndicator::Describe() const {
  json groups = json::object();
  for (size_t g = 0; g < groups_.size(); ++g)
    if (group_size_[g] > 0) groups[groups_[g]] = group_size_[g];
  json left_out = json::array();
  if (terminations_.radiating_shell) left_out.push_back("radiating shell and exterior expansion");
  if (terminations_.pml) left_out.push_back("PML layers");
  for (const std::string& g : terminations_.robin_groups) left_out.push_back(g);
  const std::string recovery = "Zienkiewicz-Zhu recovery: nodal averages of E and curl E within each material and on "
                               "each side of interior boundary faces";
  json out = {
      {"method", order_ < 2 ? recovery
                            : "magnetic part: element residual problems in ND_{p+1} on the complement of the local "
                              "gradients, with averaged tractions; electric part: the error's projection onto the "
                              "gradients of the global H1_{p+1}, with the complex eps where the field materials' loss "
                              "phases differ; a(e, e) per element with sign. The recovery (" +
                                  recovery + ") gives the surface term and, as a diagnostic, "
                                             "recovery_magnetic_relative_error"},
      {"norm", "int nu |curl E|^2 + omega^2 |eps_c| |E|^2 dV over the field elements"},
      {"recovery_order", order_},
      {"field_elements", field_elements_},
      {"field_elements_by_material", groups},
      {"excluded_elements", {{"perfect_conductor", excluded_conductor_}, {"shell", excluded_shell_},
                             {"pml", excluded_pml_}}},
      {"conductor_faces", conductor_faces_},
      {"cut_faces", cut_faces_},
      {"cut_groups", cut_groups_},
      {"terminations_left_out", left_out},
      {"setup_seconds", setup_seconds_},
      {"scope", "the discretization error of the field in the field elements (outside perfect conductors, the infinity "
                "shell and PML layers) on the given geometry, not the geometry's own approximation. The Z and L "
                "estimates hold a's volume terms only: the shell, the exterior expansion, PML layers and Robin groups "
                "are left out (with PML or Robin groups the Z and L estimates are withheld). The surface term covers "
                "faces to meshed conductors, not PEC boundary groups"},
      {"reliability", order_ < 2 ? RELIABILITY_ORDER_1 : RELIABILITY_ORDER_2}};
  if (order_ >= 2) out["local_order"] = order_ + 1;
  return out;
}

}  // namespace axf
