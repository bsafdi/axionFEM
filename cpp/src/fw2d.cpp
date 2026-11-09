// 2D full-wave model on an imported mesh (planar or axisymmetric). Beta.
// Formulation and conventions: axionfem/fw2d.hpp, docs/physics/fw2d.md and docs/comsol_import.md.
#include "axionfem/fw2d.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <map>
#include <set>

#include <mfem.hpp>

#include "axionfem/core.hpp"
#include "axionfem/fullwave.hpp"
#include "axionfem/mesh_quality.hpp"
#include "axionfem/pml.hpp"
#include "axionfem/prescribed_source.hpp"
#include "axionfem/units.hpp"

namespace axf {

using json = nlohmann::json;
using cplx = std::complex<double>;
using SpMat = Eigen::SparseMatrix<double>;
using SpMatC = Eigen::SparseMatrix<cplx>;
using VecC = Eigen::VectorXcd;

namespace {

const double kEps0 = 1.0 / (units::mu0 * units::c * units::c);
const cplx kI(0.0, 1.0);

bool is_pec_mat(const spec::Material* m) {
  return m->kind == spec::MaterialKind::Superconductor || m->kind == spec::MaterialKind::Conductor;
}
double mu_of(const spec::Material* m) {
  const bool lin = m->kind == spec::MaterialKind::Linear || m->kind == spec::MaterialKind::Iron;
  return units::mu0 * (lin && m->mu_r > 0 ? m->mu_r : 1.0);
}
cplx epsc_of(const spec::Material* m, double w) {
  const double er = m->eps_r > 0 ? m->eps_r : 1.0;
  return kEps0 * er * cplx(1.0, m->tan_delta) + cplx(0.0, w > 0 ? m->sigma_S_m / w : 0.0);
}

// DOFs of an element / boundary element with Nedelec orientation signs
void signed_dofs(const mfem::Array<int>& raw, std::vector<int>& idx, std::vector<double>& sg) {
  idx.resize(raw.Size());
  sg.resize(raw.Size());
  for (int k = 0; k < raw.Size(); ++k) {
    idx[k] = raw[k] >= 0 ? raw[k] : -1 - raw[k];
    sg[k] = raw[k] >= 0 ? 1.0 : -1.0;
  }
}

}  // namespace

struct Fw2DModel::Impl {
  bool axi = true;
  double D = 1.0;  // planar thickness
  int order = 2;
  std::unique_ptr<mfem::Mesh> mesh;
  std::unique_ptr<mfem::FiniteElementCollection> h1c, ndc;
  std::unique_ptr<mfem::FiniteElementSpace> h1, nd;
  std::vector<const spec::Material*> emat;
  std::vector<std::string> ereg;
  std::vector<int> epml;  // PML layer index or -1
  std::vector<pml::Layer> layers;
  std::vector<char> esrc;  // element carries the prescribed source (source_regions)
  struct Group {
    int attr = 0;
    std::string name, type;
    json j;
    const spec::Material* mat = nullptr;  // impedance / sheet
    bool has_Zs = false;
    cplx Zs = 0;
    double thickness = 0;
    std::string wave = "plane";
    double center[2] = {0, 0};
  };
  std::map<int, Group> groups;  // by boundary attribute
  // port (in family)
  std::vector<int> port_be;       // boundary elements of the port
  std::vector<double> port_sign;  // tangent orientation per port boundary element
  json port_info;
  std::unique_ptr<PrescribedMagneticSource> source;
  // axisymmetric source: 0 evaluate at phi = 0 (axisymmetric source), 1 box sector (exact average: the in-sector
  // field times span / 360), 2 numerical azimuthal average with src_navg samples
  int src_mode = 0, src_navg = 64;
  double src_frac = 1.0, src_phi_mid = 0.0;
  // B0 at an in-plane point in the components (x, y, z) of the evaluation point (x1, 0, x2) / (x1, x2, 0): for an
  // axisymmetric design B_x = B_r, B_y = B_phi (phi-averaged when the source depends on phi)
  std::array<double, 3> source_at(double x1, double x2) const {
    if (!axi) return source->field_T({x1, x2, 0.0});
    if (src_mode == 0) return source->field_T({x1, 0.0, x2});
    if (src_mode == 1) {
      const double cp = std::cos(src_phi_mid), sp = std::sin(src_phi_mid);
      const auto b = source->field_T({x1 * cp, x1 * sp, x2});
      return {src_frac * (b[0] * cp + b[1] * sp), src_frac * (-b[0] * sp + b[1] * cp), src_frac * b[2]};
    }
    return phi_average(x1, x2, src_navg);
  }
  std::array<double, 3> phi_average(double x1, double x2, int n) const {
    std::array<double, 3> s{0, 0, 0};
    for (int k = 0; k < n; ++k) {
      const double ph = 2 * units::pi * (k + 0.5) / n, cp = std::cos(ph), sp = std::sin(ph);
      const auto b = source->field_T({x1 * cp, x1 * sp, x2});
      s[0] += (b[0] * cp + b[1] * sp) / n;
      s[1] += (-b[0] * sp + b[1] * cp) / n;
      s[2] += b[2] / n;
    }
    return s;
  }
  // stored solutions
  std::string sweep_family;  // "in"
  std::vector<double> sweep_w;
  std::vector<VecC> sweep_u;  // full DOF vectors per frequency (per ampere)
  struct Mode {
    std::string family, name;  // family "out" | "in" (dual H); name TE/TM
    double omega = 0;
    Eigen::VectorXd u;  // full H1 DOFs, M-normalized
  };
  std::vector<Mode> modes;

  // H1 DOFs per element for the eigen families; the dual H family splits the DOFs on interior PEC sheets (H_t
  // jumps by the sheet current; each side gets its own copy; at a free sheet end the sides join)
  std::map<std::string, std::vector<std::vector<int>>> hd;
  std::map<std::string, int> hn;
  std::set<int> sheet_faces;  // mesh faces of interior PEC boundaries (both neighbours field elements)
  void h1dofs(const std::string& fam, int e, mfem::Array<int>& raw) const {
    auto it = hd.find(fam);
    if (it == hd.end() || it->second.empty()) {
      h1->GetElementDofs(e, raw);
      return;
    }
    const auto& v = it->second[e];
    raw.SetSize(static_cast<int>(v.size()));
    for (size_t k = 0; k < v.size(); ++k) raw[static_cast<int>(k)] = v[k];
  }

  double weight(double x1) const { return axi ? 2 * units::pi * x1 : D; }
  mfem::FiniteElementSpace& fes(const std::string& fam) { return fam == "in" ? *nd : *h1; }

  // complex coefficients at a point of element e: nu' (in-plane 2x2 for the out family; [2][2] for in) and eps'
  void coeffs(int e, const double x[2], double w, cplx nu[9], cplx eps[9]) const {
    const double mu = mu_of(emat[e]);
    const cplx ec = epsc_of(emat[e], w);
    for (int k = 0; k < 9; ++k) nu[k] = eps[k] = 0.0;
    if (epml[e] < 0) {
      for (int i = 0; i < 3; ++i) { nu[4 * i] = 1.0 / mu; eps[4 * i] = ec; }
      return;
    }
    cplx Lt[9], Li[9];
    layers[epml[e]].tensors(x, w, Lt, Li);
    for (int k = 0; k < 9; ++k) { nu[k] = Li[k] / mu; eps[k] = ec * Lt[k]; }
  }

  // B-type operator of the H1 families at a point: rows (a, b) of curl(u e_n) for shape i
  void bvec(const mfem::Vector& N, const mfem::DenseMatrix& dN, double r, int i, double out[2]) const {
    if (axi) {
      out[0] = -dN(i, 1);
      out[1] = dN(i, 0) + N(i) / r;
    } else {
      out[0] = dN(i, 1);
      out[1] = -dN(i, 0);
    }
  }

  // Element matrix of the driven operator (family "out" or "in") at omega: A_e = K - omega^2 M. Lossless real
  // coefficients when `real_lossless` (eigen: K and M returned separately in Ke, Me with alpha/beta of the family;
  // dual: alpha = 1/eps', beta = mu).
  void element(const std::string& fam, int e, double w, Eigen::MatrixXcd& A, std::vector<int>& idx,
               std::vector<double>& sg) const {
    mfem::FiniteElementSpace& f = fam == "in" ? *nd : *h1;
    mfem::Array<int> raw;
    f.GetElementDofs(e, raw);
    signed_dofs(raw, idx, sg);
    const int n = raw.Size();
    A.setZero(n, n);
    const mfem::FiniteElement* fe = f.GetFE(e);
    mfem::ElementTransformation* T = mesh->GetElementTransformation(e);
    const auto& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * order + 4);
    mfem::Vector N(n), X(2);
    mfem::DenseMatrix dN(n, 2), V(n, 2), C(n, 1);
    cplx nu[9], eps[9];
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, X);
      const double x[2] = {X(0), X(1)};
      const double wq = ip.weight * T->Weight() * weight(x[0]);
      coeffs(e, x, w, nu, eps);
      if (fam == "out") {
        fe->CalcPhysShape(*T, N);
        fe->CalcPhysDShape(*T, dN);
        for (int i = 0; i < n; ++i) {
          double bi[2];
          bvec(N, dN, x[0], i, bi);
          for (int j = 0; j < n; ++j) {
            double bj[2];
            bvec(N, dN, x[0], j, bj);
            cplx k = 0;
            for (int a = 0; a < 2; ++a)
              for (int b = 0; b < 2; ++b) k += bi[a] * nu[3 * a + b] * bj[b];
            A(i, j) += wq * (k - w * w * eps[8] * N(i) * N(j));
          }
        }
      } else {
        fe->CalcVShape(*T, V);
        fe->CalcPhysCurlShape(*T, C);
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j) {
            cplx m = 0;
            for (int a = 0; a < 2; ++a)
              for (int b = 0; b < 2; ++b) m += V(i, a) * eps[3 * a + b] * V(j, b);
            A(i, j) += wq * (nu[8] * C(i, 0) * C(j, 0) - w * w * m);
          }
      }
    }
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) A(i, j) *= sg[i] * sg[j];
  }

  // Real element stiffness and mass of the H1 eigenproblems: family "out" (alpha = nu, beta = eps') or "in" (dual:
  // alpha = 1/eps', beta = mu), lossless parts.
  void element_eig(const std::string& fam, int e, Eigen::MatrixXd& K, Eigen::MatrixXd& M, std::vector<int>& idx) const {
    mfem::Array<int> raw;
    h1dofs(fam, e, raw);
    std::vector<double> sg;
    signed_dofs(raw, idx, sg);
    const int n = raw.Size();
    K.setZero(n, n);
    M.setZero(n, n);
    const mfem::FiniteElement* fe = h1->GetFE(e);
    mfem::ElementTransformation* T = mesh->GetElementTransformation(e);
    const auto& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * order + 4);
    mfem::Vector N(n), X(2);
    mfem::DenseMatrix dN(n, 2);
    const double mu = mu_of(emat[e]), ep = epsc_of(emat[e], 0.0).real();
    const double alpha = fam == "out" ? 1.0 / mu : 1.0 / ep, beta = fam == "out" ? ep : mu;
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const auto& ip = ir.IntPoint(q);
      T->SetIntPoint(&ip);
      T->Transform(ip, X);
      const double wq = ip.weight * T->Weight() * weight(X(0));
      fe->CalcPhysShape(*T, N);
      fe->CalcPhysDShape(*T, dN);
      for (int i = 0; i < n; ++i) {
        double bi[2];
        bvec(N, dN, X(0), i, bi);
        for (int j = 0; j < n; ++j) {
          double bj[2];
          bvec(N, dN, X(0), j, bj);
          K(i, j) += wq * alpha * (bi[0] * bj[0] + bi[1] * bj[1]);
          M(i, j) += wq * beta * N(i) * N(j);
        }
      }
    }
  }

  // face transformation of boundary element be with the field-side element (non-PEC) as element 1 of the pair
  struct Side {
    mfem::FaceElementTransformations* ft = nullptr;
    bool use2 = false;
    int elem = -1;
  };
  Side side(int be) const {
    Side s;
    const int fn = mesh->GetBdrElementFaceIndex(be);
    s.ft = mesh->GetFaceElementTransformations(fn);
    const int e1 = s.ft->Elem1No, e2 = s.ft->Elem2No;
    s.use2 = (e1 < 0 || is_pec_mat(emat[e1])) && e2 >= 0;
    s.elem = s.use2 ? e2 : e1;
    return s;
  }

  // Robin coefficient beta at a boundary point of group g next to element e, for family fam, edge tangent t.
  // Scattering: the first-order condition n x curl E = -i k E_t of an outgoing wave plus the curvature term of the
  // E-field formulation: spherical waves none (n x curl E = -(1/r) d(r E_t)/dr = -i k E_t exactly for e^{ikr}/r);
  // cylindrical waves (E ~ e^{ik rho}/sqrt(rho)) (nu / 2 rho) [(e.a)^2 - (e.phi)^2] for E_t along the unit vector e
  // (a: cylinder axis, phi: azimuthal direction), i.e. + for E_t along the axis, - for E_t azimuthal, 0 for E_t
  // radial (e.g. on an end cap), as the 3D tensor (nu / 2 rho)(a a^T - phi phi^T).
  cplx beta(const Group& g, int e, const double x[2], double w, const std::string& fam, const double t[2]) const {
    if (g.type == "impedance") {
      cplx Zs = g.Zs;
      if (!g.has_Zs) Zs = std::sqrt(mu_of(g.mat) / epsc_of(g.mat, w));
      return -kI * w / Zs;
    }
    if (g.type == "sheet") {
      const cplx ec = epsc_of(g.mat, w);
      const double mu = mu_of(g.mat);
      const cplx k = w * std::sqrt(mu * ec), eta = std::sqrt(mu / ec);
      const cplx Y = -2.0 * kI / eta * std::tan(0.5 * k * g.thickness);
      return -kI * w * Y;
    }
    // scattering: medium of the adjacent element
    const double mu = mu_of(emat[e]);
    const cplx ec = epsc_of(emat[e], w);
    cplx b = -kI * w * std::sqrt(ec / mu);
    if (g.wave == "cylindrical") {
      double R, ea2, ep2;
      if (axi) {  // axis z (in plane), phi out of plane
        R = x[0];
        ea2 = fam == "out" ? 0.0 : t[1] * t[1];
        ep2 = fam == "out" ? 1.0 : 0.0;
      } else {    // axis out of plane, phi in plane about the center
        const double dx = x[0] - g.center[0], dy = x[1] - g.center[1];
        R = std::hypot(dx, dy);
        const double ph[2] = {-dy / R, dx / R};
        ea2 = fam == "out" ? 1.0 : 0.0;
        ep2 = fam == "out" ? 0.0 : std::pow(t[0] * ph[0] + t[1] * ph[1], 2);
      }
      b += (ea2 - ep2) / (mu * 2.0 * R);
    }
    return b;
  }

  // Boundary matrix (Robin) or vector (port: oint t . v dl) on boundary element be.
  void boundary(const std::string& fam, int be, double w, const Group* g, Eigen::MatrixXcd* A, Eigen::VectorXd* fport,
                double tsign, std::vector<int>& idx, std::vector<double>& sg) const {
    mfem::FiniteElementSpace& f = fam == "in" ? *nd : *h1;
    Side s = side(be);
    mfem::Array<int> raw;
    f.GetElementDofs(s.elem, raw);
    signed_dofs(raw, idx, sg);
    const int n = raw.Size();
    if (A) A->setZero(n, n);
    if (fport) fport->setZero(n);
    const mfem::FiniteElement* fe = f.GetFE(s.elem);
    const auto& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * order + 4);
    mfem::Vector N(n), X(2);
    mfem::DenseMatrix V(n, 2);
    for (int q = 0; q < ir.GetNPoints(); ++q) {
      const auto& ip = ir.IntPoint(q);
      s.ft->SetAllIntPoints(&ip);
      mfem::ElementTransformation& Te = s.use2 ? *s.ft->Elem2 : *s.ft->Elem1;
      s.ft->Transform(ip, X);
      const mfem::DenseMatrix& J = s.ft->Jacobian();
      const double len = std::hypot(J(0, 0), J(1, 0));
      const double t[2] = {J(0, 0) / len, J(1, 0) / len};
      const double x[2] = {X(0), X(1)};
      const double wl = ip.weight * len;  // dl
      std::vector<double> sv(n);
      if (fam == "out") {
        fe->CalcPhysShape(Te, N);
        for (int i = 0; i < n; ++i) sv[i] = N(i);
      } else {
        fe->CalcVShape(Te, V);
        for (int i = 0; i < n; ++i) sv[i] = V(i, 0) * t[0] + V(i, 1) * t[1];
      }
      if (A) {
        const cplx b = beta(*g, s.elem, x, w, fam, t) * wl * weight(x[0]);
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < n; ++j) (*A)(i, j) += b * sv[i] * sv[j];
      }
      if (fport)
        for (int i = 0; i < n; ++i) (*fport)(i) += tsign * wl * sv[i];
    }
    for (int i = 0; i < n; ++i) {
      if (A)
        for (int j = 0; j < n; ++j) (*A)(i, j) *= sg[i] * sg[j];
      if (fport) (*fport)(i) *= sg[i];
    }
  }
};

Fw2DModel::Fw2DModel(spec::Design d) : d_(std::move(d)), p_(std::make_unique<Impl>()) {}
Fw2DModel::~Fw2DModel() = default;

void Fw2DModel::run(const SolveOptions& o) {
  solved_ = false;
  p_ = std::make_unique<Impl>();  // a rerun starts from a fresh state (port chains, split DOFs, stored solutions)
  Impl& P = *p_;
  Timer timer;
  const auto& fw = d_.fullwave;
  R_ = json::object();
  json& warnings = R_["warnings"] = json::array();
  AXF_REQUIRE(!d_.mesh.file.empty(), "a 2D full-wave model on an imported mesh needs mesh.file", "/mesh/file");
  AXF_REQUIRE(fw.enabled && fw.rf_only, "a 2D design on an imported mesh needs a fullwave section with rf_only: true",
              "/fullwave/rf_only", "Imported 2D meshes run the full-wave analysis only (no static/MQS solve).");
  AXF_REQUIRE(fw.boundary == "pec", "a 2D imported-mesh model is closed by its boundary groups: fullwave.boundary must "
              "be \"pec\"", "/fullwave/boundary");
  P.axi = d_.symmetry != "planar";
  P.D = fw.thickness_m;
  P.order = o.order > 0 ? o.order : d_.mesh.order;
  std::filesystem::create_directories(o.workdir);
  // ---- mesh and info
  {
    std::ifstream f(d_.mesh.file);
    AXF_REQUIRE(bool(f), "cannot open mesh.file '" + d_.mesh.file + "'", "/mesh/file");
  }
  P.mesh = std::make_unique<mfem::Mesh>(d_.mesh.file.c_str(), 1, 1);
  mfem::Mesh& mesh = *P.mesh;
  AXF_REQUIRE(mesh.Dimension() == 2 && mesh.SpaceDimension() == 2,
              strcat("mesh.file '", d_.mesh.file, "' is not a 2D mesh in the plane (dimension ", mesh.Dimension(),
                     ", space dimension ", mesh.SpaceDimension(), ")"), "/mesh/file");
  json info;
  {
    std::ifstream f(d_.mesh.info_file);
    AXF_REQUIRE(bool(f), "cannot open mesh.info_file '" + d_.mesh.info_file + "'", "/mesh/info_file");
    f >> info;
  }
  const int ne = mesh.GetNE();
  P.emat.assign(ne, nullptr);
  P.ereg.assign(ne, "");
  P.epml.assign(ne, -1);
  P.esrc.assign(ne, 0);
  std::map<int, int> region_of_attr;
  for (size_t k = 0; k < d_.regions.size(); ++k) region_of_attr[2 + static_cast<int>(k)] = static_cast<int>(k);
  for (int e = 0; e < ne; ++e) {
    const int a = mesh.GetAttribute(e);
    auto it = region_of_attr.find(a);
    AXF_REQUIRE(it != region_of_attr.end(),
                strcat("element ", e, " of mesh.file has attribute ", a, ", which is no region (region k owns 2 + k)"),
                "/regions");
    const auto& g = d_.regions[it->second];
    P.ereg[e] = g.name;
    P.emat[e] = &d_.materials.at(g.material);
  }
  // mesh check for a solver-failure message (worst elements), computed at the first failure only. The mesh is
  // always imported (mesh.file), so the remedy is about that file. thrown: an error, which writes no results.
  json mesh_check;
  mesh_quality::Summary mesh_summary;
  auto region_name = [&](int a) {
    auto it = region_of_attr.find(a);
    return it != region_of_attr.end() ? d_.regions[it->second].name : strcat("attribute ", a);
  };
  auto mesh_note = [&](bool thrown) {
    if (mesh_check.is_null()) {
      mesh_summary = mesh_quality::scan(mesh, 5);
      mesh_check = mesh_quality::to_json(mesh_summary, region_name);
    }
    mesh_quality::DescribeOptions opt;
    opt.three_d = false;
    opt.coords_2d = P.axi ? "(r, z)" : "(x, y)";
    opt.imported_file = d_.mesh.file;
    if (thrown) opt.listed_in.clear();
    return mesh_quality::describe(mesh_summary, region_name, opt);
  };
  double xmin = 1e300, xmax = -1e300;
  for (int v = 0; v < mesh.GetNV(); ++v) {
    xmin = std::min(xmin, mesh.GetVertex(v)[0]);
    xmax = std::max(xmax, mesh.GetVertex(v)[0]);
  }
  double ymin = 1e300, ymax = -1e300;
  for (int v = 0; v < mesh.GetNV(); ++v) {
    ymin = std::min(ymin, mesh.GetVertex(v)[1]);
    ymax = std::max(ymax, mesh.GetVertex(v)[1]);
  }
  const double size = std::max(xmax - xmin, ymax - ymin);
  if (P.axi)
    AXF_REQUIRE(xmin >= -1e-9 * size, strcat("axisymmetric mesh reaches r = ", xmin, " < 0 (the first coordinate is r)"),
                "/mesh/file");
  // boundary groups
  std::set<int> battrs;
  for (int be = 0; be < mesh.GetNBE(); ++be) battrs.insert(mesh.GetBdrAttribute(be));
  json gsum = json::array();
  for (const auto& gj : info.value("boundary_groups", json::array())) {
    Impl::Group g;
    g.attr = gj.at("attr").get<int>();
    g.name = gj.value("name", strcat("attr_", g.attr));
    g.type = gj.at("type").get<std::string>();
    g.j = gj;
    const std::string where = "/mesh/info_file boundary_groups '" + g.name + "'";
    AXF_REQUIRE(g.type == "pec" || g.type == "pmc" || g.type == "axis" || g.type == "port" || g.type == "impedance" ||
                    g.type == "sheet" || g.type == "scattering",
                "boundary group type '" + g.type + "' is not one of pec, pmc, axis, port, impedance, sheet, scattering",
                where);
    AXF_REQUIRE(battrs.count(g.attr), strcat("boundary group attribute ", g.attr, " has no boundary element"), where);
    if (g.type == "impedance" || g.type == "sheet") {
      if (gj.contains("Zs_ohm")) {
        AXF_REQUIRE(g.type == "impedance", "Zs_ohm is for impedance groups", where);
        g.has_Zs = true;
        g.Zs = cplx(gj.at("Zs_ohm").at(0).get<double>(), gj.at("Zs_ohm").at(1).get<double>());
        AXF_REQUIRE(g.Zs.real() >= 0 && std::abs(g.Zs) > 0, "Zs_ohm must have Re >= 0 (passive, e^{-i omega t})", where);
      } else {
        const std::string mn = gj.value("material", std::string());
        AXF_REQUIRE(d_.materials.count(mn), "impedance/sheet group needs a design material ('material') or Zs_ohm",
                    where);
        g.mat = &d_.materials.at(mn);
        AXF_REQUIRE(!is_pec_mat(g.mat), "the material of an impedance/sheet group must be linear (finite sigma)", where);
      }
      if (g.type == "sheet") {
        g.thickness = gj.value("thickness_m", 0.0);
        AXF_REQUIRE(g.thickness > 0, "a sheet group needs thickness_m > 0", where);
      }
    }
    if (g.type == "scattering") {
      g.wave = gj.value("wave", std::string("plane"));
      AXF_REQUIRE(g.wave == "plane" || g.wave == "spherical" || g.wave == "cylindrical",
                  "scattering wave must be plane, spherical or cylindrical", where);
      AXF_REQUIRE(!(g.wave == "spherical" && !P.axi), "a planar 2D design has no spherical scattering wave", where);
      if (gj.contains("center")) {
        g.center[0] = gj.at("center").at(0).get<double>();
        g.center[1] = gj.at("center").at(1).get<double>();
      }
    }
    if (g.type == "axis") AXF_REQUIRE(P.axi, "an axis group needs an axisymmetric design", where);
    gsum.push_back(gj);
    P.groups[g.attr] = g;
  }
  // every exterior boundary element must have a condition (an untyped one would silently be PMC)
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    const int a = mesh.GetBdrAttribute(be);
    if (P.groups.count(a)) continue;
    int e1, e2;
    mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(be), &e1, &e2);
    AXF_REQUIRE(e2 >= 0, strcat("exterior boundary attribute ", a, " has no boundary group in mesh.info_file"),
                "/mesh/info_file", "Every exterior boundary needs a type (pec, pmc, axis, port, impedance, scattering).");
  }
  // every exterior edge needs a boundary element (otherwise it would silently be a PMC wall)
  {
    std::set<int> bfaces;
    for (int be = 0; be < mesh.GetNBE(); ++be) bfaces.insert(mesh.GetBdrElementFaceIndex(be));
    int missing = 0;
    double where[2] = {0, 0};
    for (int f = 0; f < mesh.GetNumFaces(); ++f) {
      int e1, e2;
      mesh.GetFaceElements(f, &e1, &e2);
      if (e2 < 0 && !bfaces.count(f)) {
        if (missing++ == 0) {
          mfem::Array<int> vs;
          mesh.GetFaceVertices(f, vs);
          where[0] = mesh.GetVertex(vs[0])[0];
          where[1] = mesh.GetVertex(vs[0])[1];
        }
      }
    }
    AXF_REQUIRE(missing == 0, strcat(missing, " exterior edges of mesh.file have no boundary element (first near (",
                                     where[0], ", ", where[1], ")): their condition would silently be PMC"),
                "/mesh/file", "Write a boundary element (with a boundary-group attribute) for every exterior edge.");
  }
  // placement of the boundary groups: impedance / scattering / pmc / port-free walls on the exterior, sheets inside,
  // no Robin condition on a conductor element; interior PEC edges between field elements are conducting sheets
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    auto it = P.groups.find(mesh.GetBdrAttribute(be));
    if (it == P.groups.end()) continue;
    const auto& g = it->second;
    int e1, e2;
    const int fidx = mesh.GetBdrElementFaceIndex(be);
    mesh.GetFaceElements(fidx, &e1, &e2);
    const bool c1 = is_pec_mat(P.emat[e1]), c2 = e2 >= 0 && is_pec_mat(P.emat[e2]);
    const bool interior = e2 >= 0 && !c1 && !c2;   // field on both sides
    const std::string where = "/mesh/info_file boundary_groups '" + g.name + "'";
    if (g.type == "impedance" || g.type == "scattering" || g.type == "pmc")
      AXF_REQUIRE(!interior, "a " + g.type + " boundary must be exterior (field on one side only); group '" + g.name +
                  "' has field elements on both sides", where);
    if (g.type == "sheet")
      AXF_REQUIRE(interior, "a sheet boundary must have field elements on both sides; group '" + g.name + "'", where);
    if (g.type == "impedance" || g.type == "scattering" || g.type == "sheet")
      AXF_REQUIRE(!c1 && !c2, "a " + g.type + " boundary next to a conductor (PEC) element: group '" + g.name + "'",
                  where, "The conductor element's E = 0 would override the Robin condition; remove the element or the group.");
    if (g.type == "pec" && interior) P.sheet_faces.insert(fidx);
  }
  {
    int nports = 0;
    for (auto& [a, g] : P.groups) nports += g.type == "port";
    AXF_REQUIRE(nports <= 1 && (nports == 0 || (d_.pickup && !d_.pickup->port_boundary.empty())),
                strcat("mesh.info_file has ", nports, " port groups; the 2D engine drives one port, the pickup "
                       "(pickup.port_boundary): another port group would silently be an open circuit"),
                "/mesh/info_file", "Remove the extra port groups or give them a wall type.");
  }
  // PML layers
  {
    std::map<std::string, int> layer_of_region;
    for (size_t i = 0; i < fw.pml.size(); ++i) {
      pml::Layer L = pml::from_json(fw.pml[i], P.axi ? 1 : 2, strcat("/fullwave/pml/", i));
      std::string nm;
      for (const auto& r : fw.pml[i].at("regions")) {
        AXF_REQUIRE(!layer_of_region.count(r.get<std::string>()), "region '" + r.get<std::string>() + "' is in two PML layers",
                    strcat("/fullwave/pml/", i));
        layer_of_region[r.get<std::string>()] = static_cast<int>(i);
        nm += (nm.empty() ? "" : ",") + r.get<std::string>();
      }
      L.name = nm;
      P.layers.push_back(L);
    }
    for (int e = 0; e < ne; ++e)
      if (layer_of_region.count(P.ereg[e])) P.epml[e] = layer_of_region[P.ereg[e]];
    json lsum = json::array();
    for (size_t i = 0; i < P.layers.size(); ++i) {
      pml::Extents phys, reg;
      mfem::Array<int> vs;
      for (int e = 0; e < ne; ++e) {
        if (is_pec_mat(P.emat[e])) continue;
        if (P.epml[e] >= 0 && P.epml[e] != static_cast<int>(i)) continue;
        mesh.GetElementVertices(e, vs);
        for (int v : vs) {
          double q[3];
          P.layers[i].local(mesh.GetVertex(v), q);
          (P.epml[e] < 0 ? phys : reg).add(q);
        }
      }
      pml::configure(P.layers[i], phys, reg, 1e-7 * size);
      lsum.push_back(P.layers[i].summary());
    }
    if (!lsum.empty()) R_["fullwave"]["pml"] = lsum;
  }
  // source
  std::set<std::string> src_regions(fw.source_regions.begin(), fw.source_regions.end());
  if (!fw.source.is_null()) {
    P.source = std::make_unique<PrescribedMagneticSource>(PrescribedMagneticSource::from_json(fw.source));
    if (P.axi && P.source->is_uniform()) {
      const auto b = P.source->field_T({1.0, 0.0, 0.0});
      AXF_REQUIRE(b[0] == 0.0 && b[1] == 0.0, "a uniform source in an axisymmetric design must point along z",
                  "/fullwave/source");
    }
    for (int e = 0; e < ne; ++e)
      P.esrc[e] = !is_pec_mat(P.emat[e]) && P.epml[e] < 0 && (src_regions.empty() || src_regions.count(P.ereg[e]));
    if (P.axi) {
      // m = 0 fields couple to the azimuthal average of B0. Box sources by their definition: a sector's field is
      // phi-independent inside it (local components), so its average is that field times span / 360 (exact).
      // Other sources: tested for phi dependence (32 angles) at up to 400 source-element centroids spread over the
      // source elements; a phi-dependent source is averaged numerically, the number of angles doubled (64 ... 1024)
      // until the average changes by < 1e-6 at the test points.
      double sec[2];
      std::vector<int> se;
      for (int e = 0; e < ne; ++e)
        if (P.esrc[e]) se.push_back(e);
      std::vector<std::array<double, 2>> tp;
      const size_t stride = std::max<size_t>(1, se.size() / 400);
      for (size_t k = 0; k < se.size(); k += stride) {
        mfem::Vector c(2);
        mesh.GetElementTransformation(se[k])->Transform(mfem::Geometries.GetCenter(mesh.GetElementBaseGeometry(se[k])), c);
        tp.push_back({c(0), c(1)});
      }
      const std::string& kind = P.source->kind();
      if ((kind == "toroidal_box" || kind == "axial_box") && P.source->box_sector_deg(sec)) {
        const double span = sec[1] - sec[0];
        P.src_mode = 1;
        P.src_frac = std::min(span, 360.0) / 360.0;
        P.src_phi_mid = (sec[0] + 0.5 * std::min(span, 360.0)) * units::pi / 180.0;
        if (P.src_frac < 1.0)
          warnings.push_back(strcat("2D axisymmetric: the source sector [", sec[0], ", ", sec[1], "] deg couples to "
                                    "m = 0 fields through its azimuthal average: the sector field times ", P.src_frac));
      } else if (kind != "uniform" && kind != "toroidal_box" && kind != "axial_box") {
        double bmax = 0, dev = 0;
        for (const auto& c : tp) {
          std::array<double, 3> ref{};
          for (int k = 0; k < 32; ++k) {
            const double ph = 2 * units::pi * k / 32.0, cp = std::cos(ph), sp = std::sin(ph);
            const auto b = P.source->field_T({c[0] * cp, c[0] * sp, c[1]});
            const std::array<double, 3> loc{b[0] * cp + b[1] * sp, -b[0] * sp + b[1] * cp, b[2]};  // (r, phi, z)
            if (k == 0) ref = loc;
            for (int i = 0; i < 3; ++i) {
              bmax = std::max(bmax, std::fabs(loc[i]));
              dev = std::max(dev, std::fabs(loc[i] - ref[i]));
            }
          }
        }
        if (dev > 1e-9 * std::max(bmax, 1e-300)) {
          P.src_mode = 2;
          double change = 0;
          for (int n = 64; n <= 1024; n *= 2) {
            change = 0;
            double scale = 1e-300;
            for (const auto& c : tp) {
              const auto a1 = P.phi_average(c[0], c[1], n), a2 = P.phi_average(c[0], c[1], 2 * n);
              for (int i = 0; i < 3; ++i) {
                change = std::max(change, std::fabs(a2[i] - a1[i]));
                scale = std::max(scale, std::fabs(a2[i]));
              }
            }
            change /= std::max(scale, 1e-9 * std::max(bmax, 1e-300));
            P.src_navg = 2 * n;
            if (change < 1e-6) break;
          }
          warnings.push_back(strcat("2D axisymmetric: the prescribed source depends on phi (relative variation ",
                                    dev / std::max(bmax, 1e-300), "); its azimuthal average (", P.src_navg,
                                    " angles, last doubling changed it by ", change, "), the part that couples to "
                                    "m = 0 fields, is used"));
        }
      }
    }
  }
  if (!P.layers.empty()) {
    std::set<std::string> touched;
    for (int be = 0; be < mesh.GetNBE(); ++be) {
      auto it = P.groups.find(mesh.GetBdrAttribute(be));
      if (it == P.groups.end()) continue;
      const auto& t = it->second.type;
      if (t != "impedance" && t != "scattering" && t != "sheet") continue;
      int e1, e2;
      mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(be), &e1, &e2);
      if (P.epml[e1] >= 0 || (e2 >= 0 && P.epml[e2] >= 0)) touched.insert(it->second.name);
    }
    for (const auto& nm : touched)
      warnings.push_back("2D full wave: the Robin boundary '" + nm +
                         "' runs through a PML layer, where its term is not stretched (a small spurious reflection; "
                         "make it PEC inside the layer)");
  }
  // ---- spaces
  P.h1c = std::make_unique<mfem::H1_FECollection>(P.order, 2);
  P.ndc = std::make_unique<mfem::ND_FECollection>(P.order, 2);
  P.h1 = std::make_unique<mfem::FiniteElementSpace>(&mesh, P.h1c.get());
  P.nd = std::make_unique<mfem::FiniteElementSpace>(&mesh, P.ndc.get());
  const char* name_out = P.axi ? "TE" : "TM";
  const char* name_in = P.axi ? "TM" : "TE";
  json out = {{"status", "provisional"},
              {"engine", "fw2d (2D imported mesh, direct sparse LU)"},
              {"symmetry", P.axi ? "axisymmetric" : "planar"},
              {"order", P.order},
              {"families", {{name_out, P.axi ? "E = E_phi (out of plane), H1" : "E = E_z (out of plane), H1"},
                            {name_in, P.axi ? "E = (E_r, E_z) (in plane), Nedelec" : "E = (E_x, E_y) (in plane), Nedelec"}}},
              {"convention", "e^{-i omega t}; Z = V/I = R - i omega L (COMSOL's Zport is the complex conjugate); "
                             "V_oc = -i omega Psi; every quantity is the 3D one (axisymmetric: full revolution; planar: "
                             "thickness D)"},
              {"boundary_groups", gsum},
              {"elements", ne},
              {"dofs", {{name_out, P.h1->GetVSize()}, {name_in, P.nd->GetVSize()}}}};
  if (!P.axi) out["thickness_m"] = P.D;
  if (R_.contains("fullwave") && R_["fullwave"].contains("pml")) out["pml"] = R_["fullwave"]["pml"];

  // essential DOFs: driven in-plane family (Nedelec): PEC; eigen out family (H1, E): PEC, axis, impedance (PEC in
  // the eigenproblem); eigen dual in family (H1, H): PMC, axis, exterior (open) ports. PEC elements and DOFs touched
  // by no field element are removed as well.
  auto essential = [&](const std::string& fam, bool eig) {
    mfem::FiniteElementSpace& f = (fam == "in" && !eig) ? *P.nd : *P.h1;
    std::vector<char> ess(f.GetVSize(), 0), used(f.GetVSize(), 0);
    mfem::Array<int> raw;
    for (int e = 0; e < ne; ++e) {
      f.GetElementDofs(e, raw);
      for (int k = 0; k < raw.Size(); ++k) {
        const int i = raw[k] >= 0 ? raw[k] : -1 - raw[k];
        // E families: E_t = 0 on (and in) conductor elements; the dual H family has the natural condition on
        // conductor surfaces (DOFs inside conductors are removed as unused below)
        if (is_pec_mat(P.emat[e])) {
          if (!(fam == "in" && eig)) ess[i] = 1;
        } else {
          used[i] = 1;
        }
      }
    }
    for (int be = 0; be < mesh.GetNBE(); ++be) {
      auto it = P.groups.find(mesh.GetBdrAttribute(be));
      if (it == P.groups.end()) continue;
      const std::string& t = it->second.type;
      bool dir = false;
      if (!eig) {
        dir = t == "pec" || (fam == "out" && t == "axis");
      } else if (fam == "out") {
        dir = t == "pec" || t == "axis" || t == "impedance";
      } else {
        int e1, e2;
        mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(be), &e1, &e2);
        dir = t == "pmc" || t == "axis" || (t == "port" && e2 < 0);
      }
      if (!dir) continue;
      f.GetBdrElementDofs(be, raw);
      for (int k = 0; k < raw.Size(); ++k) ess[raw[k] >= 0 ? raw[k] : -1 - raw[k]] = 1;
    }
    for (size_t i = 0; i < ess.size(); ++i)
      if (!used[i]) ess[i] = 1;
    return ess;
  };

  // ---- frequency sweep
  if (!fw.frequencies.empty()) {
    AXF_REQUIRE(d_.pickup && d_.pickup->kind == spec::Pickup::Kind::Port && !d_.pickup->port_boundary.empty(),
                "a 2D frequency sweep needs a port pickup on a boundary (pickup.port_boundary)", "/pickup",
                "A COMSOL lumped port maps to {\"type\": \"port\", \"port_boundary\": \"lport1\"}.");
    // port: boundary elements, chain and orientation
    int pattr = -1;
    for (auto& [a, g] : P.groups)
      if (g.type == "port" && g.name == d_.pickup->port_boundary) pattr = a;
    AXF_REQUIRE(pattr > 0, "pickup.port_boundary '" + d_.pickup->port_boundary + "' is not a port group of mesh.info_file",
                "/pickup/port_boundary");
    std::map<int, std::vector<int>> vedges;
    std::vector<std::array<int, 2>> ev;
    for (int be = 0; be < mesh.GetNBE(); ++be) {
      if (mesh.GetBdrAttribute(be) != pattr) continue;
      mfem::Array<int> vs;
      mesh.GetBdrElementVertices(be, vs);
      P.port_be.push_back(be);
      ev.push_back({vs[0], vs[1]});
      vedges[vs[0]].push_back(static_cast<int>(P.port_be.size()) - 1);
      vedges[vs[1]].push_back(static_cast<int>(P.port_be.size()) - 1);
    }
    std::vector<int> ends;
    for (auto& [v, es] : vedges) {
      AXF_REQUIRE(es.size() <= 2, "the port boundary branches (a vertex on three port edges)", "/pickup/port_boundary");
      if (es.size() == 1) ends.push_back(v);
    }
    AXF_REQUIRE(ends.size() == 2, strcat("the port boundary must be one open chain of edges (found ", ends.size(),
                                         " chain ends)"), "/pickup/port_boundary");
    auto X = [&](int v) { return std::array<double, 2>{mesh.GetVertex(v)[0], mesh.GetVertex(v)[1]}; };
    int a0 = ends[0], b0 = ends[1];
    {
      const auto pa = X(a0), pb = X(b0);
      double dv[2] = {pb[0] - pa[0], pb[1] - pa[1]};
      double tw[2] = {d_.pickup->toward.x, d_.pickup->toward.y};
      if (tw[0] == 0 && tw[1] == 0) {
        if (std::fabs(dv[0]) >= std::fabs(dv[1])) tw[0] = 1; else tw[1] = 1;
      }
      if (dv[0] * tw[0] + dv[1] * tw[1] < 0) std::swap(a0, b0);
    }
    // walk the chain from a0: oriented vertex order of each edge
    P.port_sign.assign(P.port_be.size(), 1.0);
    std::vector<char> seen(P.port_be.size(), 0);
    int cur = a0;
    double length = 0;
    for (size_t step = 0; step < P.port_be.size(); ++step) {
      int k = -1;
      for (int c : vedges[cur])
        if (!seen[c]) k = c;
      AXF_REQUIRE(k >= 0, "the port boundary chain is broken", "/pickup/port_boundary");
      seen[k] = 1;
      const int nxt = ev[k][0] == cur ? ev[k][1] : ev[k][0];
      // tangent sign: compare the face parametrization with cur -> nxt
      Impl::Side s = P.side(P.port_be[k]);
      mfem::IntegrationPoint i0, i1;
      i0.x = 0.0;
      i1.x = 1.0;
      mfem::Vector x0(2), x1(2);
      s.ft->Transform(i0, x0);
      s.ft->Transform(i1, x1);
      const auto pc = X(cur), pn = X(nxt);
      const double dd = (x1(0) - x0(0)) * (pn[0] - pc[0]) + (x1(1) - x0(1)) * (pn[1] - pc[1]);
      P.port_sign[k] = dd >= 0 ? 1.0 : -1.0;
      length += std::hypot(pn[0] - pc[0], pn[1] - pc[1]);
      cur = nxt;
    }
    AXF_REQUIRE(cur == b0, "the port boundary chain does not end at its other end", "/pickup/port_boundary");
    // terminals on PEC
    std::set<int> pec_v;
    for (int be = 0; be < mesh.GetNBE(); ++be) {
      auto it = P.groups.find(mesh.GetBdrAttribute(be));
      if (it == P.groups.end() || (it->second.type != "pec" && it->second.type != "impedance")) continue;
      mfem::Array<int> vs;
      mesh.GetBdrElementVertices(be, vs);
      for (int v : vs) pec_v.insert(v);
    }
    const bool ta = pec_v.count(a0), tb = pec_v.count(b0);
    if (!ta || !tb)
      warnings.push_back(strcat("2D port '", d_.pickup->port_boundary, "': chain end(s) ", (!ta ? "start " : ""),
                                (!tb ? "end " : ""), "not on a PEC or impedance boundary group (a lumped port joins two conductors)"));
    const auto pa = X(a0), pb = X(b0);
    P.port_info = {{"name", d_.pickup->port_boundary}, {"edges", P.port_be.size()}, {"length_m", length},
                   {"from", {pa[0], pa[1]}}, {"to", {pb[0], pb[1]}}, {"terminals_on_pec", ta && tb},
                   {"feed", P.axi ? "K = I/(2 pi r) t across the port (uniform current per circle); V = -int t.E dl"
                                  : "K = I/D t across the port; V = -int t.E dl"},
                   {"family", name_in}};
    out["ports"] = json::array({P.port_info});
    // assemble pattern and solve
    std::vector<char> ess = essential("in", false);
    const int n = P.nd->GetVSize();
    std::vector<int> red(n, -1);
    int nf = 0;
    for (int i = 0; i < n; ++i)
      if (!ess[i]) red[i] = nf++;
    AXF_REQUIRE(nf > 0, "no free degrees of freedom in the in-plane family", "/fullwave");
    // port vector
    Eigen::VectorXd fport = Eigen::VectorXd::Zero(n);
    std::vector<int> idx;
    std::vector<double> sg;
    for (size_t k = 0; k < P.port_be.size(); ++k) {
      Eigen::VectorXd fe;
      P.boundary("in", P.port_be[k], 0.0, nullptr, nullptr, &fe, P.port_sign[k], idx, sg);
      for (size_t i = 0; i < idx.size(); ++i) fport(idx[i]) += fe(i);
    }
    double fport_free = 0;
    for (int i = 0; i < n; ++i)
      if (red[i] >= 0) fport_free += fport(i) * fport(i);
    AXF_REQUIRE(fport_free > 0, "the port has no free degrees of freedom (is it on a PEC boundary?)", "/pickup");
    // source vector (B0 . shape)
    Eigen::VectorXd bsrc = Eigen::VectorXd::Zero(n), bsrc_out = Eigen::VectorXd::Zero(P.h1->GetVSize());
    double Bin_max = 0, Bout_max = 0;
    if (P.source) {
      for (int e = 0; e < ne; ++e) {
        if (!P.esrc[e]) continue;
        mfem::Array<int> raw;
        P.nd->GetElementDofs(e, raw);
        signed_dofs(raw, idx, sg);
        const mfem::FiniteElement* fe = P.nd->GetFE(e);
        mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
        const auto& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * P.order + 4);
        mfem::DenseMatrix V(raw.Size(), 2);
        mfem::Vector Xq(2);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          const auto& ip = ir.IntPoint(q);
          T->SetIntPoint(&ip);
          T->Transform(ip, Xq);
          const auto B = P.source_at(Xq(0), Xq(1));
          const double b1 = B[0], b2 = P.axi ? B[2] : B[1], bn = P.axi ? B[1] : B[2];
          Bin_max = std::max(Bin_max, std::hypot(b1, b2));
          Bout_max = std::max(Bout_max, std::fabs(bn));
          fe->CalcVShape(*T, V);
          const double wq = ip.weight * T->Weight() * P.weight(Xq(0));
          for (int i = 0; i < raw.Size(); ++i) bsrc(idx[i]) += sg[i] * wq * (V(i, 0) * b1 + V(i, 1) * b2);
        }
      }
      out["source"] = P.source->summary();
      if (!src_regions.empty()) out["source_regions"] = fw.source_regions;
      if (Bout_max > 1e-12 * std::max(Bin_max, 1e-300))
        warnings.push_back(strcat("2D full wave: the prescribed source has an out-of-plane component (max ", Bout_max,
                                  " T) that does not couple to the in-plane port family (", name_in,
                                  ") and is dropped from Psi"));
    }
    // sparsity pattern from one assembly
    std::vector<Eigen::Triplet<cplx>> trip;
    auto assemble = [&](double w, SpMatC& A) {
      trip.clear();
      Eigen::MatrixXcd Ae;
      for (int e = 0; e < ne; ++e) {
        if (is_pec_mat(P.emat[e])) continue;
        P.element("in", e, w, Ae, idx, sg);
        for (size_t i = 0; i < idx.size(); ++i) {
          if (red[idx[i]] < 0) continue;
          for (size_t j = 0; j < idx.size(); ++j)
            if (red[idx[j]] >= 0) trip.emplace_back(red[idx[i]], red[idx[j]], Ae(i, j));
        }
      }
      for (int be = 0; be < mesh.GetNBE(); ++be) {
        auto it = P.groups.find(mesh.GetBdrAttribute(be));
        if (it == P.groups.end()) continue;
        const auto& t = it->second.type;
        if (t != "impedance" && t != "sheet" && t != "scattering") continue;
        P.boundary("in", be, w, &it->second, &Ae, nullptr, 1.0, idx, sg);
        for (size_t i = 0; i < idx.size(); ++i) {
          if (red[idx[i]] < 0) continue;
          for (size_t j = 0; j < idx.size(); ++j)
            if (red[idx[j]] >= 0) trip.emplace_back(red[idx[i]], red[idx[j]], Ae(i, j));
        }
      }
      A.resize(nf, nf);
      A.setFromTriplets(trip.begin(), trip.end());
      A.makeCompressed();
    };
    Eigen::SparseLU<SpMatC, Eigen::COLAMDOrdering<int>> lu;
    bool analyzed = false;
    json sweep = json::array();
    for (double f : fw.frequencies) {
      Timer ft;
      const double w = 2 * units::pi * f;
      SpMatC A;
      assemble(w, A);
      if (!analyzed) { lu.analyzePattern(A); analyzed = true; }
      lu.factorize(A);
      if (lu.info() != Eigen::Success) {
        throw Error(strcat("2D full wave: the sparse LU factorization failed at f = ", f,
                           " Hz (singular operator: a resonance of a lossless closed domain?)"),
                    "/fullwave/frequencies", mesh_note(true).substr(1));
      }
      VecC b(nf);
      for (int i = 0; i < n; ++i)
        if (red[i] >= 0) b(red[i]) = kI * w * fport(i);  // I = 1 A
      VecC x = lu.solve(b);
      const double res = (A * x - b).norm() / std::max(b.norm(), 1e-300);
      VecC u = VecC::Zero(n);
      for (int i = 0; i < n; ++i)
        if (red[i] >= 0) u(i) = x(red[i]);
      cplx ft_u = 0, bs_u = 0;
      for (int i = 0; i < n; ++i) { ft_u += fport(i) * u(i); bs_u += bsrc(i) * u(i); }
      const cplx Z = -ft_u;  // V = -int t.E dl, I = 1
      const cplx Psi = bs_u / (kI * w);
      // power by part: P = -Im(u^H A_part u) / (2 omega)
      std::map<std::string, double> p_region, p_group;
      double p_pml = 0;
      Eigen::MatrixXcd Ae;
      auto quad = [&](const Eigen::MatrixXcd& M) {
        VecC ue(idx.size());
        for (size_t i = 0; i < idx.size(); ++i) ue(i) = u(idx[i]);
        return -(ue.adjoint() * M * ue)(0, 0).imag() / (2 * w);
      };
      for (int e = 0; e < ne; ++e) {
        if (is_pec_mat(P.emat[e])) continue;
        P.element("in", e, w, Ae, idx, sg);
        const double pe = quad(Ae);
        if (P.epml[e] >= 0) p_pml += pe;
        else p_region[P.ereg[e]] += pe;
      }
      for (int be = 0; be < mesh.GetNBE(); ++be) {
        auto it = P.groups.find(mesh.GetBdrAttribute(be));
        if (it == P.groups.end()) continue;
        const auto& t = it->second.type;
        if (t != "impedance" && t != "sheet" && t != "scattering") continue;
        P.boundary("in", be, w, &it->second, &Ae, nullptr, 1.0, idx, sg);
        p_group[it->second.name] += quad(Ae);
      }
      const double Pin = 0.5 * Z.real();
      double Psum = p_pml;
      json pr = json::object(), pg = json::object();
      for (auto& [k, v] : p_region) { pr[k] = v; Psum += v; }
      for (auto& [k, v] : p_group) { pg[k] = v; Psum += v; }
      const double L = -Z.imag() / w;
      json row = {{"f_Hz", f},
                  {"Z_re_ohm", Z.real()},
                  {"Z_im_ohm", Z.imag()},
                  {"L_H", L},
                  {"Psi_re", Psi.real()},
                  {"Psi_im", Psi.imag()},
                  {"Psi_abs", std::abs(Psi)},
                  {"V_oc_re", (-kI * w * Psi).real()},
                  {"V_oc_im", (-kI * w * Psi).imag()},
                  {"power_W_per_A2", {{"input", Pin}, {"bulk_by_region", pr}, {"boundary_by_group", pg}, {"pml", p_pml},
                                      {"balance_rel", std::fabs(Pin - Psum) / std::max(std::fabs(Pin), 1e-300)},
                                      {"definition", "-Im(u^H A_part u)/(2 omega); input 1/2 Re Z |I|^2"}}},
                  {"solver", {{"dofs", nf}, {"relative_residual", res}, {"seconds", ft.seconds()}}}};
      if (res > 1e-8)
        warnings.push_back(strcat("2D full wave at f = ", f, " Hz: relative residual ", res, " (ill-conditioned).",
                                  mesh_note(false)));
      sweep.push_back(row);
      P.sweep_w.push_back(w);
      P.sweep_u.push_back(u);
      log(1, strcat("  fw2d: f = ", f, " Hz  Z = ", Z.real(), " ", (Z.imag() < 0 ? "-" : "+"), " i ", std::fabs(Z.imag()),
                    " ohm  (", ft.seconds(), " s)"));
    }
    P.sweep_family = "in";
    out["sweep"] = sweep;
  }

  // ---- eigenmodes (closed, lossless; impedance walls as PEC + perturbative wall Q)
  if (fw.modes > 0) {
    for (auto& [a, g] : P.groups)
      AXF_REQUIRE(g.type != "scattering" && g.type != "sheet",
                  "eigenmodes of a 2D model with a scattering or sheet boundary are not supported (complex modes)",
                  "/fullwave/modes", "Remove fullwave.modes, or close the model with PEC/PMC/impedance walls.");
    AXF_REQUIRE(P.layers.empty(), "eigenmodes with PML layers are not supported (complex modes)", "/fullwave/modes");
    std::vector<std::string> fams = fw.families;
    if (fams.empty()) fams = {name_out, name_in};
    json eig = json::object();
    for (const auto& fname : fams) {
      const std::string fam = fname == name_out ? "out" : "in";
      std::vector<char> ess = essential(fam, true);
      int n = P.h1->GetVSize();
      int nsplit = 0;
      if (fam == "in" && !P.sheet_faces.empty()) {
        // split the H DOFs on interior PEC sheets into one copy per fan of elements around them
        std::vector<std::vector<int>> hdv(ne);
        mfem::Array<int> raw;
        for (int e = 0; e < ne; ++e) {
          P.h1->GetElementDofs(e, raw);
          hdv[e].assign(raw.begin(), raw.end());
        }
        std::set<int> sdofs;
        for (int be = 0; be < mesh.GetNBE(); ++be)
          if (P.sheet_faces.count(mesh.GetBdrElementFaceIndex(be))) {
            P.h1->GetBdrElementDofs(be, raw);
            for (int d0 : raw) sdofs.insert(d0);
          }
        std::map<int, std::vector<int>> dof_elems;
        for (int e = 0; e < ne; ++e) {
          if (is_pec_mat(P.emat[e])) continue;
          for (int d0 : hdv[e])
            if (sdofs.count(d0)) dof_elems[d0].push_back(e);
        }
        std::vector<int> src;
        for (auto& [d0, els] : dof_elems) {
          // union-find over elements sharing an edge that is not a sheet
          std::vector<int> par(els.size());
          for (size_t a = 0; a < els.size(); ++a) par[a] = static_cast<int>(a);
          std::function<int(int)> find = [&](int a) { return par[a] == a ? a : par[a] = find(par[a]); };
          std::vector<std::set<int>> eds(els.size());
          for (size_t a = 0; a < els.size(); ++a) {
            mfem::Array<int> es, os;
            mesh.GetElementEdges(els[a], es, os);
            eds[a].insert(es.begin(), es.end());
          }
          for (size_t a = 0; a < els.size(); ++a)
            for (size_t b = a + 1; b < els.size(); ++b)
              for (int x : eds[a])
                if (eds[b].count(x) && !P.sheet_faces.count(x)) { par[find(static_cast<int>(a))] = find(static_cast<int>(b)); break; }
          std::map<int, int> copy_of_root;
          for (size_t a = 0; a < els.size(); ++a) {
            const int r = find(static_cast<int>(a));
            if (!copy_of_root.count(r)) {
              if (copy_of_root.empty()) copy_of_root[r] = d0;
              else { copy_of_root[r] = n + static_cast<int>(src.size()); src.push_back(d0); }
            }
            for (auto& x : hdv[els[a]])
              if (x == d0) x = copy_of_root[r];
          }
        }
        nsplit = static_cast<int>(src.size());
        for (int d0 : src) ess.push_back(ess[d0]);
        n += nsplit;
        P.hd[fam] = std::move(hdv);
        P.hn[fam] = n;
      }
      std::vector<int> red(n, -1);
      int nf = 0;
      for (int i = 0; i < n; ++i)
        if (!ess[i]) red[i] = nf++;
      AXF_REQUIRE(nf > 0, "2D eigenmodes: no free degrees of freedom in family " + fname, "/fullwave/modes");
      std::vector<Eigen::Triplet<double>> tk, tm, tkf, tmf;
      Eigen::MatrixXd Ke, Me;
      std::vector<int> idx;
      for (int e = 0; e < ne; ++e) {
        if (is_pec_mat(P.emat[e])) continue;
        P.element_eig(fam, e, Ke, Me, idx);
        if (fam == "out")
          for (size_t i = 0; i < idx.size(); ++i)
            for (size_t j = 0; j < idx.size(); ++j) {
              tkf.emplace_back(idx[i], idx[j], Ke(i, j));
              tmf.emplace_back(idx[i], idx[j], Me(i, j));
            }
        for (size_t i = 0; i < idx.size(); ++i) {
          if (red[idx[i]] < 0) continue;
          for (size_t j = 0; j < idx.size(); ++j) {
            if (red[idx[j]] < 0) continue;
            tk.emplace_back(red[idx[i]], red[idx[j]], Ke(i, j));
            tm.emplace_back(red[idx[i]], red[idx[j]], Me(i, j));
          }
        }
      }
      SpMat K(nf, nf), M(nf, nf);
      K.setFromTriplets(tk.begin(), tk.end());
      M.setFromTriplets(tm.begin(), tm.end());
      // Static modes: a connected field component with no Dirichlet DOF (bounded only by natural walls: PEC for the
      // dual H family, PMC for E) has a zero-frequency field (H_phi = G/r in a coaxial region, constant H_z or E_z in
      // a closed planar region) that the polynomial space represents only approximately (a spurious low mode).
      // Faraday/Ampere around the boundary of the meridional cross-section (where the tangential field of the other
      // kind vanishes) gives the exact constraint int beta u dA = 0 (unweighted: the flux through the cross-section)
      // for every true mode; it is imposed with a Lagrange multiplier in the shift-invert solves.
      std::vector<Eigen::VectorXd> cons;
      {
        std::vector<int> comp(ne, -1);
        int nc = 0;
        for (int e0 = 0; e0 < ne; ++e0) {
          if (comp[e0] >= 0 || is_pec_mat(P.emat[e0])) continue;
          std::vector<int> st{e0}, members;
          comp[e0] = nc;
          bool pinned = false;
          while (!st.empty()) {
            const int e = st.back();
            st.pop_back();
            members.push_back(e);
            mfem::Array<int> raw;
            P.h1dofs(fam, e, raw);
            for (int k = 0; k < raw.Size(); ++k)
              if (ess[raw[k]]) pinned = true;
            mfem::Array<int> fs, ors;
            mesh.GetElementEdges(e, fs, ors);
            for (int fidx : fs) {
              if (P.sheet_faces.count(fidx)) continue;  // a conducting sheet separates the two sides
              int e1, e2;
              mesh.GetFaceElements(fidx, &e1, &e2);
              for (int nb : {e1, e2})
                if (nb >= 0 && comp[nb] < 0 && !is_pec_mat(P.emat[nb])) { comp[nb] = nc; st.push_back(nb); }
            }
          }
          ++nc;
          if (pinned) continue;
          Eigen::VectorXd c = Eigen::VectorXd::Zero(nf);
          for (int e : members) {
            mfem::Array<int> raw;
            P.h1dofs(fam, e, raw);
            const mfem::FiniteElement* fe = P.h1->GetFE(e);
            mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
            const auto& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * P.order + 2);
            mfem::Vector N(raw.Size());
            const double beta = fam == "out" ? epsc_of(P.emat[e], 0.0).real() : mu_of(P.emat[e]);
            for (int q = 0; q < ir.GetNPoints(); ++q) {
              T->SetIntPoint(&ir.IntPoint(q));
              fe->CalcPhysShape(*T, N);
              for (int k = 0; k < raw.Size(); ++k)
                if (red[raw[k]] >= 0) c(red[raw[k]]) += beta * N(k) * ir.IntPoint(q).weight * T->Weight();
            }
          }
          cons.push_back(c / c.norm());
        }
      }
      const int ncons = static_cast<int>(cons.size());
      // shift: below the spectrum (lowest modes) or about target_f_Hz
      const double sigma = fw.target_f_Hz > 0 ? std::pow(2 * units::pi * fw.target_f_Hz, 2) : 0.0;
      double shift = sigma;
      if (sigma == 0) {
        // slightly below zero: -1e-6 x the median diagonal ratio K_ii / M_ii (~ (1/h)^2)
        std::vector<double> rat;
        for (int i = 0; i < nf; ++i)
          if (M.coeff(i, i) > 0) rat.push_back(K.coeff(i, i) / M.coeff(i, i));
        std::nth_element(rat.begin(), rat.begin() + rat.size() / 2, rat.end());
        shift = rat.empty() ? -1.0 : -1e-6 * rat[rat.size() / 2];
      }
      SpMat S = K - shift * M;
      if (ncons > 0) {  // bordered system [S C; C^T 0]
        std::vector<Eigen::Triplet<double>> ts;
        for (int k = 0; k < S.outerSize(); ++k)
          for (SpMat::InnerIterator it(S, k); it; ++it) ts.emplace_back(it.row(), it.col(), it.value());
        const double scale = S.diagonal().cwiseAbs().maxCoeff();
        for (int c = 0; c < ncons; ++c)
          for (int i = 0; i < nf; ++i)
            if (cons[c](i) != 0) {
              ts.emplace_back(i, nf + c, scale * cons[c](i));
              ts.emplace_back(nf + c, i, scale * cons[c](i));
            }
        S.resize(nf + ncons, nf + ncons);
        S.setFromTriplets(ts.begin(), ts.end());
      }
      Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>> slu;
      slu.compute(S);
      AXF_REQUIRE(slu.info() == Eigen::Success, "2D eigenmodes: the shifted factorization failed", "/fullwave/modes");
      json diag;
      auto inv = [&](const Eigen::VectorXd& b) -> Eigen::VectorXd {
        if (ncons == 0) return slu.solve(b);
        Eigen::VectorXd bb = Eigen::VectorXd::Zero(nf + ncons);
        bb.head(nf) = b;
        return slu.solve(bb).head(nf);
      };
      auto ms = axi::lanczos(M, inv, shift, fw.modes, diag, sigma == 0);
      if (static_cast<int>(ms.size()) < fw.modes)
        warnings.push_back(strcat("2D eigenmodes (", fname, "): ", ms.size(), " of ", fw.modes, " modes converged"));
      // out family: wall current by consistent-flux recovery. The weak residual (K - lambda M) u at the Dirichlet
      // (PEC / impedance wall) DOFs is -oint nu (n x curl E) N_i w dl = oint g N_i w dl; solving the boundary mass
      // system for g gives |H_t| = |g| / omega to O(h^{2p}) (a one-sided curl converges only as O(h^p)).
      SpMat Kf, Mf;
      std::vector<int> wall_dofs;
      std::vector<int> wpos;
      Eigen::SimplicialLDLT<SpMat> mb_ldlt;
      bool have_mb = false;
      if (fam == "out") {
        Kf.resize(n, n);
        Mf.resize(n, n);
        Kf.setFromTriplets(tkf.begin(), tkf.end());
        Mf.setFromTriplets(tmf.begin(), tmf.end());
        wpos.assign(n, -1);
        std::vector<int> wbe;
        for (int be = 0; be < mesh.GetNBE(); ++be) {
          auto it = P.groups.find(mesh.GetBdrAttribute(be));
          if (it == P.groups.end() || (it->second.type != "impedance" && it->second.type != "pec")) continue;
          wbe.push_back(be);
        }
        bool any_imp = false;
        for (auto& [a, g] : P.groups) any_imp = any_imp || g.type == "impedance";
        if (any_imp) {
          std::vector<Eigen::Triplet<double>> tb;
          for (int be : wbe) {
            Impl::Side sd = P.side(be);
            mfem::Array<int> raw, braw;
            P.h1dofs(fam, sd.elem, raw);
            P.h1->GetBdrElementDofs(be, braw);
            std::set<int> on(braw.begin(), braw.end());
            for (int d0 : braw)
              if (wpos[d0] < 0) { wpos[d0] = static_cast<int>(wall_dofs.size()); wall_dofs.push_back(d0); }
            const mfem::FiniteElement* fe = P.h1->GetFE(sd.elem);
            const auto& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * P.order + 4);
            mfem::Vector N(raw.Size()), Xq(2);
            for (int q = 0; q < ir.GetNPoints(); ++q) {
              const auto& ip = ir.IntPoint(q);
              sd.ft->SetAllIntPoints(&ip);
              mfem::ElementTransformation& Te = sd.use2 ? *sd.ft->Elem2 : *sd.ft->Elem1;
              sd.ft->Transform(ip, Xq);
              const mfem::DenseMatrix& J = sd.ft->Jacobian();
              const double wl = ip.weight * std::hypot(J(0, 0), J(1, 0)) * P.weight(Xq(0));
              fe->CalcPhysShape(Te, N);
              for (int i = 0; i < raw.Size(); ++i) {
                if (!on.count(raw[i])) continue;
                for (int j = 0; j < raw.Size(); ++j)
                  if (on.count(raw[j])) tb.emplace_back(wpos[raw[i]], wpos[raw[j]], wl * N(i) * N(j));
              }
            }
          }
          // DOFs on the axis carry zero weight (w = 2 pi r = 0): regularize their diagonal
          const int nw = static_cast<int>(wall_dofs.size());
          SpMat Mb(nw, nw);
          Mb.setFromTriplets(tb.begin(), tb.end());
          double dmax = 0;
          for (int i = 0; i < nw; ++i) dmax = std::max(dmax, Mb.coeff(i, i));
          for (int i = 0; i < nw; ++i)
            if (Mb.coeff(i, i) <= 1e-14 * dmax) Mb.coeffRef(i, i) += 1e-12 * dmax;
          mb_ldlt.compute(Mb);
          have_mb = mb_ldlt.info() == Eigen::Success;
          if (!have_mb) warnings.push_back("2D eigenmodes: wall-current recovery failed; one-sided curl used for H_t");
        }
      }
      json list = json::array();
      const int dropped = ncons;
      for (auto& m : ms) {
        if (static_cast<int>(list.size()) >= fw.modes) break;
        Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; ++i)
          if (red[i] >= 0) u(i) = m.x[red[i]];
        const double w = m.omega;
        // true residual of the reduced problem (constraint directions removed)
        double res_true = 0;
        {
          Eigen::VectorXd xr = Eigen::Map<const Eigen::VectorXd>(m.x.data(), nf);
          Eigen::VectorXd kx = K * xr, r = kx - w * w * (M * xr);
          for (const auto& c : cons) r -= c * c.dot(r);
          res_true = r.norm() / std::max(kx.norm(), 1e-300);
          if (res_true > 1e-6)
            warnings.push_back(strcat("2D eigenmode ", fname, " at ", w / (2 * units::pi), " Hz: residual ", res_true));
        }
        Eigen::VectorXd gwall;
        if (have_mb) {
          Eigen::VectorXd rr = Kf * u - w * w * (Mf * u), rw(wall_dofs.size());
          for (size_t i = 0; i < wall_dofs.size(); ++i) rw(i) = rr(wall_dofs[i]);
          Eigen::VectorXd gw = mb_ldlt.solve(rw);
          gwall = Eigen::VectorXd::Zero(n);
          for (size_t i = 0; i < wall_dofs.size(); ++i) gwall(wall_dofs[i]) = gw(i);
        }
        // energies and perturbative losses (u is M-normalized: u^T M u = 1)
        double U = 0, Pd = 0;
        std::map<std::string, double> pw;
        for (int e = 0; e < ne; ++e) {
          if (is_pec_mat(P.emat[e])) continue;
          mfem::Array<int> raw;
          P.h1dofs(fam, e, raw);
          const mfem::FiniteElement* fe = P.h1->GetFE(e);
          mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
          const auto& ir = mfem::IntRules.Get(fe->GetGeomType(), 2 * P.order + 4);
          mfem::Vector N(raw.Size()), Xq(2);
          mfem::DenseMatrix dN(raw.Size(), 2);
          const cplx ec = epsc_of(P.emat[e], w);
          for (int q = 0; q < ir.GetNPoints(); ++q) {
            const auto& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            T->Transform(ip, Xq);
            fe->CalcPhysShape(*T, N);
            fe->CalcPhysDShape(*T, dN);
            double val = 0, bb[2] = {0, 0};
            for (int i = 0; i < raw.Size(); ++i) {
              val += u(raw[i]) * N(i);
              double bi[2];
              P.bvec(N, dN, Xq(0), i, bi);
              bb[0] += u(raw[i]) * bi[0];
              bb[1] += u(raw[i]) * bi[1];
            }
            const double wq = ip.weight * T->Weight() * P.weight(Xq(0));
            // |E|^2: out: val^2; dual: |curl H|^2 / (omega eps')^2
            const double E2 = fam == "out" ? val * val : (bb[0] * bb[0] + bb[1] * bb[1]) / std::pow(w * ec.real(), 2);
            U += 0.5 * ec.real() * E2 * wq;  // time-averaged total energy = 2 x electric
            Pd += 0.5 * w * ec.imag() * E2 * wq;
          }
        }
        for (int be = 0; be < mesh.GetNBE(); ++be) {
          auto it = P.groups.find(mesh.GetBdrAttribute(be));
          if (it == P.groups.end() || it->second.type != "impedance") continue;
          const auto& g = it->second;
          Impl::Side s = P.side(be);
          mfem::Array<int> raw;
          P.h1dofs(fam, s.elem, raw);
          const mfem::FiniteElement* fe = P.h1->GetFE(s.elem);
          const auto& ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * P.order + 4);
          mfem::Vector N(raw.Size()), Xq(2);
          mfem::DenseMatrix dN(raw.Size(), 2);
          const cplx Zs = g.has_Zs ? g.Zs : std::sqrt(mu_of(g.mat) / epsc_of(g.mat, w));
          for (int q = 0; q < ir.GetNPoints(); ++q) {
            const auto& ip = ir.IntPoint(q);
            s.ft->SetAllIntPoints(&ip);
            mfem::ElementTransformation& Te = s.use2 ? *s.ft->Elem2 : *s.ft->Elem1;
            s.ft->Transform(ip, Xq);
            const mfem::DenseMatrix& J = s.ft->Jacobian();
            const double len = std::hypot(J(0, 0), J(1, 0));
            const double t[2] = {J(0, 0) / len, J(1, 0) / len};
            fe->CalcPhysShape(Te, N);
            fe->CalcPhysDShape(Te, dN);
            double val = 0, bb[2] = {0, 0}, gq = 0;
            for (int i = 0; i < raw.Size(); ++i) {
              val += u(raw[i]) * N(i);
              if (have_mb) gq += gwall(raw[i]) * N(i);
              double bi[2];
              P.bvec(N, dN, Xq(0), i, bi);
              bb[0] += u(raw[i]) * bi[0];
              bb[1] += u(raw[i]) * bi[1];
            }
            // |H_t|: out family from the recovered wall current g (|H_t| = |g| / omega), else the one-sided
            // curl E / (i omega mu) along t; dual: H_t = u
            const double mu = mu_of(P.emat[s.elem]);
            const double Ht = fam == "out" ? (have_mb ? gq / w : (bb[0] * t[0] + bb[1] * t[1]) / (w * mu)) : val;
            pw[g.name] += 0.5 * Zs.real() * Ht * Ht * ip.weight * len * P.weight(Xq(0));
          }
        }
        double Pwall = 0;
        json pwj = json::object();
        for (auto& [k, v] : pw) { Pwall += v; pwj[k] = v; }
        json row = {{"f_Hz", w / (2 * units::pi)}, {"omega", w}, {"residual_true", res_true},
                    {"stored_energy_J", U}};
        if (Pwall > 0) {
          row["Q_wall"] = w * U / Pwall;
          row["wall_power_W"] = pwj;
          row["Q_wall_note"] = "perturbative: impedance walls are PEC in the eigenproblem; P = 1/2 oint Re(Z_s) |H_t|^2";
        }
        if (Pd > 0) row["Q_dielectric"] = w * U / Pd;
        list.push_back(row);
        Impl::Mode md;
        md.family = fam;
        md.name = fname;
        md.omega = w;
        md.u = u;
        P.modes.push_back(md);
      }
      eig[fname] = {{"modes", list}, {"dofs", nf}, {"static_mode_constraints", dropped}, {"sheet_split_dofs", nsplit}, {"lanczos", diag},
                    {"formulation", fam == "out" ? "E out of plane (H1), PEC/impedance: E = 0"
                                                 : "dual H out of plane (H1), PMC/axis/open exterior port: H = 0"}};
      if (fam == "in" && !P.port_be.empty())
        eig[fname]["note"] = "lumped ports are open circuits in the eigenproblem (no reference impedance load)";
    }
    out["eigenmodes"] = eig;
    bool lossy = false;
    for (int e = 0; e < ne; ++e)
      lossy = lossy || (!is_pec_mat(P.emat[e]) && (P.emat[e]->tan_delta > 0 || P.emat[e]->sigma_S_m > 0));
    if (lossy) warnings.push_back("2D eigenmodes are computed with the lossless parts of the materials; dielectric "
                                  "losses enter only Q_dielectric (perturbative)");
  }
  out["seconds"] = timer.seconds();
  R_["fullwave"] = out;
  R_["mesh"] = {{"file", d_.mesh.file}, {"info", info}};
  if (!mesh_check.is_null()) R_["mesh"]["info"]["worst_elements"] = mesh_check;
  std::ofstream(std::filesystem::path(o.workdir) / "results.json") << R_.dump(2);
  solved_ = true;
}

json Fw2DModel::sample(const std::string& what, const std::string& family, int index, const std::vector<double>& x1,
                       const std::vector<double>& x2) const {
  const Impl& P = *p_;
  AXF_REQUIRE(solved_, "the model is not solved");
  AXF_REQUIRE(x1.size() == x2.size(), "x1 and x2 must have the same length");
  const int np = static_cast<int>(x1.size());
  mfem::DenseMatrix pts(2, np);
  for (int k = 0; k < np; ++k) { pts(0, k) = x1[k]; pts(1, k) = x2[k]; }
  mfem::Array<int> eid;
  mfem::Array<mfem::IntegrationPoint> ips;
  P.mesh->FindPoints(pts, eid, ips, false);
  json E = json::array(), H = json::array(), valid = json::array();
  const Impl::Mode* md = nullptr;
  double w = 0;
  const VecC* us = nullptr;
  if (what == "sweep") {
    AXF_REQUIRE(index >= 0 && index < static_cast<int>(P.sweep_u.size()), "sweep index out of range");
    us = &P.sweep_u[index];
    w = P.sweep_w[index];
  } else {
    int k = 0;
    for (const auto& m : P.modes)
      if (m.name == family) {
        if (k == index) md = &m;
        ++k;
      }
    AXF_REQUIRE(md, "no mode " + family + " #" + std::to_string(index));
    w = md->omega;
  }
  for (int k = 0; k < np; ++k) {
    const int e = eid[k];
    std::array<cplx, 3> Ev{0, 0, 0}, Hv{0, 0, 0};
    const bool ok = e >= 0 && !is_pec_mat(P.emat[e]);
    if (ok) {
      mfem::ElementTransformation* T = P.mesh->GetElementTransformation(e);
      T->SetIntPoint(&ips[k]);
      const double r = x1[k];
      const double mu = mu_of(P.emat[e]);
      if (us) {  // in-plane driven field (Nedelec)
        mfem::Array<int> raw;
        P.nd->GetElementDofs(e, raw);
        std::vector<int> idx;
        std::vector<double> sg;
        signed_dofs(raw, idx, sg);
        const mfem::FiniteElement* fe = P.nd->GetFE(e);
        mfem::DenseMatrix V(raw.Size(), 2), C(raw.Size(), 1);
        fe->CalcVShape(*T, V);
        fe->CalcPhysCurlShape(*T, C);
        cplx c = 0;
        for (int i = 0; i < raw.Size(); ++i) {
          const cplx ui = sg[i] * (*us)(idx[i]);
          Ev[0] += ui * V(i, 0);
          Ev[1] += ui * V(i, 1);
          c += ui * C(i, 0);
        }
        // planar: B_z = c / (i omega); axisymmetric: B_phi = -c / (i omega)
        const cplx Bn = (P.axi ? -c : c) / (kI * w);
        if (P.epml[e] < 0) {
          Hv[2] = Bn / mu;
        } else {  // H = nu' B in the stretched medium
          cplx nu[9], eps[9];
          const double xx[2] = {x1[k], x2[k]};
          P.coeffs(e, xx, w, nu, eps);
          Hv[2] = nu[8] * Bn;
        }
      } else {
        mfem::Array<int> raw;
        P.h1dofs(md->family, e, raw);
        const mfem::FiniteElement* fe = P.h1->GetFE(e);
        mfem::Vector N(raw.Size());
        mfem::DenseMatrix dN(raw.Size(), 2);
        fe->CalcPhysShape(*T, N);
        fe->CalcPhysDShape(*T, dN);
        double val = 0, d1 = 0, d2 = 0;
        for (int i = 0; i < raw.Size(); ++i) {
          val += md->u(raw[i]) * N(i);
          d1 += md->u(raw[i]) * dN(i, 0);
          d2 += md->u(raw[i]) * dN(i, 1);
        }
        // in-plane curl of u e_n; axisymmetric on the axis (u = 0 there): u/r -> du/dr
        double bb[2];
        if (P.axi) {
          bb[0] = -d2;
          bb[1] = d1 + (r > 0 ? val / r : d1);
        } else {
          bb[0] = d2;
          bb[1] = -d1;
        }
        if (md->family == "out") {  // E = u e_n (real), H = curl E / (i omega mu)
          Ev[2] = val;
          Hv[0] = bb[0] / (kI * w * mu);
          Hv[1] = bb[1] / (kI * w * mu);
        } else {  // dual: H = u e_n, E = curl H / (-i omega eps)
          const double ep = epsc_of(P.emat[e], 0.0).real();
          Hv[2] = val;
          Ev[0] = bb[0] / (-kI * w * ep);
          Ev[1] = bb[1] / (-kI * w * ep);
        }
      }
    }
    json ej = json::array(), hj = json::array();
    for (int c = 0; c < 3; ++c) {
      ej.push_back({Ev[c].real(), Ev[c].imag()});
      hj.push_back({Hv[c].real(), Hv[c].imag()});
    }
    E.push_back(ej);
    H.push_back(hj);
    valid.push_back(ok ? 1 : 0);
  }
  return {{"E", E}, {"H", H}, {"valid", valid}, {"omega", w},
          {"components", P.axi ? json::array({"r", "z", "phi"}) : json::array({"x", "y", "z"})},
          {"normalization", us ? "per port ampere" : "eigenmode, u^T M u = 1 (H1 family variable)"}};
}

}  // namespace axf
