// Python boundary of the 3D fluids API (beta):
// axionfem._core._fluids.Mesh3D and solve_stokes_3d, wrapped by python/axionfem/fluids/stokes3d.py.
//
//  * Mesh3D.from_arrays: tetrahedra / hexahedra (basix vertex order), boundary faces with attributes, optional curved
//    order-2 geometry nodes (basix node order); the serial mesh lives on every rank.
//  * solve_stokes_3d: partitions the mesh over the communicator (METIS, deterministic), solves on the ParMesh, and keeps
//    for every LOCAL element its INPUT id and the affine map between the input reference cell and MFEM's.
//  * Solution3D evaluates on the local elements only, at INPUT reference points: geometry, velocity (+ gradient),
//    pressure; element_ids() are the input ids. Every rank covers its own elements: together exactly once.
//  * diagnostics add comm_size (of the solve) and shared_dofs (velocity + pressure DOFs a rank holds without owning
//    them, summed over the ranks: 0 for an unpartitioned solve).
// Coefficients are vectorized Python callables f(X, Y, Z) (arrays) or constants, tabulated once on exactly the
// quadrature rules the solver uses (domain: viscous and load rules; boundary: the boundary rule).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "axionfem/core.hpp"
#include "axionfem/fluids/meshes3d.hpp"
#include "axionfem/fluids/stokes.hpp"
#include "axionfem/sym_solver.hpp"

namespace nb = nanobind;
using namespace axf::fluids;

namespace {

using DArr = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using IArr = nb::ndarray<const int, nb::c_contig, nb::device::cpu>;

nb::ndarray<nb::numpy, double> np3(std::vector<double> v, std::vector<size_t> shape) {
  auto* data = new std::vector<double>(std::move(v));
  nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  return nb::ndarray<nb::numpy, double>(data->data(), shape.size(), shape.data(), owner);
}
nb::ndarray<nb::numpy, long> np3l(std::vector<long> v, std::vector<size_t> shape) {
  auto* data = new std::vector<long>(std::move(v));
  nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<long>*>(p); });
  return nb::ndarray<nb::numpy, long>(data->data(), shape.size(), shape.data(), owner);
}
nb::ndarray<nb::numpy, int> np3i(std::vector<int> v, std::vector<size_t> shape) {
  auto* data = new std::vector<int>(std::move(v));
  nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<int>*>(p); });
  return nb::ndarray<nb::numpy, int>(data->data(), shape.size(), shape.data(), owner);
}

void ensure_mpi3() {
  int init = 0;
  MPI_Initialized(&init);
  if (!init) mfem::Mpi::Init();
  static bool hy = false;
  if (!hy) {
    mfem::Hypre::Init();
    hy = true;
  }
}

// --- vectorized callables f(X, Y, Z) -> (n,) or (n, vdim) (python/axionfem/fluids/stokes3d.py normalizes them)
struct PyFn3 {
  nb::object f;
  int vdim = 1;
  void many(const double* X, const double* Y, const double* Z, size_t n, double* out) const {
    std::vector<double> xs(X, X + n), ys(Y, Y + n), zs(Z, Z + n);
    size_t shp[1] = {n};
    nb::ndarray<nb::numpy, double> xa(xs.data(), 1, shp, nb::handle()), ya(ys.data(), 1, shp, nb::handle()),
        za(zs.data(), 1, shp, nb::handle());
    nb::object r = f(xa, ya, za);
    auto arr = nb::cast<DArr>(r);
    AXF_REQUIRE(arr.size() == n * static_cast<size_t>(vdim),
                axf::strcat("a callable returned ", arr.size(), " values for ", n, " points x ", vdim, " components"));
    std::copy(arr.data(), arr.data() + n * vdim, out);
  }
};

// values tabulated on a rule of `order`, per domain element or per boundary element
struct Table3 {
  int order = -1;
  bool boundary = false;
  std::vector<std::vector<double>> vals;
  void build(const PyFn3& fn, mfem::Mesh& mesh, int ord, bool bdr) {
    order = ord;
    boundary = bdr;
    const int n = bdr ? mesh.GetNBE() : mesh.GetNE();
    vals.assign(n, {});
    std::vector<double> X, Y, Z;
    std::vector<int> nqs(n);
    for (int e = 0; e < n; ++e) {
      const mfem::Geometry::Type g = bdr ? mesh.GetBdrElementGeometry(e) : mesh.GetElementBaseGeometry(e);
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(g, ord);
      mfem::ElementTransformation* T = bdr ? mesh.GetBdrElementTransformation(e) : mesh.GetElementTransformation(e);
      nqs[e] = ir.GetNPoints();
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        double x[3] = {0, 0, 0};
        mfem::Vector xv(x, 3);
        T->Transform(ir.IntPoint(q), xv);
        X.push_back(x[0]);
        Y.push_back(x[1]);
        Z.push_back(x[2]);
      }
    }
    std::vector<double> out(X.size() * fn.vdim);
    if (!X.empty()) fn.many(X.data(), Y.data(), Z.data(), X.size(), out.data());
    size_t k = 0;
    for (int e = 0; e < n; ++e) {
      vals[e].assign(out.begin() + k * fn.vdim, out.begin() + (k + nqs[e]) * fn.vdim);
      k += nqs[e];
    }
  }
  const double* find(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) const {
    if (order < 0) return nullptr;
    const auto want = boundary ? mfem::ElementTransformation::BDR_ELEMENT : mfem::ElementTransformation::ELEMENT;
    if (T.ElementType != want) return nullptr;
    const int e = T.ElementNo;
    if (e < 0 || e >= static_cast<int>(vals.size())) return nullptr;
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(T.GetGeometryType(), order);
    if (ip.index < 0 || ip.index >= ir.GetNPoints()) return nullptr;
    const mfem::IntegrationPoint& r = ir.IntPoint(ip.index);
    if (r.x != ip.x || r.y != ip.y || r.z != ip.z) return nullptr;
    return vals[e].data() + static_cast<size_t>(ip.index) * (vals[e].size() / ir.GetNPoints());
  }
};

class PyScalar3 : public mfem::Coefficient {
 public:
  explicit PyScalar3(PyFn3 f) : f_(std::move(f)) {}
  std::vector<Table3> tables;
  bool check_positive = false;
  const PyFn3& fn() const { return f_; }
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    for (const auto& t : tables)
      if (const double* v = t.find(T, ip)) return v[0];
    double x[3] = {0, 0, 0}, out = 0;
    mfem::Vector X(x, 3);
    T.Transform(ip, X);
    f_.many(&x[0], &x[1], &x[2], 1, &out);
    if (check_positive)
      AXF_REQUIRE(std::isfinite(out) && out > 0, axf::strcat("the viscosity callable returned ", out, " at (", x[0],
                                                              ", ", x[1], ", ", x[2], "): it must be positive"), "nu");
    return out;
  }

 private:
  PyFn3 f_;
};

class PyVector3 : public mfem::VectorCoefficient {
 public:
  explicit PyVector3(PyFn3 f) : mfem::VectorCoefficient(f.vdim), f_(std::move(f)) {}
  std::vector<Table3> tables;
  const PyFn3& fn() const { return f_; }
  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    V.SetSize(vdim);
    for (const auto& t : tables)
      if (const double* v = t.find(T, ip)) {
        for (int i = 0; i < vdim; ++i) V(i) = v[i];
        return;
      }
    double x[3] = {0, 0, 0};
    mfem::Vector X(x, 3);
    T.Transform(ip, X);
    f_.many(&x[0], &x[1], &x[2], 1, V.GetData());
  }

 private:
  PyFn3 f_;
};

std::unique_ptr<mfem::Coefficient> scalar3(nb::object o) {
  if (nb::isinstance<nb::float_>(o) || nb::isinstance<nb::int_>(o))
    return std::make_unique<mfem::ConstantCoefficient>(nb::cast<double>(o));
  AXF_REQUIRE(nb::hasattr(o, "__call__"), "a scalar coefficient must be a number or a callable f(X, Y, Z)");
  return std::make_unique<PyScalar3>(PyFn3{o, 1});
}

std::unique_ptr<mfem::VectorCoefficient> vector3(nb::object o) {
  if (nb::hasattr(o, "__call__")) return std::make_unique<PyVector3>(PyFn3{o, 3});
  auto v = nb::cast<std::vector<double>>(o);
  AXF_REQUIRE(v.size() == 3, axf::strcat("a constant vector has ", v.size(), " components; 3 expected"));
  mfem::Vector mv(3);
  for (int i = 0; i < 3; ++i) mv(i) = v[i];
  return std::make_unique<mfem::VectorConstantCoefficient>(mv);
}

// --- mesh
struct PyMesh3D {
  Mesh3DInput in;
  std::shared_ptr<mfem::Mesh> serial;
  int n_elements() const { return serial->GetNE(); }
  int nn() const { return in.vpe == 4 ? 10 : 27; }

  nb::ndarray<nb::numpy, double> vertices() const {
    std::vector<double> v(3 * serial->GetNV());
    for (int i = 0; i < serial->GetNV(); ++i)
      for (int d = 0; d < 3; ++d) v[3 * i + d] = serial->GetVertex(i)[d];
    return np3(std::move(v), {static_cast<size_t>(serial->GetNV()), 3});
  }
  // the solver's connectivity: tetrahedra in MFEM's (possibly reoriented) order, hexahedra back in basix order
  nb::ndarray<nb::numpy, int> elements() const {
    static const int basix_of_mfem[8] = {0, 1, 3, 2, 4, 5, 7, 6};
    std::vector<int> e;
    mfem::Array<int> vs;
    for (int i = 0; i < serial->GetNE(); ++i) {
      serial->GetElementVertices(i, vs);
      AXF_REQUIRE(vs.Size() == in.vpe, "elements(): unexpected element type");
      if (in.vpe == 4) {
        for (int j : vs) e.push_back(j);
      } else {
        int h[8];
        for (int k = 0; k < 8; ++k) h[basix_of_mfem[k]] = vs[k];
        e.insert(e.end(), h, h + 8);
      }
    }
    return np3i(std::move(e), {static_cast<size_t>(serial->GetNE()), static_cast<size_t>(in.vpe)});
  }
  // the solver's geometry nodes at the order-2 basix reference nodes of each element in the vertex order of
  // elements(): the values of the mesh's order-2 node grid function (curved meshes), else the vertex map
  nb::ndarray<nb::numpy, double> geometry_nodes() const {
    const auto& R = ref_nodes2(in.vpe);
    std::vector<double> out(static_cast<size_t>(serial->GetNE()) * R.size() * 3, 0.0);
    const mfem::GridFunction* nodes = serial->GetNodes();
    mfem::Array<int> vd;
    for (int e = 0; e < serial->GetNE(); ++e) {
      double* oe = out.data() + static_cast<size_t>(e) * R.size() * 3;
      if (nodes) {
        const mfem::FiniteElementSpace* nfes = nodes->FESpace();
        nfes->GetElementVDofs(e, vd);
        const mfem::IntegrationRule& nir = nfes->GetFE(e)->GetNodes();
        const int nd = nir.GetNPoints();
        AXF_REQUIRE(nd == static_cast<int>(R.size()), "geometry_nodes(): the mesh nodes are not of order 2");
        for (int k = 0; k < nd; ++k) {
          const mfem::IntegrationPoint& ip = nir.IntPoint(k);
          int idx = -1;  // MFEM's and basix's reference cells coincide; the hex vertex order is converted
          for (int j = 0; j < nd; ++j)
            if (std::fabs(R[j][0] - ip.x) < 1e-12 && std::fabs(R[j][1] - ip.y) < 1e-12 && std::fabs(R[j][2] - ip.z) < 1e-12)
              idx = j;
          AXF_REQUIRE(idx >= 0, "geometry_nodes(): internal error (node not at a basix reference node)");
          for (int c = 0; c < 3; ++c) oe[3 * idx + c] = (*nodes)(vd[c * nd + k]);
        }
      } else {
        mfem::ElementTransformation* T = serial->GetElementTransformation(e);
        for (size_t j = 0; j < R.size(); ++j) {
          mfem::IntegrationPoint ip;
          ip.Set3(R[j][0], R[j][1], R[j][2]);
          mfem::Vector X(oe + 3 * j, 3);
          T->Transform(ip, X);
        }
      }
    }
    return np3(std::move(out), {static_cast<size_t>(serial->GetNE()), R.size(), 3});
  }
  nb::dict taylor_hood_check() const {
    const auto r = taylor_hood_counts_3d(in.elements, in.vpe, static_cast<int>(in.vertices.size() / 3));
    nb::dict d;
    if (in.vpe == 4) {
      d["tets_without_interior_vertex"] = r.tets_without_interior_vertex;
      d["tets_violating_weak_boffi"] = r.tets_violating_weak_boffi;
      d["tets_all_p2_nodes_on_boundary"] = r.tets_all_p2_nodes_on_boundary;
    } else {
      d["hexes_all_vertices_on_boundary"] = r.hexes_all_vertices_on_boundary;
    }
    d["checked_elements"] = r.tets + r.hexes;
    return d;
  }
};

PyMesh3D mesh3d_from_arrays(DArr V, IArr E, IArr F, IArr FA, nb::object eattr, nb::object nodes) {
  AXF_REQUIRE(V.ndim() == 2 && V.shape(1) == 3, "vertices must be an (nv, 3) array");
  AXF_REQUIRE(E.ndim() == 2 && (E.shape(1) == 4 || E.shape(1) == 8), "elements must be (ne, 4) or (ne, 8)");
  AXF_REQUIRE(F.ndim() == 2 && (F.shape(1) == 3 || F.shape(1) == 4), "boundary faces must be (nb, 3) or (nb, 4)");
  PyMesh3D m;
  m.in.vertices.assign(V.data(), V.data() + V.size());
  m.in.elements.assign(E.data(), E.data() + E.size());
  m.in.vpe = static_cast<int>(E.shape(1));
  m.in.bdr_faces.assign(F.data(), F.data() + F.size());
  m.in.fv = static_cast<int>(F.shape(1));
  m.in.bdr_attributes.assign(FA.data(), FA.data() + FA.size());
  if (!eattr.is_none()) m.in.elem_attributes = nb::cast<std::vector<int>>(eattr);
  if (!nodes.is_none()) {
    auto arr = nb::cast<DArr>(nodes);
    m.in.nodes.assign(arr.data(), arr.data() + arr.size());
  }
  m.serial = std::make_shared<mfem::Mesh>(mesh_from_arrays_3d(m.in, true));
  return m;
}

// deterministic partition: METIS, with a coordinate-slab fallback when METIS leaves a part empty
std::vector<int> partition(mfem::Mesh& mesh, int np) {
  const int ne = mesh.GetNE();
  std::vector<int> part(ne, 0);
  if (np == 1) return part;
  {
    int* p = nullptr;
    {  // METIS swaps process-wide signal handlers: under the lock every METIS call takes (sym_solver.hpp)
      std::lock_guard<std::mutex> metis(axf::metis_mutex());
      p = mesh.GeneratePartitioning(np, 1);
    }
    part.assign(p, p + ne);
    delete[] p;
  }
  std::vector<int> cnt(np, 0);
  for (int q : part) ++cnt[q];
  if (ne >= np && std::count(cnt.begin(), cnt.end(), 0) > 0) {
    std::vector<std::array<double, 4>> c(ne);
    mfem::Array<int> v;
    for (int e = 0; e < ne; ++e) {
      mesh.GetElementVertices(e, v);
      double s[3] = {0, 0, 0};
      for (int k : v)
        for (int d = 0; d < 3; ++d) s[d] += mesh.GetVertex(k)[d] / v.Size();
      c[e] = {s[0], s[1], s[2], static_cast<double>(e)};
    }
    std::sort(c.begin(), c.end());
    for (int i = 0; i < ne; ++i) part[static_cast<int>(c[i][3])] = static_cast<int>(static_cast<long>(i) * np / ne);
  }
  return part;
}

BCKind bc_kind3(const std::string& s) {
  if (s == "dirichlet") return BCKind::Dirichlet;
  if (s == "do_nothing") return BCKind::DoNothing;
  if (s == "traction") return BCKind::Traction;
  if (s == "slip") return BCKind::Slip;
  if (s == "symmetry") return BCKind::Symmetry;
  throw axf::Error("unknown 3D boundary-condition kind '" + s + "'", "bcs",
                   "dirichlet, do_nothing, traction, slip, symmetry");
}

struct PySolution3D {
  std::shared_ptr<mfem::Mesh> serial;      // the serial mesh the ParMesh was built from
  std::shared_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::Coefficient> nu;
  std::unique_ptr<mfem::VectorCoefficient> force;
  std::vector<std::unique_ptr<mfem::VectorCoefficient>> bc_values;
  std::unique_ptr<StokesSolver> S;
  std::vector<long> gid;          // input id of every local element
  std::vector<RefMap3D> maps;     // input reference cell -> MFEM reference cell, per local element
  long comm_size = 1, shared_dofs = 0, n_essential = 0;
  bool solved = false;

  nb::dict diagnostics() const {
    const auto& d = S->diagnostics();
    nb::dict r;
    r["converged"] = d.converged;
    r["relative_residual"] = d.rel_residual;
    r["relative_residual_all_rows"] = d.rel_residual_all_rows;
    r["residual_momentum"] = d.res_momentum;
    r["residual_continuity"] = d.res_continuity;
    r["iterations"] = d.iterations;
    r["nullspace_detected"] = d.nullspace_detected;
    r["bc_enclosed"] = d.bc_enclosed;
    r["kernel_defect"] = d.kernel_defect;
    r["nullspace"] = d.nullspace;
    r["flux_defect"] = d.flux_defect;
    r["flux_defect_abs"] = d.flux_defect_abs;
    r["multiplier"] = d.multiplier;
    r["solver"] = d.solver;
    r["viscous_form"] = d.viscous_form;
    r["grad_div"] = d.grad_div;
    r["slip_nodes"] = d.n_slip_nodes;
    r["slip_corner_nodes"] = d.n_slip_corner_nodes;
    r["slip_max_averaged_kink_deg"] = d.slip_max_averaged_kink_deg;
    r["dofs_velocity"] = d.n_velocity;
    r["dofs_pressure"] = d.n_pressure;
    r["dofs_field"] = d.n_total;
    r["lu_residual_before_refinement"] = d.lu.rel_residual_before;
    r["lu_residual"] = d.lu.rel_residual;
    r["lu_refine_steps"] = d.lu.refine_steps;
    r["lu_krylov_iterations"] = d.lu.krylov_iterations;
    r["lu_krylov_fallback"] = d.lu.krylov_fallback;
    r["lu_matching"] = d.lu.matching;
    r["lu_matching_retry"] = d.lu_matching_retry;
    r["lu_matching_retry_reason"] = d.lu_matching_retry_reason;
    r["lu_first_attempt_residual_before"] = d.lu_first_attempt_residual_before;
    r["lu_first_attempt_krylov_iterations"] = d.lu_first_attempt_krylov_iterations;
    r["lu_backend"] = d.lu.backend;
    r["lu_factor_nnz"] = d.lu.factor_nnz;
    r["ranks"] = d.ranks;
    r["threads"] = d.threads;
    r["lu_threads"] = d.lu.threads;
    r["comm_size"] = comm_size;
    r["shared_dofs"] = shared_dofs;
    r["essential_velocity_dofs"] = n_essential;
    r["n_local_elements"] = static_cast<long>(gid.size());
    r["direct_tol"] = S->options().direct_tol;
    r["seconds"] = d.assemble_seconds + d.setup_seconds + d.solve_seconds;
    r["assemble_seconds"] = d.assemble_seconds;
    r["solve_seconds"] = d.setup_seconds + d.solve_seconds;
    return r;
  }

  nb::ndarray<nb::numpy, long> element_ids() const { return np3l(gid, {gid.size()}); }

  template <class F>
  void each(DArr ref, F&& f) const {
    AXF_REQUIRE(ref.ndim() == 2 && ref.shape(1) == 3, "reference points must be a (q, 3) array");
    const size_t nq = ref.shape(0);
    for (size_t j = 0; j < gid.size(); ++j) {
      mfem::ElementTransformation* T = mesh->GetElementTransformation(static_cast<int>(j));
      for (size_t q = 0; q < nq; ++q) {
        double xm[3];
        maps[j].to_mfem(ref.data() + 3 * q, xm);
        mfem::IntegrationPoint ip;
        ip.Set3(xm[0], xm[1], xm[2]);
        T->SetIntPoint(&ip);
        f(*T, ip, maps[j]);
      }
    }
  }
  nb::tuple geometry(DArr ref) const {
    std::vector<double> X, J;
    each(ref, [&](mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip, const RefMap3D& m) {
      double x[3];
      mfem::Vector xv(x, 3);
      T.Transform(ip, xv);
      X.insert(X.end(), x, x + 3);
      J.push_back(T.Weight() / m.det());  // d x / d x_input = J_mfem M^{-1}
    });
    const size_t m = gid.size(), q = ref.shape(0);
    return nb::make_tuple(np3(std::move(X), {m, q, 3}), np3(std::move(J), {m, q}));
  }
  nb::tuple velocity(DArr ref) const {
    const auto& u = S->velocity();
    std::vector<double> U, G;
    mfem::Vector uv(3);
    mfem::DenseMatrix g(3, 3);
    each(ref, [&](mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip, const RefMap3D&) {
      u.GetVectorValue(T, ip, uv);
      u.GetVectorGradient(T, g);
      for (int i = 0; i < 3; ++i) U.push_back(uv(i));
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) G.push_back(g(i, j));
    });
    const size_t m = gid.size(), q = ref.shape(0);
    return nb::make_tuple(np3(std::move(U), {m, q, 3}), np3(std::move(G), {m, q, 3, 3}));
  }
  nb::ndarray<nb::numpy, double> pressure(DArr ref) const {
    const auto& p = S->pressure();
    std::vector<double> P;
    each(ref, [&](mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip, const RefMap3D&) {
      P.push_back(p.GetValue(T, ip));
    });
    return np3(std::move(P), {gid.size(), static_cast<size_t>(ref.shape(0))});
  }

  // Babuška–Miller reaction ('solved': the physical force; 'stress'/'laplacian': the raw residual of that form)
  nb::dict reaction(std::vector<int> attrs, const std::string& form) const {
    AXF_REQUIRE(form == "solved" || form == "laplacian" || form == "stress",
                "reaction residual_form must be 'solved', 'stress' or 'laplacian'");
    const ViscousForm f = form == "solved" ? S->options().form
                                           : (form == "laplacian" ? ViscousForm::Laplacian : ViscousForm::SymmetricStress);
    nb::dict d;
    if (form == "solved") {
      // 'solved' is the PHYSICAL force, as on the 2D/axisymmetric solution (the raw residual of the solved form is the
      // pseudo-traction for the Laplacian form; a null-Lagrangian correction turns it into the Cauchy traction on
      // moving and slip walls)
      const Reaction r = S->physical_reaction(attrs, {});
      d["force"] = std::vector<double>(r.force, r.force + 3);
      d["residual_form"] = r.residual_form;
      return d;
    }
    const Reaction r = S->reaction(attrs, {}, f);
    d["force"] = std::vector<double>(r.force, r.force + 3);
    d["residual_form"] = r.residual_form;
    return d;
  }

  static nb::dict csr(const mfem::HypreParMatrix& M0) {
    mfem::HypreParMatrix& M = const_cast<mfem::HypreParMatrix&>(M0);
    mfem::SparseMatrix s;
    M.MergeDiagAndOffd(s);
    nb::dict d;
    const size_t nr = static_cast<size_t>(s.Height()), nnz = static_cast<size_t>(s.NumNonZeroElems());
    d["indptr"] = np3i(std::vector<int>(s.GetI(), s.GetI() + nr + 1), {nr + 1});
    d["indices"] = np3i(std::vector<int>(s.GetJ(), s.GetJ() + nnz), {nnz});
    d["data"] = np3(std::vector<double>(s.GetData(), s.GetData() + nnz), {nnz});
    d["shape"] = nb::make_tuple(static_cast<long>(M.GetGlobalNumRows()), static_cast<long>(M.GetGlobalNumCols()));
    return d;
  }
  nb::dict blocks() const {
    AXF_REQUIRE(comm_size == 1, "blocks() needs a one-rank solve");
    nb::dict d;
    d["A"] = csr(S->viscous_block());
    d["B"] = csr(S->divergence_block());
    d["Mp"] = csr(S->pressure_mass());
    auto G = S->velocity_h1_gram();
    d["H1_gram"] = csr(*G);
    const auto& ess = S->essential_velocity_dofs();
    d["essential_velocity_dofs"] = std::vector<int>(ess.begin(), ess.end());
    return d;
  }
};

std::unique_ptr<PySolution3D> solve_stokes_3d(PyMesh3D& mesh, nb::object nu, nb::object force, nb::list bcs,
                                              const std::string& form, const std::string& solver,
                                              const std::string& lu, const std::string& nullspace, int order,
                                              double rtol, double direct_tol, double grad_div,
                                              int pressure_order, bool allow_unstable_pair, int min_refine_steps,
                                              bool do_solve, int lu_threads, double slip_corner_angle_deg,
                                              bool lu_krylov_fallback, bool lu_matching) {
  ensure_mpi3();
  auto sol = std::make_unique<PySolution3D>();
  StokesOptions o;
  o.velocity_order = order;
  o.pressure_order = pressure_order;
  o.allow_unstable_pair = allow_unstable_pair;
  o.grad_div = grad_div;
  o.min_refine_steps = min_refine_steps;
  o.lu_threads = lu_threads;
  o.slip_corner_angle_deg = slip_corner_angle_deg;
  o.lu_krylov_fallback = lu_krylov_fallback;
  o.lu_matching = lu_matching;
  AXF_REQUIRE(form == "laplacian" || form == "stress", "form must be 'laplacian' or 'stress'");
  o.form = form == "laplacian" ? ViscousForm::Laplacian : ViscousForm::SymmetricStress;
  AXF_REQUIRE(solver == "direct" || solver == "minres", "solver must be 'direct' or 'minres'");
  o.solver = solver == "direct" ? LinearSolverKind::Direct : LinearSolverKind::MinresBlockDiag;
  o.lu = lu_backend_from_string(lu);
  if (nullspace == "auto") o.nullspace = Nullspace::Auto;
  else if (nullspace == "pin") o.nullspace = Nullspace::PinPressure;
  else if (nullspace == "multiplier") o.nullspace = Nullspace::MeanMultiplier;
  else if (nullspace == "ortho") o.nullspace = Nullspace::Ortho;
  else if (nullspace == "none") o.nullspace = Nullspace::None;
  else throw axf::Error("unknown nullspace '" + nullspace + "'", "", "auto, pin, multiplier, ortho, none");
  o.krylov_rtol = rtol;
  o.direct_tol = direct_tol;
  StokesBoundary b;
  for (nb::handle h : bcs) {
    nb::dict d = nb::cast<nb::dict>(h);
    for (nb::handle k : d.keys()) {
      const std::string key = nb::cast<std::string>(k);
      AXF_REQUIRE(key == "attribute" || key == "kind" || key == "value" || key == "normal",
                  "unknown key '" + key + "' in a 3D boundary condition", "bcs", "attribute, kind, value, normal");
    }
    BoundaryCondition c;
    c.attribute = nb::cast<int>(d["attribute"]);
    c.kind = bc_kind3(nb::cast<std::string>(d["kind"]));
    if (d.contains("value") && !d["value"].is_none()) {
      AXF_REQUIRE(c.kind == BCKind::Dirichlet || c.kind == BCKind::Traction,
                  axf::strcat("attribute ", c.attribute, ": 'value' is only for dirichlet and traction"), "bcs");
      sol->bc_values.push_back(vector3(nb::borrow(d["value"])));
      c.value = sol->bc_values.back().get();
    }
    if (d.contains("normal") && !d["normal"].is_none()) {
      AXF_REQUIRE(c.kind == BCKind::Slip || c.kind == BCKind::Symmetry,
                  axf::strcat("attribute ", c.attribute, ": 'normal' is only for slip and symmetry"), "bcs");
      sol->bc_values.push_back(vector3(nb::borrow(d["normal"])));
      c.normal = sol->bc_values.back().get();
    }
    b.conditions.push_back(c);
  }

  // the mesh of the solve: the input mesh, partitioned over COMM_WORLD
  sol->serial = mesh.serial;
  MPI_Comm comm = MPI_COMM_WORLD;
  int np = 1, me = 0;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &me);
  std::vector<int> part = partition(*sol->serial, np);
  sol->mesh = std::make_shared<mfem::ParMesh>(comm, *sol->serial, part.data());
  for (int i = 0; i < static_cast<int>(part.size()); ++i)
    if (part[i] == me) sol->gid.push_back(i);
  AXF_REQUIRE(static_cast<int>(sol->gid.size()) == sol->mesh->GetNE(),
              "solve_stokes_3d: the local elements do not match the partition (internal error)");
  const int vpe = mesh.in.vpe;
  sol->maps.reserve(sol->gid.size());
  for (size_t j = 0; j < sol->gid.size(); ++j) {
    double xin[24];
    for (int k = 0; k < vpe; ++k)
      for (int d = 0; d < 3; ++d)
        xin[3 * k + d] = mesh.in.vertices[3 * mesh.in.elements[static_cast<size_t>(vpe) * sol->gid[j] + k] + d];
    sol->maps.push_back(element_ref_map(*sol->mesh, static_cast<int>(j), xin, vpe));
  }

  // coefficients, tabulated on exactly the rules the solver uses
  sol->nu = scalar3(nu);
  o.nu = sol->nu.get();
  if (auto* ps = dynamic_cast<PyScalar3*>(sol->nu.get())) {
    ps->check_positive = true;
    ps->tables.emplace_back();
    ps->tables.back().build(ps->fn(), *sol->mesh, o.viscous_quad_order(), false);
    for (const auto& e : ps->tables.back().vals)
      for (double v : e)
        AXF_REQUIRE(std::isfinite(v) && v > 0, axf::strcat("the viscosity callable returned ", v,
                                                            " at a quadrature point: it must be positive and finite"),
                    "nu");
  }
  if (!force.is_none()) {
    sol->force = vector3(force);
    o.force = sol->force.get();
    if (auto* pv = dynamic_cast<PyVector3*>(sol->force.get())) {
      pv->tables.emplace_back();
      pv->tables.back().build(pv->fn(), *sol->mesh, o.load_quad_order(), false);
    }
  }
  for (size_t i = 0; i < b.conditions.size(); ++i)
    if (b.conditions[i].kind == BCKind::Traction && b.conditions[i].value)
      if (auto* pv = dynamic_cast<PyVector3*>(b.conditions[i].value)) {
        pv->tables.emplace_back();
        pv->tables.back().build(pv->fn(), *sol->mesh, o.boundary_quad_order(), true);
      }

  sol->S = std::make_unique<StokesSolver>(*sol->mesh, o, b);
  if (do_solve) {
    sol->S->solve();
    sol->solved = true;
  } else {
    sol->S->assemble();
  }
  // rank proof: the solve's communicator and its shared (ghost) DOFs
  {
    int cs = 1;
    MPI_Comm_size(sol->mesh->GetComm(), &cs);
    sol->comm_size = cs;
    long ghost = (sol->S->velocity_space().GetVSize() - sol->S->velocity_space().GetTrueVSize()) +
                 (sol->S->pressure_space().GetVSize() - sol->S->pressure_space().GetTrueVSize());
    MPI_Allreduce(MPI_IN_PLACE, &ghost, 1, MPI_LONG, MPI_SUM, sol->mesh->GetComm());
    sol->shared_dofs = ghost;
    long ness = sol->S->essential_velocity_dofs().Size();
    MPI_Allreduce(MPI_IN_PLACE, &ness, 1, MPI_LONG, MPI_SUM, sol->mesh->GetComm());
    sol->n_essential = ness;
  }
  return sol;
}

}  // namespace

namespace axf::fluids {

void register_python_3d(nb::module_& m) {
  nb::class_<PyMesh3D>(m, "Mesh3D")
      .def_static("from_arrays", &mesh3d_from_arrays, nb::arg("vertices"), nb::arg("elements"),
                  nb::arg("boundary_faces"), nb::arg("boundary_attributes"), nb::arg("element_attributes") = nb::none(),
                  nb::arg("nodes") = nb::none(),
                  "3D mesh: (nv,3) vertices, (ne,4) tetrahedra or (ne,8) hexahedra (basix lexicographic vertex order), "
                  "(nb,3|4) boundary faces (quadrilaterals as cycles) with attributes >= 1; nodes: (ne,10|27,3) curved "
                  "order-2 geometry in basix node order.")
      .def_prop_ro("n_elements", &PyMesh3D::n_elements)
      .def("vertices", &PyMesh3D::vertices)
      .def("elements", &PyMesh3D::elements, "the solver's connectivity (tetrahedra possibly reoriented; hexahedra in "
                                            "basix order)")
      .def("geometry_nodes", &PyMesh3D::geometry_nodes,
           "(ne, 10|27, 3): the solver's element map at the basix order-2 nodes of the cells of elements()")
      .def("taylor_hood_check", &PyMesh3D::taylor_hood_check,
           "Taylor-Hood mesh condition counts: tets_without_interior_vertex (a), tets_violating_weak_boffi (b), "
           "tets_all_p2_nodes_on_boundary (c); hexes_all_vertices_on_boundary for hexahedra");

  nb::class_<PySolution3D>(m, "StokesSolution3D")
      .def_prop_ro("diagnostics", &PySolution3D::diagnostics)
      .def("element_ids", &PySolution3D::element_ids, "input ids of this rank's elements")
      .def("geometry", &PySolution3D::geometry, nb::arg("ref_points"),
           "(x (m,q,3), detJ (m,q)) at INPUT reference points of this rank's elements")
      .def("velocity", &PySolution3D::velocity, nb::arg("ref_points"), "(u (m,q,3), grad u (m,q,3,3))")
      .def("pressure", &PySolution3D::pressure, nb::arg("ref_points"), "p (m,q)")
      .def("reaction", &PySolution3D::reaction, nb::arg("attributes"), nb::arg("residual_form") = "solved",
           "Babuška–Miller force exerted by the fluid on the body attributes (collective)")
      .def("blocks", &PySolution3D::blocks, "A, B, Mp, H1_gram (CSR dicts) and the essential velocity DOFs (one rank)");

  // MPI helpers for distributed 3D runs (two-pass reductions inside one MPI job; COMM_WORLD)
  m.def("world_rank", []() {
    ensure_mpi3();
    int r = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
  });
  m.def("world_size", []() {
    ensure_mpi3();
    int s = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
  });
  m.def("allreduce_sum", [](DArr a) {
    ensure_mpi3();
    std::vector<double> v(a.data(), a.data() + a.size());
    MPI_Allreduce(MPI_IN_PLACE, v.data(), static_cast<int>(v.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return np3(std::move(v), {static_cast<size_t>(a.size())});
  }, nb::arg("values"), "element-wise sum over MPI_COMM_WORLD (collective)");
  m.def("barrier", []() {
    ensure_mpi3();
    MPI_Barrier(MPI_COMM_WORLD);
  });
  m.def("solve_stokes_3d", &solve_stokes_3d, nb::arg("mesh"), nb::arg("nu"), nb::arg("force") = nb::none(),
        nb::arg("bcs") = nb::list(), nb::arg("form") = "stress", nb::arg("solver") = "direct",
        nb::arg("lu") = "strumpack", nb::arg("nullspace") = "auto", nb::arg("order") = 2, nb::arg("rtol") = 1e-10,
        nb::arg("direct_tol") = 1e-10, nb::arg("grad_div") = 0.0,
        nb::arg("pressure_order") = -1, nb::arg("allow_unstable_pair") = false, nb::arg("min_refine_steps") = 0,
        nb::arg("solve") = true, nb::arg("lu_threads") = 1, nb::arg("slip_corner_angle_deg") = 30.0,
        nb::arg("lu_krylov_fallback") = true, nb::arg("lu_matching") = false,
        "Taylor–Hood Stokes on a 3D mesh, partitioned over MPI_COMM_WORLD (axionfem.fluids.solve_stokes_3d).");
}

}  // namespace axf::fluids
