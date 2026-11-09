// Python boundary of the fluids library: the submodule axionfem._core._fluids (wrapped by python/axionfem/fluids).
// It is registered by cpp/python/bindings.cpp in MPI builds, so that the one compiled module (_core) carries the
// fluids API and its build provenance (beta). It provides meshes from arrays and back, coefficients as Python
// callables, explicit BCs per attribute, a raw pressure point, diagnostics including the residual of the consistent
// system and the flux defect, point evaluation at reference points of elements, reactions, and the assembled blocks.
// Built only with AXF_MPI; evaluation and blocks need one MPI rank.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <string>
#include <vector>

#include <mfem.hpp>

#include "axionfem/core.hpp"
#include "axionfem/fluids/boussinesq.hpp"
#include "axionfem/fluids/gls.hpp"
#include "axionfem/fluids/mesh_checks.hpp"
#include "axionfem/fluids/meshes.hpp"
#include "axionfem/fluids/navier_stokes.hpp"
#include "axionfem/fluids/stokes.hpp"
#include "axionfem/fluids/transient_ns.hpp"
#include "axionfem/transport/scalar_transport.hpp"
#include "axionfem/sym_solver.hpp"

namespace nb = nanobind;
using namespace axf::fluids;

namespace {

using Arr2 = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using IArr2 = nb::ndarray<const int, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using IArr1 = nb::ndarray<const int, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

nb::ndarray<nb::numpy, double> np_array(std::vector<double> v, std::vector<size_t> shape) {
  auto* data = new std::vector<double>(std::move(v));
  nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
  return nb::ndarray<nb::numpy, double>(data->data(), shape.size(), shape.data(), owner);
}
nb::ndarray<nb::numpy, int> np_iarray(std::vector<int> v, std::vector<size_t> shape) {
  auto* data = new std::vector<int>(std::move(v));
  nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<int>*>(p); });
  return nb::ndarray<nb::numpy, int>(data->data(), shape.size(), shape.data(), owner);
}

void ensure_mpi();

int world_size() {
  int s = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
}

// MPI and hypre are initialized lazily (importing _core must not start MPI for the EM-only paths)
void ensure_mpi() {
  int init = 0;
  MPI_Initialized(&init);
  if (!init) mfem::Mpi::Init();
  static bool hy = false;
  if (!hy) {
    mfem::Hypre::Init();
    hy = true;
  }
}

// ParMesh(comm, m): on more than one rank MFEM partitions a conforming mesh with METIS (Mesh::GeneratePartitioning).
// The same partition, computed under axf::metis_mutex() (METIS swaps process-wide signal handlers, so every METIS call
// in the process takes that lock); a nonconforming mesh keeps MFEM's own partitioner.
std::shared_ptr<mfem::ParMesh> par_mesh(MPI_Comm comm, mfem::Mesh& m) {
  int np = 1;
  MPI_Comm_size(comm, &np);
  if (np == 1 || m.Nonconforming()) return std::make_shared<mfem::ParMesh>(comm, m);
  int* part = nullptr;
  {
    std::lock_guard<std::mutex> metis(axf::metis_mutex());
    part = m.GeneratePartitioning(np, 1);
  }
  auto pm = std::make_shared<mfem::ParMesh>(comm, m, part);
  delete[] part;
  return pm;
}

using DArr = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

// --- coefficients from Python: a number or a callable. Scalar mode: f(x, y) -> float / sequence. Vectorized mode
// (python/axionfem/fluids normalizes the callable): f(X, Y) with 1-D arrays -> array (n,) or (n, vdim).
struct PyFn {
  nb::object f;
  bool vec = false;
  int vdim = 1;
  // values at one point
  void at(double x, double y, double* out) const {
    AXF_REQUIRE(f.is_valid(), "a coefficient callable was released by the garbage collector (tp_clear)");
    if (!vec) {
      nb::object r = f(x, y);
      if (vdim == 1) {
        out[0] = nb::cast<double>(r);
        return;
      }
      auto seq = nb::cast<std::vector<double>>(r);
      AXF_REQUIRE(static_cast<int>(seq.size()) == vdim, axf::strcat("a vector callable returned ", seq.size(),
                                                               " components; ", vdim, " expected"));
      for (int i = 0; i < vdim; ++i) out[i] = seq[i];
      return;
    }
    many(&x, &y, 1, out);
  }
  // values at n points (vectorized mode), out[n * vdim]
  void many(const double* X, const double* Y, size_t n, double* out) const {
    AXF_REQUIRE(f.is_valid(), "a coefficient callable was released by the garbage collector (tp_clear)");
    std::vector<double> xs(X, X + n), ys(Y, Y + n);
    size_t shp[1] = {n};
    nb::ndarray<nb::numpy, double> xa(xs.data(), 1, shp, nb::handle()), ya(ys.data(), 1, shp, nb::handle());
    nb::object r = f(xa, ya);
    auto arr = nb::cast<DArr>(r);
    AXF_REQUIRE(arr.size() == n * static_cast<size_t>(vdim),
                axf::strcat("a vectorized callable returned ", arr.size(), " values for ", n, " points x ", vdim,
                            " components"));
    std::copy(arr.data(), arr.data() + n * vdim, out);
  }
};

// Values tabulated once (one vectorized call) at the points of the rule of `order` on every element; any other
// point (another rule, a boundary, a defect that moves the point) falls back to a direct call.
struct Table {
  int order = -1;
  std::vector<std::vector<double>> vals;  // per element: nq * vdim
  void build(const PyFn& fn, mfem::ParMesh& mesh, int ord) {
    order = ord;
    vals.assign(mesh.GetNE(), {});
    std::vector<double> X, Y;
    std::vector<int> nqs(mesh.GetNE());
    for (int e = 0; e < mesh.GetNE(); ++e) {
      const mfem::IntegrationRule& ir = mfem::IntRules.Get(mesh.GetElementBaseGeometry(e), ord);
      mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
      nqs[e] = ir.GetNPoints();
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        double x[3];
        mfem::Vector xv(x, mesh.SpaceDimension());
        T->Transform(ir.IntPoint(q), xv);
        X.push_back(x[0]);
        Y.push_back(x[1]);
      }
    }
    std::vector<double> out(X.size() * fn.vdim);
    if (!X.empty()) fn.many(X.data(), Y.data(), X.size(), out.data());
    size_t k = 0;
    for (int e = 0; e < mesh.GetNE(); ++e) {
      vals[e].assign(out.begin() + k * fn.vdim, out.begin() + (k + nqs[e]) * fn.vdim);
      k += nqs[e];
    }
  }
  const double* find(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) const {
    if (order < 0 || T.ElementType != mfem::ElementTransformation::ELEMENT) return nullptr;
    const int e = T.ElementNo;
    if (e < 0 || e >= static_cast<int>(vals.size())) return nullptr;
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(T.GetGeometryType(), order);
    if (ip.index < 0 || ip.index >= ir.GetNPoints()) return nullptr;
    const mfem::IntegrationPoint& r = ir.IntPoint(ip.index);
    if (r.x != ip.x || r.y != ip.y) return nullptr;
    const size_t vd = vals[e].size() / ir.GetNPoints();
    return vals[e].data() + ip.index * vd;
  }
};

class PyScalar : public mfem::Coefficient {
 public:
  explicit PyScalar(PyFn f) : f_(std::move(f)) {}
  Table table;
  bool check_positive = false;
  const PyFn& fn() const { return f_; }
  nb::object& py() { return f_.f; }  // for the GC (StokesSolution tp_traverse / tp_clear)
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    if (const double* v = table.find(T, ip)) return v[0];
    double x[3], out = 0;
    mfem::Vector X(x, T.GetSpaceDim());
    T.Transform(ip, X);
    f_.at(x[0], x[1], &out);
    if (check_positive)  // the viscosity (per-point callables are validated too)
      AXF_REQUIRE(std::isfinite(out) && out > 0, axf::strcat("the viscosity callable returned ", out, " at (", x[0],
                                                              ", ", x[1], "): it must be positive and finite"), "nu");
    return out;
  }

 private:
  PyFn f_;
};

class PyVector : public mfem::VectorCoefficient {
 public:
  explicit PyVector(PyFn f) : mfem::VectorCoefficient(f.vdim), f_(std::move(f)) {}
  Table table;
  const PyFn& fn() const { return f_; }
  nb::object& py() { return f_.f; }  // for the GC (StokesSolution tp_traverse / tp_clear)
  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    V.SetSize(vdim);
    if (const double* v = table.find(T, ip)) {
      for (int i = 0; i < vdim; ++i) V(i) = v[i];
      return;
    }
    double x[3];
    mfem::Vector X(x, T.GetSpaceDim());
    T.Transform(ip, X);
    f_.at(x[0], x[1], V.GetData());
  }

 private:
  PyFn f_;
};

// Traction data that depend on the outward unit normal of the (possibly curved) boundary at the quadrature point:
// g(x, y, n_x, n_y) (e.g. sigma_form n of an exact field on an isoparametric circle). Boundary elements only.
class PyVectorNormal : public mfem::VectorCoefficient {
 public:
  PyVectorNormal(nb::object f, int vdim, bool vec, mfem::ParMesh* mesh)
      : mfem::VectorCoefficient(vdim), f_(std::move(f)), vec_(vec), mesh_(mesh) {
    orientation(0);  // eagerly: the lazy version would move the mesh's shared boundary transformation mid-assembly
  }
  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    AXF_REQUIRE(T.ElementType == mfem::ElementTransformation::BDR_ELEMENT,
                "a normal-dependent traction is evaluated on boundary elements only", "bcs (normal_dependent)");
    AXF_REQUIRE(f_.is_valid(), "a traction callable was released by the garbage collector (tp_clear)");
    V.SetSize(vdim);
    double x[3], nr[3];
    mfem::Vector X(x, T.GetSpaceDim()), n(nr, T.GetSpaceDim());
    T.SetIntPoint(&ip);
    T.Transform(ip, X);
    mfem::CalcOrtho(T.Jacobian(), n);
    const double len = n.Norml2();
    AXF_REQUIRE(len > 0, "degenerate boundary element (zero normal)", "bcs (normal_dependent)");
    n *= orientation(T.ElementNo) / len;
    if (!vec_) {
      auto seq = nb::cast<std::vector<double>>(f_(x[0], x[1], nr[0], nr[1]));
      AXF_REQUIRE(static_cast<int>(seq.size()) == vdim, "a normal-dependent traction returned the wrong size");
      for (int i = 0; i < vdim; ++i) V(i) = seq[i];
      return;
    }
    size_t shp[1] = {1};
    nb::ndarray<nb::numpy, double> xa(&x[0], 1, shp, nb::handle()), ya(&x[1], 1, shp, nb::handle()),
        nxa(&nr[0], 1, shp, nb::handle()), nya(&nr[1], 1, shp, nb::handle());
    auto arr = nb::cast<DArr>(f_(xa, ya, nxa, nya));
    AXF_REQUIRE(arr.size() == static_cast<size_t>(vdim), "a normal-dependent traction returned the wrong size");
    for (int i = 0; i < vdim; ++i) V(i) = arr.data()[i];
  }

 private:
  nb::object f_;
  bool vec_;
  mfem::ParMesh* mesh_;

 public:
  nb::object& py() { return f_; }  // for the GC (StokesSolution tp_traverse / tp_clear)

 private:
  std::vector<signed char> sign_;
  // +1 if CalcOrtho of boundary element be points out of its adjacent element, -1 otherwise (computed once)
  double orientation(int be) {
    if (sign_.empty() && mesh_->GetNBE() > 0) {
      sign_.assign(mesh_->GetNBE(), 1);
      for (int b = 0; b < mesh_->GetNBE(); ++b) {
        int el = -1, info = 0;
        mesh_->GetBdrElementAdjacentElement(b, el, info);
        mfem::ElementTransformation* Tb = mesh_->GetBdrElementTransformation(b);
        const mfem::IntegrationPoint& cb = mfem::Geometries.GetCenter(Tb->GetGeometryType());
        Tb->SetIntPoint(&cb);
        double xb[3], nb_[3], xe[3];
        mfem::Vector XB(xb, 2), NB(nb_, 2), XE(xe, 2);
        Tb->Transform(cb, XB);
        mfem::CalcOrtho(Tb->Jacobian(), NB);
        mfem::ElementTransformation* Te = mesh_->GetElementTransformation(el);
        Te->Transform(mfem::Geometries.GetCenter(Te->GetGeometryType()), XE);
        sign_[b] = ((xb[0] - xe[0]) * nb_[0] + (xb[1] - xe[1]) * nb_[1]) >= 0 ? 1 : -1;
      }
    }
    return mesh_->GetNBE() ? sign_.at(be) : 1.0;
  }
};

std::unique_ptr<mfem::Coefficient> scalar_coef(nb::object o, bool vec) {
  if (nb::isinstance<nb::float_>(o) || nb::isinstance<nb::int_>(o))
    return std::make_unique<mfem::ConstantCoefficient>(nb::cast<double>(o));
  AXF_REQUIRE(nb::hasattr(o, "__call__"), "a scalar coefficient must be a number or a callable f(x, y)");
  return std::make_unique<PyScalar>(PyFn{o, vec, 1});
}

std::unique_ptr<mfem::VectorCoefficient> vector_coef(nb::object o, int vdim, bool vec) {
  if (nb::hasattr(o, "__call__")) return std::make_unique<PyVector>(PyFn{o, vec, vdim});
  auto v = nb::cast<std::vector<double>>(o);
  AXF_REQUIRE(static_cast<int>(v.size()) == vdim, axf::strcat("a constant vector has ", v.size(), " components; ", vdim,
                                                          " expected"));
  mfem::Vector mv(vdim);
  for (int i = 0; i < vdim; ++i) mv(i) = v[i];
  return std::make_unique<mfem::VectorConstantCoefficient>(mv);
}

// --- mesh
struct PyMesh {
  std::shared_ptr<mfem::Mesh> serial;
  std::shared_ptr<mfem::ParMesh> par_;
  explicit PyMesh(mfem::Mesh&& m) : serial(std::make_shared<mfem::Mesh>(std::move(m))) {}
  std::shared_ptr<mfem::ParMesh> par() {
    ensure_mpi();
    if (!par_) par_ = par_mesh(MPI_COMM_WORLD, *serial);
    return par_;
  }
  int n_elements() const { return serial->GetNE(); }
  nb::ndarray<nb::numpy, double> vertices() const {
    std::vector<double> v(2 * serial->GetNV());
    for (int i = 0; i < serial->GetNV(); ++i) {
      v[2 * i] = serial->GetVertex(i)[0];
      v[2 * i + 1] = serial->GetVertex(i)[1];
    }
    return np_array(std::move(v), {static_cast<size_t>(serial->GetNV()), 2});
  }
  nb::ndarray<nb::numpy, int> elements() const {
    const int k = serial->GetElement(0)->GetNVertices();
    std::vector<int> e;
    mfem::Array<int> vs;
    for (int i = 0; i < serial->GetNE(); ++i) {
      serial->GetElementVertices(i, vs);
      AXF_REQUIRE(vs.Size() == k, "elements(): mixed element types");
      for (int j : vs) e.push_back(j);
    }
    return np_iarray(std::move(e), {static_cast<size_t>(serial->GetNE()), static_cast<size_t>(k)});
  }
  nb::ndarray<nb::numpy, double> p2_nodes() const {
    auto v = element_p2_nodes(*serial);
    return np_array(std::move(v), {static_cast<size_t>(serial->GetNE()), 6, 2});
  }
  nb::tuple boundary() const {
    std::vector<int> seg, at;
    mfem::Array<int> vs;
    for (int i = 0; i < serial->GetNBE(); ++i) {
      serial->GetBdrElementVertices(i, vs);
      seg.push_back(vs[0]);
      seg.push_back(vs[1]);
      at.push_back(serial->GetBdrAttribute(i));
    }
    const size_t nb_ = at.size();
    return nb::make_tuple(np_iarray(std::move(seg), {nb_, 2}), np_iarray(std::move(at), {nb_}));
  }
  nb::dict taylor_hood_check() const {
    const auto r = check_taylor_hood_mesh(*serial);
    nb::dict d;
    d["checked_elements"] = r.checked_elements;
    d["all_boundary_vertices"] = r.all_boundary_vertices;
    d["tensor_all_boundary_vertices"] = r.tensor_all_boundary_vertices;
    return d;
  }
};

PyMesh mesh_from_arrays_py(Arr2 xy, IArr2 elements, IArr2 bdr, IArr1 battr, nb::object eattr, nb::object p2) {
  AXF_REQUIRE(xy.shape(1) == 2, "vertices must be an (n, 2) array");
  std::vector<double> v(xy.data(), xy.data() + xy.size());
  std::vector<int> e(elements.data(), elements.data() + elements.size());
  std::vector<int> b(bdr.data(), bdr.data() + bdr.size()), ba(battr.data(), battr.data() + battr.size());
  std::vector<int> ea;
  if (!eattr.is_none()) ea = nb::cast<std::vector<int>>(eattr);
  std::vector<double> q;
  if (!p2.is_none()) {
    auto arr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(p2);
    q.assign(arr.data(), arr.data() + arr.size());
  }
  const int vpe = static_cast<int>(elements.shape(1));
  return PyMesh(mesh_from_arrays(v, e, vpe, b, ba, ea, q));
}

// --- solution
BCKind bc_kind(const std::string& s) {
  if (s == "dirichlet") return BCKind::Dirichlet;
  if (s == "do_nothing") return BCKind::DoNothing;
  if (s == "traction") return BCKind::Traction;
  if (s == "axis") return BCKind::Axis;
  if (s == "slip") return BCKind::Slip;
  if (s == "symmetry") return BCKind::Symmetry;
  if (s == "periodic") return BCKind::Periodic;
  throw axf::Error("unknown boundary-condition kind '" + s + "'", "bcs",
                   "dirichlet, do_nothing, traction, axis, slip, symmetry, periodic");
}

// the problem data a solver keeps pointers to (mesh, coefficients, boundary values): owned by the solution object
struct ProblemData {
  std::shared_ptr<mfem::ParMesh> mesh;
  std::unique_ptr<mfem::Coefficient> nu;
  std::unique_ptr<mfem::VectorCoefficient> force;
  std::vector<std::unique_ptr<mfem::VectorCoefficient>> bc_values;
  // inductionless MHD: conductivity, imposed field, current source, electric condition values
  std::unique_ptr<mfem::Coefficient> sigma, qsrc;
  std::unique_ptr<mfem::VectorCoefficient> Bfield;
  std::vector<std::unique_ptr<mfem::Coefficient>> e_values;
  MhdOptions mopt;
  bool has_mhd = false;
  nb::object cb;  // the time callback of a transient solver (held for the GC)
  int vdim = 2;

  // evaluation at reference points of the given (local == global on one rank) elements
  template <class F>
  void each_point(IArr1 elems, Arr2 ref, F&& f) const {
    AXF_REQUIRE(world_size() == 1, "point evaluation needs one MPI rank (element numbers are rank-local)");
    AXF_REQUIRE(ref.shape(1) == 2, "reference points must be an (q, 2) array");
    for (size_t a = 0; a < elems.shape(0); ++a) {
      const int e = elems.data()[a];
      AXF_REQUIRE(e >= 0 && e < mesh->GetNE(), axf::strcat("element ", e, " out of range"));
      mfem::ElementTransformation* T = mesh->GetElementTransformation(e);
      for (size_t q = 0; q < ref.shape(0); ++q) {
        mfem::IntegrationPoint ip;
        ip.Set2(ref.data()[2 * q], ref.data()[2 * q + 1]);
        T->SetIntPoint(&ip);
        f(*T, ip);
      }
    }
  }

  nb::tuple velocity_of(const mfem::ParGridFunction& u, IArr1 elems, Arr2 ref) const {
    std::vector<double> U, G;
    U.reserve(elems.shape(0) * ref.shape(0) * vdim);
    G.reserve(elems.shape(0) * ref.shape(0) * vdim * 2);
    mfem::Vector uv(vdim);
    mfem::DenseMatrix g(vdim, 2);
    each_point(elems, ref, [&](mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
      u.GetVectorValue(T, ip, uv);
      u.GetVectorGradient(T, g);
      for (int i = 0; i < vdim; ++i) U.push_back(uv(i));
      for (int i = 0; i < vdim; ++i)
        for (int j = 0; j < 2; ++j) G.push_back(g(i, j));
    });
    const size_t m = elems.shape(0), q = ref.shape(0), vd = static_cast<size_t>(vdim);
    return nb::make_tuple(np_array(std::move(U), {m, q, vd}), np_array(std::move(G), {m, q, vd, 2}));
  }
  nb::ndarray<nb::numpy, double> pressure_of(const mfem::ParGridFunction& p, IArr1 elems, Arr2 ref) const {
    std::vector<double> P;
    each_point(elems, ref, [&](mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
      P.push_back(p.GetValue(T, ip));
    });
    return np_array(std::move(P), {elems.shape(0), ref.shape(0)});
  }
  nb::tuple geometry(IArr1 elems, Arr2 ref) const {
    std::vector<double> X, J;
    each_point(elems, ref, [&](mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
      double x[3];
      mfem::Vector xv(x, T.GetSpaceDim());
      T.Transform(ip, xv);
      X.push_back(x[0]);
      X.push_back(x[1]);
      J.push_back(T.Weight());
    });
    const size_t m = elems.shape(0), q = ref.shape(0);
    return nb::make_tuple(np_array(std::move(X), {m, q, 2}), np_array(std::move(J), {m, q}));
  }

  // the velocity DOF values at the nodes of the boundary elements of `attributes` (each node once per boundary
  // element side it is seen from; coordinates from that element's geometry): for checks of nodal Dirichlet data
  nb::tuple boundary_nodes_of(const mfem::ParGridFunction& u, std::vector<int> attrs) const {
    AXF_REQUIRE(world_size() == 1, "boundary_nodes() needs one MPI rank");
    mfem::ParFiniteElementSpace* fes = u.ParFESpace();
    std::vector<double> X, U;
    std::vector<int> seen(fes->GetNDofs(), 0);
    mfem::Array<int> dofs;
    for (int be = 0; be < mesh->GetNBE(); ++be) {
      if (std::find(attrs.begin(), attrs.end(), mesh->GetBdrAttribute(be)) == attrs.end()) continue;
      fes->GetBdrElementDofs(be, dofs);
      const mfem::IntegrationRule& nodes = fes->GetBE(be)->GetNodes();
      mfem::ElementTransformation* T = mesh->GetBdrElementTransformation(be);
      for (int k = 0; k < dofs.Size(); ++k) {
        if (seen[dofs[k]]++) continue;
        double x[3];
        mfem::Vector xv(x, 2);
        T->Transform(nodes.IntPoint(k), xv);
        X.push_back(x[0]);
        X.push_back(x[1]);
        for (int c = 0; c < vdim; ++c) U.push_back(u(fes->DofToVDof(dofs[k], c)));
      }
    }
    const size_t n = X.size() / 2;
    return nb::make_tuple(np_array(std::move(X), {n, 2}), np_array(std::move(U), {n, static_cast<size_t>(vdim)}));
  }
};

struct PySolution : ProblemData {
  std::unique_ptr<StokesSolver> S;
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
    r["pinned_dof"] = d.pinned_dof;
    r["pressure_point_value"] = d.pressure_point_value;
    r["solver"] = d.solver;
    r["viscous_form"] = d.viscous_form;
    r["grad_div"] = d.grad_div;
    r["slip_nodes"] = d.n_slip_nodes;
    r["slip_corner_nodes"] = d.n_slip_corner_nodes;
    r["axis_point_nodes"] = d.n_axis_point_nodes;
    r["slip_max_averaged_kink_deg"] = d.slip_max_averaged_kink_deg;
    r["periodic_pressure_gradient"] = d.periodic_pressure_gradient;
    r["periodic_flow_rate"] = d.periodic_flow_rate;
    r["dofs_velocity"] = d.n_velocity;
    r["dofs_pressure"] = d.n_pressure;
    r["dofs_field"] = d.n_total;
    r["lu_residual_before_refinement"] = d.lu.rel_residual_before;
    r["lu_residual"] = d.lu.rel_residual;
    r["lu_refine_steps"] = d.lu.refine_steps;
    r["lu_backend"] = d.lu.backend;
    r["ranks"] = d.ranks;
    r["threads"] = d.threads;
    r["velocity_nullspace"] = d.velocity_nullspace;
    r["velocity_nullspace_residual"] = d.velocity_nullspace_residual;
    r["velocity_nullspace_discrete_residual"] = d.velocity_nullspace_discrete_residual;
    r["velocity_nullspace_near"] = d.velocity_nullspace_near;
    r["lu_threads"] = d.lu.threads;
    r["direct_tol"] = S->options().direct_tol;
    r["seconds"] = d.assemble_seconds + d.setup_seconds + d.solve_seconds;
    r["assemble_seconds"] = d.assemble_seconds;
    r["solve_seconds"] = d.setup_seconds + d.solve_seconds;
    return r;
  }

  nb::tuple velocity(IArr1 elems, Arr2 ref) const { return velocity_of(S->velocity(), elems, ref); }
  nb::ndarray<nb::numpy, double> pressure(IArr1 elems, Arr2 ref) const { return pressure_of(S->pressure(), elems, ref); }
  nb::tuple boundary_nodes(std::vector<int> attrs) const { return boundary_nodes_of(S->velocity(), attrs); }

  nb::dict reaction(std::vector<int> attrs, std::vector<double> center, const std::string& form) const {
    AXF_REQUIRE(form == "solved" || form == "laplacian" || form == "stress",
                "reaction residual_form must be 'solved', 'stress' or 'laplacian'");
    const Reaction r = form == "solved" ? S->physical_reaction(attrs, center)
                                        : S->reaction(attrs, center, form == "laplacian" ? ViscousForm::Laplacian
                                                                                         : ViscousForm::SymmetricStress);
    nb::dict d;
    d["force"] = std::vector<double>(r.force, r.force + 3);
    d["torque"] = r.torque;
    d["residual_form"] = r.residual_form;
    return d;
  }

  static nb::dict csr(const mfem::HypreParMatrix& M0) {
    mfem::HypreParMatrix& M = const_cast<mfem::HypreParMatrix&>(M0);
    mfem::SparseMatrix s;
    M.MergeDiagAndOffd(s);
    nb::dict d;
    const size_t nr = static_cast<size_t>(s.Height()), nnz = static_cast<size_t>(s.NumNonZeroElems());
    d["indptr"] = np_iarray(std::vector<int>(s.GetI(), s.GetI() + nr + 1), {nr + 1});
    d["indices"] = np_iarray(std::vector<int>(s.GetJ(), s.GetJ() + nnz), {nnz});
    d["data"] = np_array(std::vector<double>(s.GetData(), s.GetData() + nnz), {nnz});
    d["shape"] = nb::make_tuple(static_cast<long>(M.GetGlobalNumRows()), static_cast<long>(M.GetGlobalNumCols()));
    return d;
  }
  nb::dict blocks() const {
    AXF_REQUIRE(world_size() == 1, "blocks() needs one MPI rank");
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

std::vector<double> vec_or_empty(nb::handle h) {
  if (!h.is_valid() || h.is_none()) return {};
  return nb::cast<std::vector<double>>(h);
}

// The Python callables a solution holds (viscosity, force, boundary data) are shown to the garbage collector: a
// callable defined in a module (or a closure) refers back to the solution through its globals, and without
// tp_traverse that cycle is invisible to the GC and leaks at interpreter exit (nanobind's leak report). Same
// approach as the chemistry bindings; no global suppression of nanobind's leak warnings.
template <typename F>
void for_each_held_callable(ProblemData& s, F&& f) {
  auto one = [&](auto* c) {
    if (!c) return;
    if (auto* a = dynamic_cast<PyScalar*>(c)) f(a->py());
    if (auto* b = dynamic_cast<PyVector*>(c)) f(b->py());
    if (auto* n = dynamic_cast<PyVectorNormal*>(c)) f(n->py());
  };
  if (s.cb.is_valid()) f(s.cb);
  one(static_cast<mfem::Coefficient*>(s.nu.get()));
  one(static_cast<mfem::VectorCoefficient*>(s.force.get()));
  one(static_cast<mfem::Coefficient*>(s.sigma.get()));
  one(static_cast<mfem::Coefficient*>(s.qsrc.get()));
  one(static_cast<mfem::VectorCoefficient*>(s.Bfield.get()));
  for (auto& v : s.e_values) one(static_cast<mfem::Coefficient*>(v.get()));
  for (auto& v : s.bc_values) one(static_cast<mfem::VectorCoefficient*>(v.get()));
}

template <class Sol>
int solution_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  int rc = 0;
  for_each_held_callable(*nb::inst_ptr<Sol>(self), [&](nb::object& o) {
    if (!rc && o.ptr()) rc = visit(o.ptr(), arg);
  });
  return rc;
}

template <class Sol>
int solution_tp_clear(PyObject* self) {
  if (nb::inst_ready(self))
    for_each_held_callable(*nb::inst_ptr<Sol>(self), [](nb::object& o) { o = nb::object(); });
  return 0;
}

// (the same slots for NavierStokesSolution, which owns the same ProblemData)
PyType_Slot solution_slots[] = {{Py_tp_traverse, (void*)solution_tp_traverse<PySolution>},
                                {Py_tp_clear, (void*)solution_tp_clear<PySolution>}, {0, nullptr}};

// Options, conditions and coefficients from the Python arguments (shared by the Stokes and Navier-Stokes entry points)
void setup_problem(ProblemData& sol, StokesOptions& o, StokesBoundary& b, PyMesh& mesh, nb::object nu,
                   nb::object force, nb::list bcs, const std::string& form, const std::string& geometry,
                   const std::string& solver, const std::string& lu, const std::string& nullspace,
                   nb::object pressure_point, int order, double rtol, double direct_tol, double grad_div,
                   int pressure_order, bool allow_unstable_pair, bool vectorized, int min_refine_steps, int lu_threads,
                   double slip_corner_angle_deg, bool refuse_velocity_nullspace, double velocity_nullspace_tol) {
  o.velocity_order = order;
  o.pressure_order = pressure_order;
  o.allow_unstable_pair = allow_unstable_pair;
  o.grad_div = grad_div;
  o.min_refine_steps = min_refine_steps;
  o.lu_threads = lu_threads;
  o.slip_corner_angle_deg = slip_corner_angle_deg;
  o.refuse_velocity_nullspace = refuse_velocity_nullspace;
  o.velocity_nullspace_tol = velocity_nullspace_tol;
  o.form = form == "laplacian" ? ViscousForm::Laplacian : ViscousForm::SymmetricStress;
  AXF_REQUIRE(form == "laplacian" || form == "stress", "form must be 'laplacian' or 'stress'");
  AXF_REQUIRE(geometry == "planar" || geometry == "axisymmetric" || geometry == "planar25d",
              "geometry must be 'planar', 'axisymmetric' or 'planar25d'");
  o.geometry = geometry == "axisymmetric" ? Geometry::Axisymmetric
               : geometry == "planar25d"  ? Geometry::Planar25D
                                          : Geometry::Planar;
  sol.vdim = o.geometry == Geometry::Planar ? 2 : 3;
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
  if (!pressure_point.is_none()) {
    nb::dict pp = nb::cast<nb::dict>(pressure_point);
    o.pressure_point.active = true;
    o.pressure_point.x0 = nb::cast<std::vector<double>>(pp["x0"]);
    o.pressure_point.value = nb::cast<double>(pp["value"]);
  }
  std::vector<PeriodicPair> pairs;
  std::vector<std::pair<int, nb::object>> normal_values;  // traction data g(x, y, n): bound after the mesh is known
  for (nb::handle h : bcs) {
    nb::dict d = nb::cast<nb::dict>(h);
    BoundaryCondition c;
    c.attribute = nb::cast<int>(d["attribute"]);
    c.kind = bc_kind(nb::cast<std::string>(d["kind"]));
    for (nb::handle k : d.keys()) {
      const std::string key = nb::cast<std::string>(k);
      static const char* known[] = {"attribute", "kind", "value", "normal", "partner", "translation",
                                    "pressure_drop", "flow_rate", "normal_dependent"};
      AXF_REQUIRE(std::find_if(std::begin(known), std::end(known), [&](const char* s) { return key == s; }) !=
                      std::end(known),
                  "unknown key '" + key + "' in a boundary condition", "bcs",
                  "attribute, kind, value, normal (slip/symmetry), partner, translation, pressure_drop, flow_rate "
                  "(periodic)");
    }
    if (d.contains("value") && !d["value"].is_none()) {
      AXF_REQUIRE(c.kind == BCKind::Dirichlet || c.kind == BCKind::Traction,
                  axf::strcat("attribute ", c.attribute, ": 'value' is only for dirichlet and traction"), "bcs");
      const bool by_normal = d.contains("normal_dependent") && nb::cast<bool>(d["normal_dependent"]);
      if (by_normal) {
        AXF_REQUIRE(c.kind == BCKind::Traction && nb::hasattr(d["value"], "__call__"),
                    axf::strcat("attribute ", c.attribute, ": normal_dependent needs a traction callable "
                                "g(x, y, n_x, n_y)"), "bcs");
        normal_values.push_back({c.attribute, nb::borrow(d["value"])});
      } else {
        sol.bc_values.push_back(vector_coef(nb::borrow(d["value"]), sol.vdim, vectorized));
        c.value = sol.bc_values.back().get();
      }
    }
    if (d.contains("normal") && !d["normal"].is_none()) {
      AXF_REQUIRE(c.kind == BCKind::Slip || c.kind == BCKind::Symmetry,
                  axf::strcat("attribute ", c.attribute, ": 'normal' is only for slip and symmetry"), "bcs");
      sol.bc_values.push_back(vector_coef(nb::borrow(d["normal"]), sol.vdim, vectorized));
      c.normal = sol.bc_values.back().get();
    }
    if (c.kind == BCKind::Periodic) {
      AXF_REQUIRE(d.contains("partner") && d.contains("translation"),
                  axf::strcat("periodic attribute ", c.attribute, " needs 'partner' and 'translation'"), "bcs");
      c.partner = nb::cast<int>(d["partner"]);
      c.translation = vec_or_empty(d["translation"]);
      const bool dp = d.contains("pressure_drop") && !d["pressure_drop"].is_none();
      const bool q = d.contains("flow_rate") && !d["flow_rate"].is_none();
      AXF_REQUIRE(!(dp && q), axf::strcat("periodic attribute ", c.attribute,
                                          ": prescribe either pressure_drop or flow_rate, not both"), "bcs");
      if (dp) {
        c.drive = PeriodicDrive::PressureDrop;
        c.drive_value = nb::cast<double>(d["pressure_drop"]);
      } else if (q) {
        c.drive = PeriodicDrive::FlowRate;
        c.drive_value = nb::cast<double>(d["flow_rate"]);
      }
      pairs.push_back({c.attribute, c.partner, c.translation});
    } else {
      AXF_REQUIRE(!d.contains("partner") && !d.contains("translation") && !d.contains("pressure_drop") &&
                      !d.contains("flow_rate"),
                  axf::strcat("attribute ", c.attribute, ": partner/translation/pressure_drop/flow_rate are only for "
                              "periodic conditions"), "bcs");
    }
    b.conditions.push_back(c);
  }
  // the mesh the solver sees: the input mesh, made periodic if needed
  if (!pairs.empty()) {
    mfem::Mesh base(*mesh.serial, true);
    mfem::Mesh m = make_periodic_mesh(base, pairs);
    sol.mesh = par_mesh(MPI_COMM_WORLD, m);
  } else {
    sol.mesh = mesh.par();
  }
  for (auto& nv : normal_values) {
    sol.bc_values.push_back(std::make_unique<PyVectorNormal>(nv.second, sol.vdim, vectorized, sol.mesh.get()));
    for (auto& c : b.conditions)
      if (c.attribute == nv.first) c.value = sol.bc_values.back().get();
  }
  // coefficients after the options: tabulated on exactly the quadrature rules the solver uses (vectorized mode)
  sol.nu = scalar_coef(nu, vectorized);
  o.nu = sol.nu.get();
  if (auto* ps = dynamic_cast<PyScalar*>(sol.nu.get())) ps->check_positive = true;
  if (vectorized)
    if (auto* ps = dynamic_cast<PyScalar*>(sol.nu.get())) {
      ps->table.build(ps->fn(), *sol.mesh, o.viscous_quad_order());
      for (const auto& e : ps->table.vals)
        for (double v : e)
          AXF_REQUIRE(std::isfinite(v) && v > 0, axf::strcat("the viscosity callable returned ", v,
                                                              " at a quadrature point: it must be positive and finite"),
                      "nu");
    }
  if (!force.is_none()) {
    sol.force = vector_coef(force, sol.vdim, vectorized);
    o.force = sol.force.get();
    if (vectorized)
      if (auto* pv = dynamic_cast<PyVector*>(sol.force.get())) pv->table.build(pv->fn(), *sol.mesh, o.load_quad_order());
  }
}

std::unique_ptr<PySolution> solve_stokes(PyMesh& mesh, nb::object nu, nb::object force, nb::list bcs,
                                         const std::string& form, const std::string& geometry,
                                         const std::string& solver, const std::string& lu,
                                         const std::string& nullspace, nb::object pressure_point, int order,
                                         double rtol, double direct_tol, double grad_div,
                                         int pressure_order, bool allow_unstable_pair, bool vectorized,
                                         int min_refine_steps, bool do_solve, int lu_threads,
                                         double slip_corner_angle_deg, bool refuse_velocity_nullspace,
                                         double velocity_nullspace_tol) {
  ensure_mpi();
  // (an MFEM verification failure aborts the process unless MFEM is built with MFEM_USE_EXCEPTIONS=YES; the inputs
  // MFEM would reject are validated here)
  auto sol = std::make_unique<PySolution>();
  StokesOptions o;
  StokesBoundary b;
  setup_problem(*sol, o, b, mesh, nu, force, bcs, form, geometry, solver, lu, nullspace, pressure_point, order, rtol,
                direct_tol, grad_div, pressure_order, allow_unstable_pair, vectorized, min_refine_steps, lu_threads,
                slip_corner_angle_deg, refuse_velocity_nullspace, velocity_nullspace_tol);
  sol->S = std::make_unique<StokesSolver>(*sol->mesh, o, b);
  if (do_solve) {
    sol->S->solve();
    sol->solved = true;
  } else {
    sol->S->assemble();
  }
  return sol;
}

// --- steady Navier–Stokes
nb::dict csr_dict(const DistCsr& D) {
  nb::dict d;
  const size_t nr = static_cast<size_t>(D.local_rows), nnz = D.val.size();
  d["indptr"] = np_iarray(std::vector<int>(D.rowptr.begin(), D.rowptr.end()), {nr + 1});
  d["indices"] = np_iarray(std::vector<int>(D.col.begin(), D.col.end()), {nnz});
  d["data"] = np_array(std::vector<double>(D.val.begin(), D.val.end()), {nnz});
  d["shape"] = nb::make_tuple(static_cast<long>(D.n_global), static_cast<long>(D.n_global));
  return d;
}

// the diagnostics dict of a Navier-Stokes solve (steady solution, or the last step of a transient solver)
nb::dict ns_diagnostics_dict(const NavierStokesSolver* S, bool has_mhd) {
  const auto& n = S->diagnostics();
  const auto& d = n.linear;
  nb::dict r;
  r["converged"] = n.converged;
  r["relative_residual"] = n.relative_residual;
  r["residual_lift"] = n.residual_lift;
  r["iterations"] = n.iterations;
  r["line_search_min_step_accepts"] = n.line_search_min_step_accepts;
  r["convection"] = n.convection;
  r["strategy"] = n.strategy;
  r["failure"] = n.failure;
  r["quad_orders"] = n.quad_orders;
  r["viscosity_scales"] = n.viscosity_scales;
  r["seconds"] = n.seconds;
  r["nullspace_detected"] = d.nullspace_detected;
  r["bc_enclosed"] = d.bc_enclosed;
  r["kernel_defect"] = d.kernel_defect;
  r["nullspace"] = d.nullspace;
  r["pinned_dof"] = d.pinned_dof;
  r["flux_defect"] = d.flux_defect;
  r["flux_defect_abs"] = d.flux_defect_abs;
  r["solver"] = d.solver;
  r["viscous_form"] = d.viscous_form;
  r["axis_point_nodes"] = d.n_axis_point_nodes;
  r["dofs_velocity"] = d.n_velocity;
  r["dofs_pressure"] = d.n_pressure;
  r["dofs_field"] = d.n_total;
  r["lu_backend"] = d.lu.backend;
  r["lu_residual_before_refinement"] = d.lu.rel_residual_before;
  r["lu_residual"] = d.lu.rel_residual;
  r["lu_refine_steps"] = d.lu.refine_steps;
  r["lu_threads"] = d.lu.threads;
  r["lu_fallbacks"] = n.lu_fallbacks;
  r["lu_fallback_backend"] = n.lu_fallback_backend;
  r["lu_factor_seconds"] = d.lu.factor_seconds;
  r["lu_analyze_seconds"] = d.lu.analyze_seconds;
  r["lu_solve_seconds"] = d.lu.solve_seconds;
  r["lu_n"] = d.lu.n;
  r["lu_nnz"] = d.lu.nnz;
  r["ranks"] = d.ranks;
  r["threads"] = d.threads;
  r["rho"] = S->options().rho;
  r["newton_rtol"] = S->options().nonlinear.newton_rtol;
  if (has_mhd) {
    const auto& m = S->mhd_diagnostics();
    r["mhd"] = true;
    r["dofs_potential"] = m.n_potential;
    r["potential_order"] = m.potential_order;
    r["phi_nullspace_detected"] = m.phi_nullspace_detected;
    r["phi_nullspace_declared"] = m.phi_nullspace_declared;
    r["phi_kernel_defect"] = m.phi_kernel_defect;
    r["phi_flux_defect_abs"] = m.phi_flux_defect_abs;
    r["phi_pinned_dof"] = m.phi_pinned_dof;
    r["phi_dirichlet_dofs"] = m.n_potential_dirichlet;
    r["mhd_quad_order"] = m.quad_order;
    r["mhd_quad_order_boundary"] = m.quad_order_boundary;
    r["mhd_aux_unknowns"] = m.n_aux;
    r["current_form"] = m.current_form;
    r["mhd_assemble_seconds"] = m.assemble_seconds;
    r["mhd_phi_scale"] = m.phi_scale;
    r["mhd_row_scale"] = m.row_scale;
  } else {
    r["mhd"] = false;
  }
  return r;
}

struct PyNSSolution : ProblemData {
  std::unique_ptr<NavierStokesSolver> S;

  nb::dict diagnostics() const { return ns_diagnostics_dict(S.get(), has_mhd); }
  nb::tuple potential(IArr1 elems, Arr2 ref) const {
    const mfem::GridFunction& g = S->potential();
    std::vector<double> P, G;
    each_point(elems, ref, [&](mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) {
      P.push_back(g.GetValue(T, ip));
      mfem::Vector gr;
      g.GetGradient(T, gr);
      G.push_back(gr(0));
      G.push_back(gr(1));
    });
    const size_t m = elems.shape(0), q = ref.shape(0);
    return nb::make_tuple(np_array(std::move(P), {m, q}), np_array(std::move(G), {m, q, 2}));
  }
  nb::tuple potential_boundary_nodes(std::vector<int> attrs) const {
    AXF_REQUIRE(world_size() == 1, "potential_boundary_nodes() needs one MPI rank");
    const mfem::GridFunction& g = S->potential();
    const mfem::FiniteElementSpace* fes = g.FESpace();
    std::vector<double> X, U;
    std::vector<int> seen(fes->GetNDofs(), 0);
    mfem::Array<int> dofs;
    for (int be = 0; be < mesh->GetNBE(); ++be) {
      if (std::find(attrs.begin(), attrs.end(), mesh->GetBdrAttribute(be)) == attrs.end()) continue;
      fes->GetBdrElementDofs(be, dofs);
      const mfem::IntegrationRule& nodes = fes->GetBE(be)->GetNodes();
      mfem::ElementTransformation* T = mesh->GetBdrElementTransformation(be);
      for (int k = 0; k < dofs.Size(); ++k) {
        if (seen[dofs[k]]++) continue;
        double x[3];
        mfem::Vector xv(x, 2);
        T->Transform(nodes.IntPoint(k), xv);
        X.push_back(x[0]);
        X.push_back(x[1]);
        U.push_back(g(dofs[k]));
      }
    }
    const size_t n = U.size();
    return nb::make_tuple(np_array(std::move(X), {n, 2}), np_array(std::move(U), {n}));
  }
  nb::list history() const {
    nb::list l;
    for (const auto& h : S->diagnostics().history) {
      nb::dict d;
      d["kind"] = h.kind;
      d["residual"] = h.residual;
      d["step"] = h.step;
      d["dt"] = h.dt;
      d["stage"] = h.stage;
      d["viscosity_scale"] = h.viscosity_scale;
      d["lu_refine"] = h.lu_refine;
      l.append(d);
    }
    return l;
  }
  nb::tuple velocity(IArr1 elems, Arr2 ref) const { return velocity_of(S->velocity(), elems, ref); }
  nb::ndarray<nb::numpy, double> pressure(IArr1 elems, Arr2 ref) const { return pressure_of(S->pressure(), elems, ref); }
  nb::tuple boundary_nodes(std::vector<int> attrs) const { return boundary_nodes_of(S->velocity(), attrs); }
  static nb::ndarray<nb::numpy, double> arr(std::vector<double> v) {
    const size_t n = v.size();
    return np_array(std::move(v), {n});
  }
  nb::ndarray<nb::numpy, double> state() const { return arr(S->state()); }
  nb::ndarray<nb::numpy, double> residual(std::vector<double> x, double s) const {
    return arr(S->residual(x, s));
  }
  nb::ndarray<nb::numpy, double> jacobian_action(std::vector<double> x, std::vector<double> dvec, double s) const {
    return arr(S->jacobian_action(x, dvec, s));
  }
  nb::dict jacobian(std::vector<double> x, double s) const {
    return csr_dict(S->jacobian(x, s));
  }
  double dual_norm(std::vector<double> R) const { return S->dual_norm(R); }
  nb::tuple layout() const {
    const auto l = S->layout();
    return nb::make_tuple(l.first, l.second);
  }
  nb::dict full_residual() const {
    const FullResidual f = S->full_residual();
    nb::dict d;
    d["r_u"] = arr(f.r_u);
    d["r_p"] = arr(f.r_p);
    d["node_xy"] = np_array(f.node_xy, {f.node_xy.size() / 2, 2});
    nb::dict an;
    for (const auto& kv : f.attribute_nodes) {
      std::vector<int> v(kv.second.begin(), kv.second.end());
      const size_t n = v.size();
      an[nb::int_(kv.first)] = np_iarray(std::move(v), {n});
    }
    d["attribute_nodes"] = an;
    return d;
  }
  nb::dict reaction(std::vector<int> attrs, std::vector<double> center, const std::string& form) const {
    AXF_REQUIRE(form == "solved" || form == "laplacian" || form == "stress",
                "reaction residual_form must be 'solved', 'stress' or 'laplacian'");
    const ViscousForm f = form == "solved" ? S->options().stokes.form
                                           : (form == "laplacian" ? ViscousForm::Laplacian : ViscousForm::SymmetricStress);
    const Reaction r = S->reaction(attrs, center, f);
    nb::dict d;
    d["force"] = std::vector<double>(r.force, r.force + 3);
    d["torque"] = r.torque;
    d["residual_form"] = r.residual_form;
    return d;
  }
};

// the GC slots of NavierStokesSolution (it owns the same Python callables as StokesSolution)
PyType_Slot ns_solution_slots[] = {{Py_tp_traverse, (void*)solution_tp_traverse<PyNSSolution>},
                                   {Py_tp_clear, (void*)solution_tp_clear<PyNSSolution>}, {0, nullptr}};

NonlinearStrategy strategy_from(const std::string& s) {
  if (s == "newton") return NonlinearStrategy::Newton;
  if (s == "picard") return NonlinearStrategy::PicardOnly;
  if (s == "ptc") return NonlinearStrategy::PseudoTransient;
  throw axf::Error("unknown nonlinear strategy '" + s + "'", "strategy", "newton, picard, ptc");
}

std::unique_ptr<PyNSSolution> solve_navier_stokes(
    PyMesh& mesh, nb::object mu, double rho, nb::object force, nb::list bcs, const std::string& form,
    const std::string& convection, const std::string& geometry, int order, const std::string& lu,
    const std::string& nullspace, const std::string& strategy, int n_picard, bool line_search, double newton_rtol,
    int newton_maxit, int picard_maxit, double ptc_dt0, double ptc_release_dt, int ptc_maxit,
    std::vector<double> viscosity_scales, int quad_order_convection, double direct_tol, int lu_threads,
    bool vectorized, nb::object mhd, bool do_solve) {
  ensure_mpi();
  auto sol = std::make_unique<PyNSSolution>();
  NavierStokesOptions o;
  StokesBoundary b;
  setup_problem(*sol, o.stokes, b, mesh, mu, force, bcs, form, geometry, "direct", lu, nullspace, nb::none(), order,
                1e-10, direct_tol, 0.0, -1, false, vectorized, 0, lu_threads, 30.0, true, 1e-3);
  AXF_REQUIRE(convection == "standard" || convection == "skew", "convection must be 'standard' or 'skew'");
  o.rho = rho;
  o.convection = convection == "skew" ? ConvectiveForm::Skew : ConvectiveForm::Standard;
  o.quad_order_convection = quad_order_convection;
  auto& nl = o.nonlinear;
  nl.strategy = strategy_from(strategy);
  nl.n_picard = n_picard;
  nl.line_search = line_search;
  nl.newton_rtol = newton_rtol;
  nl.newton_maxit = newton_maxit;
  nl.picard_maxit = picard_maxit;
  nl.ptc_dt0 = ptc_dt0;
  nl.ptc_release_dt = ptc_release_dt;
  nl.ptc_maxit = ptc_maxit;
  nl.viscosity_scales = std::move(viscosity_scales);
  if (!mhd.is_none()) {
    nb::dict d = nb::cast<nb::dict>(mhd);
    static const char* known[] = {"sigma", "B", "E_z", "q", "electric", "potential_order", "quad_order"};
    for (nb::handle k : d.keys()) {
      const std::string key = nb::cast<std::string>(k);
      AXF_REQUIRE(std::find_if(std::begin(known), std::end(known), [&](const char* x) { return key == x; }) !=
                      std::end(known),
                  "unknown key '" + key + "' in the mhd options", "mhd",
                  "sigma, B, E_z, q, electric, potential_order, quad_order");
    }
    AXF_REQUIRE(d.contains("sigma") && d.contains("B") && d.contains("electric"),
                "the mhd options need sigma, B and electric", "mhd");
    sol->has_mhd = true;
    MhdOptions& m = sol->mopt;
    m.potential_order = d.contains("potential_order") ? nb::cast<int>(d["potential_order"]) : 2;
    m.E_z = d.contains("E_z") ? nb::cast<double>(d["E_z"]) : 0.0;
    const int qd = d.contains("quad_order") ? nb::cast<int>(d["quad_order"]) : 0;
    const int quad = qd > 0 ? qd : 2 * std::max(order, m.potential_order) + 4;
    m.quad_order = quad;
    sol->sigma = scalar_coef(nb::borrow(d["sigma"]), vectorized);
    m.sigma = sol->sigma.get();
    if (auto* ps = dynamic_cast<PyScalar*>(sol->sigma.get()))
      if (vectorized) ps->table.build(ps->fn(), *sol->mesh, quad);
    sol->Bfield = vector_coef(nb::borrow(d["B"]), 3, vectorized);
    m.B = sol->Bfield.get();
    if (vectorized)
      if (auto* pv = dynamic_cast<PyVector*>(sol->Bfield.get())) pv->table.build(pv->fn(), *sol->mesh, quad);
    if (d.contains("q") && !d["q"].is_none()) {
      sol->qsrc = scalar_coef(nb::borrow(d["q"]), vectorized);
      m.q = sol->qsrc.get();
      if (vectorized)
        if (auto* ps = dynamic_cast<PyScalar*>(sol->qsrc.get())) ps->table.build(ps->fn(), *sol->mesh, quad);
    }
    for (nb::handle h : nb::cast<nb::list>(d["electric"])) {
      nb::dict e = nb::cast<nb::dict>(h);
      ElectricCondition c;
      c.attribute = nb::cast<int>(e["attribute"]);
      const std::string k = nb::cast<std::string>(e["kind"]);
      if (k == "insulating") c.kind = ElectricKind::Insulating;
      else if (k == "normal_current") c.kind = ElectricKind::NormalCurrent;
      else if (k == "potential") c.kind = ElectricKind::Potential;
      else throw axf::Error("unknown electric condition '" + k + "'", "mhd.electric",
                            "insulating, normal_current, potential");
      if (e.contains("value") && !e["value"].is_none()) {
        sol->e_values.push_back(scalar_coef(nb::borrow(e["value"]), vectorized));
        c.value = sol->e_values.back().get();
      }
      m.electric.push_back(c);
    }
    o.mhd = &sol->mopt;
  }
  sol->S = std::make_unique<NavierStokesSolver>(*sol->mesh, o, b);
  if (do_solve) sol->S->solve();
  return sol;
}

// --- transient Navier-Stokes (transient_ns.hpp)
// A discrete state (velocity, pressure) as an Evaluator-ready field; the grid functions live on the solver's spaces, so
// the Python wrapper keeps the solver alive for as long as a snapshot is in use.
struct PyField : ProblemData {
  std::unique_ptr<mfem::ParGridFunction> ug, pg;
  nb::tuple velocity(IArr1 elems, Arr2 ref) const { return velocity_of(*ug, elems, ref); }
  nb::ndarray<nb::numpy, double> pressure(IArr1 elems, Arr2 ref) const { return pressure_of(*pg, elems, ref); }
  nb::tuple boundary_nodes(std::vector<int> attrs) const { return boundary_nodes_of(*ug, attrs); }
};

struct PyTransient : ProblemData {
  std::unique_ptr<TransientNavierStokes> T;
  bool vectorized = false;
  int load_order = 0;

  static nb::dict step_dict(const StepResult& r) {
    nb::dict d;
    d["converged"] = r.converged;
    d["relative_residual"] = r.relative_residual;
    d["iterations"] = r.iterations;
    d["lu_refine"] = r.lu_refine;
    d["t"] = r.t;
    d["dt"] = r.dt;
    d["order"] = r.order;
    d["has_estimate"] = r.has_estimate;
    d["est_norm"] = r.est_norm;
    d["ctrl_norm"] = r.ctrl_norm;
    d["failure"] = r.failure;
    return d;
  }
  std::unique_ptr<PyField> field(const mfem::Vector& u, const mfem::Vector& p, bool as_pressure_raw = false) const {
    auto f = std::make_unique<PyField>();
    f->mesh = mesh;
    f->vdim = vdim;
    NavierStokesSolver& core = T->core();
    f->ug.reset(new mfem::ParGridFunction(&core.velocity_space()));
    f->pg.reset(new mfem::ParGridFunction(&core.pressure_space()));
    f->ug->SetFromTrueDofs(u);
    mfem::Vector pn(p);
    if (!as_pressure_raw) core.normalize_pressure(pn);
    f->pg->SetFromTrueDofs(pn);
    return f;
  }
  void retabulate() {
    if (!vectorized) return;
    if (auto* pv = dynamic_cast<PyVector*>(force.get())) pv->table.build(pv->fn(), *mesh, load_order);
  }
  nb::dict diagnostics() const { return ns_diagnostics_dict(&T->core(), false); }
};

PyType_Slot transient_slots[] = {{Py_tp_traverse, (void*)solution_tp_traverse<PyTransient>},
                                 {Py_tp_clear, (void*)solution_tp_clear<PyTransient>}, {0, nullptr}};

std::unique_ptr<PyTransient> make_transient(
    PyMesh& mesh, nb::object mu, double rho, nb::object force, nb::list bcs, const std::string& form,
    const std::string& geometry, int order, const std::string& lu, const std::string& nullspace,
    const std::string& scheme, int max_order, double newton_rtol, int newton_maxit, int quad_order_convection,
    double direct_tol, int lu_threads, bool vectorized, nb::object time_cb) {
  ensure_mpi();
  auto sol = std::make_unique<PyTransient>();
  sol->vectorized = vectorized;
  sol->cb = time_cb;
  TransientOptions o;
  StokesBoundary b;
  setup_problem(*sol, o.ns.stokes, b, mesh, mu, force, bcs, form, geometry, "direct", lu, nullspace, nb::none(), order,
                1e-10, direct_tol, 0.0, -1, false, vectorized, 0, lu_threads, 30.0, true, 1e-3);
  AXF_REQUIRE(scheme == "newton" || scheme == "semi", "scheme must be 'newton' or 'semi'");
  o.scheme = scheme == "newton" ? TimeScheme::Newton : TimeScheme::SemiImplicit;
  o.max_order = max_order;
  o.ns.rho = rho;
  o.ns.convection = ConvectiveForm::Standard;
  o.ns.quad_order_convection = quad_order_convection;
  o.ns.nonlinear.newton_rtol = newton_rtol;
  o.ns.nonlinear.newton_maxit = newton_maxit;
  sol->load_order = o.ns.stokes.load_quad_order();
  PyTransient* raw = sol.get();
  auto hook = [raw](double tf, double tg, double ttr) {
    AXF_REQUIRE(raw->cb.is_valid(), "the time callback was released by the garbage collector (tp_clear)");
    raw->cb(tf, tg, ttr);
    raw->retabulate();
  };
  // first retabulation happens in the hook; the force table of setup_problem is built at t = 0 of the callback state
  sol->T = std::make_unique<TransientNavierStokes>(*sol->mesh, o, b, hook);
  return sol;
}


// ======================================================================================================== transport
// Scalar (temperature) transport with SUPG and Boussinesq natural convection.
// Data callables are vectorized: f(X, Y) with 1-D arrays returning (n,) for scalars and (n, 2) for 2-vectors (the
// Python layer normalizes), called ONCE per kind of point set by the operator's finalize().
struct DataStore {
  std::vector<nb::object> objs;
};
using StorePtr = std::shared_ptr<DataStore>;

void call_vec(DataStore& st, size_t idx, const double* xy, int n, int ncomp, double* out) {
  nb::object& f = st.objs[idx];
  AXF_REQUIRE(f.is_valid(), "a transport data callable was released by the garbage collector (tp_clear)");
  std::vector<double> xs(n), ys(n);
  for (int i = 0; i < n; ++i) {
    xs[i] = xy[2 * i];
    ys[i] = xy[2 * i + 1];
  }
  size_t shp[1] = {static_cast<size_t>(n)};
  nb::ndarray<nb::numpy, double> xa(xs.data(), 1, shp, nb::handle()), ya(ys.data(), 1, shp, nb::handle());
  nb::object r = f(xa, ya);
  auto arr = nb::cast<DArr>(r);
  AXF_REQUIRE(arr.size() == static_cast<size_t>(n) * ncomp,
              axf::strcat("a vectorized callable returned ", arr.size(), " values for ", n, " points x ", ncomp,
                          " components"));
  std::copy(arr.data(), arr.data() + static_cast<size_t>(n) * ncomp, out);
}

axf::transport::ScalarData to_scalar_data(const StorePtr& st, nb::handle h) {
  axf::transport::ScalarData d;
  if (!h.is_valid() || h.is_none()) return d;
  if (nb::hasattr(h, "__call__")) {
    st->objs.push_back(nb::borrow(h));
    const size_t idx = st->objs.size() - 1;
    d.fn = [st, idx](const double* xy, int n, double* out) { call_vec(*st, idx, xy, n, 1, out); };
  } else {
    d.value = nb::cast<double>(h);
  }
  return d;
}
axf::transport::BatchVec to_batch_vec(const StorePtr& st, nb::handle h) {
  axf::transport::BatchVec f;
  if (!h.is_valid() || h.is_none()) return f;
  AXF_REQUIRE(nb::hasattr(h, "__call__"), "a vector datum must be a callable f(X, Y) -> (n, 2)");
  st->objs.push_back(nb::borrow(h));
  const size_t idx = st->objs.size() - 1;
  f = [st, idx](const double* xy, int n, int nc, double* out) { call_vec(*st, idx, xy, n, nc, out); };
  return f;
}
axf::transport::BatchFn to_batch_fn(const StorePtr& st, nb::handle h) {
  axf::transport::BatchFn f;
  if (!h.is_valid() || h.is_none()) return f;
  AXF_REQUIRE(nb::hasattr(h, "__call__"), "a scalar datum must be a callable f(X, Y) -> (n,)");
  st->objs.push_back(nb::borrow(h));
  const size_t idx = st->objs.size() - 1;
  f = [st, idx](const double* xy, int n, double* out) { call_vec(*st, idx, xy, n, 1, out); };
  return f;
}

axf::transport::BcKind tbc_kind(const std::string& s) {
  using K = axf::transport::BcKind;
  if (s == "dirichlet") return K::Dirichlet;
  if (s == "flux") return K::Flux;
  if (s == "robin") return K::Robin;
  if (s == "total_flux") return K::TotalFlux;
  if (s == "zero" || s == "symmetry" || s == "insulation" || s == "outflow") return K::Zero;
  if (s == "axis") return K::Axis;
  throw axf::Error("unknown transport boundary-condition kind '" + s + "'", "bcs",
                   "dirichlet, flux (inward n.k grad T = g), robin (n.k grad T = h (T_ext - T)), total_flux, zero, axis");
}

struct TBcSpec {
  std::vector<int> attrs;
  axf::transport::BcKind kind;
  axf::transport::BcData data;
};
// conservative_flux: the natural datum is the TOTAL flux, so "zero" (and its aliases) is a zero total-flux wall; that is
// right for walls and symmetry lines (beta.n = 0) and wrong for an outflow, which would block the advective flux
// so "outflow" is refused there.
std::vector<TBcSpec> parse_tbcs(const StorePtr& st, nb::list bcs, bool conservative_flux = false) {
  std::vector<TBcSpec> out;
  for (nb::handle h : bcs) {
    nb::dict d = nb::cast<nb::dict>(h);
    for (nb::handle k : d.keys()) {
      const std::string key = nb::cast<std::string>(k);
      static const char* known[] = {"attribute", "attributes", "kind", "value", "h", "T_ext"};
      AXF_REQUIRE(std::find_if(std::begin(known), std::end(known), [&](const char* x) { return key == x; }) !=
                      std::end(known),
                  "unknown key '" + key + "' in a transport boundary condition", "bcs",
                  "attribute(s), kind, value, h, T_ext");
    }
    TBcSpec b;
    if (d.contains("attributes")) b.attrs = nb::cast<std::vector<int>>(d["attributes"]);
    else b.attrs = {nb::cast<int>(d["attribute"])};
    const std::string kind_s = nb::cast<std::string>(d["kind"]);
    AXF_REQUIRE(!(conservative_flux && kind_s == "outflow"),
                "'outflow' is refused with form='conservative_flux': the natural datum of that form is the TOTAL flux, so a "
                "zero condition would block the advective flux through the outlet instead of letting it pass",
                "bcs", "Use form='convective' (or 'conservative_divergence'), or give the outlet kind 'total_flux' "
                "with the advective flux you intend.");
    b.kind = tbc_kind(kind_s);
    if (d.contains("value")) b.data.g = to_scalar_data(st, d["value"]);
    if (d.contains("h")) b.data.h = to_scalar_data(st, d["h"]);
    if (d.contains("T_ext")) b.data.ext = to_scalar_data(st, d["T_ext"]);
    if (b.kind == axf::transport::BcKind::Robin)
      AXF_REQUIRE(d.contains("h") && d.contains("T_ext"), "a robin condition needs h and T_ext", "bcs");
    out.push_back(std::move(b));
  }
  return out;
}

// point evaluation of a scalar grid function (T, physical gradient) at reference points of elements
nb::tuple scalar_values(mfem::Mesh& mesh, const mfem::GridFunction& g, IArr1 elems, Arr2 ref) {
  AXF_REQUIRE(world_size() == 1, "point evaluation needs one MPI rank (element numbers are rank-local)");
  AXF_REQUIRE(ref.shape(1) == 2, "reference points must be an (q, 2) array");
  std::vector<double> V, G;
  mfem::Vector gv(2);
  for (size_t a = 0; a < elems.shape(0); ++a) {
    const int e = elems.data()[a];
    AXF_REQUIRE(e >= 0 && e < mesh.GetNE(), axf::strcat("element ", e, " out of range"));
    mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
    for (size_t q = 0; q < ref.shape(0); ++q) {
      mfem::IntegrationPoint ip;
      ip.Set2(ref.data()[2 * q], ref.data()[2 * q + 1]);
      T->SetIntPoint(&ip);
      V.push_back(g.GetValue(*T, ip));
      g.GetGradient(*T, gv);
      G.push_back(gv(0));
      G.push_back(gv(1));
    }
  }
  const size_t m = elems.shape(0), q = ref.shape(0);
  return nb::make_tuple(np_array(std::move(V), {m, q}), np_array(std::move(G), {m, q, 2}));
}
nb::tuple geometry_of(mfem::Mesh& mesh, IArr1 elems, Arr2 ref) {
  AXF_REQUIRE(world_size() == 1, "point evaluation needs one MPI rank (element numbers are rank-local)");
  std::vector<double> X, J;
  for (size_t a = 0; a < elems.shape(0); ++a) {
    const int e = elems.data()[a];
    AXF_REQUIRE(e >= 0 && e < mesh.GetNE(), axf::strcat("element ", e, " out of range"));
    mfem::ElementTransformation* T = mesh.GetElementTransformation(e);
    for (size_t q = 0; q < ref.shape(0); ++q) {
      mfem::IntegrationPoint ip;
      ip.Set2(ref.data()[2 * q], ref.data()[2 * q + 1]);
      T->SetIntPoint(&ip);
      double x[3];
      mfem::Vector xv(x, T->GetSpaceDim());
      T->Transform(ip, xv);
      X.push_back(x[0]);
      X.push_back(x[1]);
      J.push_back(T->Weight());
    }
  }
  const size_t m = elems.shape(0), q = ref.shape(0);
  return nb::make_tuple(np_array(std::move(X), {m, q, 2}), np_array(std::move(J), {m, q}));
}
nb::tuple scalar_boundary_nodes(mfem::Mesh& mesh, const mfem::FiniteElementSpace& fes, const mfem::Vector& T,
                                std::vector<int> attrs) {
  std::vector<double> X, U;
  std::vector<int> seen(fes.GetNDofs(), 0);
  mfem::Array<int> dofs;
  for (int be = 0; be < mesh.GetNBE(); ++be) {
    if (std::find(attrs.begin(), attrs.end(), mesh.GetBdrAttribute(be)) == attrs.end()) continue;
    fes.GetBdrElementDofs(be, dofs);
    const mfem::IntegrationRule& nodes = fes.GetBE(be)->GetNodes();
    mfem::ElementTransformation* Tr = mesh.GetBdrElementTransformation(be);
    for (int k = 0; k < dofs.Size(); ++k) {
      if (seen[dofs[k]]++) continue;
      double x[3];
      mfem::Vector xv(x, 2);
      Tr->Transform(nodes.IntPoint(k), xv);
      X.push_back(x[0]);
      X.push_back(x[1]);
      U.push_back(T(dofs[k]));
    }
  }
  const size_t n = U.size();
  return nb::make_tuple(np_array(std::move(X), {n, 2}), np_array(std::move(U), {n}));
}

template <class Sol>
int store_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) return 0;
  int rc = 0;
  Sol& s = *nb::inst_ptr<Sol>(self);
  for_each_held_callable(s, [&](nb::object& o) {
    if (!rc && o.ptr()) rc = visit(o.ptr(), arg);
  });
  if (s.store)
    for (auto& o : s.store->objs)
      if (!rc && o.ptr()) rc = visit(o.ptr(), arg);
  return rc;
}
template <class Sol>
int store_clear(PyObject* self) {
  if (nb::inst_ready(self)) {
    Sol& s = *nb::inst_ptr<Sol>(self);
    for_each_held_callable(s, [](nb::object& o) { o = nb::object(); });
    if (s.store)
      for (auto& o : s.store->objs) o = nb::object();
  }
  return 0;
}

struct PyScalarSolution : ProblemData {
  StorePtr store;
  std::unique_ptr<axf::transport::ScalarTransport> tr;
  mfem::Vector T, u;
  bool has_u = false;
  bool solved = false;
  axf::transport::SolveDiagnostics sd;
  std::unique_ptr<mfem::GridFunction> Tg;
  nb::object keepalive;  // the velocity solution (its space and mesh)
  std::string form, tau_rule, geometry, lu;
  int order = 2;
  bool supg = false;

  nb::dict diagnostics() const {
    nb::dict r;
    r["converged"] = sd.converged;
    r["relative_residual"] = sd.relative_residual;
    r["residual_lift"] = sd.residual_lift;
    r["iterations"] = sd.iterations;
    r["lu_residual"] = sd.lu_residual;
    r["failure"] = sd.failure;
    r["dofs_field"] = sd.dofs;
    r["dofs_free"] = sd.free_dofs;
    r["geometry"] = geometry;
    r["order"] = order;
    r["advective_form"] = form;
    r["supg"] = supg;
    r["tau_rule"] = supg ? tau_rule : std::string("none");
    r["quad_order"] = tr->quad_order();
    r["lu_backend"] = lu;
    return r;
  }
  void need_solved() const {
    AXF_REQUIRE(solved && sd.converged, "the transport solve did not converge; refusing to return its field", "",
                axf::strcat("final steady relative residual ", sd.relative_residual, " (", sd.failure, ")"));
  }
  nb::tuple scalar(IArr1 elems, Arr2 ref) const {
    need_solved();
    return scalar_values(*mesh, *Tg, elems, ref);
  }
  nb::tuple geometry_at(IArr1 elems, Arr2 ref) const { return geometry_of(*mesh, elems, ref); }
  nb::ndarray<nb::numpy, double> residual_full() const {
    need_solved();
    mfem::Vector R;
    tr->residual_full(T, has_u ? &u : nullptr, R);
    return np_array(std::vector<double>(R.GetData(), R.GetData() + R.Size()), {static_cast<size_t>(R.Size())});
  }
  nb::ndarray<nb::numpy, double> boundary_weights(std::vector<int> attrs) const {
    auto w = tr->boundary_weights(attrs);
    const size_t n = w.size();
    return np_array(std::move(w), {n});
  }
  nb::ndarray<nb::numpy, double> field() const {
    need_solved();
    return np_array(std::vector<double>(T.GetData(), T.GetData() + T.Size()), {static_cast<size_t>(T.Size())});
  }
  nb::tuple tau_samples() const {
    need_solved();
    auto v = tr->tau_samples(T, has_u ? &u : nullptr);
    std::vector<int> el;
    std::vector<double> xy, tau;
    for (const auto& t : v) {
      el.push_back(t.element);
      xy.push_back(t.x);
      xy.push_back(t.y);
      tau.push_back(t.tau);
    }
    const size_t n = el.size();
    return nb::make_tuple(np_iarray(std::move(el), {n}), np_array(std::move(xy), {n, 2}), np_array(std::move(tau), {n}));
  }
  nb::tuple boundary_nodes(std::vector<int> attrs) const {
    need_solved();
    return scalar_boundary_nodes(*mesh, tr->space(), T, attrs);
  }
  double surface_flux(std::vector<int> attrs) const {
    need_solved();
    return tr->surface_flux(T, attrs);
  }
  double dual_norm(std::vector<double> R) const {
    mfem::Vector r(static_cast<int>(R.size()));
    for (size_t i = 0; i < R.size(); ++i) r(i) = R[i];
    return tr->dual_norm(r);
  }
};

std::unique_ptr<PyScalarSolution> solve_scalar_transport(
    PyMesh& mesh, const std::string& geometry, int order, const std::string& form, bool supg,
    const std::string& tau_rule, double capacity, nb::object kappa, nb::object grad_kappa, nb::object velocity,
    nb::object div_velocity, double velocity_scale, nb::object velocity_solution, nb::object source, nb::list bcs,
    double shift, const std::string& lu, double rtol, int maxit, bool do_solve) {
  ensure_mpi();
  auto sol = std::make_unique<PyScalarSolution>();
  sol->store = std::make_shared<DataStore>();
  sol->geometry = geometry;
  sol->order = order;
  sol->supg = supg;
  sol->tau_rule = tau_rule;
  sol->form = form;
  sol->lu = lu;
  AXF_REQUIRE(geometry == "planar" || geometry == "axisymmetric", "geometry must be 'planar' or 'axisymmetric'");
  AXF_REQUIRE(form == "convective" || form == "conservative_flux" || form == "conservative_divergence",
              "advective form must be 'convective', 'conservative_flux' or 'conservative_divergence'");
  axf::transport::TransportOptions to;
  to.geometry = geometry == "axisymmetric" ? axf::transport::Geometry::Axisymmetric : axf::transport::Geometry::Planar;
  to.order = order;
  to.supg = supg;
  to.tau_rule = tau_rule;
  to.form = form == "convective" ? axf::transport::AdvectiveForm::Convective
            : form == "conservative_flux" ? axf::transport::AdvectiveForm::ConservativeFlux
                                          : axf::transport::AdvectiveForm::ConservativeDivergence;
  sol->mesh = mesh.par();
  sol->tr.reset(new axf::transport::ScalarTransport(*sol->mesh, to));
  auto& tr = *sol->tr;
  tr.set_capacity(capacity);
  tr.set_shift(shift);
  tr.set_diffusivity(to_scalar_data(sol->store, kappa), to_batch_vec(sol->store, grad_kappa));
  if (!source.is_none()) tr.set_source(to_scalar_data(sol->store, source));
  axf::transport::Velocity v;
  v.scale = velocity_scale;
  if (!velocity_solution.is_none()) {
    PyNSSolution& ns = nb::cast<PyNSSolution&>(velocity_solution);
    AXF_REQUIRE(ns.mesh.get() == sol->mesh.get(), "the velocity solution lives on another mesh (no interpolation)");
    v.fes = &ns.S->velocity_space();
    ns.S->velocity().GetTrueDofs(sol->u);
    sol->has_u = true;
    sol->keepalive = velocity_solution;
  } else if (!velocity.is_none()) {
    v.fn = to_batch_vec(sol->store, velocity);
    v.div_fn = to_batch_fn(sol->store, div_velocity);
  }
  if (v.is_set()) tr.set_velocity(v);
  for (auto& b : parse_tbcs(sol->store, bcs, form == "conservative_flux")) tr.add_bc(b.attrs, b.kind, b.data);
  tr.finalize();
  if (do_solve) {
    sol->sd = tr.solve(sol->T, sol->has_u ? &sol->u : nullptr, lu_backend_from_string(lu), rtol, maxit);
    sol->solved = true;
    sol->Tg.reset(new mfem::GridFunction(&tr.space()));
    *sol->Tg = sol->T;
  }
  return sol;
}

PyType_Slot scalar_solution_slots[] = {{Py_tp_traverse, (void*)store_traverse<PyScalarSolution>},
                                       {Py_tp_clear, (void*)store_clear<PyScalarSolution>}, {0, nullptr}};

struct PyBousSolution : ProblemData {
  StorePtr store;
  std::unique_ptr<BoussinesqSolver> S;

  nb::dict diagnostics() const {
    const auto& n = S->diagnostics();
    const auto& d = n.linear;
    nb::dict r;
    r["converged"] = n.converged;
    r["relative_residual"] = n.relative_residual;
    r["residual_lift"] = n.residual_lift;
    r["iterations"] = n.iterations;
    r["line_search_min_step_accepts"] = n.line_search_min_step_accepts;
    r["strategy"] = n.strategy;
    r["start"] = n.start;
    r["failure"] = n.failure;
    r["buoyancy_scales"] = n.buoyancy_scales;
    r["tau_rule"] = n.tau_rule;
    r["supg"] = S->options().supg;
    r["energy_form"] = n.energy_form;
    r["convection"] = n.convection;
    r["quad_orders"] = n.quad_orders;
    r["seconds"] = n.seconds;
    r["nullspace_detected"] = d.nullspace_detected;
    r["bc_enclosed"] = d.bc_enclosed;
    r["kernel_defect"] = d.kernel_defect;
    r["nullspace"] = d.nullspace;
    r["pinned_dof"] = d.pinned_dof;
    r["flux_defect"] = d.flux_defect;
    r["flux_defect_abs"] = d.flux_defect_abs;
    r["solver"] = d.solver;
    r["viscous_form"] = d.viscous_form;
    r["dofs_velocity"] = n.n_velocity;
    r["dofs_pressure"] = n.n_pressure;
    r["dofs_temperature"] = n.n_temperature;
    r["dofs_field"] = n.n_velocity + n.n_pressure + n.n_temperature;
    r["lu_backend"] = axf::fluids::to_string(S->options().ns.stokes.lu);
    r["lu_residual"] = n.lu_residual;
    r["lu_threads"] = d.lu.threads;
    r["ranks"] = d.ranks;
    r["threads"] = d.threads;
    r["rho0"] = S->options().ns.rho;
    r["cp"] = S->options().cp;
    r["beta_g"] = S->options().beta_g;
    r["T_ref"] = S->options().t_ref;
    r["newton_rtol"] = S->options().ns.nonlinear.newton_rtol;
    return r;
  }
  nb::list history() const {
    nb::list l;
    for (const auto& h : S->diagnostics().history) {
      nb::dict d;
      d["kind"] = h.kind;
      d["residual"] = h.residual;
      d["step"] = h.step;
      d["dt"] = h.dt;
      d["stage"] = h.stage;
      d["buoyancy_scale"] = h.viscosity_scale;
      d["lu_refine"] = h.lu_refine;
      d["lu_residual"] = h.lu_residual;
      l.append(d);
    }
    return l;
  }
  nb::tuple velocity(IArr1 elems, Arr2 ref) const { return velocity_of(S->velocity(), elems, ref); }
  nb::ndarray<nb::numpy, double> pressure(IArr1 elems, Arr2 ref) const { return pressure_of(S->pressure(), elems, ref); }
  nb::tuple scalar(IArr1 elems, Arr2 ref) const { return scalar_values(*mesh, S->temperature(), elems, ref); }
  nb::tuple geometry_at(IArr1 elems, Arr2 ref) const { return geometry_of(*mesh, elems, ref); }
  nb::tuple boundary_nodes(std::vector<int> attrs) const { return boundary_nodes_of(S->velocity(), attrs); }
  nb::tuple temperature_boundary_nodes(std::vector<int> attrs) const {
    const mfem::GridFunction& g = S->temperature();
    mfem::Vector t(g.Size());
    for (int i = 0; i < g.Size(); ++i) t(i) = g(i);
    return scalar_boundary_nodes(*mesh, S->transport().space(), t, attrs);
  }
  static nb::ndarray<nb::numpy, double> arr(std::vector<double> v) {
    const size_t n = v.size();
    return np_array(std::move(v), {n});
  }
  nb::ndarray<nb::numpy, double> state() const { return arr(S->state()); }
  nb::ndarray<nb::numpy, double> residual(std::vector<double> x, double s) const {
    return arr(S->residual(x, s));
  }
  nb::ndarray<nb::numpy, double> jacobian_action(std::vector<double> x, std::vector<double> d, double s) const {
    return arr(S->jacobian_action(x, d, s));
  }
  double dual_norm(std::vector<double> R) const { return S->dual_norm(R); }
  nb::tuple layout() const {
    const auto l = S->layout();
    return nb::make_tuple(l[0], l[1], l[2]);
  }
  nb::ndarray<nb::numpy, double> full_residual_T() const {
    return arr(S->full_residual_T());
  }
  nb::ndarray<nb::numpy, double> boundary_weights(std::vector<int> attrs) const {
    return arr(S->boundary_weights(attrs));
  }
  double surface_flux(std::vector<int> attrs) const {
    const mfem::GridFunction& g = S->temperature();
    mfem::Vector t(g.Size());
    for (int i = 0; i < g.Size(); ++i) t(i) = g(i);
    return S->transport().surface_flux(t, attrs);
  }
};

PyType_Slot bous_solution_slots[] = {{Py_tp_traverse, (void*)store_traverse<PyBousSolution>},
                                     {Py_tp_clear, (void*)store_clear<PyBousSolution>}, {0, nullptr}};

std::unique_ptr<PyBousSolution> solve_boussinesq(
    PyMesh& mesh, nb::object mu, double rho0, nb::object force, nb::list bcs, const std::string& form,
    const std::string& convection, const std::string& geometry, int order, const std::string& lu,
    const std::string& nullspace, int newton_maxit, double newton_rtol, bool line_search, int quad_order_convection,
    double direct_tol, int lu_threads, bool vectorized, double cp, double beta_g, nb::object t_ref, int up_sign,
    nb::object k, nb::object grad_k, nb::object heat_source, nb::list t_bcs, bool supg, int t_order,
    std::vector<double> buoyancy_scales, const std::string& strategy, int n_picard, const std::string& start,
    bool transient, bool do_solve) {
  ensure_mpi();
  auto sol = std::make_unique<PyBousSolution>();
  sol->store = std::make_shared<DataStore>();
  BoussinesqOptions o;
  StokesBoundary b;
  setup_problem(*sol, o.ns.stokes, b, mesh, mu, force, bcs, form, geometry, "direct", lu, nullspace, nb::none(), order,
                1e-10, direct_tol, 0.0, -1, false, vectorized, 0, lu_threads, 30.0, true, 1e-3);
  AXF_REQUIRE(convection == "standard" || convection == "skew", "convection must be 'standard' or 'skew'");
  o.ns.rho = rho0;
  o.ns.convection = convection == "skew" ? ConvectiveForm::Skew : ConvectiveForm::Standard;
  o.ns.quad_order_convection = quad_order_convection;
  auto& nl = o.ns.nonlinear;
  nl.strategy = strategy_from(strategy);
  nl.n_picard = n_picard;
  nl.line_search = line_search;
  nl.newton_rtol = newton_rtol;
  nl.newton_maxit = newton_maxit;
  o.cp = cp;
  o.beta_g = beta_g;
  if (!t_ref.is_none()) {
    o.t_ref_set = true;
    o.t_ref = nb::cast<double>(t_ref);
  }
  o.up_sign = up_sign;
  o.k = to_scalar_data(sol->store, k);
  o.grad_k = to_batch_vec(sol->store, grad_k);
  if (!heat_source.is_none()) o.heat_source = to_scalar_data(sol->store, heat_source);
  o.t_order = t_order;
  o.supg = supg;
  o.buoyancy_scales = std::move(buoyancy_scales);
  o.start = start;
  o.transient = transient;
  for (auto& tb : parse_tbcs(sol->store, t_bcs)) o.t_bcs.push_back({tb.attrs, tb.kind, tb.data});
  sol->S = std::make_unique<BoussinesqSolver>(*sol->mesh, o, b);
  if (do_solve) sol->S->solve();
  return sol;
}


// --- stabilized P1/P1 GLS (cpp/include/axionfem/fluids/gls.hpp)
std::vector<double> to_vec(const nb::object& o) {
  std::vector<double> v;
  if (o.is_none()) return v;
  auto a = nb::cast<DArr>(o);
  v.assign(a.data(), a.data() + a.size());
  return v;
}
std::vector<int> to_ivec(const nb::object& o) {
  std::vector<int> v;
  if (o.is_none()) return v;
  auto a = nb::cast<nb::ndarray<const int, nb::c_contig, nb::device::cpu>>(o);
  v.assign(a.data(), a.data() + a.size());
  return v;
}

struct PyGls {
  std::unique_ptr<GlsProblem> P;

  nb::dict diagnostics() const {
    const auto& d = P->diagnostics();
    nb::dict r;
    r["converged"] = d.converged;
    r["relative_residual"] = d.relative_residual;
    r["residual_lift"] = d.residual_lift;
    r["iterations"] = d.iterations;
    r["line_search_min_step_accepts"] = d.line_search_min_step_accepts;
    r["failure"] = d.failure;
    r["nullspace"] = d.nullspace;
    r["nullspace_detected"] = d.nullspace_detected;
    r["multiplier"] = d.multiplier;
    r["viscosity_scales"] = d.viscosity_scales;
    r["seconds"] = d.seconds;
    r["lu_backend"] = d.lu_backend;
    r["lu_residual"] = d.lu_residual;
    r["lu_residual_before_refinement"] = d.lu_residual_before;
    r["lu_refine_steps"] = d.lu_refine_steps;
    r["dofs_velocity"] = d.n_free_u;
    r["dofs_pressure"] = d.n_pressure;
    r["dofs_field"] = d.n_total;
    r["solver"] = "gls_p1p1";
    r["stabilization"] = nb::make_tuple(P->options().stab.supg, P->options().stab.pspg, P->options().stab.lsic);
    r["C_I"] = 36.0;
    r["CG"] = 1.0;
    return r;
  }
  nb::list history() const {
    nb::list l;
    for (const auto& h : P->diagnostics().history) {
      nb::dict d;
      d["kind"] = h.kind;
      d["residual"] = h.residual;
      d["step"] = h.step;
      d["stage"] = h.stage;
      d["viscosity_scale"] = h.viscosity_scale;
      d["lu_refine"] = h.lu_refine;
      d["lu_residual"] = h.lu_residual;
      d["lu_residual_before_refinement"] = h.lu_residual_before;
      l.append(d);
    }
    return l;
  }
  nb::ndarray<nb::numpy, double> u() const {
    return np_array(P->u(), {static_cast<size_t>(P->n_vertices()), 2});
  }
  nb::ndarray<nb::numpy, double> p() const {
    return np_array(P->p(), {static_cast<size_t>(P->n_vertices())});
  }
  nb::ndarray<nb::numpy, int> dirichlet_nodes() const {
    return np_iarray(P->dirichlet_nodes(), {P->dirichlet_nodes().size()});
  }
  nb::tuple residual_full(DArr u, DArr p, double s) const {
    std::vector<double> ru, rp;
    P->residual_full(std::vector<double>(u.data(), u.data() + u.size()), std::vector<double>(p.data(), p.data() + p.size()),
                     ru, rp, s);
    const size_t nv = static_cast<size_t>(P->n_vertices());
    return nb::make_tuple(np_array(std::move(ru), {nv, 2}), np_array(std::move(rp), {nv}));
  }
  nb::tuple layout() const {
    auto l = P->layout();
    return nb::make_tuple(l.first, l.second);
  }
  nb::ndarray<nb::numpy, double> state() const {
    auto x = P->state();
    return np_array(x, {x.size()});
  }
  nb::ndarray<nb::numpy, double> residual(DArr x, double s) const {
    auto r = P->residual(std::vector<double>(x.data(), x.data() + x.size()), s);
    return np_array(r, {r.size()});
  }
  nb::ndarray<nb::numpy, double> jacobian_action(DArr x, DArr d, double s) const {
    auto r = P->jacobian_action(std::vector<double>(x.data(), x.data() + x.size()),
                                std::vector<double>(d.data(), d.data() + d.size()), s);
    return np_array(r, {r.size()});
  }
  nb::tuple tau_at(DArr w) const {
    std::vector<double> a, b, c;
    P->tau_at(std::vector<double>(w.data(), w.data() + w.size()), a, b, c);
    const size_t ne = static_cast<size_t>(P->n_elements());
    return nb::make_tuple(np_array(std::move(a), {ne, 6}), np_array(std::move(b), {ne, 6}),
                          np_array(std::move(c), {ne, 6}));
  }
  double dual_norm(DArr ru, DArr rp) const {
    return P->dual_norm(std::vector<double>(ru.data(), ru.data() + ru.size()),
                        std::vector<double>(rp.data(), rp.data() + rp.size()));
  }
};

std::unique_ptr<PyGls> gls_problem(nb::object V, nb::object T, nb::object mu_q, nb::object gmu_q, nb::object mu_nodal,
                                   nb::object f_q, nb::object adv_q, nb::object hist1, nb::object hist2,
                                   nb::object dirichlet_nodes, nb::object dirichlet_values, double rho,
                                   const std::string& mode, const std::string& form, double dt, int bdf,
                                   bool supg, bool pspg, bool lsic, const std::string& lu, int lu_threads,
                                   double newton_rtol, int newton_maxit, bool line_search,
                                   std::vector<double> viscosity_scales, bool do_solve) {
  ensure_mpi();
  GlsData D;
  D.V = to_vec(V);
  D.T = to_ivec(T);
  D.mu_q = to_vec(mu_q);
  D.gmu_q = to_vec(gmu_q);
  D.mu_nodal = to_vec(mu_nodal);
  D.f_q = to_vec(f_q);
  D.adv_q = to_vec(adv_q);
  D.hist1 = to_vec(hist1);
  D.hist2 = to_vec(hist2);
  D.dirichlet_nodes = to_ivec(dirichlet_nodes);
  D.dirichlet_values = to_vec(dirichlet_values);
  GlsOptions o;
  o.rho = rho;
  if (mode == "stokes") o.mode = GlsMode::Stokes;
  else if (mode == "oseen") o.mode = GlsMode::Oseen;
  else if (mode == "navier_stokes") o.mode = GlsMode::NavierStokes;
  else AXF_REQUIRE(false, "mode must be stokes, oseen or navier_stokes (got '" + mode + "')", "mode", "");
  if (form == "laplacian") o.form = GlsForm::Laplacian;
  else if (form == "stress") o.form = GlsForm::Stress;
  else AXF_REQUIRE(false, "form must be laplacian or stress (got '" + form + "')", "form", "");
  o.dt = dt;
  o.bdf = bdf;
  o.stab.supg = supg;
  o.stab.pspg = pspg;
  o.stab.lsic = lsic;
  o.lu = lu_backend_from_string(lu);
  o.lu_threads = lu_threads;
  o.nl.newton_rtol = newton_rtol;
  o.nl.newton_maxit = newton_maxit;
  o.nl.line_search = line_search;
  o.nl.viscosity_scales = std::move(viscosity_scales);
  auto g = std::make_unique<PyGls>();
  g->P = std::make_unique<GlsProblem>(std::move(D), std::move(o));
  if (do_solve) g->P->solve();
  return g;
}

}  // namespace

namespace axf::fluids {

void register_python_3d(nb::module_& m);  // 3D: fluids_bindings3d.cpp (Mesh3D, solve_stokes_3d)

void register_python(nb::module_& core) {
  nb::module_ m = core.def_submodule("_fluids", "axionFEM fluids library (beta). "
                                                "Wrapped by axionfem.fluids.");
  register_python_3d(m);
  nb::class_<PyMesh>(m, "Mesh")
      .def_static("from_arrays", &mesh_from_arrays_py, nb::arg("vertices"), nb::arg("elements"),
                  nb::arg("boundary_segments"), nb::arg("boundary_attributes"),
                  nb::arg("element_attributes") = nb::none(), nb::arg("p2_nodes") = nb::none(),
                  "Mesh from (nv,2) vertices, (ne,3|4) counter-clockwise elements, (nb,2) boundary segments and their "
                  "attributes (>= 1); p2_nodes: (ne,6,2) isoparametric P2 geometry [v0,v1,v2,m01,m12,m20] per "
                  "triangle (input vertex order).")
      .def_static("unit_square", [](int n, bool quad) {
        return PyMesh(mfem::Mesh::MakeCartesian2D(n, n, quad ? mfem::Element::QUADRILATERAL : mfem::Element::TRIANGLE,
                                                  true, 1.0, 1.0));
      }, nb::arg("n"), nb::arg("quad") = false, "MFEM's structured unit square (attributes 1 bottom, 2 right, 3 top, 4 left).")
      .def_static("annulus", [](int nr, int nt, double R1, double R2, int order) {
        return PyMesh(annulus_mesh(nr, nt, R1, R2, order));
      }, nb::arg("nr"), nb::arg("nt"), nb::arg("r1"), nb::arg("r2"), nb::arg("order") = 2,
                  "Annulus with exact isoparametric circles (attributes 1 inner, 2 outer).")
      .def_prop_ro("n_elements", &PyMesh::n_elements)
      .def("vertices", &PyMesh::vertices)
      .def("elements", &PyMesh::elements)
      .def("p2_nodes", &PyMesh::p2_nodes, "(ne, 6, 2) geometry at the P2 reference nodes of every triangle")
      .def("boundary", &PyMesh::boundary, "(segments (nb,2), attributes (nb,))")
      .def("taylor_hood_check", &PyMesh::taylor_hood_check);

  nb::class_<PySolution>(m, "StokesSolution", nb::type_slots(solution_slots))
      .def_prop_ro("diagnostics", &PySolution::diagnostics)
      .def_prop_ro("n_elements", [](const PySolution& s) { return s.mesh->GetNE(); })
      .def("velocity", &PySolution::velocity, nb::arg("elements"), nb::arg("ref_points"),
           "(u (m,q,vdim), grad u (m,q,vdim,2)) at reference points of the elements; refused unless converged")
      .def("pressure", &PySolution::pressure, nb::arg("elements"), nb::arg("ref_points"),
           "p_h at reference points (with a periodic pressure drop: the periodic part p_per)")
      .def("geometry", &PySolution::geometry, nb::arg("elements"), nb::arg("ref_points"), "(x (m,q,2), detJ (m,q))")
      .def("boundary_nodes", &PySolution::boundary_nodes, nb::arg("attributes"),
           "(points (n,2), values (n,vdim)): the velocity DOFs on the boundary elements of the attributes")
      .def("boundary_power", [](const PySolution& s, std::vector<int> a) { return s.S->boundary_power(a); },
           nb::arg("attributes"),
           "power put into the fluid by the prescribed velocity on these Dirichlet attributes (Babuška–Miller)")
      .def("reaction", &PySolution::reaction, nb::arg("attributes"), nb::arg("center") = std::vector<double>{},
           nb::arg("residual_form") = "solved",
           "Babuška–Miller force/torque exerted by the fluid on the body attributes (kinematic; a dict with 'force' "
           "(3 components; axisymmetric: F_z in force[2], times 2 pi) and 'torque' (planar: about `center`; "
           "axisymmetric: about the axis, times 2 pi)).\n"
           "residual_form:\n"
           "  'solved' (default): the PHYSICAL force (Cauchy traction) from the residual of the solved operator, "
           "grad-div included, valid on any wall: at rest, moving (rotating) or slip. For a Laplacian-form solve "
           "it adds nu[(grad u^T, grad w) - (div u, div w)], which turns the pseudo-traction into the traction "
           "without losing Galerkin consistency; it needs a CONSTANT viscosity (refused otherwise). "
           "Caveat: a Laplacian-form slip (or traction) wall imposes zero PSEUDO-traction, not free slip (zero "
           "Cauchy shear), so on curved slip or traction walls the solved problem itself is not the physical one: "
           "solve with form='stress' there.\n"
           "  'stress': the raw residual of the stress-form operator (+ grad-div). For a stress-form solve it equals "
           "'solved'; for a Laplacian-form solve it is not Galerkin-consistent (it carries -nu (div u_h, div w)) "
           "and converges at a lower order: a diagnostic.\n"
           "  'laplacian': the raw residual of the Laplacian operator, i.e. the PSEUDO-traction nu du/dn - p n. It "
           "is the force only on no-slip walls AT REST; on moving or slip walls it is wrong (a rotating sphere: 2/3 "
           "of the torque; a rotating cylinder: 5/8). A diagnostic / negative control.")
      .def("blocks", &PySolution::blocks, "A, B, Mp, H1_gram (CSR dicts) and the essential velocity DOFs");

  m.def("solve_stokes", &solve_stokes, nb::arg("mesh"), nb::arg("nu"), nb::arg("force") = nb::none(),
        nb::arg("bcs") = nb::list(), nb::arg("form") = "stress", nb::arg("geometry") = "planar",
        nb::arg("solver") = "direct", nb::arg("lu") = "strumpack", nb::arg("nullspace") = "auto",
        nb::arg("pressure_point") = nb::none(), nb::arg("order") = 2, nb::arg("rtol") = 1e-10,
        nb::arg("direct_tol") = 1e-10, nb::arg("grad_div") = 0.0,
        nb::arg("pressure_order") = -1, nb::arg("allow_unstable_pair") = false, nb::arg("vectorized") = false,
        nb::arg("min_refine_steps") = 0, nb::arg("solve") = true, nb::arg("lu_threads") = 1,
        nb::arg("slip_corner_angle_deg") = 30.0, nb::arg("refuse_velocity_nullspace") = true,
        nb::arg("velocity_nullspace_tol") = 1e-3,
        "Solve Taylor–Hood Stokes (see axionfem.fluids.solve_stokes). Never raises on non-convergence.");
  nb::class_<PyNSSolution>(m, "NavierStokesSolution", nb::type_slots(ns_solution_slots))
      .def_prop_ro("diagnostics", &PyNSSolution::diagnostics)
      .def_prop_ro("history", &PyNSSolution::history,
                   "list of {kind, residual, step, dt, stage, viscosity_scale, lu_refine}; residual = steady relative "
                   "dual-norm residual AFTER the step")
      .def_prop_ro("n_elements", [](const PyNSSolution& s) { return s.mesh->GetNE(); })
      .def("velocity", &PyNSSolution::velocity, nb::arg("elements"), nb::arg("ref_points"),
           "(u (m,q,vdim), grad u (m,q,vdim,2)); axisymmetric components (u_r, u_z, u_theta); refused unless converged")
      .def("pressure", &PyNSSolution::pressure, nb::arg("elements"), nb::arg("ref_points"))
      .def("geometry", &PyNSSolution::geometry, nb::arg("elements"), nb::arg("ref_points"))
      .def("boundary_nodes", &PyNSSolution::boundary_nodes, nb::arg("attributes"))
      .def("potential", &PyNSSolution::potential, nb::arg("elements"), nb::arg("ref_points"),
           "(phi (m,q), grad phi (m,q,2)); MHD solves only; refused unless converged")
      .def("potential_boundary_nodes", &PyNSSolution::potential_boundary_nodes, nb::arg("attributes"))
      .def("state", &PyNSSolution::state, "the free-DOF state x_free of the solution (internal pressure)")
      .def("residual", &PyNSSolution::residual, nb::arg("x"), nb::arg("viscosity_scale") = 1.0)
      .def("jacobian_action", &PyNSSolution::jacobian_action, nb::arg("x"), nb::arg("d"),
           nb::arg("viscosity_scale") = 1.0)
      .def("jacobian", &PyNSSolution::jacobian, nb::arg("x"), nb::arg("viscosity_scale") = 1.0)
      .def("dual_norm", &PyNSSolution::dual_norm, nb::arg("R"))
      .def("layout", &PyNSSolution::layout)
      .def("full_residual", &PyNSSolution::full_residual)
      .def("reaction", &PyNSSolution::reaction, nb::arg("attributes"), nb::arg("center") = std::vector<double>{},
           nb::arg("residual_form") = "stress");
  m.def("solve_navier_stokes", &solve_navier_stokes, nb::arg("mesh"), nb::arg("mu"), nb::arg("rho"),
        nb::arg("force") = nb::none(), nb::arg("bcs") = nb::list(), nb::arg("form") = "stress",
        nb::arg("convection") = "standard", nb::arg("geometry") = "planar", nb::arg("order") = 2,
        nb::arg("lu") = "strumpack", nb::arg("nullspace") = "auto", nb::arg("strategy") = "newton",
        nb::arg("n_picard") = 3, nb::arg("line_search") = true, nb::arg("newton_rtol") = 1e-12,
        nb::arg("newton_maxit") = 30, nb::arg("picard_maxit") = 200, nb::arg("ptc_dt0") = 0.05,
        nb::arg("ptc_release_dt") = 1e12, nb::arg("ptc_maxit") = 200,
        nb::arg("viscosity_scales") = std::vector<double>{}, nb::arg("quad_order_convection") = 0,
        nb::arg("direct_tol") = 1e-10, nb::arg("lu_threads") = 1, nb::arg("vectorized") = false,
        nb::arg("mhd") = nb::none(), nb::arg("solve") = true,
        "Steady Navier-Stokes (see axionfem.fluids.solve_navier_stokes). Never raises on non-convergence.");
  nb::class_<PyField>(m, "FieldSnapshot")
      .def("velocity", &PyField::velocity, nb::arg("elements"), nb::arg("ref_points"))
      .def("pressure", &PyField::pressure, nb::arg("elements"), nb::arg("ref_points"))
      .def("geometry", &PyField::geometry, nb::arg("elements"), nb::arg("ref_points"))
      .def_prop_ro("n_elements", [](const PyField& s) { return s.mesh->GetNE(); })
      .def("boundary_nodes", &PyField::boundary_nodes, nb::arg("attributes"));
  nb::class_<PyTransient>(m, "TransientNavierStokes", nb::type_slots(transient_slots))
      .def_prop_ro("diagnostics", &PyTransient::diagnostics)
      .def_prop_ro("n_elements", [](const PyTransient& s) { return s.mesh->GetNE(); })
      .def("initialize", [](PyTransient& s, nb::object u0, double t0, bool project) {
        auto c = vector_coef(u0, s.vdim, s.vectorized);
        s.bc_values.push_back(std::move(c));  // keeps the callable alive (and visible to the GC)
        return s.T->initialize(t0, *s.bc_values.back(), project);
      }, nb::arg("u0"), nb::arg("t0") = 0.0, nb::arg("project") = true)
      .def("set_history", [](PyTransient& s, std::vector<double> times, nb::list coefs) {
        std::vector<mfem::Vector> st;
        for (nb::handle h : coefs) {
          auto c = vector_coef(nb::borrow(h), s.vdim, s.vectorized);
          st.push_back(s.T->interpolate(*c));
        }
        s.T->set_history(times, st);
      }, nb::arg("times"), nb::arg("states"))
      .def("step_trial", [](PyTransient& s, double dt, int force_order) {
        return PyTransient::step_dict(s.T->step_trial(dt, force_order));
      }, nb::arg("dt"), nb::arg("force_order") = 0)
      .def("commit", [](PyTransient& s) { s.T->commit(); })
      .def("advance", [](PyTransient& s, double dt) {
        return PyTransient::step_dict(s.T->advance(dt));
      }, nb::arg("dt"))
      .def("adapt", [](PyTransient& s, double T, double rtol, double S_u, double h0) {
        const AdaptiveResult r = s.T->run_adaptive(T, rtol, S_u, h0);
        nb::dict d;
        d["times"] = r.times;
        d["orders"] = r.orders;
        d["rejected"] = r.rejected;
        d["failed_solves"] = r.failed_solves;
        d["dt_max_ratio"] = r.dt_max_ratio;
        d["dt_min_ratio"] = r.dt_min_ratio;
        d["dt_max"] = r.dt_max;
        d["dt_min"] = r.dt_min;
        d["reached"] = r.reached;
        return d;
      }, nb::arg("T"), nb::arg("rtol"), nb::arg("S_u"), nb::arg("h0"))
      .def("time", [](const PyTransient& s) { return s.T->time(); })
      .def("snapshot", [](const PyTransient& s) { return s.field(s.T->state_u(), s.T->state_p()); })
      .def("trial_snapshot", [](const PyTransient& s) { return s.field(s.T->trial_u(), s.T->trial_p()); })
      .def("estimate_snapshot", [](const PyTransient& s) {
        mfem::Vector z(s.T->core().pressure_size());
        z = 0.0;
        return s.field(s.T->estimate(), z, true);
      })
      .def_prop_ro("stats", [](const PyTransient& s) {
        const RunStats& r = s.T->stats();
        nb::dict d;
        d["all_converged"] = r.all_converged;
        d["max_relative_residual"] = r.max_relative_residual;
        d["newton_iterations"] = r.newton_iterations;
        d["solves"] = r.solves;
        d["lu_fallbacks"] = r.lu_fallbacks;
        d["lu_fallback_backend"] = r.lu_fallback_backend;
        d["failure"] = r.failure;
        return d;
      })
      .def("reset_stats", [](PyTransient& s) { s.T->reset_stats(); })
      .def_prop_ro("last_history", [](const PyTransient& s) {
        nb::list l;
        for (const auto& h : s.T->core().diagnostics().history) l.append(h.residual);
        return l;
      });
  m.def("make_transient", &make_transient, nb::arg("mesh"), nb::arg("mu"), nb::arg("rho"), nb::arg("force").none(),
        nb::arg("bcs"), nb::arg("form"), nb::arg("geometry"), nb::arg("order"), nb::arg("lu"), nb::arg("nullspace"),
        nb::arg("scheme"), nb::arg("max_order"), nb::arg("newton_rtol"), nb::arg("newton_maxit"),
        nb::arg("quad_order_convection"), nb::arg("direct_tol"), nb::arg("lu_threads"), nb::arg("vectorized"),
        nb::arg("time_cb"));
  nb::class_<PyGls>(m, "GlsProblem")
      .def_prop_ro("diagnostics", &PyGls::diagnostics)
      .def_prop_ro("history", &PyGls::history)
      .def("u", &PyGls::u)
      .def("p", &PyGls::p)
      .def("dirichlet_nodes", &PyGls::dirichlet_nodes)
      .def("residual_full", &PyGls::residual_full, nb::arg("u"), nb::arg("p"), nb::arg("viscosity_scale") = 1.0)
      .def("layout", &PyGls::layout)
      .def("state", &PyGls::state)
      .def("residual", &PyGls::residual, nb::arg("x"), nb::arg("viscosity_scale") = 1.0)
      .def("jacobian_action", &PyGls::jacobian_action, nb::arg("x"), nb::arg("d"), nb::arg("viscosity_scale") = 1.0)
      .def("tau_at", &PyGls::tau_at, nb::arg("w"))
      .def("dual_norm", &PyGls::dual_norm, nb::arg("ru"), nb::arg("rp"));
  m.def("gls_problem", &gls_problem, nb::arg("vertices"), nb::arg("triangles"), nb::arg("mu_q") = nb::none(),
        nb::arg("gmu_q") = nb::none(), nb::arg("mu_nodal") = nb::none(), nb::arg("f_q") = nb::none(),
        nb::arg("adv_q") = nb::none(), nb::arg("hist1") = nb::none(), nb::arg("hist2") = nb::none(),
        nb::arg("dirichlet_nodes") = nb::none(), nb::arg("dirichlet_values") = nb::none(), nb::arg("rho") = 1.0,
        nb::arg("mode") = "navier_stokes", nb::arg("form") = "stress", nb::arg("dt") = 0.0, nb::arg("bdf") = 1,
        nb::arg("supg") = true, nb::arg("pspg") = true, nb::arg("lsic") = true, nb::arg("lu") = "strumpack",
        nb::arg("lu_threads") = 1, nb::arg("newton_rtol") = 1e-12, nb::arg("newton_maxit") = 40,
        nb::arg("line_search") = true, nb::arg("viscosity_scales") = std::vector<double>{},
        nb::arg("solve") = true,
        "Stabilized P1/P1 GLS problem on raw arrays (see axionfem.fluids.gls_problem). Never raises on non-convergence.");
  m.def("gls_rule", []() {
    const double* pt = gls_rule_points();
    const double* w = gls_rule_weights();
    return nb::make_tuple(np_array(std::vector<double>(pt, pt + 2 * GLS_NQ), {static_cast<size_t>(GLS_NQ), 2}),
                          np_array(std::vector<double>(w, w + GLS_NQ), {static_cast<size_t>(GLS_NQ)}));
  }, "the fixed 6-point degree-4 rule (reference points on (0,0),(1,0),(0,1); weights summing to 1/2)");

  nb::class_<PyScalarSolution>(m, "ScalarTransportSolution", nb::type_slots(scalar_solution_slots))
      .def_prop_ro("diagnostics", &PyScalarSolution::diagnostics)
      .def_prop_ro("n_elements", [](const PyScalarSolution& s) { return s.mesh->GetNE(); })
      .def("scalar", &PyScalarSolution::scalar, nb::arg("elements"), nb::arg("ref_points"),
           "(T (m,q), grad T (m,q,2)) at reference points of the elements; refused unless converged")
      .def("geometry", &PyScalarSolution::geometry_at, nb::arg("elements"), nb::arg("ref_points"))
      .def("residual_full", &PyScalarSolution::residual_full,
           "R_i = R(T_h; phi_i) over EVERY dof, natural and SUPG terms included, Dirichlet rows not replaced")
      .def("boundary_weights", &PyScalarSolution::boundary_weights, nb::arg("attributes"),
           "nodal v_Gamma: 1 at every dof of the closed boundary part")
      .def("field", &PyScalarSolution::field)
      .def("tau_samples", &PyScalarSolution::tau_samples, "(element, (x, y), tau) at the assembly quadrature points")
      .def("boundary_nodes", &PyScalarSolution::boundary_nodes, nb::arg("attributes"))
      .def("surface_flux", &PyScalarSolution::surface_flux, nb::arg("attributes"))
      .def("dual_norm", &PyScalarSolution::dual_norm, nb::arg("R"));
  m.def("solve_scalar_transport", &solve_scalar_transport, nb::arg("mesh"), nb::arg("geometry"), nb::arg("order"),
        nb::arg("form"), nb::arg("supg"), nb::arg("tau_rule"), nb::arg("capacity"), nb::arg("kappa"),
        nb::arg("grad_kappa") = nb::none(), nb::arg("velocity") = nb::none(), nb::arg("div_velocity") = nb::none(),
        nb::arg("velocity_scale") = 1.0, nb::arg("velocity_solution") = nb::none(), nb::arg("source") = nb::none(),
        nb::arg("bcs") = nb::list(), nb::arg("shift") = 0.0, nb::arg("lu") = "eigen", nb::arg("rtol") = 1e-12,
        nb::arg("maxit") = 4, nb::arg("solve") = true,
        "Steady scalar transport (see axionfem.transport.ScalarTransport / axionfem.fluids.solve_heat).");
  nb::class_<PyBousSolution>(m, "BoussinesqSolution", nb::type_slots(bous_solution_slots))
      .def_prop_ro("diagnostics", &PyBousSolution::diagnostics)
      .def_prop_ro("history", &PyBousSolution::history)
      .def_prop_ro("n_elements", [](const PyBousSolution& s) { return s.mesh->GetNE(); })
      .def("velocity", &PyBousSolution::velocity, nb::arg("elements"), nb::arg("ref_points"))
      .def("pressure", &PyBousSolution::pressure, nb::arg("elements"), nb::arg("ref_points"))
      .def("temperature", &PyBousSolution::scalar, nb::arg("elements"), nb::arg("ref_points"))
      .def("geometry", &PyBousSolution::geometry_at, nb::arg("elements"), nb::arg("ref_points"))
      .def("boundary_nodes", &PyBousSolution::boundary_nodes, nb::arg("attributes"))
      .def("temperature_boundary_nodes", &PyBousSolution::temperature_boundary_nodes, nb::arg("attributes"))
      .def("state", &PyBousSolution::state)
      .def("residual", &PyBousSolution::residual, nb::arg("x"), nb::arg("buoyancy_scale") = 1.0)
      .def("jacobian_action", &PyBousSolution::jacobian_action, nb::arg("x"), nb::arg("d"),
           nb::arg("buoyancy_scale") = 1.0)
      .def("dual_norm", &PyBousSolution::dual_norm, nb::arg("R"))
      .def("layout", &PyBousSolution::layout)
      .def("full_residual_T", &PyBousSolution::full_residual_T)
      .def("boundary_weights", &PyBousSolution::boundary_weights, nb::arg("attributes"))
      .def("surface_flux", &PyBousSolution::surface_flux, nb::arg("attributes"));
  m.def("solve_boussinesq", &solve_boussinesq, nb::arg("mesh"), nb::arg("mu"), nb::arg("rho0"),
        nb::arg("force") = nb::none(), nb::arg("bcs") = nb::list(), nb::arg("form") = "stress",
        nb::arg("convection") = "standard", nb::arg("geometry") = "planar", nb::arg("order") = 2,
        nb::arg("lu") = "strumpack", nb::arg("nullspace") = "auto", nb::arg("newton_maxit") = 30,
        nb::arg("newton_rtol") = 1e-12, nb::arg("line_search") = false, nb::arg("quad_order_convection") = 0,
        nb::arg("direct_tol") = 1e-10, nb::arg("lu_threads") = 1, nb::arg("vectorized") = true,
        nb::arg("cp") = 1.0, nb::arg("beta_g") = 0.0, nb::arg("T_ref") = nb::none(), nb::arg("up_sign") = 1,
        nb::arg("k") = nb::none(), nb::arg("grad_k") = nb::none(), nb::arg("heat_source") = nb::none(),
        nb::arg("t_bcs") = nb::list(), nb::arg("supg") = false, nb::arg("t_order") = 2,
        nb::arg("buoyancy_scales") = std::vector<double>{}, nb::arg("strategy") = "newton", nb::arg("n_picard") = 0,
        nb::arg("start") = "lift", nb::arg("transient") = false,
        nb::arg("solve") = true,
        "Steady Boussinesq natural convection (see axionfem.fluids.solve_boussinesq). Never raises on non-convergence.");
  m.def("lu_backend_available", [](const std::string& s) { return lu_backend_available(lu_backend_from_string(s)); });
}

}  // namespace axf::fluids
