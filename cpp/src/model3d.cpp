#include "shell_map.hpp"
#include "axionfem/model3d.hpp"
#include "axionfem/fw2d.hpp"

#include "model3d_impl.hpp"
#include "body_constraints.hpp"
#include "surface_losses3d.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <limits>
#include <numeric>
#include <set>

#include "axionfem/adapt.hpp"
#include "axionfem/axi.hpp"
#include "axionfem/core.hpp"
#include "axionfem/cost.hpp"
#include "axionfem/device.hpp"
#include "axionfem/scan.hpp"
#include "axionfem/sym_solver.hpp"
#include "axionfem/topology.hpp"
#include "axionfem/units.hpp"

namespace axf {

using units::mu0;
using units::pi;

namespace {

constexpr int kShellAttr = 898;

// Where hypre runs, for the results ("cpu", or "cuda:0 (NVIDIA A40)"); set by configure_hypre_gpu.
std::string& hypre_device_label() {
  static std::string label = "cpu";
  return label;
}

// hypre on the GPU that $AXF_DEVICE asks for (device.hpp): device memory and execution for every hypre object created
// from here on. MFEM stays on the host and copies vectors and matrices into hypre's memory at each hypre call, so the
// host code below never reads a stale copy (docs/validation/gpu.md). Once per process.
void configure_hypre_gpu(const DeviceRequest& dev) {
  static bool done = false;
  if (done) return;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
  int count = 0;
#if defined(HYPRE_USING_CUDA)
  if (cudaGetDeviceCount(&count) != cudaSuccess) count = 0;
  (void)cudaGetLastError();  // a failed query leaves a sticky error that hypre would report later as its own
#else
  if (hipGetDeviceCount(&count) != hipSuccess) count = 0;
  (void)hipGetLastError();
#endif
  AXF_REQUIRE(count > 0, "AXF_DEVICE='" + configure_device() + "' but no GPU is visible to this process", "",
              "Run where a GPU is visible (on a cluster: in a job that requests one; docs/install.md, \"GPU builds\"), "
              "check CUDA_VISIBLE_DEVICES, or unset AXF_DEVICE to run on the CPU.");
  int nranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  AXF_REQUIRE(nranks == 1, strcat("AXF_DEVICE='", configure_device(), "' with ", nranks,
                                  " MPI ranks: hypre on a GPU is validated on one rank only"), "",
              "Two and 14 ranks sharing one GPU crashed or hung in the first 3D solve (a cudaFree error in MFEM's "
              "memory manager, an invalid rank in a halo exchange; docs/validation/gpu.md); one GPU per rank was not "
              "tried. Run one rank with the GPU, or unset AXF_DEVICE to run the ranks on the CPU.");
  const int id = dev.id < 0 ? 0 : dev.id;  // "cuda": the first visible GPU (one rank, above)
  AXF_REQUIRE(id < count, strcat("AXF_DEVICE='", configure_device(), "' asks for GPU ", id, " but ", count,
                                 " are visible to this process"), "",
              "Use a smaller index, or the backend alone for the first visible GPU.");
  std::string name;
#if defined(HYPRE_USING_CUDA)
  MFEM_GPU_CHECK(cudaSetDevice(id));
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, id) == cudaSuccess) name = prop.name;
#else
  MFEM_GPU_CHECK(hipSetDevice(id));
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, id) == hipSuccess) name = prop.name;
#endif
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
  // PMIS coarsening (BoomerAMG on a GPU) draws random numbers. From cuRAND, whose state carries over from one AMG setup
  // to the next, two identical solves in one process differ at the solver tolerance; the host generator is reseeded
  // for every setup, so repeated solves are bit-identical (docs/validation/gpu.md).
  HYPRE_SetUseGpuRand(0);
  HYPRE_DeviceInitialize();
  hypre_device_label() = strcat(dev.gpu, ":", id, " (", name, ")");
  log(1, "  hypre on " + hypre_device_label());
#else
  throw Error("AXF_DEVICE='" + configure_device() + "' but the hypre of this build has no GPU support", "",
              "Build the GPU stack with scripts/build_deps_gpu.sh (hypre with CUDA) and configure against it; see "
              "docs/install.md, \"GPU builds\".");
#endif
  done = true;
}

void ensure_mpi() {
  int inited = 0;
  MPI_Initialized(&inited);
  if (!inited) mfem::Mpi::Init();
  const DeviceRequest& dev = device_request();
  // A GPU request: hypre's own default policy (device), not MFEM's (the host, where MFEM always runs).
  if (!dev.gpu.empty()) mfem::Hypre::configure_runtime_policy_from_mfem = false;
  mfem::Hypre::Init();
  if (!dev.gpu.empty()) configure_hypre_gpu(dev);
}

int run_mesher(const std::string& mesher, const std::string& input, const std::string& log);

// ---------------------------------------------------------------- vector sources
// Piecewise source current density: per element attribute, a callable J(x).
class SourceJ : public mfem::VectorCoefficient {
 public:
  explicit SourceJ() : mfem::VectorCoefficient(3) {}
  std::map<int, std::function<void(const mfem::Vector&, mfem::ElementTransformation&, mfem::Vector&)>> by_attr;
  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    V.SetSize(3);
    V = 0.0;
    auto it = by_attr.find(T.Attribute);
    if (it == by_attr.end()) return;
    mfem::Vector x(3);
    T.Transform(ip, x);
    it->second(x, T, V);
  }
};

// A field of the serial mesh evaluated in the matching local element of the partitioned mesh (same vertex
// order, so the same reference point): projection of serial ND fields onto the parallel spaces is exact.
class SerialFieldCoefficient : public mfem::VectorCoefficient {
 public:
  SerialFieldCoefficient(mfem::Mesh& serial, const mfem::GridFunction& g, const std::vector<int>& loc2ser)
      : mfem::VectorCoefficient(3), m_(serial), g_(g), l2s_(loc2ser) {}
  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    mfem::ElementTransformation* Ts = m_.GetElementTransformation(l2s_[T.ElementNo]);
    Ts->SetIntPoint(&ip);
    g_.GetVectorValue(*Ts, ip, V);
  }

 private:
  mfem::Mesh& m_;
  const mfem::GridFunction& g_;
  const std::vector<int>& l2s_;
};

// The reverse: a field of the partitioned mesh evaluated in a serial element owned by this rank.
class LocalFieldCoefficient : public mfem::VectorCoefficient {
 public:
  LocalFieldCoefficient(mfem::ParMesh& pm, const mfem::ParGridFunction& g, const std::vector<int>& ser2loc)
      : mfem::VectorCoefficient(3), pm_(pm), g_(g), s2l_(ser2loc) {}
  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    mfem::ElementTransformation* Tl = pm_.GetElementTransformation(s2l_[T.ElementNo]);
    Tl->SetIntPoint(&ip);
    g_.GetVectorValue(*Tl, ip, V);
  }

 private:
  mfem::ParMesh& pm_;
  const mfem::ParGridFunction& g_;
  const std::vector<int>& s2l_;
};

double allsum(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return v;
}
double allmax(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return v;
}
double allmin(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  return v;
}
void allsum(double* v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); }
double gdot(const mfem::Vector& a, const mfem::Vector& b) { return mfem::InnerProduct(MPI_COMM_WORLD, a, b); }
double gnorm(const mfem::Vector& a) { return std::sqrt(std::max(0.0, gdot(a, a))); }

// unit azimuthal direction about an axis (origin o, unit direction a) and distance rho
void azimuthal(const mfem::Vector& x, const spec::Vec3& o, const spec::Vec3& a, double phi[3], double& rho) {
  double d[3] = {x(0) - o.x, x(1) - o.y, x(2) - o.z};
  double an = std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
  double u[3] = {a.x / an, a.y / an, a.z / an};
  double along = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
  double rv[3] = {d[0] - along * u[0], d[1] - along * u[1], d[2] - along * u[2]};
  rho = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
  // phi = u x r_hat
  phi[0] = (u[1] * rv[2] - u[2] * rv[1]) / std::max(rho, 1e-300);
  phi[1] = (u[2] * rv[0] - u[0] * rv[2]) / std::max(rho, 1e-300);
  phi[2] = (u[0] * rv[1] - u[1] * rv[0]) / std::max(rho, 1e-300);
}

}  // namespace

// ================================================================== implementation (model3d_impl.hpp)

Model3D::Model3D(spec::Design d) : d_(std::move(d)), p_(std::make_unique<Impl>()) {}

// gathered copy of a field on the serial mesh, identical on every rank (face-based and point post-processing):
// element-wise projection of the local field, shared DOFs averaged over their contributions
std::unique_ptr<mfem::ParGridFunction> Model3D::Impl::gather(const mfem::ParGridFunction& u) {
  if (!view_mesh) {
    view_mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_SELF, *serial);
    view_fes = std::make_unique<mfem::ParFiniteElementSpace>(view_mesh.get(), fec.get());
  }
  std::vector<int> ser2loc(serial->GetNE(), -1);
  for (size_t e = 0; e < loc2ser.size(); ++e) ser2loc[loc2ser[e]] = static_cast<int>(e);
  auto g = std::make_unique<mfem::ParGridFunction>(view_fes.get());
  mfem::Vector cnt(g->Size());
  *g = 0.0;
  cnt = 0.0;
  LocalFieldCoefficient uc(*pmesh, u, ser2loc);
  mfem::Array<int> vd;
  mfem::Vector vals;
  mfem::DofTransformation dt;
  for (size_t e = 0; e < loc2ser.size(); ++e) {
    const int s2 = loc2ser[e];
    view_fes->GetElementVDofs(s2, vd, dt);
    vals.SetSize(vd.Size());
    view_fes->GetFE(s2)->Project(uc, *view_mesh->GetElementTransformation(s2), vals);
    if (!dt.IsIdentity()) dt.TransformPrimal(vals);  // face DOFs of order >= 2 Nedelec tetrahedra
    g->AddElementVector(vd, vals);
    for (int k : vd) cnt(k >= 0 ? k : -1 - k) += 1.0;
  }
  MPI_Allreduce(MPI_IN_PLACE, g->GetData(), g->Size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, cnt.GetData(), cnt.Size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (int i = 0; i < g->Size(); ++i)
    if (cnt(i) > 0) (*g)(i) /= cnt(i);
  return g;
}
Model3D::~Model3D() = default;

std::string worst_elements_note(Model3D::Impl& P, const std::string& listed_in) {
  namespace mq = mesh_quality;
  mfem::ParMesh& pm = *P.pmesh;
  const long seq = pm.GetSequence();
  const long nodes_seq = pm.GetNodesSequence();
  auto region = [&P](int attr) {
    auto it = P.region_of_attr.find(attr);
    return it != P.region_of_attr.end() ? it->second : strcat("attribute ", attr);
  };
  mq::DescribeOptions opt;
  opt.three_d = true;
  opt.imported_file = P.mesh_info.value("imported_mesh_file", std::string());
  opt.listed_in = listed_in;
  int stale = (P.wq_mesh != &pm || P.wq_seq != seq || P.wq_nodes_seq != nodes_seq) ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &stale, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);  // every rank or none recomputes
  if (!stale) return mq::describe(P.wq_summary, region, opt);
  constexpr int keep = 5;
  constexpr int head = 5;  // number of records, elements, inverted, distorted, flat
  const int ne = pm.GetNE();
  // global element numbers: the serial ones (the partition keeps them in loc2ser), else, if any rank lacks that map,
  // a running offset over the ranks (decided collectively)
  int unmapped = P.loc2ser.size() == static_cast<size_t>(ne) ? 0 : 1;
  MPI_Allreduce(MPI_IN_PLACE, &unmapped, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  std::vector<int> global(ne);
  if (!unmapped) {
    global = P.loc2ser;
  } else {
    int offset = 0;
    MPI_Exscan(&ne, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (P.rank == 0) offset = 0;
    for (int e = 0; e < ne; ++e) global[e] = offset + e;
  }
  const mq::Summary local = mq::scan(pm, keep, &global);
  const int width = head + keep * mq::kRecordDoubles;
  std::vector<double> mine(width, 0.0);
  mine[0] = static_cast<double>(local.worst.size());
  mine[1] = static_cast<double>(local.elements);
  mine[2] = static_cast<double>(local.inverted);
  mine[3] = static_cast<double>(local.distorted);
  mine[4] = static_cast<double>(local.flat);
  for (size_t k = 0; k < local.worst.size(); ++k) mq::pack(local.worst[k], &mine[head + k * mq::kRecordDoubles]);
  std::vector<double> all(static_cast<size_t>(width) * P.nranks);
  MPI_Allgather(mine.data(), width, MPI_DOUBLE, all.data(), width, MPI_DOUBLE, MPI_COMM_WORLD);
  mq::Summary total;
  total.dim = pm.Dimension();
  for (int r = 0; r < P.nranks; ++r) {
    const double* rec = &all[static_cast<size_t>(r) * width];
    mq::Summary part;
    part.elements = static_cast<long long>(rec[1]);
    part.inverted = static_cast<long long>(rec[2]);
    part.distorted = static_cast<long long>(rec[3]);
    part.flat = static_cast<long long>(rec[4]);
    for (int k = 0; k < static_cast<int>(rec[0]); ++k)
      part.worst.push_back(mq::unpack(rec + head + k * mq::kRecordDoubles));
    mq::merge(total, part, keep);
  }
  P.wq_summary = total;
  P.mesh_info["worst_elements"] = mq::to_json(total, region);
  P.wq_mesh = &pm;
  P.wq_seq = seq;
  P.wq_nodes_seq = nodes_seq;
  const std::string text = mq::describe(total, region, opt);
  log(1, "solver failure:" + text);
  return text;
}

namespace {

int run_mesher(const std::string& mesher, const std::string& input, const std::string& logf) {
  return axi::run_process({mesher, input}, logf);
}

// axf-mesher's warnings in its log ("axf-mesher: WARNING: ..." or "axf-mesher: warning: ..."), for the results: a
// curved mesh that is not reproducible, for one, is said only there.
std::vector<std::string> mesher_warnings(const std::string& logf) {
  std::vector<std::string> out;
  std::ifstream f(logf);
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("axf-mesher: WARNING", 0) == 0 || line.rfind("axf-mesher: warning", 0) == 0)
      if (std::find(out.begin(), out.end(), line) == out.end()) out.push_back(line);
  return out;
}

// Conductor bodies: connected components (through faces) of the elements satisfying `in`.
std::vector<int> bodies_of(const mfem::Mesh& m, const std::function<bool(int)>& in, int& nb) {
  const int ne = m.GetNE();
  std::vector<int> b(ne, -1);
  const mfem::Table& el2el = const_cast<mfem::Mesh&>(m).ElementToElementTable();
  nb = 0;
  std::vector<int> st;
  for (int s = 0; s < ne; ++s) {
    if (b[s] >= 0 || !in(s)) continue;
    b[s] = nb;
    st.push_back(s);
    while (!st.empty()) {
      int e = st.back();
      st.pop_back();
      const int* row = el2el.GetRow(e);
      for (int k = 0; k < el2el.RowSize(e); ++k) {
        int n = row[k];
        if (b[n] < 0 && in(n)) {
          b[n] = nb;
          st.push_back(n);
        }
      }
    }
    ++nb;
  }
  return b;
}

// Reluctivity of a field state. Elements with a B-H law use nu(|B|) of the field `A`;
// others use the constant per-attribute value.
// "Infinity shell" [R1, R2) mapped onto [R1, infinity) by rho' = R1 (R2 - R1)/(R2 - rho) (radial stretch about
// the centre c). In the computational coordinates the vacuum reluctivity becomes the tensor
//   nu_eff = nu0 [ (1/f') (I - r r^T) + (f' rho^2 / f^2) r r^T ],  f = rho', f' = d rho'/d rho,
// and A . B integrands (Psi) are invariant. attr < 0: no shell.
struct ShellMap {
  int attr = -1;
  double R1 = 0, R2 = 0, c[3] = {0, 0, 0};
  bool in(const mfem::ElementTransformation& T) const { return attr > 0 && T.Attribute == attr; }
  void tensor(mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip, double nu0, mfem::DenseMatrix& K) const {
    mfem::Vector x(3);
    T.Transform(ip, x);
    double d[3] = {x(0) - c[0], x(1) - c[1], x(2) - c[2]};
    const double rho = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const double f = R1 * (R2 - R1) / std::max(R2 - rho, 1e-12 * R2), fp = f / std::max(R2 - rho, 1e-12 * R2);
    const double at = 1.0 / fp, ar = fp * rho * rho / (f * f);
    K.SetSize(3);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) {
        const double rr = d[i] * d[j] / (rho * rho);
        K(i, j) = nu0 * (at * ((i == j ? 1.0 : 0.0) - rr) + ar * rr);
      }
  }
};

class NuBCoefficient : public mfem::VectorCoefficient {  // nu(|B|) B (tensor in the shell), for residuals
 public:
  NuBCoefficient(const mfem::ParGridFunction& A, const std::vector<const BHLaw*>& bh, const mfem::Vector& nu_attr,
                 const ShellMap& sh)
      : mfem::VectorCoefficient(3), A_(A), bh_(bh), nu_(nu_attr), sh_(sh) {}
  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    T.SetIntPoint(&ip);
    V.SetSize(3);
    A_.GetCurl(T, V);
    if (sh_.in(T)) {
      mfem::DenseMatrix K;
      sh_.tensor(T, ip, nu_(T.Attribute - 1), K);
      mfem::Vector B(V);
      K.Mult(B, V);
      return;
    }
    const BHLaw* law = bh_[T.ElementNo];
    V *= law ? law->nu(V.Norml2()) : nu_(T.Attribute - 1);
  }

 private:
  const mfem::ParGridFunction& A_;
  const std::vector<const BHLaw*>& bh_;
  const mfem::Vector& nu_;
  const ShellMap& sh_;
};
// Differential (tangent) reluctivity nu I + 2 dnu/dB^2 B B^T, or the secant nu(|B|) I.
class NuTensorCoefficient : public mfem::MatrixCoefficient {
 public:
  // A == nullptr: linear values (iron at its initial permeability); shell elements get the mapped tensor
  NuTensorCoefficient(const mfem::ParGridFunction* A, const std::vector<const BHLaw*>& bh, const mfem::Vector& nu_attr,
                      bool secant, const ShellMap& sh)
      : mfem::MatrixCoefficient(3), A_(A), bh_(bh), nu_(nu_attr), secant_(secant), sh_(sh) {}
  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    T.SetIntPoint(&ip);
    if (sh_.in(T)) {
      sh_.tensor(T, ip, nu_(T.Attribute - 1), K);
      return;
    }
    K.SetSize(3);
    K = 0.0;
    const BHLaw* law = bh_[T.ElementNo];
    if (!law || !A_) {
      for (int i = 0; i < 3; ++i) K(i, i) = nu_(T.Attribute - 1);
      return;
    }
    mfem::Vector B(3);
    A_->GetCurl(T, B);
    const double b = B.Norml2();
    for (int i = 0; i < 3; ++i) K(i, i) = law->nu(b);
    if (secant_) return;
    const double d2 = 2.0 * law->dnu_dB2(b);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) K(i, j) += d2 * B(i) * B(j);
  }

 private:
  const mfem::ParGridFunction* A_;
  const std::vector<const BHLaw*>& bh_;
  const mfem::Vector& nu_;
  bool secant_;
  const ShellMap& sh_;
};

// ---------------------------------------------------------------- linear solver (AMS) options
// Settings of the AMS-preconditioned solves of the static/pickup 3D problems. The defaults are the ones
// validated in docs/validation/ams_robustness.md; each can be overridden from the environment so that
// benchmarks/ams_robustness.py can sweep them with one build. They are solver options only: within the
// solver tolerance the results must not depend on them.
double env_num(const char* name, double def) {
  const char* s = std::getenv(name);
  if (!s || !*s) return def;
  char* end = nullptr;
  const double v = std::strtod(s, &end);
  return end == s ? def : v;
}
struct SolverOpts {
  double gauge = -1;        // gauge mass eps_e = gauge * nu_e/h_e^2 (< 0: the default for the order)
  double theta = 0.5;       // AMG strength threshold inside AMS (hypre's default is 0.25)
  double tol = 1e-11;       // relative reduction of the preconditioned residual
  int global_gauge = 0;     // 1: one eps = gauge * nu0 / L^2 for the whole mesh (pre-fix behaviour, 1e-8)
  int h_vol = 0;            // 1: h_e = |element|^(1/3) instead of the smallest element width
  int ess_proj = 1;         // apply the preconditioner on the free DOFs only
  int cycle = 13;           // AMS cycle type
  int rlx = 2;              // AMS smoother type
  int amg_coarsen = 10;     // coarsening of the AMS AMG solves (10 HMIS; 8 PMIS, MFEM's choice on a GPU)
  int amg_relax = 8;        // relaxation of the AMS AMG solves (8 l1 hybrid symmetric GS; 18 l1-Jacobi, MFEM's GPU choice)
  int sweeps = 1;           // AMS smoother sweeps
  int agg = 0;              // aggressive-coarsening levels of the AMS AMG solves (MFEM's default is 1)
  int singular = 0;         // 1: AMS singular-problem mode (no beta Poisson matrix)
  int fgmres = 0;           // 1: FGMRES instead of PCG
  int reuse = 1;            // reuse the operator and the preconditioner between solves
  int maxit = 800;
  int print = 0;            // AMS/Krylov print level
  SolverOpts() {
    gauge = env_num("AXF_GAUGE", gauge);
    theta = env_num("AXF_AMS_THETA", theta);
    tol = env_num("AXF_SOLVER_TOL", tol);
    global_gauge = static_cast<int>(env_num("AXF_GAUGE_GLOBAL", global_gauge));
    h_vol = static_cast<int>(env_num("AXF_GAUGE_HVOL", h_vol));
    ess_proj = static_cast<int>(env_num("AXF_AMS_ESS_PROJ", ess_proj));
    cycle = env_int_checked("AXF_AMS_CYCLE", cycle);
    rlx = env_int_checked("AXF_AMS_RLX", rlx);
    amg_coarsen = env_int_checked("AXF_AMS_AMG_COARSEN", amg_coarsen);
    amg_relax = env_int_checked("AXF_AMS_AMG_RELAX", amg_relax);
    sweeps = env_int_checked("AXF_AMS_SWEEPS", sweeps);
    agg = env_int_checked("AXF_AMS_AGG", agg);
    singular = static_cast<int>(env_num("AXF_AMS_SINGULAR", singular));
    fgmres = static_cast<int>(env_num("AXF_SOLVER_FGMRES", fgmres));
    reuse = static_cast<int>(env_num("AXF_SOLVER_REUSE", reuse));
    maxit = static_cast<int>(env_num("AXF_SOLVER_MAXIT", maxit));
    print = static_cast<int>(env_num("AXF_SOLVER_PRINT", print));
  }
  json to_json(double gauge_used) const {
    return {{"gauge", gauge_used}, {"gauge_global", global_gauge != 0},
            {"ams_theta", theta}, {"ams_cycle", cycle}, {"ams_agg_levels", agg}, {"ams_sweeps", sweeps},
            {"ams_smoother", rlx}, {"ams_amg_coarsen", amg_coarsen}, {"ams_amg_relax", amg_relax},
            {"essential_projection", ess_proj != 0},
            {"singular", singular != 0}, {"fgmres", fgmres != 0}, {"reuse", reuse != 0},
            {"tol", tol}, {"max_iterations", maxit}};
  }
};

// Per-element scalar: the gauge mass eps_e.
class ElemCoefficient : public mfem::Coefficient {
 public:
  explicit ElemCoefficient(std::vector<double> v) : v_(std::move(v)) {}
  double Eval(mfem::ElementTransformation& T, const mfem::IntegrationPoint&) override { return v_[T.ElementNo]; }

 private:
  std::vector<double> v_;
};

// A preconditioner applied on the free DOFs only. Eliminating essential DOFs leaves decoupled rows whose
// residual is zero throughout the iteration; a correction placed there by the preconditioner only adds a
// spurious, badly scaled component to the Krylov space.
class FreeDofSolver : public mfem::Solver {
 public:
  FreeDofSolver(mfem::Solver& pc, const mfem::Array<int>& ess)
      : mfem::Solver(pc.Height(), pc.Width()), pc_(pc), ess_(ess) {
    iterative_mode = false;
  }
  void SetOperator(const mfem::Operator&) override {}
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
    xf_.SetSize(x.Size());
    xf_ = x;
    for (int i : ess_) xf_(i) = 0.0;
    pc_.Mult(xf_, y);
    for (int i : ess_) y(i) = 0.0;
  }

 private:
  mfem::Solver& pc_;
  const mfem::Array<int>& ess_;
  mutable mfem::Vector xf_;
};

// Unit period of a tree-gauge cochain: cotree values are integer multiples of the generator
// period. Values below 1e-6 of the largest are round-off from linear combinations and are ignored.
double unit_period(const std::vector<double>& h) {
  double vmax = 0;
  for (double v : h) vmax = std::max(vmax, std::fabs(v));
  double gmin = 0;
  for (double v : h)
    if (std::fabs(v) > 1e-6 * vmax && (gmin == 0 || std::fabs(v) < gmin)) gmin = std::fabs(v);
  return gmin;
}

// Geometric cut cochains for bodies of revolution about z: exact unit periods.
//  type 0 (azimuthal): signed crossings of the half-plane y = 0, x > 0 (positive towards +y).
//  type 1 (poloidal):  signed crossings of the plane z = zc restricted to rho < rho_mid (inner leg),
//                      positive towards +z; several zc are tried to avoid gaps.
// Only cochains that are closed on every face of the subset and not exact (nontrivial) are returned.
struct CutCochain {
  std::vector<double> h;
  int type;
};
// Local frame of a body of revolution: x' = R (x - o) with the axis direction mapped to +z.
struct AxisFrame {
  double o[3] = {0, 0, 0}, R[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  AxisFrame() = default;
  AxisFrame(const spec::Vec3& origin, const spec::Vec3& dir) {
    o[0] = origin.x;
    o[1] = origin.y;
    o[2] = origin.z;
    double z[3] = {dir.x, dir.y, dir.z};
    const double n = std::sqrt(z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
    for (double& v : z) v /= n;
    double t[3] = {1, 0, 0};  // any vector not parallel to z
    if (std::fabs(z[0]) > 0.9) t[0] = 0, t[1] = 1;
    double x[3] = {t[1] * z[2] - t[2] * z[1], t[2] * z[0] - t[0] * z[2], t[0] * z[1] - t[1] * z[0]};
    const double nx = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    for (double& v : x) v /= nx;
    const double y[3] = {z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]};
    for (int k = 0; k < 3; ++k) R[0][k] = x[k], R[1][k] = y[k], R[2][k] = z[k];
  }
  void map(const double* p, double* q) const {
    const double d[3] = {p[0] - o[0], p[1] - o[1], p[2] - o[2]};
    for (int i = 0; i < 3; ++i) q[i] = R[i][0] * d[0] + R[i][1] * d[1] + R[i][2] * d[2];
  }
};

std::vector<CutCochain> geometric_cuts(const mfem::Mesh& m, const std::vector<char>& in,
                                       const AxisFrame& frame = AxisFrame()) {
  const int ned = m.GetNEdges();
  // vertex coordinates in the body's frame (its axis along z)
  std::vector<double> X(3 * m.GetNV());
  for (int v = 0; v < m.GetNV(); ++v) frame.map(m.GetVertex(v), &X[3 * v]);
  mfem::Array<int> a, o;
  std::vector<char> edge_in(ned, 0);
  std::set<int> faces;
  double rmin = 1e300, rmax = 0, zmin = 1e300, zmax = -1e300;
  for (int e = 0; e < m.GetNE(); ++e) {
    if (!in[e]) continue;
    m.GetElementEdges(e, a, o);
    for (int x : a) edge_in[x] = 1;
    m.GetElementFaces(e, a, o);
    faces.insert(a.begin(), a.end());
    m.GetElementVertices(e, a);
    for (int v : a) {
      const double* x = &X[3 * v];
      double r = std::hypot(x[0], x[1]);
      rmin = std::min(rmin, r);
      rmax = std::max(rmax, r);
      zmin = std::min(zmin, x[2]);
      zmax = std::max(zmax, x[2]);
    }
  }
  auto closed = [&](const std::vector<double>& h) {
    double nz = 0;
    for (double v : h) nz += std::fabs(v);
    if (nz == 0) return false;
    mfem::Array<int> fe, fo;
    for (int f : faces) {
      const_cast<mfem::Mesh&>(m).GetFaceEdges(f, fe, fo);
      double sum = 0;
      for (int k = 0; k < fe.Size(); ++k) sum += fo[k] * h[fe[k]];
      if (std::fabs(sum) > 1e-9) return false;
    }
    return true;
  };
  // exact (cohomologically trivial) cochains are dh = h for a vertex potential: integrate h along a
  // spanning forest of the body's edges and compare on every edge
  auto exact = [&](const std::vector<double>& h) {
    const int nv = m.GetNV();
    std::vector<std::vector<std::pair<int, int>>> adj(nv);  // (neighbour, signed edge index + 1)
    mfem::Array<int> ev;
    for (int ed = 0; ed < ned; ++ed) {
      if (!edge_in[ed]) continue;
      m.GetEdgeVertices(ed, ev);
      adj[ev[0]].push_back({ev[1], ed + 1});
      adj[ev[1]].push_back({ev[0], -(ed + 1)});
    }
    std::vector<double> phi(nv, 0.0);
    std::vector<char> seen(nv, 0);
    for (int s0 = 0; s0 < nv; ++s0) {
      if (seen[s0] || adj[s0].empty()) continue;
      seen[s0] = 1;
      std::vector<int> st{s0};
      while (!st.empty()) {
        int v = st.back();
        st.pop_back();
        for (auto [w, se] : adj[v]) {
          if (seen[w]) continue;
          seen[w] = 1;
          phi[w] = phi[v] + (se > 0 ? h[se - 1] : -h[-se - 1]);
          st.push_back(w);
        }
      }
    }
    for (int ed = 0; ed < ned; ++ed) {
      if (!edge_in[ed]) continue;
      m.GetEdgeVertices(ed, ev);
      if (std::fabs(h[ed] - (phi[ev[1]] - phi[ev[0]])) > 1e-9) return false;
    }
    return true;
  };
  auto nontrivial = [&](const std::vector<double>& h) { return closed(h) && !exact(h); };
  std::vector<CutCochain> out;
  // azimuthal
  {
    std::vector<double> h(ned, 0.0);
    for (int ed = 0; ed < ned; ++ed) {
      if (!edge_in[ed]) continue;
      m.GetEdgeVertices(ed, a);
      const double *p0 = &X[3 * a[0]], *p1 = &X[3 * a[1]];
      if ((p0[1] < 0) == (p1[1] < 0)) continue;
      double t = p0[1] / (p0[1] - p1[1]);
      double xc = p0[0] + t * (p1[0] - p0[0]);
      if (xc > 0) h[ed] = p1[1] > p0[1] ? 1.0 : -1.0;
    }
    if (nontrivial(h)) out.push_back({h, 0});
  }
  // poloidal (inner leg), trying several heights
  const double rmid = 0.5 * (rmin + rmax);
  for (double frac : {0.5, 0.3, 0.7, 0.4, 0.6, 0.2, 0.8}) {
    const double zc = zmin + frac * (zmax - zmin);
    std::vector<double> h(ned, 0.0);
    for (int ed = 0; ed < ned; ++ed) {
      if (!edge_in[ed]) continue;
      m.GetEdgeVertices(ed, a);
      const double *p0 = &X[3 * a[0]], *p1 = &X[3 * a[1]];
      if ((p0[2] < zc) == (p1[2] < zc)) continue;
      double t = (zc - p0[2]) / (p1[2] - p0[2]);
      double xc = p0[0] + t * (p1[0] - p0[0]), yc = p0[1] + t * (p1[1] - p0[1]);
      if (std::hypot(xc, yc) < rmid) h[ed] = p1[2] > p0[2] ? 1.0 : -1.0;
    }
    if (nontrivial(h)) {
      out.push_back({h, 1});
      break;
    }
  }
  return out;
}

// A curved (second-order) element is invalid when its Jacobian determinant is not positive, or nearly
// degenerate, at the order-6 quadrature points: min det J <= 1e-3 max |det J|.
bool curved_element_invalid(mfem::Mesh& sm, int e) {
  mfem::ElementTransformation* T = sm.GetElementTransformation(e);
  const auto& ir = mfem::IntRules.Get(sm.GetElementBaseGeometry(e), 6);
  double emin = 1e300, emax = 0;
  for (int q = 0; q < ir.GetNPoints(); ++q) {
    T->SetIntPoint(&ir.IntPoint(q));
    const double w = T->Weight();
    emin = std::min(emin, w);
    emax = std::max(emax, std::fabs(w));
  }
  return !(emin > 1e-3 * emax);
}

// The two terminals of a current port. A port may consist of several disjoint gap pieces fed in parallel (a
// distributed port, e.g. readout modules spread along a slit): every face-connected piece must touch conductors on
// exactly two separate face patches. In every piece the patch further along `toward` is terminal 1 (V = 0) and the
// other terminal 0 (V = 1), so that the current crosses every piece along `toward`; for one piece this is the
// orientation rule of a single gap. The conduction solve then splits the total current between the pieces by their
// conductance G_k: with w = G / sum(G), the port sees Z = w^T Z_pieces w (Z_pieces: the pieces as separate ports).
// That equals an ideal common bus 1 / (1^T Z_pieces^-1 1) only when a symmetry of the structure permutes the pieces
// (identical AND symmetrically placed); in MQS the split always gives L >= L_bus (docs/design_format.md).
struct PortPiece {
  std::array<double, 3> centroid{};    // mean of its two terminal-patch centroids
  std::array<double, 3> crossing{};    // terminal-0 -> terminal-1 patch centroids: the current direction
  double cos_toward = 0;               // |cos| of the angle between that crossing and `toward`
};
struct PortTerminals {
  std::map<int, int> patch;     // serial terminal face -> patch
  std::map<int, int> terminal;  // serial terminal face -> 0 (V = 1) or 1 (V = 0)
  std::map<int, int> piece_of;  // serial port element -> piece
  std::vector<PortPiece> piece;
  std::vector<std::string> warnings;
  int pieces = 0;
  int patches = 0;
};

// strict: `toward` must separate the two terminals (always for RF ports; the MQS pickup port may be oriented later by
// the sign of Psi). A distributed port always needs it, since each piece is oriented separately.
PortTerminals port_terminals(mfem::Mesh& sm, const std::vector<int>& ser_attr, int pattr,
                             const std::vector<int>& tfaces, const spec::Vec3& toward, const std::string& path,
                             bool strict) {
  PortTerminals out;
  mfem::Array<int> fcs, ori;
  std::map<int, int> piece;  // port element -> piece
  for (int e0 = 0; e0 < sm.GetNE(); ++e0) {
    if (ser_attr[e0] != pattr || piece.count(e0)) continue;
    std::vector<int> pending{e0};
    piece[e0] = out.pieces;
    while (!pending.empty()) {
      const int e = pending.back();
      pending.pop_back();
      sm.GetElementFaces(e, fcs, ori);
      for (int face : fcs) {
        int e1, e2;
        sm.GetFaceElements(face, &e1, &e2);
        const int other = e1 == e ? e2 : e1;
        if (other >= 0 && ser_attr[other] == pattr && !piece.count(other)) {
          piece[other] = out.pieces;
          pending.push_back(other);
        }
      }
    }
    ++out.pieces;
  }
  AXF_REQUIRE(out.pieces > 0, "current-port region has no mesh elements", path);
  // terminal patches: the terminal faces grouped by edge adjacency
  std::map<int, std::vector<int>> edge_faces;
  mfem::Array<int> fed, fo;
  for (int f : tfaces) {
    sm.GetFaceEdges(f, fed, fo);
    for (int ed : fed) edge_faces[ed].push_back(f);
  }
  for (int f0 : tfaces) {
    if (out.patch.count(f0)) continue;
    std::vector<int> st{f0};
    out.patch[f0] = out.patches;
    while (!st.empty()) {
      const int f = st.back();
      st.pop_back();
      sm.GetFaceEdges(f, fed, fo);
      for (int ed : fed)
        for (int g : edge_faces[ed])
          if (!out.patch.count(g)) {
            out.patch[g] = out.patches;
            st.push_back(g);
          }
    }
    ++out.patches;
  }
  // each patch belongs to the piece of the port element beside it; vertex centroids for the orientation
  std::vector<int> piece_of_patch(out.patches, -1);
  std::vector<std::array<double, 4>> cent(out.patches, std::array<double, 4>{0, 0, 0, 0});
  for (const auto& [f, p] : out.patch) {
    int e1, e2;
    sm.GetFaceElements(f, &e1, &e2);
    const int pe = (e1 >= 0 && ser_attr[e1] == pattr) ? e1 : e2;
    const int pc = piece.at(pe);
    AXF_REQUIRE(piece_of_patch[p] < 0 || piece_of_patch[p] == pc, "a current-port terminal patch spans two port pieces",
                path, "Separate the pieces of a distributed port by more than one element.");
    piece_of_patch[p] = pc;
    mfem::Array<int> fv;
    sm.GetFaceVertices(f, fv);
    for (int v : fv) {
      const double* xv = sm.GetVertex(v);
      for (int i = 0; i < 3; ++i) cent[p][i] += xv[i];
      cent[p][3] += 1.0;
    }
  }
  std::vector<std::vector<int>> patches_of_piece(out.pieces);
  for (int p = 0; p < out.patches; ++p)
    if (piece_of_patch[p] >= 0) patches_of_piece[piece_of_patch[p]].push_back(p);
  const double tw[3] = {toward.x, toward.y, toward.z};
  const double twn = std::sqrt(tw[0] * tw[0] + tw[1] * tw[1] + tw[2] * tw[2]);
  const bool multi = out.pieces > 1;
  AXF_REQUIRE(!multi || twn > 0, strcat("a distributed port (", out.pieces, " pieces) needs `toward`"), path,
              "Give toward along the physical current direction across the gaps; each piece is oriented by it.");
  std::vector<int> term_of_patch(out.patches, 0);
  out.piece.resize(out.pieces);
  auto at = [](const std::array<double, 4>& c, int i) { return c[i] / c[3]; };
  for (int c = 0; c < out.pieces; ++c) {
    const auto& pp = patches_of_piece[c];
    std::array<double, 3> pc{0, 0, 0};  // where the piece is, for the messages
    for (int p : pp)
      for (int i = 0; i < 3; ++i) pc[i] += at(cent[p], i) / pp.size();
    const std::string where = multi ? strcat(" in piece ", c + 1, " of ", out.pieces, " near (", pc[0], ", ", pc[1],
                                             ", ", pc[2], ") m")
                                    : std::string();
    AXF_REQUIRE(pp.size() == 2,
                strcat("the readout port must touch conductors on exactly two separate faces (found ", pp.size(),
                       where, ")"),
                path, "The port region should fill the gap between two conductor faces (e.g. a slit or neck).");
    double sep[3], dir = 0, sn = 0;
    for (int i = 0; i < 3; ++i) {
      sep[i] = at(cent[pp[1]], i) - at(cent[pp[0]], i);
      dir += sep[i] * tw[i];
      sn += sep[i] * sep[i];
    }
    sn = std::sqrt(sn);
    const double cosv = (twn > 0 && sn > 0) ? std::fabs(dir) / (twn * sn) : 0.0;
    if (multi)  // the relative orientation of the pieces is physics: demand a clear projection on toward
      AXF_REQUIRE(cosv >= 0.5,
                  strcat("the pieces of a distributed port must all be crossed roughly along toward: piece ", c + 1,
                         " of ", out.pieces, " near (", pc[0], ", ", pc[1], ", ", pc[2], ") m is crossed along (",
                         sep[0], ", ", sep[1], ", ", sep[2], ") m, ",
                         std::acos(std::min(cosv, 1.0)) * 180 / M_PI, " degrees from toward"),
                  path, "Gaps crossed in different directions need separate fullwave.ports (one toward each).");
    else
      AXF_REQUIRE(!strict || (twn > 0 && std::fabs(dir) > 1e-12 * twn),
                  "toward is perpendicular to the separation of the two port terminals", path,
                  "Choose toward approximately along the physical current direction across the gap.");
    term_of_patch[pp[0]] = dir < 0 ? 1 : 0;  // the patch further along toward is terminal 1 (V = 0); dir 0: as found
    term_of_patch[pp[1]] = 1 - term_of_patch[pp[0]];
    const double s = term_of_patch[pp[0]] == 0 ? 1.0 : -1.0;
    out.piece[c] = {pc, {s * sep[0], s * sep[1], s * sep[2]}, cosv};
  }
  if (multi) {
    // One global toward cannot follow a gap whose crossing direction turns around the device (a radial gap: pieces
    // at 0 and 180 degrees are both aligned with toward = x but carry current outward and inward). Compare the
    // crossings in the local cylindrical frame about the z axis and warn when the pieces disagree.
    std::set<std::pair<int, int>> dominant;
    for (const auto& p : out.piece) {
      const double rho = std::hypot(p.centroid[0], p.centroid[1]);
      if (rho <= 0) continue;
      const double rx = p.centroid[0] / rho, ry = p.centroid[1] / rho;
      const double comp[3] = {p.crossing[0] * rx + p.crossing[1] * ry, -p.crossing[0] * ry + p.crossing[1] * rx,
                              p.crossing[2]};
      int k = 0;
      for (int i = 1; i < 3; ++i)
        if (std::fabs(comp[i]) > std::fabs(comp[k])) k = i;
      dominant.insert({k, comp[k] > 0 ? 1 : -1});
    }
    if (dominant.size() > 1)
      out.warnings.push_back(strcat(
          path, ": the pieces of the distributed port are crossed in different directions in the local (r, phi, z) "
          "frame about the z axis; with one global toward some may carry current in the opposite sense. Check the "
          "port's `pieces` diagnostics, or use separate fullwave.ports."));
  }
  for (const auto& [f, p] : out.patch) out.terminal[f] = term_of_patch[p];
  out.piece_of = std::move(piece);
  return out;
}

// Pieces of a distributed port that bridge two different conductor bodies must all connect them the same way round:
// no body may be terminal 0 of one piece and terminal 1 of another. (Pieces within one body, a slit in one
// conductor, carry no such information.) face_body: serial terminal face -> conductor body.
void check_piece_pairing(const PortTerminals& pt, const std::map<int, int>& face_body, mfem::Mesh& sm,
                         const std::vector<int>& ser_attr, int pattr, const std::string& path) {
  if (pt.pieces < 2) return;
  std::vector<std::array<int, 2>> pb(pt.pieces, std::array<int, 2>{-2, -2});  // per piece: body of terminal 0, 1
  for (const auto& [f, p] : pt.patch) {
    int e1, e2;
    sm.GetFaceElements(f, &e1, &e2);
    const int pe = (e1 >= 0 && ser_attr[e1] == pattr) ? e1 : e2;
    pb[pt.piece_of.at(pe)][pt.terminal.at(f)] = face_body.at(f);
  }
  std::set<int> as0, as1;
  for (const auto& b : pb)
    if (b[0] != b[1]) {
      as0.insert(b[0]);
      as1.insert(b[1]);
    }
  for (int b : as0)
    AXF_REQUIRE(!as1.count(b), "the pieces of a distributed port connect two conductor bodies in opposite senses",
                path, "Orient every piece the same way round with toward, or use separate fullwave.ports.");
}

// The current carried by each piece of a port, I_k = int_piece |grad V|^2 (sigma = 1, local sums reduced here), as
// the port's `pieces` diagnostics; warnings for a sliver-like piece and for an unequal split, which is not a bus.
json port_pieces_json(const PortTerminals& pt, std::vector<double> Ik, double Iraw, const std::string& region,
                      std::vector<std::string>& warn) {
  allsum(Ik.data(), static_cast<int>(Ik.size()));
  json out = json::array();
  double fmin = 1, fmax = 0;
  for (int c = 0; c < pt.pieces; ++c) {
    const double fr = Ik[c] / Iraw;
    fmin = std::min(fmin, fr);
    fmax = std::max(fmax, fr);
    const PortPiece& p = pt.piece[c];
    out.push_back({{"centroid_m", p.centroid}, {"crossing_m", p.crossing}, {"cos_toward", p.cos_toward},
                   {"current_fraction", fr}});
    if (pt.pieces > 1 && fr < 0.01)
      warn.push_back(strcat("port '", region, "': piece ", c + 1, " near (", p.centroid[0], ", ", p.centroid[1], ", ",
                            p.centroid[2], ") m carries ", 100 * fr, " % of the current (a sliver of the region?)"));
  }
  if (pt.pieces > 1 && fmax > 1.01 * fmin)
    warn.push_back(strcat("port '", region, "': its ", pt.pieces, " pieces carry unequal currents (fractions ", fmin,
                          " to ", fmax, "). The current is split by conductance, which equals an ideal common bus "
                          "only when a symmetry permutes the pieces; in MQS the split gives L >= L_bus."));
  return out;
}

}  // namespace


// Surface (lumped) readout port of an imported mesh (pickup.port_boundary; docs/comsol_import.md).
// The port is a set of mesh faces (a boundary group of type "port" in mesh.info_file, e.g. a COMSOL Lumped Port
// boundary).  Its terminals are the two connected chains of port-face boundary edges that lie on PEC (an edge
// of a PEC boundary group or of a PEC element).  The readout current I flows across the faces as a uniform
// surface current K = (I/w) e, with w the terminal length (mean of the two chains) and e the unit tangent of
// each face closest to the terminal-to-terminal direction (oriented along pickup.toward).  This is COMSOL's
// "Uniform" lumped port: V = (1/w) int_S E . e dS = -(1/I) int_S E . K dS, so Z = V/I by reciprocity.
// The source vector int_S K . N dS replaces the volume-port vector int J . N dV in the full-wave analysis.
namespace {
class SurfacePortK : public mfem::VectorCoefficient {
 public:
  SurfacePortK(const double t[3], double K0) : mfem::VectorCoefficient(3), K0_(K0) { for (int i = 0; i < 3; ++i) t_[i] = t[i]; }
  void Eval(mfem::Vector& K, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    K.SetSize(3);
    T.SetIntPoint(&ip);
    mfem::Vector n(3);
    mfem::CalcOrtho(T.Jacobian(), n);
    const double nn = n.Norml2();
    double e[3] = {t_[0], t_[1], t_[2]};
    if (nn > 0) {
      const double tn = (t_[0] * n(0) + t_[1] * n(1) + t_[2] * n(2)) / (nn * nn);
      for (int i = 0; i < 3; ++i) e[i] -= tn * n(i);
    }
    const double en = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    for (int i = 0; i < 3; ++i) K(i) = en > 0 ? K0_ * e[i] / en : 0.0;
  }
 private:
  double t_[3];
  double K0_;
};
}  // namespace

void prepare_surface_port(Model3D::Impl& P, const spec::Design& d, json& warnings) {
  const std::string path = "/pickup/port_boundary";
  const std::string name = d.pickup->port_boundary;
  auto& fes = *P.fes;
  mfem::Mesh& sm = *P.serial;
  int pattr = -1;
  std::set<int> pec_attrs;
  for (const auto& g : P.mesh_info.value("boundary_groups", json::array())) {
    const std::string type = g.value("type", std::string());
    // terminals: PEC or impedance (finite-conductivity) conductor boundaries
    if (type == "pec" || type == "impedance") pec_attrs.insert(g.at("attr").get<int>());
    if (g.value("name", std::string()) == name) {
      AXF_REQUIRE(type == "port", "boundary group '" + name + "' has type '" + type + "', not \"port\"", path);
      pattr = g.at("attr").get<int>();
    }
  }
  AXF_REQUIRE(pattr > 0, "pickup.port_boundary '" + name + "' is not a boundary group of mesh.info_file", path,
              "List it in boundary_groups as {\"name\": ..., \"attr\": ..., \"type\": \"port\"}.");
  // port faces and their edges (serial mesh: the same on every rank)
  std::vector<int> pbe;
  std::map<int, int> edge_use;
  mfem::Array<int> ed, eo;
  for (int i = 0; i < sm.GetNBE(); ++i) {
    if (sm.GetBdrAttribute(i) != pattr) continue;
    pbe.push_back(i);
    sm.GetBdrElementEdges(i, ed, eo);
    for (int e : ed) ++edge_use[e];
  }
  AXF_REQUIRE(!pbe.empty(), "port boundary group '" + name + "' has no faces in the mesh", path);
  std::set<int> pec_edges;
  for (int i = 0; i < sm.GetNBE(); ++i) {
    if (!pec_attrs.count(sm.GetBdrAttribute(i))) continue;
    sm.GetBdrElementEdges(i, ed, eo);
    for (int e : ed) pec_edges.insert(e);
  }
  for (int e = 0; e < sm.GetNE(); ++e) {
    auto it = P.region_of_attr.find(P.ser_attr[e]);
    const auto* region = it == P.region_of_attr.end() ? nullptr : d.find_region(it->second);
    if (!region) continue;
    const auto kind = d.materials.at(region->material).kind;
    if (kind != spec::MaterialKind::Conductor && kind != spec::MaterialKind::Superconductor) continue;
    sm.GetElementEdges(e, ed, eo);
    for (int x : ed) pec_edges.insert(x);
  }
  // terminal edges: on the outline of the port (used by one port face) and on PEC
  std::vector<int> tedges;
  for (auto& [e, n] : edge_use)
    if (n == 1 && pec_edges.count(e)) tedges.push_back(e);
  // chains (connected through shared vertices)
  std::map<int, std::vector<int>> vert_edges;
  mfem::Array<int> ev;
  for (int e : tedges) {
    sm.GetEdgeVertices(e, ev);
    for (int v : ev) vert_edges[v].push_back(e);
  }
  std::map<int, int> chain;
  int nchain = 0;
  for (int e0 : tedges) {
    if (chain.count(e0)) continue;
    std::vector<int> st{e0};
    chain[e0] = nchain;
    while (!st.empty()) {
      const int e = st.back();
      st.pop_back();
      sm.GetEdgeVertices(e, ev);
      for (int v : ev)
        for (int g : vert_edges[v])
          if (!chain.count(g)) { chain[g] = nchain; st.push_back(g); }
    }
    ++nchain;
  }
  AXF_REQUIRE(nchain == 2, strcat("the surface port must meet PEC along exactly two separate edge chains (found ",
                                  nchain, ")"), path,
              "A Uniform lumped port is a face between two conductors; check the PEC groups next to the port faces.");
  double len[2] = {0, 0}, cen[2][3] = {{0, 0, 0}, {0, 0, 0}};
  for (auto& [e, c] : chain) {
    sm.GetEdgeVertices(e, ev);
    const double *a = sm.GetVertex(ev[0]), *b = sm.GetVertex(ev[1]);
    const double l = std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
    len[c] += l;
    for (int i = 0; i < 3; ++i) cen[c][i] += 0.5 * l * (a[i] + b[i]);
  }
  for (int c = 0; c < 2; ++c) for (int i = 0; i < 3; ++i) cen[c][i] /= len[c];
  double t[3], tl = 0, tw = 0;
  const double tow[3] = {d.pickup->toward.x, d.pickup->toward.y, d.pickup->toward.z};
  for (int i = 0; i < 3; ++i) { t[i] = cen[1][i] - cen[0][i]; tl += t[i] * t[i]; tw += t[i] * tow[i]; }
  tl = std::sqrt(tl);
  AXF_REQUIRE(tl > 0, "the two port terminals coincide", path);
  const double tow_norm = std::sqrt(tow[0] * tow[0] + tow[1] * tow[1] + tow[2] * tow[2]);
  if (tow_norm > 0) {
    AXF_REQUIRE(std::fabs(tw) > 1e-6 * tl * tow_norm, "toward is perpendicular to the separation of the two port terminals",
                path, "Choose toward approximately along the physical current direction across the port.");
    if (tw < 0) for (double& x : t) x = -x;
  }
  for (double& x : t) x /= tl;
  double area = 0;
  for (int i : pbe) {  // straight-sided triangles (vertex positions)
    mfem::Array<int> bv;
    sm.GetBdrElementVertices(i, bv);
    AXF_REQUIRE(bv.Size() == 3, "surface ports need triangular faces", path);
    const double *a0 = sm.GetVertex(bv[0]), *a1 = sm.GetVertex(bv[1]), *a2 = sm.GetVertex(bv[2]);
    const double u[3] = {a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2]}, v[3] = {a2[0] - a0[0], a2[1] - a0[1], a2[2] - a0[2]};
    const double c[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    area += 0.5 * std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  }
  const double w = 0.5 * (len[0] + len[1]);
  const double rect = area / (w * tl);
  AXF_REQUIRE(std::fabs(len[0] - len[1]) <= 0.05 * w && std::fabs(rect - 1) <= 0.05,
              strcat("surface port '", name, "': terminal lengths ", len[0], " and ", len[1], " m, area/(w h) = ", rect,
                     "; the uniform-current feed needs a rectangular port between two parallel PEC edges"), path,
              "Check that only the two opposite edges of the port touch PEC (a side edge on PEC merges into a terminal).");
  if (std::fabs(len[0] - len[1]) > 0.005 * w || std::fabs(rect - 1) > 0.005)
    warnings.push_back(strcat("surface port '", name, "': terminal lengths ", len[0], " and ", len[1], " m, area/(w h) = ",
                              rect, " (not exactly rectangular)"));
  const double K0 = P.I_pick / w;
  SurfacePortK K(t, K0);
  mfem::ParMesh& pm = *P.pmesh;
  mfem::Array<int> marker(pm.bdr_attributes.Size() ? pm.bdr_attributes.Max() : 0);
  marker = 0;
  AXF_REQUIRE(pattr <= marker.Size(), "internal: port attribute exceeds the mesh boundary attributes", path);
  marker[pattr - 1] = 1;
  mfem::ParLinearForm l(&fes);
  l.AddBoundaryIntegrator(new mfem::VectorFEDomainLFIntegrator(K), marker);
  l.Assemble();
  P.port_surface_J = std::make_unique<mfem::Vector>(fes.GetTrueVSize());
  l.ParallelAssemble(*P.port_surface_J);
  P.port_diagnostics = {{"boundary", name}, {"attr", pattr}, {"faces", pbe.size()}, {"terminal_patches", nchain},
                        {"terminal_lengths_m", {len[0], len[1]}}, {"width_m", w}, {"length_m", tl}, {"area_m2", area},
                        {"area_over_width_times_length", rect}, {"direction", {t[0], t[1], t[2]}},
                        {"normalization_current_A", P.I_pick}, {"toward", {tow[0], tow[1], tow[2]}},
                        {"feed", "uniform surface current between the two PEC edges (COMSOL Uniform lumped port)"}};
}

// Explicit RF-only path: capacitor components need not have a magnetostatic
// pickup return. All old MQS/topological behavior remains in the normal path.
void prepare_rf_ports(Model3D::Impl& P,const spec::Design& d,const SolveOptions& o,json& warnings) {
  auto& pm=*P.pmesh;auto& fes=*P.fes;
  const int ne=pm.GetNE(),nser=P.serial->GetNE(),max_attr=pm.attributes.Max();
  int nb=0;
  auto body_rf_ser=bodies_of(*P.serial,[&](int e) {
    auto i=P.region_of_attr.find(P.ser_attr[e]);
    const auto* region=i==P.region_of_attr.end()?nullptr:d.find_region(i->second);
    if(!region) return false;
    auto kind=d.materials.at(region->material).kind;
    return kind==spec::MaterialKind::Conductor || kind==spec::MaterialKind::Superconductor;
  },nb);
  P.body_rf.assign(ne,-1);P.body_static.assign(ne,-1);
  for(int e=0;e<ne;++e) P.body_rf[e]=body_rf_ser[P.loc2ser[e]];
  P.I_pick=o.pickup_current;
  mfem::Vector b(fes.GetTrueVSize());b=0.0;
    auto prepare_port = [&](const std::string& region, const spec::Vec3& toward,
                            Model3D::Impl::ExtraPort* extra, mfem::Vector& source) {
    const bool primary = extra == nullptr;
    const std::string path = primary ? "/pickup/port" : "/fullwave/ports/" + extra->name;
    const int pattr = P.attr_of_region.at(region);
    auto& h1c_store = primary ? P.port_h1c : extra->h1c;
    auto& h1_store = primary ? P.port_h1 : extra->h1;
    auto& V_store = primary ? P.port_V : extra->V;
    auto& diagnostics = primary ? P.port_diagnostics : extra->diagnostics;
    // A terminal is a connected patch on a retained conductor or a named PEC boundary.
    // PMC faces are never terminals. Hole-body identity survives volume removal.
    mfem::Mesh& sm = *P.serial;
    std::vector<int> tfaces;
    std::map<int,int> terminal_body;
    std::map<int,int> boundary_identity;
    std::set<int> pec_attrs;
    for (const auto& group:P.mesh_info.value("boundary_groups",json::array())) {
      const std::string gty = group.value("type",std::string());
      if(gty=="pec" || gty=="impedance") pec_attrs.insert(group.at("attr").get<int>());  // conductor walls
    }
    for(int i=0;i<sm.GetNBE();++i) {
      const int attr=sm.GetBdrAttribute(i);
      if(pec_attrs.count(attr)) boundary_identity[sm.GetBdrElementFaceIndex(i)]=-1-attr;
    }
    mfem::Array<int> fcs, ori;
    for (int e = 0; e < nser; ++e) {
      if (P.ser_attr[e] != pattr) continue;
      sm.GetElementFaces(e, fcs, ori);
      for (int f : fcs) {
        int e1, e2;
        sm.GetFaceElements(f, &e1, &e2);
        int other = e1 == e ? e2 : e1;
        if (other >= 0 && body_rf_ser[other] >= 0) {tfaces.push_back(f);terminal_body[f]=body_rf_ser[other];}
        else if(other<0 && boundary_identity.count(f)) {tfaces.push_back(f);terminal_body[f]=boundary_identity.at(f);}
      }
    }
    // one or more gap pieces, each with two terminal patches oriented by `toward` (port_terminals)
    const PortTerminals pt = port_terminals(sm, P.ser_attr, pattr, tfaces, toward, path, true);
    const int npatch = pt.patches;
    {
      std::map<int, std::set<int>> patch_bodies;  // every terminal patch must touch exactly one conductor body
      std::set<int> tb[2];
      for (const auto& [f, p] : pt.patch) {
        patch_bodies[p].insert(terminal_body.at(f));
        tb[pt.terminal.at(f)].insert(terminal_body.at(f));
      }
      for (const auto& [p, bodies_p] : patch_bodies)
        AXF_REQUIRE(bodies_p.size() == 1, "each current-port terminal must touch exactly one connected conductor body",
                    path);
      // No DC return is required for an RF-only capacitor component. The
      // displacement current and any external load close the RF circuit.
      // [bodies of the V = 1 terminal, bodies of the V = 0 terminal]; nested lists when a terminal touches several
      diagnostics["terminal_body_ids"] =
          (tb[0].size() == 1 && tb[1].size() == 1)
              ? json{*tb[0].begin(), *tb[1].begin()}
              : json{json(std::vector<int>(tb[0].begin(), tb[0].end())),
                     json(std::vector<int>(tb[1].begin(), tb[1].end()))};
      check_piece_pairing(pt, terminal_body, sm, P.ser_attr, pattr, path);
      for (const auto& w : pt.warnings) warnings.push_back(w);
    }
    // conduction solve in the port: V = 1 on terminal 0, 0 on terminal 1 (kept for the full-wave analysis)
    h1c_store = std::make_unique<mfem::H1_FECollection>(P.order, 3);
    h1_store = std::make_unique<mfem::ParFiniteElementSpace>(&pm, h1c_store.get());
    mfem::ParFiniteElementSpace& h1 = *h1_store;
    mfem::Vector sig(max_attr);
    sig = 0.0;
    sig(pattr - 1) = 1.0;
    mfem::PWConstCoefficient sigma(sig);
    mfem::ParBilinearForm k(&h1);
    k.AddDomainIntegrator(new mfem::DiffusionIntegrator(sigma, &port_feed_rule(P.order)));
    k.Assemble();
    V_store = std::make_unique<mfem::ParGridFunction>(&h1);
    mfem::ParGridFunction& V = *V_store;
    V = 0.0;
    // port and terminal DOFs: local element faces correspond to the serial ones in element-face order
    // (the partitioned elements keep the serial vertex order); markers are OR-synchronized across ranks
    mfem::Array<int> in_port(h1.GetVSize()), term0(h1.GetVSize()), term1(h1.GetVSize());
    in_port = 0;
    term0 = 0;
    term1 = 0;
    mfem::Array<int> dofs, lf, lo;
    for (int e = 0; e < ne; ++e) {
      if (pm.GetAttribute(e) != pattr) continue;
      h1.GetElementDofs(e, dofs);
      for (int dd : dofs) in_port[dd] = 1;
      pm.GetElementFaces(e, lf, lo);
      sm.GetElementFaces(P.loc2ser[e], fcs, ori);
      for (int k = 0; k < fcs.Size(); ++k) {
        auto it = pt.terminal.find(fcs[k]);
        if (it == pt.terminal.end()) continue;
        h1.GetFaceDofs(lf[k], dofs);
        for (int dd : dofs) (it->second == 0 ? term0 : term1)[dd] = 1;
      }
    }
    h1.Synchronize(in_port);
    h1.Synchronize(term0);
    h1.Synchronize(term1);
    // terminal values: port_terminals made terminal 1 the side further along `toward` in every piece, so the
    // current J = -sigma grad V (normalized below to a positive I_pick) flows along `toward`
    const double v0 = 1.0, v1 = 0.0;
    std::vector<char> essv(h1.GetVSize(), 0);
    for (int i = 0; i < h1.GetVSize(); ++i) {
      if (term0[i]) { essv[i] = 1; V(i) = v0; }
      if (term1[i]) { essv[i] = 1; V(i) = v1; }
    }
    mfem::Array<int> ess_h1;
    for (int i = 0; i < h1.GetVSize(); ++i) {
      if (!in_port[i] || essv[i]) {
        int t = h1.GetLocalTDofNumber(i);
        if (t >= 0) ess_h1.Append(t);
      }
    }
    mfem::ParLinearForm zb(&h1);
    zb.Assemble();
    zb = 0.0;
    mfem::OperatorPtr Kop;
    mfem::Vector X, B;
    keep_eliminated_rows(k, ess_h1);  // the DOFs outside the port: zero rows (hypre on a GPU would drop them)
    k.FormLinearSystem(ess_h1, V, zb, Kop, X, B);
    mfem::HypreBoomerAMG amg(*Kop.As<mfem::HypreParMatrix>());
    amg.SetPrintLevel(0);
    mfem::HyprePCG pcg(*Kop.As<mfem::HypreParMatrix>());
    pcg.SetTol(1e-12);
    pcg.SetMaxIter(500);
    pcg.SetPrintLevel(0);
    pcg.SetPreconditioner(amg);
    pcg.Mult(B, X);
    int port_its = 0;
    pcg.GetNumIterations(port_its);
    if (port_its >= 500)
      warnings.push_back("readout port: the conduction solve reached its iteration cap." + worst_elements_note(P));
    // Refine to the rounding floor. J = -grad V enters the full-wave right-hand side, and its discrete divergence
    // at the free port nodes is this solve's residual; at low frequency the field amplifies that charge by about
    // (k h)^-2 (a 1.2e-14 relative residual gave a 12 % error in a port's electric energy at 100 Hz).
    int refine_steps = 0;
    {
      mfem::Vector r(B.Size()), dx(B.Size());
      const double nb = std::sqrt(std::max(allsum(B * B), 1e-300));
      double prev = 1e300;
      for (int step = 0; step < 4; ++step) {
        Kop->Mult(X, r);
        r -= B;
        r *= -1.0;  // r = B - K X
        const double rel = std::sqrt(allsum(r * r)) / nb;
        if (rel < 1e-17 || rel > 0.5 * prev) break;  // at the floor, or no longer improving
        prev = rel;
        dx = 0.0;
        pcg.Mult(r, dx);
        X += dx;
        ++refine_steps;
      }
    }
    mfem::Vector port_residual(B.Size());
    Kop->Mult(X, port_residual);
    port_residual -= B;
    const double port_rel = std::sqrt(allsum(port_residual * port_residual) /
                                      std::max(allsum(B * B), 1e-300));
    log(2, strcat("  port conduction solve: ", port_its, " PCG iterations, ", refine_steps,
                  " refinement steps, relative residual ", port_rel));
    if (!(std::isfinite(port_rel) && port_rel < 1e-8)) {  // port_rel is global: every rank takes this branch
      throw Error(strcat("port conduction solve failed its true-residual check (relative residual ", port_rel,
                         " after ", port_its, " PCG iterations and ", refine_steps,
                         " refinement steps; the gate is 1e-8)"),
                  path, worst_elements_note(P, "").substr(1));
    }
    k.RecoverFEMSolution(X, zb, V);
    // raw current with sigma = 1: I = int |grad V|^2 dV (unit voltage), and per piece
    double Iraw = 0;
    std::vector<double> Ik(pt.pieces, 0.0);
    mfem::Vector gV(3);
    for (int e = 0; e < ne; ++e) {
      if (pm.GetAttribute(e) != pattr) continue;
      AXF_REQUIRE(pm.GetElementBaseGeometry(e) == mfem::Geometry::TETRAHEDRON,
                  "a port region must be meshed with tetrahedra (its feed integrals share one tetrahedron rule)", "/pickup");
      mfem::ElementTransformation* T = pm.GetElementTransformation(e);
      const auto& ir = port_feed_rule(P.order);
      double Ie = 0;
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        T->SetIntPoint(&ir.IntPoint(q));
        V.GetGradient(*T, gV);
        Ie += (gV * gV) * ir.IntPoint(q).weight * T->Weight();
      }
      Iraw += Ie;
      Ik[pt.piece_of.at(P.loc2ser[e])] += Ie;
    }
    Iraw = allsum(Iraw);
    AXF_REQUIRE(Iraw > 0, "the port conduction problem carries no current", path);
    std::vector<std::string> piece_warn;
    const json pieces_json = port_pieces_json(pt, Ik, Iraw, region, piece_warn);
    for (const auto& w : piece_warn) warnings.push_back(w);
    const double scale = -P.I_pick / Iraw;  // J = -sigma grad V, normalized to I_pick
    if (primary) { P.port_scale = scale; P.port_attr = pattr; }
    else { extra->scale = scale; extra->attr = pattr; }
    const json terminal_ids=diagnostics["terminal_body_ids"];
    diagnostics = {{"terminal_body_ids",terminal_ids},{"region", region}, {"terminal_patches", npatch}, {"port_pieces", pt.pieces},
                   {"conduction_relative_residual", port_rel}, {"conduction_refinement_steps", refine_steps},
                   {"raw_conductance_m", Iraw},
                   {"normalization_current_A", P.I_pick}, {"toward", {toward.x,toward.y,toward.z}}};
    if (pt.pieces > 1) diagnostics["pieces"] = pieces_json;
    SourceJ Jp;
    mfem::ParGridFunction* Vp = &V;
    Jp.by_attr[pattr] = [Vp, scale](const mfem::Vector&, mfem::ElementTransformation& T, mfem::Vector& out) {
      mfem::Vector g(3);
      Vp->GetGradient(T, g);
      out(0) = scale * g(0);
      out(1) = scale * g(1);
      out(2) = scale * g(2);
    };
    mfem::ParLinearForm bl(&fes);
    bl.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(Jp, &port_feed_rule(P.order)));
    bl.Assemble();
    bl.ParallelAssemble(source);
    };
    if (!d.pickup) return;  // modes-only RF design (3D cavity): no readout port
    if (!d.pickup->port_boundary.empty()) {
      prepare_surface_port(P, d, warnings);
      return;
    }
    prepare_port(d.pickup->port, d.pickup->toward, nullptr, b);
    P.extra_ports.clear();
    P.extra_ports.reserve(d.fullwave.ports.size());
    for (const auto& port : d.fullwave.ports) {
      P.extra_ports.emplace_back();
      auto& extra = P.extra_ports.back();
      extra.name = port.name;
      extra.region = port.region;
      mfem::Vector unused_source(fes.GetTrueVSize());
      spec::Vec3 direction = port.toward;
      prepare_port(port.region, direction, &extra, unused_source);
    }

}

void Model3D::run(const SolveOptions& o) {
  const auto& ad = d_.mesh.adapt;
  if (o.mesh_only) {  // `axf mesh`: generate the mesh files and stop
    run_once(o);
    return;
  }
  // a solver failure's mesh check (worst_elements_note) may come after results.mesh was written
  auto publish_mesh_check = [this]() {
    if (p_ && p_->mesh_info.contains("worst_elements") && results_.contains("mesh"))
      results_["mesh"]["info"]["worst_elements"] = p_->mesh_info["worst_elements"];
  };
  if (ad.cycles <= 0) {
    run_once(o);
    if (d_.fullwave.enabled) fullwave_analysis(o);
    publish_mesh_check();
    return;
  }
  // adaptive refinement (as in 2D): re-mesh with target sizes from the indicators until Psi, L_p, W settle
  SolveOptions oc = o;
  json hist = json::array();
  bool converged = false;
  std::string why;
  for (int cyc = 0;; ++cyc) {
    if (cyc > 0) p_ = std::make_unique<Impl>();
    run_once(oc);
    const std::vector<double> eta2 = error_indicators();
    double tot = 0;
    for (double x : eta2) tot += x;
    json row = adapt_row(cyc, results_["mesh"]["elements"].get<long>(), results_["mesh"]["dofs"].get<long>(), results_,
                         std::sqrt(tot));
    if (!hist.empty()) {
      const double change = adapt_change(hist.back(), row, ad.watch);
      row["max_relative_change"] = change;
      if (change < ad.tol) converged = true;
    }
    hist.push_back(row);
    if (p_->rank == 0)
      log(1, strcat("adaptive cycle ", cyc, ": ", row["elements"].dump(), " elements, estimated error ", std::sqrt(tot)));
    if (converged) break;
    if (cyc >= ad.cycles) {
      why = "adaptive refinement stopped after " + std::to_string(ad.cycles) + " cycles before the quantities settled";
      break;
    }
    const Impl& P = *p_;
    mfem::ParMesh& vm = P.vmesh();
    const int ne = vm.GetNE(), pord = P.order;
    const double eta_t2 = ad.reduction * ad.reduction * tot / std::max(ne, 1);
    std::vector<double> hnew(ne);
    double projected = 0;
    for (int e = 0; e < ne; ++e) {
      const double h = std::cbrt(6.0 * std::sqrt(2.0) * vm.GetElementVolume(e));  // regular-tetrahedron edge
      double f = eta2[e] > 0 ? std::pow(eta_t2 / eta2[e], 0.5 / (pord + 1.5)) : 1.0;  // eta_e ~ h^(p + 3/2)
      f = std::clamp(f, 0.3, 1.0);
      hnew[e] = h * f;
      projected += 1.0 / (f * f * f);
    }
    if (projected > ad.max_elements) {
      const double g = std::cbrt(projected / ad.max_elements);
      for (double& h : hnew) h *= g;
      why = "adaptive refinement limited by mesh.adapt.max_elements";
    }
    std::vector<double> hv(vm.GetNV(), 1e300);
    mfem::Array<int> vv;
    for (int e = 0; e < ne; ++e) {
      vm.GetElementVertices(e, vv);
      for (int v : vv) hv[v] = std::min(hv[v], hnew[e]);
    }
    const std::string path = oc.workdir + "/adapt_size_" + std::to_string(cyc) + ".bin";
    if (P.rank == 0) {
      std::ofstream f(path, std::ios::binary);
      const int32_t nn = 4;
      const int64_t cnt = ne;
      f.write(reinterpret_cast<const char*>(&nn), sizeof nn);
      f.write(reinterpret_cast<const char*>(&cnt), sizeof cnt);
      for (int e = 0; e < ne; ++e) {
        vm.GetElementVertices(e, vv);
        double rec[16];
        for (int k = 0; k < 4; ++k) {
          const double* x = vm.GetVertex(vv[k]);
          for (int i = 0; i < 3; ++i) rec[3 * k + i] = x[i];
          rec[12 + k] = hv[vv[k]];
        }
        f.write(reinterpret_cast<const char*>(rec), sizeof rec);
      }
    }
    oc.size_field_file = path;
  }
  results_["adapt"] = {{"converged", converged}, {"tolerance", ad.tol}, {"watch", ad.watch}, {"history", hist}};
  if (!converged && !why.empty()) results_["warnings"].push_back(why);
  if (d_.fullwave.enabled) fullwave_analysis(o);
  publish_mesh_check();
}

std::vector<double> Model3D::error_indicators() const {
  const Impl& P = *p_;
  mfem::ParMesh& vm = P.vmesh();
  const int ne = vm.GetNE();
  const mfem::ParGridFunction* fields[2] = {P.vA_lab(), P.have_pick ? P.vA_pick() : nullptr};
  std::vector<double> eta2(ne, 0.0);
  const mfem::IntegrationRule* verts = mfem::Geometries.GetVertices(mfem::Geometry::TETRAHEDRON);
  const mfem::IntegrationRule& ir = mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * P.order + 1);
  mfem::Array<int> vv;
  mfem::Vector B(3);
  auto shell = [&](int e) { return P.sh_attr > 0 && vm.GetAttribute(e) == P.sh_attr; };
  for (const mfem::ParGridFunction* g : fields) {
    if (!g) continue;
    // recovered vertex values per (vertex, attribute): material interfaces keep their jumps
    std::map<std::pair<int, int>, std::array<double, 4>> rec;
    for (int e = 0; e < ne; ++e) {
      if (shell(e)) continue;
      vm.GetElementVertices(e, vv);
      mfem::ElementTransformation* T = vm.GetElementTransformation(e);
      for (int k = 0; k < 4; ++k) {
        T->SetIntPoint(&verts->IntPoint(k));
        g->GetCurl(*T, B);
        auto& r = rec[{vv[k], vm.GetAttribute(e)}];
        for (int i = 0; i < 3; ++i) r[i] += B(i);
        r[3] += 1;
      }
    }
    std::vector<double> el(ne, 0.0);
    double norm = 0;
    for (int e = 0; e < ne; ++e) {
      if (shell(e)) continue;
      vm.GetElementVertices(e, vv);
      std::array<double, 4> rv[4];
      for (int k = 0; k < 4; ++k) rv[k] = rec[{vv[k], vm.GetAttribute(e)}];
      mfem::ElementTransformation* T = vm.GetElementTransformation(e);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = ir.IntPoint(q);
        T->SetIntPoint(&ip);
        g->GetCurl(*T, B);
        const double lam[4] = {1 - ip.x - ip.y - ip.z, ip.x, ip.y, ip.z}, w = ip.weight * T->Weight();
        for (int i = 0; i < 3; ++i) {
          double star = 0;
          for (int k = 0; k < 4; ++k) star += lam[k] * rv[k][i] / std::max(rv[k][3], 1.0);
          el[e] += (B(i) - star) * (B(i) - star) * w;
          norm += B(i) * B(i) * w;
        }
      }
    }
    if (norm > 0)
      for (int e = 0; e < ne; ++e) eta2[e] += el[e] / norm;
    // With Gamma watched (mesh.adapt.watch), the same recovery error of the pickup field on the conductor faces of
    // pickup.losses, weighted by 1/mu_r^2 of the field side (|H|^2, as Gamma) and relative to the faces' weighted
    // norm: it refines where the loss factor's error sits on the surfaces, which the volume term weighs by volume
    // only. Assigned to the field-side element.
    const auto& watch = d_.mesh.adapt.watch;
    if (g != fields[1] || std::find(watch.begin(), watch.end(), "Gamma") == watch.end()) continue;
    // per attribute: conductor or superconductor (the faces of pickup.losses), and 1/mu_r^2 of a field side, so that
    // the term weighs |H|^2 as Gamma does (linear and iron at their mu_r)
    static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
    std::map<int, std::pair<bool, double>> attr_info;
    for (int e = 0; e < ne; ++e) {
      const int a = vm.GetAttribute(e);
      if (attr_info.count(a)) continue;
      const spec::Region* reg = d_.find_region(P.region_of_attr.count(a) ? P.region_of_attr.at(a) : "air");
      const spec::Material& m = reg ? d_.materials.at(reg->material) : vacuum;
      const bool magnetic = m.kind == spec::MaterialKind::Linear || m.kind == spec::MaterialKind::Iron;
      attr_info[a] = {m.kind == spec::MaterialKind::Superconductor || m.kind == spec::MaterialKind::Conductor,
                      magnetic ? 1.0 / (m.mu_r * m.mu_r) : 1.0};
    }
    auto rfc = [&](int e) { return attr_info.at(vm.GetAttribute(e)).first; };
    const mfem::IntegrationRule& irs = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * P.order + 2);
    std::vector<double> es(ne, 0.0);
    double snorm = 0;
    mfem::Vector nor(3);
    for (int f = 0; f < vm.GetNumFaces(); ++f) {
      int e1, e2;
      vm.GetFaceElements(f, &e1, &e2);
      if (e1 < 0 || e2 < 0 || rfc(e1) == rfc(e2)) continue;
      const int ev = rfc(e1) ? e2 : e1;
      if (shell(ev)) continue;
      vm.GetElementVertices(ev, vv);
      std::array<double, 4> rv[4];
      for (int k = 0; k < 4; ++k) rv[k] = rec[{vv[k], vm.GetAttribute(ev)}];
      mfem::FaceElementTransformations* FT = vm.GetFaceElementTransformations(f);
      for (int q = 0; q < irs.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = irs.IntPoint(q);
        FT->SetAllIntPoints(&ip);
        mfem::ElementTransformation& T = ev == e1 ? *FT->Elem1 : *FT->Elem2;
        g->GetCurl(T, B);
        mfem::CalcOrtho(FT->Jacobian(), nor);
        const mfem::IntegrationPoint& ep = T.GetIntPoint();
        const double lam[4] = {1 - ep.x - ep.y - ep.z, ep.x, ep.y, ep.z};
        const double w = ip.weight * nor.Norml2() * attr_info.at(vm.GetAttribute(ev)).second;
        for (int i = 0; i < 3; ++i) {
          double star = 0;
          for (int k = 0; k < 4; ++k) star += lam[k] * rv[k][i] / std::max(rv[k][3], 1.0);
          es[ev] += (B(i) - star) * (B(i) - star) * w;
          snorm += B(i) * B(i) * w;
        }
      }
    }
    if (snorm > 0)
      for (int e = 0; e < ne; ++e) eta2[e] += es[e] / snorm;
  }
  return eta2;
}

void Model3D::run_once(const SolveOptions& o) {
  Timer timer;
  json timing;
  double t_last = 0;
  auto lap = [&](const char* stage) {
    double t = timer.seconds();
    timing[stage] = t - t_last;
    t_last = t;
  };
  ensure_mpi();
  check_integer_solver_env();  // a bad AXF_AMS_* or AXF_FW_AMS_* stops the run here, before meshing
  Impl& P = *p_;
  auto& d = d_;
  json warnings = json::array();
  MPI_Comm_size(MPI_COMM_WORLD, &P.nranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &P.rank);
  const int saved_log = log_level();
  if (P.rank != 0) log_level() = 0;
  struct LogRestore {
    int v;
    ~LogRestore() { log_level() = v; }
  } log_restore{saved_log};
  AXF_REQUIRE(d.curves.empty(), "zero-thickness curves (sheets) are not supported in 3D designs", "/curves",
              "Model sheets as thin regions.");

  // ------------------------------------------------------------ mesh
  {
    std::error_code ec;
    std::filesystem::create_directories(o.workdir, ec);
    json in;
    in["dim"] = 3;
    in["domain_radius"] = d.mesh.domain_radius;  // <= 0: automatic in the mesher
    in["infinity_shell"] = d.fullwave.rf_only ? 0.0 : 1.25;                  // exact open boundary: shell [Rb, 1.25 Rb] mapped to infinity
    in["shell_attr"] = kShellAttr;
    in["size"] = d.mesh.size;
    in["symmetry"]=d.mesh.symmetry;
    if(!d.mesh.domain_solid.is_null()) in["domain_solid"]=d.mesh.domain_solid;
    if (d.fullwave.enabled && !d.fullwave.frequencies.empty()) {
      // full wave: the shell is a complex-scaled map (absorbing layer), resolved with more element layers;
      // at most lambda/8 per element at the highest frequency
      in["shell_layers"] = 4;
      const double fmax = *std::max_element(d.fullwave.frequencies.begin(), d.fullwave.frequencies.end());
      in["max_size"] = units::c / fmax / 8;
    }
    in["surface_size"] = d.mesh.surface_size;
    in["order"] = d.mesh.geometry_order > 0 ? d.mesh.geometry_order : std::min(d.mesh.order, 2);
    in["outer_attr"] = axi::kOuterAttr;
    in["air_attr"] = axi::kAirAttr;
    json regs = json::array(), refine = json::array();
    int attr = 2;
    for (auto& g : d.regions) {
      json r = {{"name", g.name}, {"attr", attr}, {"solid", g.solid.resolved()}};
      if (g.mesh_size > 0) r["size"] = g.mesh_size;
      if (g.surface_size > 0) r["surface_size"] = g.surface_size;  // per-region surface size (axf-mesher mesh3d)
      regs.push_back(r);
      P.attr_of_region[g.name] = attr;
      P.region_of_attr[attr] = g.name;
      auto k = d.materials.at(g.material).kind;
      bool wind = std::any_of(d.windings.begin(), d.windings.end(), [&](auto& w) { return w.region == g.name; });
      bool port = d.pickup && d.pickup->port == g.name;
      if (d.mesh.remove_pec && (k==spec::MaterialKind::Conductor || k==spec::MaterialKind::Superconductor))
        regs.back()["remove_pec"]=true;
      if (k != spec::MaterialKind::Vacuum || wind || port) refine.push_back(g.name);
      ++attr;
    }
    P.region_of_attr[axi::kAirAttr] = "air";
    P.region_of_attr[kShellAttr] = "infinity_shell";
    in["regions"] = regs;
    in["refine_regions"] = refine;
    if (!o.size_field_file.empty()) in["size_field_file"] = o.size_field_file;
    const std::string base = o.workdir + "/mesh3d";
    in["output"] = base + ".msh";
    in["info"] = base + "_info.json";
    in["threads"] = 1;  // deterministic meshes (multi-threaded HXT is not reproducible)
    if(d.source.contains("mesh"))
      for(const char* key:{"far_size","surface_layer","threads","algorithm3d","curvature_points","extend_from_boundary",
                           "optimize_netgen"})
        if(d.source["mesh"].contains(key)) in[key]=d.source["mesh"][key];
    if (P.rank == 0) std::ofstream(base + "_in.json") << in.dump(2);
    std::string mesher = axi::find_mesher(o.mesher);
    log(1, "meshing (3D)");
    int rc = 0;
    if (P.rank == 0 && d.mesh.file.empty()) rc = run_mesher(mesher, base + "_in.json", base + "_mesher.log");
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    json info;
    std::ifstream f(d.mesh.file.empty()?base + "_info.json":d.mesh.info_file);
    if (f) info = json::parse(f, nullptr, false);
    if (rc != 0 || !info.is_object() || !info.value("ok", false)) {
      std::string why = info.is_object() ? info.value("error", std::string("unknown")) : "no info file";
      throw Error("3D mesh generation failed: " + why, "/regions", "See " + base + "_mesher.log.");
    }
    P.mesh_info = info;
    log(1, strcat("mesh: ", info.value("elements_3d", 0), " tetrahedra in ", info.value("seconds", 0.0), " s"));
    if (d.mesh.file.empty())
      for (const auto& w : mesher_warnings(base + "_mesher.log")) warnings.push_back(w);
    const std::string mesh_path=d.mesh.file.empty()?base+".msh":d.mesh.file;
    if (o.mesh_only) {
      // the curved-element check the solve would apply, reported without solving (no straightening here)
      mfem::Mesh sm(mesh_path.c_str(), 1, 1, true);
      if (sm.GetNodes()) {
        std::size_t n = 0;
        for (int e = 0; e < sm.GetNE(); ++e)
          if (curved_element_invalid(sm, e)) ++n;
        P.mesh_info["curved_elements_invalid"] = n;
      }
      results_ = {{"mesh", {{"file", mesh_path}, {"info", P.mesh_info}}}, {"warnings", warnings}};
      return;
    }
    P.serial = std::make_unique<mfem::Mesh>(mesh_path.c_str(), 1, 1, true);
    if(!d.mesh.file.empty()) {
      P.mesh_info["imported_mesh_file"]=d.mesh.file;
      // every element of an imported mesh must carry the attribute of a design region (2 + region index), or 1 (air)
      std::set<int> unknown;
      for (int e = 0; e < P.serial->GetNE(); ++e) {
        const int a = P.serial->GetAttribute(e);
        if (a != axi::kAirAttr && !P.region_of_attr.count(a)) unknown.insert(a);
      }
      AXF_REQUIRE(unknown.empty(), strcat("imported mesh ", d.mesh.file, " has elements with attribute ", *unknown.begin(),
                                          ", which is no design region"), "/mesh/file",
                  "Region k of the design (0-based, disabled regions skipped) owns attribute 2 + k; attribute 1 is air.");
      // the info file may name the volume groups: they must be the design's regions in the same order
      for (const auto& g : P.mesh_info.value("groups", json::array())) {
        if (g.value("dim", 0) != 3 || !g.contains("name") || !g.contains("attr")) continue;
        const int a = g.at("attr").get<int>();
        const std::string nm = g.at("name").get<std::string>();
        auto it = P.region_of_attr.find(a);
        AXF_REQUIRE(it != P.region_of_attr.end() && it->second == nm,
                    strcat("imported mesh attribute ", a, " is '", nm, "' in ", d.mesh.info_file, " but the design gives it to '",
                           it == P.region_of_attr.end() ? std::string("(no region)") : it->second, "'"), "/regions",
                    "Keep the regions of an imported design in the order written by the importer (attribute 2 + k).");
      }
    }
    // curved (second-order) elements can be inverted near thin curved features: check det J at
    // quadrature points and fall back to straight-sided geometry if any element is invalid
    if (P.serial->GetNodes()) {
      mfem::Mesh& sm = *P.serial;
      auto invalid = [&](int e) { return curved_element_invalid(sm, e); };
      std::vector<int> bad;
      for (int e = 0; e < sm.GetNE(); ++e)
        if (invalid(e)) bad.push_back(e);
      P.mesh_info["curved_elements_invalid"] = bad.size();
      if (!bad.empty()) {
        // straighten only the offending elements: their high-order nodes move to the linear positions
        mfem::GridFunction& nodes = *sm.GetNodes();
        const mfem::FiniteElementSpace* nfes = nodes.FESpace();
        int fixed_pass = 0;
        for (int pass = 0; pass < 3 && !bad.empty(); ++pass) {
          for (int e : bad) {
            mfem::Array<int> vd, vv;
            nfes->GetElementVDofs(e, vd);
            sm.GetElementVertices(e, vv);
            const mfem::IntegrationRule& nir = nfes->GetFE(e)->GetNodes();
            const int nd = nir.GetNPoints();
            for (int i = 0; i < nd; ++i) {
              const mfem::IntegrationPoint& ip = nir.IntPoint(i);
              const double l[4] = {1 - ip.x - ip.y - ip.z, ip.x, ip.y, ip.z};
              for (int c = 0; c < 3; ++c) {
                double x = 0;
                for (int k = 0; k < 4; ++k) x += l[k] * sm.GetVertex(vv[k])[c];
                const int dof = vd[c * nd + i];
                nodes(dof >= 0 ? dof : -1 - dof) = x;
              }
            }
          }
          std::vector<int> still;
          for (int e = 0; e < sm.GetNE(); ++e)
            if (invalid(e)) still.push_back(e);
          bad.swap(still);
          fixed_pass = pass + 1;
        }
        if (!bad.empty()) {
          warnings.push_back(strcat(bad.size(), " curved elements stay invalid; using straight-sided geometry"));
          sm.SetCurvature(1, false, 3, mfem::Ordering::byVDIM);
        } else {
          log(1, strcat("mesh: straightened ", P.mesh_info["curved_elements_invalid"].get<size_t>(),
                        " inverted curved elements locally (", fixed_pass, " pass(es))"));
          warnings.push_back(strcat(P.mesh_info["curved_elements_invalid"].get<size_t>(),
                                    " inverted curved elements were straightened locally (geometry approximated there)"));
        }
      }
    }
    for (int i = 0; i < d.mesh.refinements; ++i) P.serial->UniformRefinement();
    // Check the identical serial mesh on every rank before partitioning/collectives.
    // Affine slivers have constant det(J), so the curved-geometry test cannot detect them.
    double worst_aspect = 0; int worst_element = -1;
    for (int e = 0; e < P.serial->GetNE(); ++e) {
      const double h = P.serial->GetElementSize(e, 1);
      const double hv = P.serial->GetElementSize(e, 2);
      const int attr = P.serial->GetAttribute(e);
      AXF_REQUIRE(detail::valid_element_width(h, hv),
                  strcat("degenerate 3D mesh element ", e, " in region ", P.region_of_attr[attr],
                         " (attribute ", attr, "): minimum width ", h, ", maximum width ", hv),
                  "/mesh", "Jacobian aspect ratio must be <= 1e6. Remesh with different size/surface_size or repair near-zero-volume CAD fragments.");
      if (hv/h > worst_aspect) { worst_aspect=hv/h; worst_element=e; }
    }
    P.mesh_info["maximum_jacobian_aspect_ratio"] = worst_aspect;
    P.mesh_info["worst_aspect_element"] = worst_element;
    if (worst_aspect > 1e3)
      warnings.push_back(strcat("3D mesh Jacobian aspect ratio ", worst_aspect, " at element ", worst_element,
                               " exceeds 1e3: inspect thin features and verify residuals and mesh convergence"));
    // partition (METIS); local elements keep the serial order within each rank
    const int nser = P.serial->GetNE();
    P.ser_attr.resize(nser);
    for (int s2 = 0; s2 < nser; ++s2) P.ser_attr[s2] = P.serial->GetAttribute(s2);
    P.loc2ser.clear();
    if (P.nranks > 1) {
      int* part = nullptr;
      {  // METIS swaps process-wide signal handlers: under the lock every METIS call takes (sym_solver.hpp)
        std::lock_guard<std::mutex> metis(axf::metis_mutex());
        part = P.serial->GeneratePartitioning(P.nranks, 1);
      }
      for (int s2 = 0; s2 < nser; ++s2)
        if (part[s2] == P.rank) P.loc2ser.push_back(s2);
      P.pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, *P.serial, part);
      delete[] part;
    } else {
      P.loc2ser.resize(nser);
      std::iota(P.loc2ser.begin(), P.loc2ser.end(), 0);
      P.pmesh = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, *P.serial);
    }
    AXF_REQUIRE(static_cast<int>(P.loc2ser.size()) == P.pmesh->GetNE(), "internal: partition size mismatch");
    {  // the local/serial element correspondence: same element maps at the reference vertices (curved meshes
       // reset ParMesh vertices from the nodes, so compare the maps, with a tolerance)
      mfem::Vector mn, mx, xl(3), xs(3);
      P.serial->GetBoundingBox(mn, mx);
      double Lb = 0;
      for (int i = 0; i < 3; ++i) Lb = std::max(Lb, mx(i) - mn(i));
      for (int e = 0; e < P.pmesh->GetNE(); e += std::max(1, P.pmesh->GetNE() / 16)) {
        const mfem::IntegrationRule& rv = *mfem::Geometries.GetVertices(P.pmesh->GetElementBaseGeometry(e));
        mfem::ElementTransformation* Tl = P.pmesh->GetElementTransformation(e);
        mfem::ElementTransformation* Ts = P.serial->GetElementTransformation(P.loc2ser[e]);
        for (int k = 0; k < rv.GetNPoints(); ++k) {
          Tl->Transform(rv.IntPoint(k), xl);
          Ts->Transform(rv.IntPoint(k), xs);
          xl -= xs;
          AXF_REQUIRE(xl.Normlinf() <= 1e-9 * Lb, "internal: partitioned element order differs from the serial mesh");
        }
      }
    }
    if (P.nranks > 1) log(1, strcat("3D mesh partitioned over ", P.nranks, " MPI ranks"));
  }
  lap("mesh_s");
  mfem::ParMesh& pm = *P.pmesh;
  const int ne = pm.GetNE();
  P.order = o.order > 0 ? o.order : d.mesh.order;
  if (P.order >= 3)
    warnings.push_back("3D order >= 3 is not validated: AMS-PCG can stall next to thin features and slits, and its gauge regularization costs 1e-7 of Psi instead of 1e-8; prefer order 2");
  for (auto& w : d.windings)
    if (!w.conductor.empty())
      warnings.push_back("winding '" + w.name + "' names a conductor ('" + w.conductor +
                         "'), but critical-current margins are computed by the axisymmetric engine only");
  P.fec = std::make_unique<mfem::ND_FECollection>(P.order, 3);
  P.fes = std::make_unique<mfem::ParFiniteElementSpace>(&pm, P.fec.get());
  P.fec1 = std::make_unique<mfem::ND_FECollection>(1, 3);
  P.fes1 = std::make_unique<mfem::ParFiniteElementSpace>(&pm, P.fec1.get());
  P.fes1s = std::make_unique<mfem::FiniteElementSpace>(P.serial.get(), P.fec1.get());
  mfem::ParFiniteElementSpace& fes = *P.fes;
  log(1, strcat("3D space: ND order ", P.order, ", ", fes.GlobalTrueVSize(), " DOFs"));

  // ------------------------------------------------------------ element data
  static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
  P.elem_region.resize(ne);
  P.elem_mat.resize(ne);
  P.elem_bh.assign(ne, nullptr);
  std::map<std::string, const BHLaw*> bh_of;
  for (auto& [name, m] : d.materials) {
    if (m.kind == spec::MaterialKind::Iron) {
      P.bh.push_back(make_bh(m.bh));
      bh_of[name] = P.bh.back().get();
    }
  }
  for (int e = 0; e < ne; ++e) {
    int a = pm.GetAttribute(e);
    std::string reg = P.region_of_attr.count(a) ? P.region_of_attr[a] : "air";
    P.elem_region[e] = reg;
    const spec::Region* g = d.find_region(reg);
    P.elem_mat[e] = g ? &d.materials.at(g->material) : &vacuum;
    if (g && P.elem_mat[e]->kind == spec::MaterialKind::Iron) P.elem_bh[e] = bh_of.at(g->material);
  }
  auto is_sc = [&](int e) { return P.elem_mat[e]->kind == spec::MaterialKind::Superconductor; };
  auto is_cond = [&](int e) { return P.elem_mat[e]->kind == spec::MaterialKind::Conductor; };
  // serial element materials (topology on the serial mesh)
  const int nser = P.serial->GetNE();
  std::vector<const spec::Material*> ser_mat(nser);
  std::vector<std::string> ser_region(nser);
  for (int s2 = 0; s2 < nser; ++s2) {
    const int a = P.ser_attr[s2];
    ser_region[s2] = P.region_of_attr.count(a) ? P.region_of_attr[a] : "air";
    const spec::Region* g = d.find_region(ser_region[s2]);
    ser_mat[s2] = g ? &d.materials.at(g->material) : &vacuum;
  }
  auto ser_sc = [&](int s2) { return ser_mat[s2]->kind == spec::MaterialKind::Superconductor; };
  auto ser_cond = [&](int s2) { return ser_mat[s2]->kind == spec::MaterialKind::Conductor; };
  std::vector<int> ser2loc(nser, -1);
  for (int e = 0; e < ne; ++e) ser2loc[P.loc2ser[e]] = e;
  const bool nonlinear = std::any_of(ser_mat.begin(), ser_mat.end(), [](auto* m) { return m->kind == spec::MaterialKind::Iron; });
  if (!d.background.type.empty()) {
    AXF_REQUIRE(!nonlinear, "background fields cannot be combined with nonlinear iron yet", "/background");
    P.bg = std::make_unique<BackgroundField>(d.background);
    if (d.pickup || !d.windings.empty())  // a design with neither (an eigenmode design) has no W_mag or cost
      warnings.push_back("a background field is present: W_mag and cost exclude the background magnet");
  }

  if(d.fullwave.rf_only) {
    AXF_REQUIRE(d.mesh.adapt.cycles==0,"rf_only does not run magnetostatic adaptation","/mesh/adapt");
    prepare_rf_ports(P,d,o,warnings);
    // Zero placeholders are never promoted to an MQS anchor or B0 solution.
    P.A_lab=std::make_unique<mfem::ParGridFunction>(&fes);*P.A_lab=0.0;
    P.A_pick=std::make_unique<mfem::ParGridFunction>(&fes);*P.A_pick=0.0;
    P.have_lab=true;P.have_pick=false;
    if(P.nranks>1) P.view_A_lab=P.gather(*P.A_lab);
    lap("rf_port_setup_s");timing["total_s"]=timer.seconds();
    results_={{"version",kVersion},{"design",{{"name",d.name},{"parameters",d.parameters},{"symmetry","3d"}}},
      {"warnings",warnings},{"static",{{"status","not computed; RF-only prescribed source"}}},
      {"coupling",{{"status","no MQS anchor computed"},{"device_radius_m",P.mesh_info.value("device_radius",0.0)}}},
      {"mesh",{{"elements",nser},{"order",P.order},{"mpi_ranks",P.nranks},{"dofs",fes.GlobalTrueVSize()},{"info",P.mesh_info}}},
      {"timing",timing},{"solver",{{"path","rf_only"},{"hypre_device",hypre_device_label()}}},
      {"benchmarks",json::array()}};
    return;
  }

  // reluctivity per attribute (linear)
  const int max_attr = pm.attributes.Max();
  mfem::Vector nu_attr(max_attr);
  nu_attr = 1.0 / mu0;
  for (int s2 = 0; s2 < nser; ++s2) {  // all attributes, also those absent on this rank
    const auto* m = ser_mat[s2];
    if (m->kind == spec::MaterialKind::Linear || m->kind == spec::MaterialKind::Iron)
      nu_attr(P.ser_attr[s2] - 1) = 1.0 / (mu0 * m->mu_r);
  }
  mfem::PWConstCoefficient nu(nu_attr);
  ShellMap shell;
  if (P.mesh_info.value("shell_attr", -1) > 0 && P.mesh_info.value("shell_outer_radius", 0.0) >
                                                     P.mesh_info.value("shell_inner_radius", 0.0)) {
    shell.attr = P.mesh_info["shell_attr"];
    shell.R1 = P.mesh_info["shell_inner_radius"];
    shell.R2 = P.mesh_info["shell_outer_radius"];
    const json c = P.mesh_info.value("domain_center", json::array({0, 0, 0}));
    for (int i = 0; i < 3; ++i) shell.c[i] = c[i].get<double>();
    P.sh_attr = shell.attr;
    P.sh_R1 = shell.R1;
    P.sh_R2 = shell.R2;
    for (int i = 0; i < 3; ++i) P.sh_c[i] = shell.c[i];
  }
  NuTensorCoefficient base_nu(nullptr, P.elem_bh, nu_attr, false, shell);  // linear, with the shell map
  auto in_shell = [&](int e) { return shell.attr > 0 && pm.GetAttribute(e) == shell.attr; };
  // background field added to B at the current integration point of T (physical elements only)
  auto add_bg = [&](mfem::ElementTransformation& T, mfem::Vector& Bv) {
    if (!P.bg || (shell.attr > 0 && T.Attribute == shell.attr)) return;
    mfem::Vector x(3);
    T.Transform(T.GetIntPoint(), x);
    double b[3];
    P.bg_at(x, b);
    for (int k = 0; k < 3; ++k) Bv(k) += b[k];
  };
  // ---- gauge regularization.  curl(nu curl .) is singular on gradients; a mass term eps |u|^2 makes the
  // system definite.  eps must be small against the *local* curl-curl scale nu_e/h_e^2, not against the
  // global one: with a single eps = 1e-8 nu0/L^2 (L the size of the whole mesh including the shell) the
  // gauge term is 1e-8 (h_e/L)^2 of the local stiffness, which next to a slit or a thin feature is at or
  // below round-off, so the gradient block is numerically singular there and AMS-PCG stalls.  Per element eps_e = kappa nu_e / h_e^2 makes the relative size of the gauge term exactly
  // kappa in every element.  Too small a kappa is lost in the round-off of the curl-curl entries, AMS's
  // gradient-space (beta) matrix becomes numerically singular and the solve breaks down loudly; too large
  // a kappa biases the solution (the bias is linear in kappa: 6e-8 of Psi at kappa = 1e-12).
  const SolverOpts so;
  // kappa of the gauge mass. Its lower bound is where the term disappears into the round-off of the
  // curl-curl entries, and that floor grows with the conditioning of the element matrices, hence with the
  // order: the toroid-with-slit case stalls below 1e-13 at ND2 and below 1e-11 at ND3, so kappa is
  // calibrated per order (docs/validation/ams_robustness.md).
  const double kappa = so.gauge >= 0 ? so.gauge : (P.order >= 3 ? 1e-11 : 1e-12);
  double L = 0;
  {
    mfem::Vector mn, mx;
    P.serial->GetBoundingBox(mn, mx);
    for (int i = 0; i < 3; ++i) L = std::max(L, mx(i) - mn(i));
  }
  // nu_e / h_e^2 per element (h_e: smallest element width; in the shell the mean of the mapped tensor)
  std::vector<double> gauge_unit(ne), elem_h(ne);
  {
    mfem::DenseMatrix K;
    for (int e = 0; e < ne; ++e) {
      elem_h[e] = pm.GetElementSize(e, so.h_vol ? 0 : 1);
      double nue = nu_attr(pm.GetAttribute(e) - 1);
      if (P.elem_bh[e]) nue = 1.0 / mu0;  // iron: nu(|B|) rises towards nu0 as it saturates; use the largest
      if (in_shell(e)) {  // the mapped tensor varies within the element (and vanishes tangentially at R2)
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2);
        double s = 0;
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          T->SetIntPoint(&ir.IntPoint(q));
          shell.tensor(*T, ir.IntPoint(q), nue, K);
          s += K.Trace() / 3.0;
        }
        nue = s / ir.GetNPoints();
      }
      gauge_unit[e] = nue / (elem_h[e] * elem_h[e]);
    }
  }
  std::vector<double> eps_e(ne);
  for (int e = 0; e < ne; ++e)
    eps_e[e] = so.global_gauge ? kappa / (mu0 * L * L) : kappa * gauge_unit[e];
  ElemCoefficient eps(eps_e);

  // ------------------------------------------------------------ outer boundary
  mfem::Array<int> bdr(pm.bdr_attributes.Max());
  bdr = 0;
  if (pm.bdr_attributes.Size()) {
    for (int i = 0; i < bdr.Size(); ++i) if (i + 1 == axi::kOuterAttr) bdr[i] = 1;
  }
  mfem::Array<int> outer_tdofs;
  fes.GetEssentialTrueDofs(bdr, outer_tdofs);

  // ------------------------------------------------------------ bodies and cohomology lifts
  struct Body {
    std::vector<std::string> names;
    std::vector<int> elements;         // local elements
    std::vector<int> ser_elements;     // serial elements
    std::vector<mfem::Vector> lifts;   // per generator, full ND_p L-vector values (body DOFs; the ND1 -> ND_p prolongation also fills neighbouring free DOFs, which are solved again)
    std::vector<double> period_az;     // azimuthal period of each generator
    std::vector<int> geo_type;         // -1: cohomology basis; 0/1: geometric azimuthal/poloidal cut
  };
  std::vector<int> body_static_ser;  // serial element -> static body
  auto build_bodies = [&](bool rf, std::vector<int>& body_of, std::vector<int>* body_ser_out = nullptr) {
    int nb = 0;
    std::vector<int> body_ser = bodies_of(*P.serial, [&](int s2) { return ser_sc(s2) || (rf && ser_cond(s2)); }, nb);
    body_of.assign(ne, -1);
    for (int e = 0; e < ne; ++e) body_of[e] = body_ser[P.loc2ser[e]];
    std::vector<Body> bodies(nb);
    for (int e = 0; e < ne; ++e) if (body_of[e] >= 0) bodies[body_of[e]].elements.push_back(e);
    for (int s2 = 0; s2 < nser; ++s2) if (body_ser[s2] >= 0) bodies[body_ser[s2]].ser_elements.push_back(s2);
    if (body_ser_out) *body_ser_out = body_ser;
    for (int b = 0; b < nb; ++b) {
      std::set<std::string> nm;
      for (int s2 : bodies[b].ser_elements) nm.insert(ser_region[s2]);
      bodies[b].names.assign(nm.begin(), nm.end());
      std::vector<char> in(nser, 0);
      for (int s2 : bodies[b].ser_elements) in[s2] = 1;
      auto coh = topo::cohomology_h1(*P.serial, in);
      AxisFrame frame;  // a superconductor `axis` sets the body's axis of revolution (default: z)
      for (auto& sc : d.superconductors)
        if (sc.has_axis && std::find(bodies[b].names.begin(), bodies[b].names.end(), sc.body) != bodies[b].names.end())
          frame = AxisFrame(sc.axis_origin, sc.axis_direction);
      auto cuts = geometric_cuts(*P.serial, in, frame);
      std::vector<int> gtype;
      if (coh.betti1 > 0 && static_cast<int>(cuts.size()) == coh.betti1) {
        coh.cochains.clear();
        for (auto& c : cuts) { coh.cochains.push_back(c.h); gtype.push_back(c.type); }  // exact unit periods, known types
        log(2, strcat("body ", bodies[b].names.front(), ": using ", cuts.size(), " geometric cut cochain(s)"));
      } else if (coh.betti1 > 0) {
        log(1, strcat("body ", bodies[b].names.front(), ": b1 = ", coh.betti1, " but ", cuts.size(),
                      " closed geometric cuts; using the cohomology basis (normalization approximate)"));
      }
      for (auto& h : coh.cochains) {
        // unit period: in the tree gauge the cotree values are periods of fundamental loops,
        // integer multiples of the generator period; take the smallest nonzero magnitude.
        const double gmin = unit_period(h);
        if (gmin > 0) for (double& v : h) v /= gmin;
        // serial ND1 field from the cochain (edge DOF = cochain value in the global edge orientation)
        mfem::GridFunction g1(P.fes1s.get());
        g1 = 0.0;
        mfem::Array<int> dofs;
        for (int ed = 0; ed < P.serial->GetNEdges(); ++ed) {
          if (h[ed] == 0.0) continue;
          P.fes1s->GetEdgeDofs(ed, dofs);
          int dd = dofs[0];
          if (dd >= 0) g1(dd) = h[ed];
          else g1(-1 - dd) = -h[ed];
        }
        // azimuthal period: int h.phi dV / int dV/(2 pi rho)  (z axis through the origin), on the serial mesh
        double num = 0, den = 0;
        mfem::Vector hv(3), x(3);
        for (int e : bodies[b].ser_elements) {
          mfem::ElementTransformation* T = P.serial->GetElementTransformation(e);
          const auto& ir = mfem::IntRules.Get(P.serial->GetElementBaseGeometry(e), 4);
          for (int q = 0; q < ir.GetNPoints(); ++q) {
            T->SetIntPoint(&ir.IntPoint(q));
            T->Transform(ir.IntPoint(q), x);
            g1.GetVectorValue(*T, ir.IntPoint(q), hv);
            double rho = std::hypot(x(0), x(1));
            if (rho < 1e-12) continue;
            double w = ir.IntPoint(q).weight * T->Weight();
            num += w * (-hv(0) * x(1) + hv(1) * x(0)) / rho;
            den += w / (2 * pi * rho);
          }
        }
        double paz = den > 0 ? num / den : 0.0;
        const int gt = gtype.empty() ? -1 : gtype[bodies[b].lifts.size()];
        if (gt == 0) paz = 1.0;        // geometric azimuthal cut: period exactly +1 about +z
        else if (gt == 1) paz = 0.0;   // geometric poloidal cut
        bodies[b].geo_type.push_back(gt);
        // project onto the partitioned ND_p space (exact: ND1 is a subspace)
        mfem::ParGridFunction gp(P.fes.get());
        SerialFieldCoefficient g1c(*P.serial, g1, P.loc2ser);
        gp.ProjectCoefficient(g1c);
        bodies[b].lifts.push_back(gp);
        bodies[b].period_az.push_back(paz);
      }
      log(2, strcat("body ", bodies[b].names.front(), ": ", bodies[b].ser_elements.size(), " elements, b1 = ", coh.betti1,
                    " (", coh.parameters, " parameters)"));
    }
    return bodies;
  };

  // background: body DOFs of the reaction field are -Pi(A_bg) (plus the cohomology lifts), so that the total
  // field vanishes inside superconductors. True-dof vector, averaged over the ranks that hold body elements.
  auto bg_body_offset = [&](const std::vector<int>& body_of) {
    mfem::Vector tv(fes.GetTrueVSize());
    tv = 0.0;
    if (!P.bg) return tv;
    mfem::ParGridFunction val(&fes), cnt(&fes);
    val = 0.0;
    cnt = 0.0;
    class BgA : public mfem::VectorCoefficient {
     public:
      explicit BgA(const Model3D::Impl& p) : mfem::VectorCoefficient(3), p_(p) {}
      void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
        mfem::Vector x(3);
        T.Transform(ip, x);
        double b[3], a[3];
        p_.bg_at(x, b, a);
        V.SetSize(3);
        for (int k = 0; k < 3; ++k) V(k) = a[k];
      }
     private:
      const Model3D::Impl& p_;
    } bga(P);
    mfem::Array<int> vd;
    mfem::Vector vals;
    mfem::DofTransformation dt;
    for (int e = 0; e < ne; ++e) {
      if (body_of[e] < 0) continue;
      fes.GetElementVDofs(e, vd, dt);
      vals.SetSize(vd.Size());
      fes.GetFE(e)->Project(bga, *pm.GetElementTransformation(e), vals);
      if (!dt.IsIdentity()) dt.TransformPrimal(vals);  // face DOFs of order >= 2 Nedelec tetrahedra
      for (int k = 0; k < vd.Size(); ++k) {
        const int i = vd[k] >= 0 ? vd[k] : -1 - vd[k];
        val(i) = vd[k] >= 0 ? vals(k) : -vals(k);
        cnt(i) = 1.0;
      }
    }
    mfem::Vector tc(fes.GetTrueVSize());
    fes.GetProlongationMatrix()->MultTranspose(val, tv);  // sums the copies of shared DOFs
    fes.GetProlongationMatrix()->MultTranspose(cnt, tc);
    for (int i = 0; i < tv.Size(); ++i) tv(i) = tc(i) > 0 ? -tv(i) / tc(i) : 0.0;
    return tv;
  };

  // essential true DOFs: outer boundary + every DOF of body elements
  auto essential = [&](const std::vector<int>& body_of, mfem::Array<int>& ess) {
    mfem::Array<int> mark(fes.GetVSize());
    mark = 0;
    mfem::Array<int> dofs;
    for (int e = 0; e < ne; ++e) {
      if (body_of[e] < 0) continue;
      fes.GetElementDofs(e, dofs);
      for (int dd : dofs) mark[dd >= 0 ? dd : -1 - dd] = 1;
    }
    fes.Synchronize(mark);  // DOFs shared with body elements of other ranks
    std::set<int> s(outer_tdofs.begin(), outer_tdofs.end());
    for (int i = 0; i < fes.GetVSize(); ++i) {
      if (!mark[i]) continue;
      int t = fes.GetLocalTDofNumber(i);
      if (t >= 0) s.insert(t);
    }
    ess.SetSize(0);
    for (int t : s) ess.Append(t);
  };

  // linear solve: given essential values (in x) and RHS form b, return the solution
  // quadrature for field-dependent reluctivities (all elements are tetrahedra)
  const mfem::IntegrationRule& irq = mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * P.order + 1);
  mfem::ParBilinearForm a(&fes);
  if (shell.attr > 0) {
    auto* cc = new mfem::CurlCurlIntegrator(base_nu);
    cc->SetIntRule(&irq);
    a.AddDomainIntegrator(cc);
  } else {
    a.AddDomainIntegrator(new mfem::CurlCurlIntegrator(nu));
  }
  a.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(eps));
  a.Assemble();
  a.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> Kfull(a.ParallelAssemble());
  mfem::HypreParMatrix* Kcur = Kfull.get();   // operator for reactions (secant operator in the pickup mode)
  mfem::MatrixCoefficient* cur_nu_mat = nullptr;  // tangent/secant reluctivity (null: linear nu)
  int op_epoch = 0;                               // bumped by set_nu_mat: invalidates the solver cache
  auto set_nu_mat = [&](mfem::MatrixCoefficient* m) {
    cur_nu_mat = m;
    ++op_epoch;
  };
  int total_its = 0;
  double max_rel_res = 0;
  // ---- divergence cleaning of right-hand sides: remove components along discrete gradients
  // of H1 functions that vanish on the essential set (numerical incompatibility of J).
  mfem::H1_FECollection h1c(P.order, 3);
  mfem::ParFiniteElementSpace h1(&pm, &h1c);
  mfem::ParDiscreteLinearOperator gradop(&h1, &fes);
  gradop.AddDomainInterpolator(new mfem::GradientInterpolator);
  gradop.Assemble();
  gradop.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> G(gradop.ParallelAssemble());
  mfem::ParBilinearForm mform(&fes);
  mfem::ConstantCoefficient one(1.0);
  mform.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(one));
  mform.Assemble();
  mform.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> Mnd(mform.ParallelAssemble());
  mfem::ParBilinearForm meform(&fes);  // the gauge mass itself (eps varies per element)
  meform.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(eps));
  meform.Assemble();
  meform.Finalize();
  std::unique_ptr<mfem::HypreParMatrix> Meps(meform.ParallelAssemble());
  // Everything in the cleaning that does not depend on the right-hand side: the H1 operator with its
  // essential rows eliminated, its AMG preconditioner, the body-gradient modes and their Gram matrix. All of
  // it is shared by the repeated solves of one phase (the lift solves, each Newton step), where it would
  // otherwise cost an AMG setup and one H1 solve per body per linear solve.
  struct CleanCache {
    std::vector<int> body;
    mfem::Array<int> ess, ess_h1;
    bool valid = false;
    std::unique_ptr<mfem::HypreParMatrix> L, Le;
    std::unique_ptr<mfem::HypreBoomerAMG> amg;
    std::unique_ptr<mfem::HyprePCG> pcg;
    std::vector<mfem::Vector> gk, Mg;
    Eigen::FullPivLU<Eigen::MatrixXd> lu;
    int nbody = 0;
    void clear() {
      pcg.reset();
      amg.reset();
      Le.reset();
      L.reset();
      gk.clear();
      Mg.clear();
      ess_h1.SetSize(0);
      nbody = 0;
      valid = false;
    }
  } cc;
  auto clean = [&](const std::vector<int>& body_of, const mfem::Array<int>& ess_nd, mfem::Vector& B) {
    bool cc_same = so.reuse && cc.valid && cc.body == body_of && cc.ess.Size() == ess_nd.Size() &&
                   std::equal(ess_nd.begin(), ess_nd.end(), cc.ess.begin());
    // every rank must agree: the setup below runs MPI collectives, and a rank whose own essential DOFs did
    // not change may still have neighbours whose did
    cc_same = allmax(cc_same ? 0.0 : 1.0) == 0.0;
    if (!cc_same) {
      cc.clear();
      cc.body = body_of;
      cc.ess = ess_nd;
      // H1 essential: outer boundary + DOFs of body elements
      {
        mfem::Array<int> ob;
        h1.GetEssentialTrueDofs(bdr, ob);
        std::set<int> sset(ob.begin(), ob.end());
        mfem::Array<int> mk(h1.GetVSize()), dofs;
        mk = 0;
        for (int e = 0; e < ne; ++e) {
          if (body_of.empty() || body_of[e] < 0) continue;
          h1.GetElementDofs(e, dofs);
          for (int dd : dofs) mk[dd] = 1;
        }
        h1.Synchronize(mk);
        for (int dd = 0; dd < h1.GetVSize(); ++dd) {
          if (!mk[dd]) continue;
          int t = h1.GetLocalTDofNumber(dd);
          if (t >= 0) sset.insert(t);
        }
        for (int t : sset) cc.ess_h1.Append(t);
      }
      cc.L.reset(mfem::RAP(Mnd.get(), G.get()));
      ensure_diagonal(cc.L, cc.ess_h1);
      cc.Le.reset(cc.L->EliminateRowsCols(cc.ess_h1));
      cc.amg = std::make_unique<mfem::HypreBoomerAMG>(*cc.L);
      cc.amg->SetPrintLevel(0);
      cc.pcg = std::make_unique<mfem::HyprePCG>(*cc.L);
      cc.pcg->SetTol(1e-12);
      cc.pcg->SetMaxIter(500);
      cc.pcg->SetPrintLevel(0);
      cc.pcg->SetPreconditioner(*cc.amg);
      // Each eliminated body k also leaves a discrete harmonic gradient G u_k (u_k = 1 on body k, 0 on the
      // outer boundary and the other bodies) in the null space of the constrained curl-curl operator; only
      // eps regularizes it. Sources carry no net current into a body, so remove the round-off component.
      for (int bo : body_of) cc.nbody = std::max(cc.nbody, bo + 1);
      cc.nbody = static_cast<int>(allmax(cc.nbody));  // bodies absent on this rank still enter the Gram system
      if (cc.nbody > 0) {
        const int nbody = cc.nbody;
        cc.gk.resize(nbody);
        cc.Mg.resize(nbody);
        std::vector<std::vector<int>> nodes(nbody);
        {
          mfem::Array<int> dofs;
          for (int k = 0; k < nbody; ++k) {
            mfem::Array<int> mk(h1.GetVSize());
            mk = 0;
            for (int e = 0; e < ne; ++e) {
              if (body_of[e] != k) continue;
              h1.GetElementDofs(e, dofs);
              for (int dd : dofs) mk[dd] = 1;
            }
            h1.Synchronize(mk);
            for (int dd = 0; dd < h1.GetVSize(); ++dd) {
              if (!mk[dd]) continue;
              int t = h1.GetLocalTDofNumber(dd);
              if (t >= 0) nodes[k].push_back(t);
            }
          }
        }
        for (int k = 0; k < nbody; ++k) {
          mfem::Vector u(h1.GetTrueVSize()), rhs(h1.GetTrueVSize());
          u = 0.0;
          rhs = 0.0;
          for (int t : nodes[k]) u(t) = 1.0;
          cc.L->EliminateBC(*cc.Le, cc.ess_h1, u, rhs);  // rhs = -Le u on free rows, u on essential rows
          cc.pcg->Mult(rhs, u);
          cc.gk[k].SetSize(fes.GetTrueVSize());
          G->Mult(u, cc.gk[k]);
          cc.Mg[k].SetSize(fes.GetTrueVSize());
          Mnd->Mult(cc.gk[k], cc.Mg[k]);
          for (int i : ess_nd) cc.Mg[k](i) = 0.0;
        }
        Eigen::MatrixXd Gm(nbody, nbody);
        for (int j = 0; j < nbody; ++j)
          for (int k = 0; k < nbody; ++k) Gm(j, k) = gdot(cc.gk[j], cc.Mg[k]);
        cc.lu.compute(Gm);
      }
      cc.valid = true;
    }
    mfem::Vector Bt(B);
    for (int i : ess_nd) Bt(i) = 0.0;
    const double nb0 = gnorm(Bt);
    mfem::Vector c(h1.GetTrueVSize());
    G->MultTranspose(Bt, c);
    for (int i : cc.ess_h1) c(i) = 0.0;
    mfem::Vector phi(h1.GetTrueVSize());
    phi = 0.0;
    cc.pcg->Mult(c, phi);
    {
      int it;
      cc.pcg->GetNumIterations(it);
      if (it >= 500)
        warnings.push_back("divergence cleaning: the H1 solve reached its iteration cap; results may be inaccurate." +
                           worst_elements_note(P));
    }
    mfem::Vector g(fes.GetTrueVSize()), y(fes.GetTrueVSize());
    G->Mult(phi, g);
    Mnd->Mult(g, y);
    std::vector<char> ess(B.Size(), 0);
    for (int i : ess_nd) ess[i] = 1;
    for (int i = 0; i < B.Size(); ++i) if (!ess[i]) B(i) -= y(i);
    if (cc.nbody > 0) {
      Eigen::VectorXd r(cc.nbody);
      mfem::Vector Bf(B);
      for (int i : ess_nd) Bf(i) = 0.0;
      for (int j = 0; j < cc.nbody; ++j) r(j) = gdot(cc.gk[j], Bf);
      Eigen::VectorXd cf = cc.lu.solve(r);
      for (int k = 0; k < cc.nbody; ++k)
        for (int i = 0; i < B.Size(); ++i)
          if (!ess[i]) B(i) -= cf(k) * cc.Mg[k](i);
    }
    mfem::Vector Bc(B);
    for (int i : ess_nd) Bc(i) = 0.0;
    log(2, strcat("  divergence cleaning: |b| ", nb0, " -> ", gnorm(Bc)));
  };
  const std::vector<int>* cur_body_of = nullptr;
  // Operator and preconditioner of the last solve. Successive solves often share both (the lift solves of
  // solve_with_bodies, the extra lift solves of a Newton step, the pickup solves): assembly and the AMS
  // setup cost more than the iterations, so they are reused whenever the operator and the essential set
  // are unchanged. `op_epoch` is bumped whenever the reluctivity of the operator changes.
  struct SolverCache {
    std::unique_ptr<mfem::HypreParMatrix> Ap, Ae;
    std::unique_ptr<mfem::HypreAMS> ams;
    std::unique_ptr<FreeDofSolver> pcfree;
    mfem::Array<int> ess;
    int epoch = -1;
    void clear() {  // destruction order: each object references the next one
      pcfree.reset();
      ams.reset();
      Ae.reset();
      Ap.reset();
      epoch = -1;
    }
  } cache;
  json solve_log = json::array();
  json body_constraint_solves = json::array();
  // linear solve: x carries the essential values (and the initial guess); Btrue is the assembled right-hand
  // side as a true-dof vector (ParLinearForm::ParallelAssemble)
  auto solve = [&](const mfem::Array<int>& ess, mfem::ParGridFunction& x, const mfem::Vector& Btrue) {
    const double t0 = timer.seconds();
    bool same = so.reuse && cache.epoch == op_epoch && cache.ess.Size() == ess.Size() &&
                std::equal(ess.begin(), ess.end(), cache.ess.begin());
    same = allmax(same ? 0.0 : 1.0) == 0.0;  // all ranks or none (collectives; global essential set)
    double t_setup = 0;
    if (!same) {
      cache.clear();
      cache.ess = ess;
      cache.epoch = op_epoch;
      mfem::ParBilinearForm a2(&fes);
      if (cur_nu_mat || shell.attr > 0) {
        auto* cc = new mfem::CurlCurlIntegrator(cur_nu_mat ? *cur_nu_mat : static_cast<mfem::MatrixCoefficient&>(base_nu));
        cc->SetIntRule(&irq);
        a2.AddDomainIntegrator(cc);
      } else {
        a2.AddDomainIntegrator(new mfem::CurlCurlIntegrator(nu));
      }
      a2.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(eps));
      a2.Assemble();
      a2.Finalize();
      cache.Ap.reset(a2.ParallelAssemble());
      ensure_diagonal(cache.Ap, ess);
      cache.Ae.reset(cache.Ap->EliminateRowsCols(ess));
      cache.ams = std::make_unique<mfem::HypreAMS>(*cache.Ap, &fes);
      cache.ams->SetPrintLevel(so.print);
      if (so.singular) cache.ams->SetSingularProblem();
      {  // AMS options (the values below are hypre's/MFEM's defaults unless overridden)
        AXF_REQUIRE(so.cycle >= 10 && so.cycle <= 14, "AXF_AMS_CYCLE must be 10..14", "",
                    "MFEM builds AMS's interpolation matrices for that family of cycle types; the others "
                    "need a single Pi block and crash.");
        HYPRE_Solver h = *cache.ams;
        HYPRE_AMSSetCycleType(h, so.cycle);
        HYPRE_AMSSetSmoothingOptions(h, so.rlx, so.sweeps, 1.0, 1.0);
        HYPRE_AMSSetAlphaAMGOptions(h, so.amg_coarsen, so.agg, so.amg_relax, so.theta, 6, 4);
        HYPRE_AMSSetBetaAMGOptions(h, so.amg_coarsen, so.agg, so.amg_relax, so.theta, 6, 4);
        // the coarsest level too: MFEM sets it to its own default, l1-Jacobi (18) when hypre runs on a GPU and 8 on the
        // host, which is this default
        HYPRE_AMSSetAlphaAMGCoarseRelaxType(h, so.amg_relax);
        HYPRE_AMSSetBetaAMGCoarseRelaxType(h, so.amg_relax);
      }
      if (so.ess_proj) cache.pcfree = std::make_unique<FreeDofSolver>(*cache.ams, cache.ess);
      t_setup = timer.seconds() - t0;
    }
    mfem::HypreParMatrix& A = *cache.Ap;
    mfem::Vector X(fes.GetTrueVSize()), B(Btrue);
    x.GetTrueDofs(X);
    A.EliminateBC(*cache.Ae, ess, X, B);  // B -= Ae X on free rows, B = A(i,i) X on essential rows
    const double t_clean0 = timer.seconds();
    clean(cur_body_of ? *cur_body_of : std::vector<int>{}, ess, B);
    const double t_clean = timer.seconds() - t_clean0;
    mfem::Solver& pc = cache.pcfree ? static_cast<mfem::Solver&>(*cache.pcfree)
                                    : static_cast<mfem::Solver&>(*cache.ams);
    std::unique_ptr<mfem::IterativeSolver> kry;
    if (so.fgmres) {
      auto* g = new mfem::FGMRESSolver(MPI_COMM_WORLD);
      g->SetKDim(50);
      kry.reset(g);
    } else {
      kry.reset(new mfem::CGSolver(MPI_COMM_WORLD));
    }
    kry->iterative_mode = true;
    kry->SetRelTol(so.tol);
    kry->SetMaxIter(so.maxit);
    kry->SetPrintLevel(so.print);
    kry->SetOperator(A);  // before SetPreconditioner: that order does not forward SetOperator to it
    kry->SetPreconditioner(pc);
    kry->Mult(B, X);
    const int its = kry->GetNumIterations();
    total_its += its;
    // true residual: a breakdown or stall must not pass silently
    mfem::Vector Rres(B.Size());
    A.Mult(X, Rres);
    Rres -= B;
    const double bn = gnorm(B), rel = bn > 0 ? gnorm(Rres) / bn : 0.0;
    max_rel_res = std::max(max_rel_res, rel);
    log(1, strcat("  AMS-PCG iterations: ", its, ", relative residual ", rel));
    solve_log.push_back({{"iterations", its}, {"relative_residual", rel}, {"setup_s", t_setup},
                         {"clean_s", t_clean}, {"solve_s", timer.seconds() - t0 - t_setup - t_clean}});
    if (!(rel < 1e-6)) {
      warnings.push_back(strcat("3D linear solve did not converge (", its, " AMS-PCG iterations, relative residual ", rel,
                                "); results are NOT reliable. A breakdown within a few iterations, or a stall at "
                                "the iteration cap, usually means the gauge regularization is too small for this "
                                "mesh (or a badly shaped element: see the mesh check below): raise AXF_GAUGE (default 1e-12, or 1e-11 from order 3; see "
                                "docs/validation/ams_robustness.md).", worst_elements_note(P)));
    } else if (its >= so.maxit) {
      warnings.push_back(strcat("3D linear solve reached the iteration cap (", its,
                                " AMS-PCG iterations, relative residual ", rel, "); accuracy is reduced.",
                                worst_elements_note(P)));
    }
    x.SetFromTrueDofs(X);
  };
  // reaction of a solution on a lift: h^T (K u - b)   (true-dof vectors)
  auto reaction = [&](const mfem::ParGridFunction& u, const mfem::Vector* Bt, const mfem::ParGridFunction& h) {
    mfem::Vector U(fes.GetTrueVSize()), H(fes.GetTrueVSize()), KU(fes.GetTrueVSize());
    u.GetTrueDofs(U);
    h.GetTrueDofs(H);
    Kcur->Mult(U, KU);
    if (Bt) KU -= *Bt;
    return gdot(H, KU);
  };
  // Magnetic energy. mode 0: W = 1/2 int nu |curl u|^2 with the linear nu;
  // mode 1: stored energy int w(|B|) dV of a nonlinear static field (w = int_0^B H db in iron);
  // mode 2: 1/2 int nu_sec |curl u|^2 with the secant reluctivity of iron at the static field.
  auto energy = [&](const mfem::ParGridFunction& u, int mode = 0) {
    double W = 0;
    mfem::Vector c(3), cb(3);
    for (int e = 0; e < ne; ++e) {
      mfem::ElementTransformation* T = pm.GetElementTransformation(e);
      const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P.order + 1);
      const double nu_e = nu_attr(pm.GetAttribute(e) - 1);
      const BHLaw* law = mode > 0 ? P.elem_bh[e] : nullptr;
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        T->SetIntPoint(&ir.IntPoint(q));
        u.GetCurl(*T, c);
        const double w = ir.IntPoint(q).weight * T->Weight(), B2 = c * c;
        if (in_shell(e)) {  // mapped exterior: 1/2 B . nu_eff B
          mfem::DenseMatrix K;
          shell.tensor(*T, ir.IntPoint(q), nu_e, K);
          W += 0.5 * K.InnerProduct(c, c) * w;
          continue;
        }
        if (!law) {
          W += 0.5 * nu_e * B2 * w;
        } else if (mode == 1) {
          W += law->energy(std::sqrt(B2)) * w;
        } else {
          P.A_lab->GetCurl(*T, cb);
          W += 0.5 * law->nu(cb.Norml2()) * B2 * w;
        }
      }
    }
    return allsum(W);
  };

  // generic solve with bodies and masters
  struct MasterSpec {
    bool fixed;
    double value;  // fluxoid (fixed) or prescribed reaction (net current)
  };
  auto solve_with_bodies = [&](const std::vector<Body>& bodies, const std::vector<int>& body_of,
                               const std::vector<std::vector<MasterSpec>>& spec_of, const mfem::Vector& b,
                               mfem::ParGridFunction& u, json& jbodies, bool with_bg) {
    mfem::Array<int> ess;
    essential(body_of, ess);
    cur_body_of = &body_of;
    // base: fixed masters, sources (and the background offset of the body DOFs)
    u = 0.0;
    if (with_bg && P.bg) u.SetFromTrueDofs(bg_body_offset(body_of));
    std::vector<std::pair<int, int>> free;  // (body, generator)
    for (size_t bi = 0; bi < bodies.size(); ++bi) {
      for (size_t j = 0; j < bodies[bi].lifts.size(); ++j) {
        const MasterSpec& ms = spec_of[bi][j];
        if (ms.fixed) u.Add(ms.value, bodies[bi].lifts[j]);
        else free.push_back({static_cast<int>(bi), static_cast<int>(j)});
      }
    }
    solve(ess, u, b);
    const int nf = static_cast<int>(free.size());
    std::vector<double> alpha(nf, 0.0);
    if (nf > 0) {
      std::vector<std::unique_ptr<mfem::ParGridFunction>> uj(nf);
      mfem::Vector zero(fes.GetTrueVSize());
      zero = 0.0;
      for (int k = 0; k < nf; ++k) {
        uj[k] = std::make_unique<mfem::ParGridFunction>(&fes);
        *uj[k] = bodies[free[k].first].lifts[free[k].second];
        solve(ess, *uj[k], zero);
      }
      Eigen::MatrixXd R(nf, nf);
      Eigen::VectorXd r0(nf);
      for (int j = 0; j < nf; ++j) {
        const auto& hj = bodies[free[j].first].lifts[free[j].second];
        mfem::ParGridFunction hg(&fes);
        hg = hj;
        r0(j) = reaction(u, &b, hg);
        for (int k = 0; k < nf; ++k) R(j, k) = reaction(*uj[k], nullptr, hg);
      }
      Eigen::VectorXd rhs(nf);
      for (int j = 0; j < nf; ++j) rhs(j) = spec_of[free[j].first][free[j].second].value - r0(j);
      Eigen::VectorXd al;
      if (nf == 1) {
        al = R.ldlt().solve(rhs);  // Preserve the existing scalar path.
      } else {
        detail::BodyConstraintSolution cs;
        try {
          cs = detail::solve_body_constraints(R, rhs);
        } catch (const std::exception& exc) {
          throw Error(exc.what(), "/superconductors",
                      "Inspect loop independence, mesh quality and the linear-solver residuals before retrying.");
        }
        al = cs.coefficients;
        json matrix = json::array(), mapping = json::array();
        std::vector<double> initial(nf), target(nf), coefficients(nf), residual(nf);
        for (int j = 0; j < nf; ++j) {
          std::vector<double> row(nf);
          for (int k = 0; k < nf; ++k) row[k] = R(j, k);
          matrix.push_back(row);
          initial[j] = r0(j);
          target[j] = spec_of[free[j].first][free[j].second].value;
          coefficients[j] = al(j);
          residual[j] = cs.residual(j);
          mapping.push_back({{"body", free[j].first}, {"generator", free[j].second}});
        }
        body_constraint_solves.push_back({{"method", "full-pivot LU"}, {"loops", mapping},
          {"reaction_matrix_A_per_Wb", matrix}, {"initial_reactions_A", initial},
          {"target_reactions_A", target}, {"coefficients_Wb", coefficients},
          {"algebraic_residual_A", residual}, {"backward_error", cs.backward_error},
          {"relative_asymmetry", cs.asymmetry}, {"reciprocal_condition", cs.reciprocal_condition}});
      }
      for (int k = 0; k < nf; ++k) {
        alpha[k] = al(k);
        u.Add(al(k), *uj[k]);
      }
    }
    jbodies = json::array();
    for (size_t bi = 0; bi < bodies.size(); ++bi) {
      json gens = json::array();
      for (size_t j = 0; j < bodies[bi].lifts.size(); ++j) {
        double al = spec_of[bi][j].fixed ? spec_of[bi][j].value : 0.0;
        for (int k = 0; k < nf; ++k) if (free[k].first == (int)bi && free[k].second == (int)j) al = alpha[k];
        mfem::ParGridFunction hg(&fes);
        hg = bodies[bi].lifts[j];
        gens.push_back({{"azimuthal_period", bodies[bi].period_az[j]},
                        {"fluxoid_coefficient", al},
                        {"net_current_A", reaction(u, &b, hg)},
                        {"constraint", spec_of[bi][j].fixed ? "fixed fluxoid" : "net current"}});
        if (nf > 1) {
          mfem::Vector U(fes.GetTrueVSize()), H(fes.GetTrueVSize()), KU(fes.GetTrueVSize());
          u.GetTrueDofs(U);
          hg.GetTrueDofs(H);
          Kcur->Mult(U, KU);
          KU -= b;
          double absolute_sum = 0.;
          for (int k = 0; k < H.Size(); ++k) absolute_sum += std::abs(H(k) * KU(k));
          gens.back()["reaction_abs_sum_A"] = allsum(absolute_sum);
          if (!spec_of[bi][j].fixed) {
            gens.back()["constraint_reaction_target_A"] = spec_of[bi][j].value;
            gens.back()["constraint_reaction_residual_A"] =
                gens.back()["net_current_A"].get<double>() - spec_of[bi][j].value;
          }
        }
      }
      jbodies.push_back({{"names", bodies[bi].names}, {"elements", bodies[bi].ser_elements.size()},
                         {"betti1", bodies[bi].lifts.size()}, {"generators", gens}});
    }
  };

  // ---- nonlinear iron (static field): Newton on the body-constrained system.
  //   residual  N(u) - b,  N(u) = int nu(|B|) B . curl v + eps u . v
  //   tangent   nu_d = nu I + 2 dnu/dB^2 B B^T
  // Increments keep fixed fluxoids; prescribed net currents (free masters) are linearized as in
  // solve_with_bodies. Step length from the energy functional
  //   Pi(u) = int w(|B|) + eps/2 |u|^2 - b.u - sum_j I_j alpha_j.
  json newton_diag;
  auto newton_static = [&](const std::vector<Body>& bodies, const std::vector<int>& body_of,
                           const std::vector<std::vector<MasterSpec>>& spec_of, const mfem::Vector& Bsrc,
                           mfem::ParGridFunction& u, json& jbodies) {
    mfem::Array<int> ess;
    essential(body_of, ess);
    cur_body_of = &body_of;
    const int nt = fes.GetTrueVSize();
    mfem::Vector Braw(Bsrc);
    std::vector<std::pair<int, int>> free;
    std::vector<double> alpha, target;
    for (size_t bi = 0; bi < bodies.size(); ++bi)
      for (size_t j = 0; j < bodies[bi].lifts.size(); ++j)
        if (!spec_of[bi][j].fixed) {
          free.push_back({static_cast<int>(bi), static_cast<int>(j)});
          alpha.push_back(jbodies[bi]["generators"][j]["fluxoid_coefficient"].get<double>());
          target.push_back(spec_of[bi][j].value);
        }
    const int nf = static_cast<int>(free.size());
    std::vector<mfem::Vector> H(nf);
    for (int k = 0; k < nf; ++k) {
      mfem::ParGridFunction hg(&fes);
      hg = bodies[free[k].first].lifts[free[k].second];
      H[k].SetSize(nt);
      hg.GetTrueDofs(H[k]);
    }
    auto residual = [&](const mfem::ParGridFunction& x, mfem::Vector& Rt) {
      NuBCoefficient F(x, P.elem_bh, nu_attr, shell);
      mfem::ParLinearForm n(&fes);
      auto* li = new mfem::VectorFEDomainLFCurlIntegrator(F);
      li->SetIntRule(&irq);
      n.AddDomainIntegrator(li);
      n.Assemble();
      Rt.SetSize(nt);
      n.ParallelAssemble(Rt);
      mfem::Vector X(nt), MX(nt);
      x.GetTrueDofs(X);
      Meps->Mult(X, MX);
      Rt += MX;
      Rt -= Braw;
    };
    auto functional = [&](const mfem::ParGridFunction& x, const std::vector<double>& al) {
      mfem::Vector X(nt), MX(nt);
      x.GetTrueDofs(X);
      Meps->Mult(X, MX);
      double Pi = energy(x, 1) + 0.5 * gdot(X, MX) - gdot(Braw, X);
      for (int k = 0; k < nf; ++k) Pi -= target[k] * al[k];
      return Pi;
    };
    auto free_norm = [&](const mfem::Vector& Rt) {
      mfem::Vector Rc(Rt);
      clean(body_of, ess, Rc);
      for (int i : ess) Rc(i) = 0.0;
      return gnorm(Rc);
    };
    const double ref = std::max(free_norm(Braw), 1e-300);
    const double tol = 1e-9, res_tol = 1e-10;
    // update size in the curl-curl energy norm: insensitive to gauge (gradient) components, which the
    // eps-regularized solves amplify from round-off without changing B
    auto energy_norm = [&](const mfem::ParGridFunction& x) {
      mfem::Vector X(nt), KX(nt);
      x.GetTrueDofs(X);
      Kfull->Mult(X, KX);
      return std::sqrt(std::max(gdot(X, KX), 0.0));
    };
    const int max_newton = 40;
    mfem::Vector Rt;
    double rel_step = 1, res = 0;
    int it = 0;
    bool converged = false;
    for (; it < max_newton; ++it) {
      residual(u, Rt);
      res = free_norm(Rt) / ref;
      if (it == 0) {  // diagnostics: field and reluctivity contrast in the iron
        double bmax = 0, rmax = 1;
        int nfe = 0;
        mfem::Vector Bq(3);
        for (int e = 0; e < ne; ++e) {
          const BHLaw* law = P.elem_bh[e];
          if (!law) continue;
          ++nfe;
          mfem::ElementTransformation* T = pm.GetElementTransformation(e);
          for (int q = 0; q < irq.GetNPoints(); ++q) {
            T->SetIntPoint(&irq.IntPoint(q));
            u.GetCurl(*T, Bq);
            const double bb = Bq.Norml2();
            bmax = std::max(bmax, bb);
            rmax = std::max(rmax, law->nu(bb) / law->nu0());
          }
        }
        bmax = allmax(bmax);
        rmax = allmax(rmax);
        log(2, strcat("  iron: ", static_cast<long>(allsum(nfe)), " elements, initial max |B| ", bmax, " T, max nu/nu0 ", rmax));
      }
      if (res < res_tol) {
        converged = true;
        break;
      }
      NuTensorCoefficient tangent(&u, P.elem_bh, nu_attr, false, shell);
      set_nu_mat(&tangent);
      mfem::ParBilinearForm ad(&fes);
      auto* cc = new mfem::CurlCurlIntegrator(tangent);
      cc->SetIntRule(&irq);
      ad.AddDomainIntegrator(cc);
      ad.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(eps));
      ad.Assemble();
      ad.Finalize();
      std::unique_ptr<mfem::HypreParMatrix> Kd(ad.ParallelAssemble());
      mfem::ParGridFunction du(&fes);
      du = 0.0;
      mfem::Vector rhs(Rt);
      rhs.Neg();
      solve(ess, du, rhs);
      std::vector<double> beta(nf, 0.0);
      if (nf > 0) {
        std::vector<std::unique_ptr<mfem::ParGridFunction>> dk(nf);
        mfem::Vector zero(nt), KD(nt), T0(nt);
        zero = 0.0;
        for (int k = 0; k < nf; ++k) {
          dk[k] = std::make_unique<mfem::ParGridFunction>(&fes);
          *dk[k] = bodies[free[k].first].lifts[free[k].second];
          solve(ess, *dk[k], zero);
        }
        mfem::Vector D(nt);
        du.GetTrueDofs(D);
        Kd->Mult(D, T0);
        T0 += Rt;
        Eigen::MatrixXd Rm(nf, nf);
        Eigen::VectorXd rv(nf);
        for (int k = 0; k < nf; ++k) {
          dk[k]->GetTrueDofs(D);
          Kd->Mult(D, KD);
          for (int j = 0; j < nf; ++j) Rm(j, k) = gdot(H[j], KD);
        }
        for (int j = 0; j < nf; ++j) rv(j) = target[j] - gdot(H[j], T0);
        Eigen::VectorXd bt = Rm.fullPivLu().solve(rv);
        for (int k = 0; k < nf; ++k) {
          beta[k] = bt(k);
          du.Add(bt(k), *dk[k]);
        }
      }
      set_nu_mat(nullptr);
      // line search on Pi
      const double Pi0 = functional(u, alpha);
      double t = 1.0;
      mfem::ParGridFunction ut(&fes);
      std::vector<double> at(nf);
      for (int ls = 0; ls < 12; ++ls, t *= 0.5) {
        ut = u;
        ut.Add(t, du);
        for (int k = 0; k < nf; ++k) at[k] = alpha[k] + t * beta[k];
        if (functional(ut, at) <= Pi0 + 1e-12 * std::fabs(Pi0)) break;
        // near convergence the energy change is below round-off (and gauge noise in du perturbs the eps
        // term): accept the step if the residual decreases
        mfem::Vector Rtt;
        residual(ut, Rtt);
        if (free_norm(Rtt) / ref < res) break;
      }
      u = ut;
      alpha = at;
      rel_step = t * energy_norm(du) / std::max(energy_norm(u), 1e-300);
      log(2, strcat("  newton ", it, ": residual ", res, ", step ", t, ", |du|_K/|u|_K ", rel_step));
      if (rel_step < tol && t >= 0.25) {
        converged = true;
        ++it;
        break;
      }
    }
    residual(u, Rt);
    res = free_norm(Rt) / ref;
    if (!converged)
      warnings.push_back(strcat("3D Newton (nonlinear iron) did not converge in ", max_newton,
                                " iterations: relative step ", rel_step, ", relative residual ", res, ".",
                                worst_elements_note(P)));
    log(1, strcat("  nonlinear iron: ", it, " Newton iterations, relative residual ", res));
    newton_diag = {{"newton_iterations", it}, {"converged", converged}, {"relative_residual", res},
                   {"relative_last_step", rel_step}};
    // reactions from the nonlinear residual
    for (size_t bi = 0; bi < bodies.size(); ++bi)
      for (size_t j = 0; j < bodies[bi].lifts.size(); ++j) {
        mfem::ParGridFunction hg(&fes);
        hg = bodies[bi].lifts[j];
        mfem::Vector Hj(nt);
        hg.GetTrueDofs(Hj);
        auto& g = jbodies[bi]["generators"][j];
        g["net_current_A"] = gdot(Hj, Rt);
        for (int k = 0; k < nf; ++k)
          if (free[k].first == static_cast<int>(bi) && free[k].second == static_cast<int>(j)) g["fluxoid_coefficient"] = alpha[k];
      }
  };

  auto state_of = [&](const std::vector<std::string>& names) -> const spec::Superconductor* {
    const spec::Superconductor* found = nullptr;
    for (auto& s : d.superconductors) {
      if (std::find(names.begin(), names.end(), s.body) == names.end()) continue;
      if (found && (found->azimuthal.kind != s.azimuthal.kind || found->azimuthal.value != s.azimuthal.value ||
                    found->poloidal.kind != s.poloidal.kind || found->poloidal.value != s.poloidal.value ||
                    found->azimuthal_slit != s.azimuthal_slit))
        throw Error("superconducting parts '" + found->body + "' and '" + s.body + "' touch (one body) but have different states",
                    "/superconductors", "Give touching parts identical states or separate them.");
      found = &s;
    }
    return found;
  };

  // gathered copy of a field on the serial mesh, identical on every rank (face-based and point post-processing):
  // element-wise projection of the local field, shared DOFs averaged over their contributions
  auto gather = [&](const mfem::ParGridFunction& u) { return P.gather(u); };

  lap("setup_s");
  // persistent circuits (3D): per circuit, the assembled unit-current source b_c (so that the flux
  // linkage of a field u is b_c . u), and the circuit currents found in the static solve
  struct Circ3 {
    std::string name;
    double target = 0, current = 0;
    mfem::Vector b;  // true dofs
    std::vector<std::pair<int, std::function<void(const mfem::Vector&, mfem::Vector&)>>> unit_J;  // (attr, J at x)
  };
  std::vector<Circ3> circs;
  json circuits_json = json::array();
  // ------------------------------------------------------------ static field
  json static_json;
  {
    SourceJ J;
    std::map<std::string, double> J0;  // azimuthal current density scale per region
    std::vector<std::unique_ptr<mfem::ParGridFunction>> keep_alive;
    std::vector<std::unique_ptr<mfem::H1_FECollection>> keep_fec;
    std::vector<std::unique_ptr<mfem::ParFiniteElementSpace>> keep_fes;
    auto circuit_of = [&](const std::string& n) -> const spec::Circuit* {
      for (auto& c : d.circuits) if (c.name == n) return &c;
      return nullptr;
    };
    for (auto& w : d.windings) {
      const spec::Circuit* wc = w.circuit.empty() ? nullptr : circuit_of(w.circuit);
      AXF_REQUIRE(w.circuit.empty() || wc, "winding '" + w.name + "' refers to unknown circuit '" + w.circuit + "'",
                  "/windings");
      AXF_REQUIRE(!wc || w.type == spec::WindingType::Azimuthal,
                  "3D circuits support azimuthal windings only", "/windings");
      if (w.type == spec::WindingType::Poloidal || w.type == spec::WindingType::Loop) {
        // harmonic current in the winding volume: around its cross-section (poloidal) or around its single loop
        // (loop: racetrack, saddle or any closed coil)
        const bool loopw = w.type == spec::WindingType::Loop;
        const int wattr = P.attr_of_region.at(w.region);
        std::vector<char> in(ne, 0), in_ser(nser, 0);
        for (int e = 0; e < ne; ++e) in[e] = (pm.GetAttribute(e) == wattr);
        for (int s2 = 0; s2 < nser; ++s2) in_ser[s2] = (P.ser_attr[s2] == wattr);
        auto coh = topo::cohomology_h1(*P.serial, in_ser);
        AXF_REQUIRE(coh.betti1 >= 1, std::string(loopw ? "loop" : "poloidal") + " winding '" + w.name +
                                         "' has no loop for its current (it must enclose a hole)", "/windings");
        AXF_REQUIRE(!loopw || coh.betti1 == 1,
                    strcat("loop winding '", w.name, "' has ", coh.betti1, " independent loops; it needs exactly one"),
                    "/windings", "Model each closed coil as its own region.");
        // ND1 fields (serial mesh, global edge numbering) and azimuthal periods of the generators
        auto nd1_of = [&](const std::vector<double>& h) {
          auto g = std::make_unique<mfem::GridFunction>(P.fes1s.get());
          *g = 0.0;
          mfem::Array<int> dd;
          for (int ed = 0; ed < P.serial->GetNEdges(); ++ed) {
            if (h[ed] == 0.0) continue;
            P.fes1s->GetEdgeDofs(ed, dd);
            if (dd[0] >= 0) (*g)(dd[0]) = h[ed];
            else (*g)(-1 - dd[0]) = -h[ed];
          }
          return g;
        };
        auto paz_of = [&](mfem::GridFunction& g) {  // identical on every rank (serial mesh)
          double num = 0, den = 0;
          mfem::Vector hv(3), x(3);
          for (int e = 0; e < nser; ++e) {
            if (!in_ser[e]) continue;
            mfem::ElementTransformation* T = P.serial->GetElementTransformation(e);
            const auto& ir = mfem::IntRules.Get(P.serial->GetElementBaseGeometry(e), 4);
            for (int q = 0; q < ir.GetNPoints(); ++q) {
              T->SetIntPoint(&ir.IntPoint(q));
              T->Transform(ir.IntPoint(q), x);
              g.GetVectorValue(*T, ir.IntPoint(q), hv);
              double rho = std::hypot(x(0), x(1));
              if (rho < 1e-12) continue;
              double wq = ir.IntPoint(q).weight * T->Weight();
              num += wq * (-hv(0) * x(1) + hv(1) * x(0)) / rho;
              den += wq / (2 * pi * rho);
            }
          }
          return den > 0 ? num / den : 0.0;
        };
        std::vector<double> h(P.serial->GetNEdges(), 0.0);
        auto wcuts = geometric_cuts(*P.serial, in_ser);
        auto pc = std::find_if(wcuts.begin(), wcuts.end(), [&](const CutCochain& c) { return c.type == (loopw ? 0 : 1); });
        if (loopw) {
          h = (pc != wcuts.end() && coh.betti1 == 1) ? pc->h : coh.cochains[0];
        } else if (pc != wcuts.end()) {
          h = pc->h;  // exact poloidal cut cochain, unit period
          log(2, "  winding uses the geometric poloidal cut");
        } else if (coh.betti1 == 1) {
          h = coh.cochains[0];
        } else {
          // combination with zero azimuthal period (purely poloidal)
          std::vector<double> paz;
          for (auto& c : coh.cochains) paz.push_back(paz_of(*nd1_of(c)));
          log(2, strcat("  winding generators: P_az = ", paz[0], ", ", paz[1]));
          int k = static_cast<int>(std::max_element(paz.begin(), paz.end(), [](double x, double y) {
                    return std::fabs(x) < std::fabs(y); }) - paz.begin());
          int j = k == 0 ? 1 : 0;
          for (size_t ed = 0; ed < h.size(); ++ed) h[ed] = paz[k] * coh.cochains[j][ed] - paz[j] * coh.cochains[k][ed];
          if (coh.betti1 > 2) warnings.push_back("winding '" + w.name + "' has more than two independent loops; using a poloidal combination of the first two");
        }
        const double gmin = unit_period(h);
        AXF_REQUIRE(gmin > 0, strcat("winding '", w.name, "': empty poloidal cochain"), "/windings");
        for (double& v : h) v /= gmin;
        auto E0s = nd1_of(h);
        const double paz_w = paz_of(*E0s);
        auto E0 = std::make_unique<mfem::ParGridFunction>(P.fes1.get());
        {
          SerialFieldCoefficient E0c(*P.serial, *E0s, P.loc2ser);
          E0->ProjectCoefficient(E0c);  // exact: the same ND1 space on the partitioned mesh
        }
        if (!loopw) {
          AXF_REQUIRE(std::fabs(paz_w) < 1e-2,
                      strcat("poloidal winding '", w.name, "': its region has no loop around its cross-section (the only "
                             "current loop found circulates around the axis, period ", paz_w, ")"),
                      "/windings", "A poloidal winding needs a hollow (ring-shaped) cross-section; use type 'azimuthal' "
                      "for a current around the axis.");
        }
        // harmonic projection: solve int grad phi . grad v = int E0 . grad v on the winding (Neumann)
        keep_fec.push_back(std::make_unique<mfem::H1_FECollection>(std::max(1, P.order), 3));
        keep_fes.push_back(std::make_unique<mfem::ParFiniteElementSpace>(&pm, keep_fec.back().get()));
        mfem::ParFiniteElementSpace& h1 = *keep_fes.back();
        mfem::Vector sig(max_attr);
        sig = 0.0;
        sig(wattr - 1) = 1.0;
        mfem::PWConstCoefficient sigma(sig);
        mfem::ParBilinearForm kf(&h1);
        kf.AddDomainIntegrator(new mfem::DiffusionIntegrator(sigma));
        kf.Assemble();
        mfem::VectorGridFunctionCoefficient E0c(E0.get());
        mfem::ScalarVectorProductCoefficient sE0(sigma, E0c);
        mfem::ParLinearForm rf(&h1);
        rf.AddDomainIntegrator(new mfem::DomainLFGradIntegrator(sE0));
        rf.Assemble();
        auto phi = std::make_unique<mfem::ParGridFunction>(&h1);
        *phi = 0.0;
        mfem::Array<int> inw(h1.GetVSize());
        inw = 0;
        mfem::Array<int> dd;
        for (int e = 0; e < ne; ++e) {
          if (!in[e]) continue;
          h1.GetElementDofs(e, dd);
          for (int x : dd) inw[x] = 1;
        }
        h1.Synchronize(inw);  // winding DOFs shared with other ranks
        // anchor: the winding DOF with the smallest global number
        long long anchor_g = std::numeric_limits<long long>::max();
        for (int i = 0; i < h1.GetVSize(); ++i)
          if (inw[i]) anchor_g = std::min<long long>(anchor_g, h1.GetGlobalTDofNumber(i));
        MPI_Allreduce(MPI_IN_PLACE, &anchor_g, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        mfem::Array<int> ess;
        for (int i = 0; i < h1.GetVSize(); ++i)
          if (!inw[i] || h1.GetGlobalTDofNumber(i) == anchor_g) {
            int t = h1.GetLocalTDofNumber(i);
            if (t >= 0) ess.Append(t);
          }
        ess.Sort();
        ess.Unique();
        mfem::OperatorPtr Kop;
        mfem::Vector X, B;
        keep_eliminated_rows(kf, ess);  // the DOFs outside the winding: zero rows (hypre on a GPU would drop them)
        kf.FormLinearSystem(ess, *phi, rf, Kop, X, B);
        mfem::HypreBoomerAMG amg(*Kop.As<mfem::HypreParMatrix>());
        amg.SetPrintLevel(0);
        mfem::HyprePCG pcg(*Kop.As<mfem::HypreParMatrix>());
        pcg.SetTol(1e-12);
        pcg.SetMaxIter(800);
        pcg.SetPrintLevel(0);
        pcg.SetPreconditioner(amg);
        pcg.Mult(B, X);
        {
          int it;
          pcg.GetNumIterations(it);
          if (it >= 800)
            warnings.push_back("poloidal winding '" + w.name +
                               "': the current-distribution solve reached its iteration cap." + worst_elements_note(P));
        }
        kf.RecoverFEMSolution(X, rf, *phi);
        // current = int |E|^2 dV (sigma = 1, unit period); sign from J_z on the inner leg
        double pw = 0, jz_inner = 0, rmin = 1e300, rmax = 0, mdir = 0, mv[3] = {0, 0, 0};  // mdir: moment . axis (loop windings)
        double ax[3] = {w.axis_direction.x, w.axis_direction.y, w.axis_direction.z};
        mfem::Vector ev(3), gp(3), x(3);
        mfem::Array<int> vv;
        for (int e = 0; e < ne; ++e) {
          if (!in[e]) continue;
          pm.GetElementVertices(e, vv);
          for (int v : vv) {
            const double* xv = pm.GetVertex(v);
            double r = std::hypot(xv[0], xv[1]);
            rmin = std::min(rmin, r);
            rmax = std::max(rmax, r);
          }
        }
        rmin = allmin(rmin);
        rmax = allmax(rmax);
        const double rmid = 0.5 * (rmin + rmax);
        for (int e = 0; e < ne; ++e) {
          if (!in[e]) continue;
          mfem::ElementTransformation* T = pm.GetElementTransformation(e);
          const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P.order + 2);
          for (int q = 0; q < ir.GetNPoints(); ++q) {
            T->SetIntPoint(&ir.IntPoint(q));
            T->Transform(ir.IntPoint(q), x);
            E0->GetVectorValue(*T, ir.IntPoint(q), ev);
            phi->GetGradient(*T, gp);
            ev -= gp;
            double wq = ir.IntPoint(q).weight * T->Weight();
            pw += (ev * ev) * wq;
            if (std::hypot(x(0), x(1)) < rmid) jz_inner += ev(2) * wq;
            const double xo[3] = {x(0) - w.axis_origin.x, x(1) - w.axis_origin.y, x(2) - w.axis_origin.z};
            const double mx[3] = {xo[1] * ev(2) - xo[2] * ev(1), xo[2] * ev(0) - xo[0] * ev(2), xo[0] * ev(1) - xo[1] * ev(0)};
            mdir += 0.5 * (mx[0] * ax[0] + mx[1] * ax[1] + mx[2] * ax[2]) * wq;
            for (int k = 0; k < 3; ++k) mv[k] += 0.5 * mx[k] * wq;
          }
        }
        pw = allsum(pw);
        jz_inner = allsum(jz_inner);
        mdir = allsum(mdir);
        allsum(mv, 3);
        if (loopw) {
          const double an = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
          const double mn = std::sqrt(mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]);
          AXF_REQUIRE(std::fabs(mdir) >= 0.1 * mn * an,
                      strcat("loop winding '", w.name, "': its magnetic moment (", mv[0], ", ", mv[1], ", ", mv[2],
                             ") is nearly perpendicular to axis.direction, so the current direction is undefined"),
                      "/windings", "Set axis.direction along the coil's axis (the moment of a right-handed current).");
        }
        AXF_REQUIRE(pw > 0, "poloidal winding '" + w.name + "' carries no current", "/windings");
        log(2, strcat("  winding '", w.name, "': gmin ", gmin, ", int|E|^2 = ", pw, ", P_az(final) = ", paz_w,
                      ", inner-leg Jz sign ", jz_inner));
        const double sgn = loopw ? (mdir >= 0 ? 1.0 : -1.0) : (jz_inner >= 0 ? 1.0 : -1.0);
        const double scale = sgn * w.ampere_turns / pw;
        mfem::ParGridFunction* E0p = E0.get();
        mfem::ParGridFunction* php = phi.get();
        J.by_attr[wattr] = [E0p, php, scale](const mfem::Vector&, mfem::ElementTransformation& T, mfem::Vector& V) {
          mfem::Vector a(3), g(3);
          E0p->GetVectorValue(T, T.GetIntPoint(), a);
          php->GetGradient(T, g);
          for (int i = 0; i < 3; ++i) V(i) = scale * (a(i) - g(i));
        };
        keep_alive.push_back(std::move(E0));
        keep_alive.push_back(std::move(phi));
        log(1, strcat(loopw ? "loop" : "poloidal", " winding '", w.name, "': b1 = ", coh.betti1,
                      ", harmonic current normalized"));
        continue;
      }
      // J0 = NI / int dV / (2 pi rho)
      double denom = 0;
      mfem::Vector x(3);
      for (int e = 0; e < ne; ++e) {
        if (P.elem_region[e] != w.region) continue;
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 4);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          T->SetIntPoint(&ir.IntPoint(q));
          T->Transform(ir.IntPoint(q), x);
          double phi[3], rho;
          azimuthal(x, w.axis_origin, w.axis_direction, phi, rho);
          denom += ir.IntPoint(q).weight * T->Weight() / (2 * pi * std::max(rho, 1e-300));
        }
      }
      denom = allsum(denom);
      AXF_REQUIRE(denom > 0, "winding region '" + w.region + "' has no volume", "/windings");
      auto o2 = w.axis_origin;
      auto a2 = w.axis_direction;
      if (wc && wc->kind == spec::Circuit::Kind::Persistent) {
        // unit circuit current: ampere-turns = turns
        const double ju = w.turns / denom;
        auto it = std::find_if(circs.begin(), circs.end(), [&](const Circ3& c) { return c.name == wc->name; });
        if (it == circs.end()) {
          circs.push_back({wc->name, wc->flux_linkage, 0.0, mfem::Vector(), {}});
          it = circs.end() - 1;
        }
        it->unit_J.push_back({P.attr_of_region.at(w.region), [ju, o2, a2](const mfem::Vector& xx, mfem::Vector& V) {
                                double phi[3], rho;
                                azimuthal(xx, o2, a2, phi, rho);
                                for (int k = 0; k < 3; ++k) V(k) = ju * phi[k];
                              }});
        continue;
      }
      const double NI = wc ? w.turns * wc->current : w.ampere_turns;  // powered circuit: a fixed source
      double j0 = NI / denom;
      J.by_attr[P.attr_of_region.at(w.region)] = [j0, o2, a2](const mfem::Vector& xx, mfem::ElementTransformation&,
                                                               mfem::Vector& V) {
        double phi[3], rho;
        azimuthal(xx, o2, a2, phi, rho);
        V(0) = j0 * phi[0];
        V(1) = j0 * phi[1];
        V(2) = j0 * phi[2];
      };
    }
    AXF_REQUIRE(circs.empty() || !nonlinear, "persistent circuits with nonlinear iron are not available in 3D yet",
                "/circuits", "Use the axisymmetric engine, or linear iron (material kind 'linear').");
    // assembled unit sources of the persistent circuits
    for (auto& c : circs) {
      SourceJ Jc;
      for (auto& [attr, f] : c.unit_J) {
        auto ff = f;
        Jc.by_attr[attr] = [ff](const mfem::Vector& xx, mfem::ElementTransformation&, mfem::Vector& V) { ff(xx, V); };
      }
      mfem::ParLinearForm bc(&fes);
      bc.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(Jc));
      bc.Assemble();
      c.b.SetSize(fes.GetTrueVSize());
      bc.ParallelAssemble(c.b);
    }
    mfem::Vector b(fes.GetTrueVSize());  // assembled source (true dofs)
    {
      mfem::ParLinearForm bl(&fes);
      bl.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(J));
      bl.Assemble();
      bl.ParallelAssemble(b);
    }
    if (P.bg) {  // linear magnetic materials: curl((nu - nu0) B_bg) drives the reaction field
      class BgMag : public mfem::VectorCoefficient {
       public:
        BgMag(const Model3D::Impl& p, const mfem::Vector& nu, int shell)
            : mfem::VectorCoefficient(3), p_(p), nu_(nu), shell_(shell) {}
        void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
          V.SetSize(3);
          V = 0.0;
          if (shell_ > 0 && T.Attribute == shell_) return;
          const double dn = nu_(T.Attribute - 1) - 1.0 / mu0;
          if (dn == 0.0) return;
          mfem::Vector x(3);
          T.Transform(ip, x);
          double bb[3];
          p_.bg_at(x, bb);
          for (int k = 0; k < 3; ++k) V(k) = -dn * bb[k];
        }
       private:
        const Model3D::Impl& p_;
        const mfem::Vector& nu_;
        int shell_;
      } bgm(P, nu_attr, shell.attr);
      mfem::ParLinearForm bl(&fes);
      bl.AddDomainIntegrator(new mfem::VectorFEDomainLFCurlIntegrator(bgm));
      bl.Assemble();
      mfem::Vector tb(fes.GetTrueVSize());
      bl.ParallelAssemble(tb);
      b += tb;
    }
    std::vector<Body> bodies = build_bodies(false, P.body_static, &body_static_ser);
    {
      mfem::Array<int> ess0;
      essential(P.body_static, ess0);
      mfem::Vector B0(b), B1;
      for (int i : ess0) B0(i) = 0.0;
      B1 = B0;
      clean(P.body_static, ess0, B1);
      for (int i : ess0) B1(i) = 0.0;
      mfem::Vector D(B0);
      D -= B1;
      const double nB0 = gnorm(B0), incompat = nB0 > 0 ? gnorm(D) / nB0 : 0.0;
      log(1, strcat("  static source: non-solenoidal fraction ", incompat));
      if (incompat > 1e-2)
        warnings.push_back(strcat("winding currents are not divergence-free (fraction ", incompat,
                                  " removed); azimuthal windings must be closed rings about their axis. Results are NOT reliable"));
    }
    std::vector<std::vector<MasterSpec>> ms(bodies.size());
    for (size_t bi = 0; bi < bodies.size(); ++bi) {
      const spec::Superconductor* st = state_of(bodies[bi].names);
      spec::Constraint az = st ? st->azimuthal : spec::Constraint{}, pol = st ? st->poloidal : spec::Constraint{};
      auto nontrivial = [](const spec::Constraint& c) { return c.kind == spec::Constraint::Kind::Flux || c.value != 0; };
      for (size_t j = 0; j < bodies[bi].lifts.size(); ++j) {
        double paz = bodies[bi].period_az[j];
        bool is_az = std::fabs(paz) > 0.5;
        if (bodies[bi].geo_type[j] < 0 && (nontrivial(az) || nontrivial(pol))) {
          throw Error("superconducting body '" + bodies[bi].names.front() + "': its loops are not loops about the z axis "
                      "(no exact geometric cut), so azimuthal/poloidal constraints cannot be assigned", "/superconductors",
                      "Only the default state (zero net currents) is supported for such bodies in 3D.");
        }
        const spec::Constraint& c = is_az ? az : pol;
        if (bodies[bi].lifts.size() > 1 && c.value != 0 && bodies[bi].geo_type[j] < 0) {
          throw Error("superconducting body '" + bodies[bi].names.front() + "' has several independent loops; "
                      "only zero net currents or zero fluxes are supported for it in 3D", "/superconductors");
        }
        if (c.kind == spec::Constraint::Kind::Flux) {
          AXF_REQUIRE(is_az || c.value == 0, "nonzero poloidal flux constraints are not supported in 3D yet", "/superconductors");
          ms[bi].push_back({true, is_az ? c.value / paz : 0.0});
        } else {
          AXF_REQUIRE(is_az || c.value == 0, "nonzero poloidal net currents are not supported in 3D yet", "/superconductors");
          // reaction conjugate to the lift coefficient = current x period
          ms[bi].push_back({false, c.value * (is_az ? paz : 1.0)});
        }
      }
    }
    P.A_lab = std::make_unique<mfem::ParGridFunction>(&fes);
    json jb;
    log(1, "solving static field (3D)");
    solve_with_bodies(bodies, P.body_static, ms, b, *P.A_lab, jb, true);
    if (nonlinear) newton_static(bodies, P.body_static, ms, b, *P.A_lab, jb);
    if (!circs.empty()) {
      // A = A0 + sum_c I_c u_c, with u_c the unit-current field of circuit c (homogeneous body states);
      // the linkages b_c . A must equal the targets
      const int nc = static_cast<int>(circs.size());
      auto ms0 = ms;
      for (auto& v : ms0)
        for (auto& x : v) x.value = 0.0;
      std::vector<std::unique_ptr<mfem::ParGridFunction>> uc(nc);
      for (int c = 0; c < nc; ++c) {
        uc[c] = std::make_unique<mfem::ParGridFunction>(&fes);
        json jtmp;
        log(1, strcat("  persistent circuit '", circs[c].name, "': unit-current solve"));
        solve_with_bodies(bodies, P.body_static, ms0, circs[c].b, *uc[c], jtmp, false);
      }
      Eigen::MatrixXd M(nc, nc);
      Eigen::VectorXd r(nc);
      std::vector<double> lscale(nc);  // linkage scale for the residual check
      mfem::Vector U(fes.GetTrueVSize());
      P.A_lab->GetTrueDofs(U);
      std::vector<double> lam_bg(nc, 0.0);  // background linkage: int J_unit . A_bg dV
      if (P.bg) {
        for (int c = 0; c < nc; ++c) {
          mfem::Vector x(3), Ju(3);
          for (auto& [attr, f] : circs[c].unit_J) {
            for (int e = 0; e < ne; ++e) {
              if (pm.GetAttribute(e) != attr) continue;
              mfem::ElementTransformation* T = pm.GetElementTransformation(e);
              const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P.order + 2);
              for (int q = 0; q < ir.GetNPoints(); ++q) {
                T->SetIntPoint(&ir.IntPoint(q));
                T->Transform(ir.IntPoint(q), x);
                f(x, Ju);
                double bb[3], aa[3];
                P.bg_at(x, bb, aa);
                lam_bg[c] += (Ju(0) * aa[0] + Ju(1) * aa[1] + Ju(2) * aa[2]) * ir.IntPoint(q).weight * T->Weight();
              }
            }
          }
          lam_bg[c] = allsum(lam_bg[c]);
        }
      }
      for (int c = 0; c < nc; ++c) {
        const double bu = gdot(circs[c].b, U) + lam_bg[c];
        r(c) = circs[c].target - bu;
        lscale[c] = std::max(std::fabs(circs[c].target), std::fabs(bu));
        for (int k = 0; k < nc; ++k) {
          mfem::Vector Uk(fes.GetTrueVSize());
          uc[k]->GetTrueDofs(Uk);
          M(c, k) = gdot(circs[c].b, Uk);
        }
      }
      Eigen::VectorXd I = M.fullPivLu().solve(r);
      for (int c = 0; c < nc; ++c) {
        circs[c].current = I(c);
        P.A_lab->Add(I(c), *uc[c]);
      }
      P.A_lab->GetTrueDofs(U);
      for (int ci = 0; ci < nc; ++ci) {
        auto& c = circs[ci];
        const double got = gdot(c.b, U) + lam_bg[ci];
        circuits_json.push_back({{"name", c.name}, {"current_A", c.current}, {"flux_linkage_Wb", got},
                                 {"target_Wb", c.target}});
        if (std::fabs(got - c.target) > 1e-8 * std::max({lscale[ci], std::fabs(got), 1e-300}))
          warnings.push_back(strcat("circuit '", c.name, "': flux linkage ", got, " Wb differs from its target ",
                                    c.target, " Wb"));
        // the circuit current is part of the static source (forces, reports)
        for (auto& [attr, f] : c.unit_J) {
          auto ff = f;
          const double Ic = c.current;
          J.by_attr[attr] = [ff, Ic](const mfem::Vector& xx, mfem::ElementTransformation&, mfem::Vector& V) {
            ff(xx, V);
            V *= Ic;
          };
        }
      }
      jb = json();  // body reactions of the superposed solution are not recomputed
    }
    P.have_lab = true;
    P.view_body_static = body_static_ser;
    if (P.nranks > 1) P.view_A_lab = gather(*P.A_lab);
    static_json = {{"energy_J", energy(*P.A_lab, nonlinear ? 1 : 0)}, {"bodies", jb}};
    if (!circuits_json.empty()) static_json["circuits"] = circuits_json;
    if (nonlinear) static_json["nonlinear"] = newton_diag;

    // ---- forces and torques (about the origin) from the static field
    //   windings: F = int J x B dV;  superconductors: F = -oint B^2/(2 mu0) n dA (n out of the body,
    //   B on the vacuum side; B . n = 0 there). Iron forces are not computed.
    json fw = json::object(), fs = json::object();
    double Fsum[3] = {0, 0, 0};
    auto vec3 = [](const double v[3]) { return json::array({v[0], v[1], v[2]}); };
    mfem::Vector Bq(3), Jq(3), xq(3);
    for (auto& w : d.windings) {
      if (!P.attr_of_region.count(w.region)) continue;
      const int wa = P.attr_of_region.at(w.region);
      double F[3] = {0, 0, 0}, Tq[3] = {0, 0, 0};
      for (int e = 0; e < ne; ++e) {
        if (pm.GetAttribute(e) != wa) continue;
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P.order + 2);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(q);
          T->SetIntPoint(&ip);
          T->Transform(ip, xq);
          J.Eval(Jq, *T, ip);
          P.A_lab->GetCurl(*T, Bq);
          add_bg(*T, Bq);
          const double wq = ip.weight * T->Weight();
          const double f[3] = {Jq(1) * Bq(2) - Jq(2) * Bq(1), Jq(2) * Bq(0) - Jq(0) * Bq(2), Jq(0) * Bq(1) - Jq(1) * Bq(0)};
          for (int k = 0; k < 3; ++k) F[k] += wq * f[k];
          Tq[0] += wq * (xq(1) * f[2] - xq(2) * f[1]);
          Tq[1] += wq * (xq(2) * f[0] - xq(0) * f[2]);
          Tq[2] += wq * (xq(0) * f[1] - xq(1) * f[0]);
        }
      }
      allsum(F, 3);
      allsum(Tq, 3);
      for (int k = 0; k < 3; ++k) Fsum[k] += F[k];
      fw[w.name.empty() ? w.region : w.name] = {{"F_N", vec3(F)}, {"torque_Nm", vec3(Tq)}};
    }
    if (!bodies.empty()) {  // on the gathered serial view (every rank, no shared faces)
      mfem::ParMesh& vm = P.vmesh();
      const mfem::ParGridFunction& vA = *P.vA_lab();
      const std::vector<int>& vbs = P.vbody_static();
      std::vector<std::array<double, 6>> FT(bodies.size(), {0, 0, 0, 0, 0, 0});
      const auto& irf = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * P.order + 2);
      mfem::Vector nor(3), xf(3);
      for (int f = 0; f < vm.GetNumFaces(); ++f) {
        int e1, e2;
        vm.GetFaceElements(f, &e1, &e2);
        if (e1 < 0 || e2 < 0) continue;
        const int b1 = vbs[e1], b2 = vbs[e2];
        if ((b1 >= 0) == (b2 >= 0)) continue;
        const int bi = b1 >= 0 ? b1 : b2, ev = b1 >= 0 ? e2 : e1, side = b1 >= 0 ? 2 : 1;
        const double sgn = b1 >= 0 ? 1.0 : -1.0;  // CalcOrtho points from element 1 to element 2
        mfem::FaceElementTransformations* FTr = vm.GetFaceElementTransformations(f);
        for (int q = 0; q < irf.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = irf.IntPoint(q);
          FTr->SetAllIntPoints(&ip);
          mfem::CalcOrtho(FTr->Jacobian(), nor);
          FTr->Transform(ip, xf);
          const mfem::IntegrationPoint eip = side == 1 ? FTr->GetElement1IntPoint() : FTr->GetElement2IntPoint();
          const double n3[3] = {sgn * nor(0), sgn * nor(1), sgn * nor(2)}, x3[3] = {xf(0), xf(1), xf(2)};
          mfem::ElementTransformation* Te = vm.GetElementTransformation(ev);
          Te->SetIntPoint(&eip);
          vA.GetCurl(*Te, Bq);
          add_bg(*Te, Bq);
          const double pr = (Bq * Bq) / (2 * mu0) * ip.weight;
          const double fq[3] = {-pr * n3[0], -pr * n3[1], -pr * n3[2]};
          for (int k = 0; k < 3; ++k) FT[bi][k] += fq[k];
          FT[bi][3] += x3[1] * fq[2] - x3[2] * fq[1];
          FT[bi][4] += x3[2] * fq[0] - x3[0] * fq[2];
          FT[bi][5] += x3[0] * fq[1] - x3[1] * fq[0];
        }
      }
      for (size_t bi = 0; bi < bodies.size(); ++bi) {
        const double F[3] = {FT[bi][0], FT[bi][1], FT[bi][2]}, Tq[3] = {FT[bi][3], FT[bi][4], FT[bi][5]};
        for (int k = 0; k < 3; ++k) Fsum[k] += F[k];
        fs[bodies[bi].names.front()] = {{"F_N", vec3(F)}, {"torque_Nm", vec3(Tq)}};
      }
    }
    // magnetic materials (iron, linear): Maxwell stress of the vacuum field around the body. Default: the volume
    // ("eggshell") form F = -int_layer T . grad g dV, T = (B B^T - B^2/2 I)/mu0, over the vacuum elements touching
    // the body, g = the P1 function that is 1 on the body's vertices and 0 elsewhere; it avoids the field at the
    // body's own faces and edges (singular there), which made the surface integral settle 2.7 % high. The surface integral oint T n dA on the faces (B from the non-magnetic side) is
    // the fallback when the layer touches a winding, superconductor or another magnetic body (div T != 0 there).
    // Gathered serial view; interfaces between two magnetic regions are not counted.
    json fm = json::object();
    std::map<std::string, std::array<double, 6>> FTegg;
    std::set<std::string> egg_bad;
    {
      mfem::ParMesh& vm = P.vmesh();
      const mfem::ParGridFunction& vA = *P.vA_lab();
      auto magnetic = [&](int s2) {
        return ser_mat[s2]->kind == spec::MaterialKind::Iron || ser_mat[s2]->kind == spec::MaterialKind::Linear;
      };
      std::set<std::string> winding_regions;
      for (auto& w : d.windings) winding_regions.insert(w.region);
      // vertex -> magnetic regions it belongs to
      std::vector<std::vector<std::string>> vreg(vm.GetNV());
      mfem::Array<int> vv;
      for (int e = 0; e < vm.GetNE(); ++e) {
        if (!magnetic(e)) continue;
        vm.GetElementVertices(e, vv);
        for (int v : vv)
          if (std::find(vreg[v].begin(), vreg[v].end(), ser_region[e]) == vreg[v].end()) vreg[v].push_back(ser_region[e]);
      }
      mfem::H1_TetrahedronElement p1(1);
      mfem::DenseMatrix dsh(4, 3);
      mfem::Vector xq(3);
      for (int e = 0; e < vm.GetNE(); ++e) {
        if (magnetic(e) || vm.GetElementGeometry(e) != mfem::Geometry::TETRAHEDRON) continue;
        vm.GetElementVertices(e, vv);
        std::set<std::string> touched;
        for (int v : vv)
          for (auto& r : vreg[v]) touched.insert(r);
        if (touched.empty()) continue;
        const bool vacuum_layer = ser_mat[e]->kind == spec::MaterialKind::Vacuum && !winding_regions.count(ser_region[e]) &&
                                  !(shell.attr > 0 && P.ser_attr[e] == shell.attr);
        if (!vacuum_layer || touched.size() > 1) {
          for (auto& r : touched) egg_bad.insert(r);
          continue;
        }
        const std::string& reg = *touched.begin();
        auto& acc = FTegg[reg];
        mfem::ElementTransformation* T = vm.GetElementTransformation(e);
        const auto& ir = mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * P.order + 2);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(q);
          T->SetIntPoint(&ip);
          T->Transform(ip, xq);
          p1.CalcPhysDShape(*T, dsh);
          double gg[3] = {0, 0, 0};
          for (int k = 0; k < 4; ++k) {
            const bool in = std::find(vreg[vv[k]].begin(), vreg[vv[k]].end(), reg) != vreg[vv[k]].end();
            if (!in) continue;
            for (int c = 0; c < 3; ++c) gg[c] += dsh(k, c);
          }
          vA.GetCurl(*T, Bq);
          add_bg(*T, Bq);
          const double b2 = Bq * Bq, bg = Bq(0) * gg[0] + Bq(1) * gg[1] + Bq(2) * gg[2];
          const double wq = ip.weight * T->Weight();
          double fq[3];
          for (int c = 0; c < 3; ++c) fq[c] = -(Bq(c) * bg - 0.5 * b2 * gg[c]) / mu0 * wq;
          for (int c = 0; c < 3; ++c) acc[c] += fq[c];
          acc[3] += xq(1) * fq[2] - xq(2) * fq[1];
          acc[4] += xq(2) * fq[0] - xq(0) * fq[2];
          acc[5] += xq(0) * fq[1] - xq(1) * fq[0];
        }
      }
    }
    {
      mfem::ParMesh& vm = P.vmesh();
      const mfem::ParGridFunction& vA = *P.vA_lab();
      auto magnetic = [&](int s2) {
        return ser_mat[s2]->kind == spec::MaterialKind::Iron || ser_mat[s2]->kind == spec::MaterialKind::Linear;
      };
      std::map<std::string, std::array<double, 6>> FTm;
      const auto& irf = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * P.order + 2);
      mfem::Vector nor(3), xf(3);
      for (int f = 0; f < vm.GetNumFaces(); ++f) {
        int e1, e2;
        vm.GetFaceElements(f, &e1, &e2);
        if (e1 < 0 || e2 < 0) continue;
        const bool m1 = magnetic(e1), m2 = magnetic(e2);
        if (m1 == m2) continue;
        const int em = m1 ? e1 : e2, ev = m1 ? e2 : e1, side = m1 ? 2 : 1;
        if (shell.attr > 0 && P.ser_attr[ev] == shell.attr) continue;
        const double sgn = m1 ? 1.0 : -1.0;  // CalcOrtho points from element 1 to element 2
        auto& acc = FTm[ser_region[em]];
        mfem::FaceElementTransformations* FTr = vm.GetFaceElementTransformations(f);
        for (int q = 0; q < irf.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = irf.IntPoint(q);
          FTr->SetAllIntPoints(&ip);
          mfem::CalcOrtho(FTr->Jacobian(), nor);
          FTr->Transform(ip, xf);
          const mfem::IntegrationPoint eip = side == 1 ? FTr->GetElement1IntPoint() : FTr->GetElement2IntPoint();
          mfem::ElementTransformation* Te = vm.GetElementTransformation(ev);
          Te->SetIntPoint(&eip);
          vA.GetCurl(*Te, Bq);
          add_bg(*Te, Bq);
          double n3[3], bn = 0, b2 = 0;
          for (int k = 0; k < 3; ++k) {
            n3[k] = sgn * nor(k);
            bn += Bq(k) * n3[k];
            b2 += Bq(k) * Bq(k);
          }
          double fq[3];
          for (int k = 0; k < 3; ++k) fq[k] = (Bq(k) * bn - 0.5 * b2 * n3[k]) / mu0 * ip.weight;
          for (int k = 0; k < 3; ++k) acc[k] += fq[k];
          acc[3] += xf(1) * fq[2] - xf(2) * fq[1];
          acc[4] += xf(2) * fq[0] - xf(0) * fq[2];
          acc[5] += xf(0) * fq[1] - xf(1) * fq[0];
        }
      }
      for (auto& [name, a0] : FTm) {
        const bool egg = !egg_bad.count(name) && FTegg.count(name);
        const auto& a = egg ? FTegg[name] : a0;
        const double F[3] = {a[0], a[1], a[2]}, Tq[3] = {a[3], a[4], a[5]}, Fs[3] = {a0[0], a0[1], a0[2]};
        for (int k = 0; k < 3; ++k) Fsum[k] += F[k];
        fm[name] = {{"F_N", vec3(F)}, {"torque_Nm", vec3(Tq)},
                    {"method", egg ? "volume (eggshell) Maxwell stress" : "surface Maxwell stress"},
                    {"F_surface_N", vec3(Fs)}};
        if (!egg)
          warnings.push_back("magnetic body '" + name + "' touches a winding, superconductor or other magnetic body, so "
                             "its force uses the surface Maxwell stress, which converges slowly (a few percent)");
      }
    }
    double Fscale = 0;
    for (auto* grp : {&fw, &fs, &fm})
      for (auto& [k, v] : grp->items()) for (int i = 0; i < 3; ++i) Fscale = std::max(Fscale, std::fabs(v["F_N"][i].get<double>()));
    static_json["forces"] = {{"windings", fw}, {"superconductors", fs}, {"magnetic_materials", fm},
                             {"sum_F_N", vec3(Fsum)},
                             {"note", "forces on windings (J x B), superconductors and magnetic materials (Maxwell "
                                      "stress); for an isolated system the sum vanishes up to discretization "
                                      "(closure check)"},
                             {"largest_component_N", Fscale}};
    if (d.mechanics.enabled) {  // 3D mechanics (model3d_mech.cpp) reads J after this block
      P.J_static = std::make_unique<SourceJ>(J);
      P.J_keep = std::move(keep_alive), P.J_keep_fes = std::move(keep_fes), P.J_keep_fec = std::move(keep_fec);
    }
  }

  lap("static_s");
  // ------------------------------------------------------------ pickup mode
  // azimuthal: a closed conductor carries the readout current around the z axis (prescribed net current on
  //            its loop about the axis; the 3D form of the axisymmetric ideal slit);
  // port:      the readout current crosses a gap region between two faces of one conductor.
  json pick_json;
  if (d.pickup) {
    const bool az_pick = d.pickup->kind == spec::Pickup::Kind::Azimuthal;
    AXF_REQUIRE(az_pick || d.pickup->kind == spec::Pickup::Kind::Port,
                "3D designs use pickup type 'azimuthal' or 'port'", "/pickup/type",
                "'azimuthal': {\"type\": \"azimuthal\", \"conductor\": NAME} for a closed ring carrying the readout "
                "current around the z axis; 'port': a gap region bridging two faces of one conductor.");
    std::vector<int> body_rf_ser;
    std::vector<Body> bodies = build_bodies(true, P.body_rf, &body_rf_ser);
    P.I_pick = o.pickup_current;
    mfem::Vector b(fes.GetTrueVSize());  // assembled readout source (true dofs)
    b = 0.0;
    int pick_body = -1;
    if (az_pick) {
      for (size_t bi = 0; bi < bodies.size(); ++bi)
        if (std::find(bodies[bi].names.begin(), bodies[bi].names.end(), d.pickup->conductor) != bodies[bi].names.end())
          pick_body = static_cast<int>(bi);
      AXF_REQUIRE(pick_body >= 0, "pickup conductor '" + d.pickup->conductor + "' is not a conductor body", "/pickup/conductor");
    } else {
    auto prepare_port = [&](const std::string& region, const spec::Vec3& toward,
                            Model3D::Impl::ExtraPort* extra, mfem::Vector& source) {
    const bool primary = extra == nullptr;
    const std::string path = primary ? "/pickup/port" : "/fullwave/ports/" + extra->name;
    const int pattr = P.attr_of_region.at(region);
    auto& h1c_store = primary ? P.port_h1c : extra->h1c;
    auto& h1_store = primary ? P.port_h1 : extra->h1;
    auto& V_store = primary ? P.port_V : extra->V;
    auto& diagnostics = primary ? P.port_diagnostics : extra->diagnostics;
    // terminal patches (serial mesh faces): port-element faces shared with RF-conductor elements, grouped by
    // edge adjacency
    mfem::Mesh& sm = *P.serial;
    std::vector<int> tfaces;
    mfem::Array<int> fcs, ori;
    for (int e = 0; e < nser; ++e) {
      if (P.ser_attr[e] != pattr) continue;
      sm.GetElementFaces(e, fcs, ori);
      for (int f : fcs) {
        int e1, e2;
        sm.GetFaceElements(f, &e1, &e2);
        int other = e1 == e ? e2 : e1;
        if (other >= 0 && body_rf_ser[other] >= 0) tfaces.push_back(f);
      }
    }
    // one or more gap pieces, each with two terminal patches oriented by `toward` (port_terminals)
    const PortTerminals pt = port_terminals(sm, P.ser_attr, pattr, tfaces, toward, path, !primary);
    const int npatch = pt.patches;
    {
      std::map<int, std::set<int>> patch_bodies;  // every terminal patch must touch exactly one conductor body
      std::set<int> all_bodies;
      std::map<int, int> face_body;
      for (const auto& [f, p] : pt.patch) {
        int e1, e2;
        sm.GetFaceElements(f, &e1, &e2);
        const int other = P.ser_attr[e1] == pattr ? e2 : e1;
        patch_bodies[p].insert(body_rf_ser[other]);
        all_bodies.insert(body_rf_ser[other]);
        face_body[f] = body_rf_ser[other];
      }
      for (const auto& [p, bodies_p] : patch_bodies)
        AXF_REQUIRE(bodies_p.size() == 1, "each current-port terminal must touch exactly one connected conductor body",
                    path);
      AXF_REQUIRE(!primary || all_bodies.size() == 1,
                  "the readout port connects two different conductors: the readout current has no closed path",
                  path, "Join the conductors on the two sides of the port (e.g. with a short).");
      check_piece_pairing(pt, face_body, sm, P.ser_attr, pattr, path);
      for (const auto& w : pt.warnings) warnings.push_back(w);
    }
    // conduction solve in the port: V = 1 on terminal 0, 0 on terminal 1 (kept for the full-wave analysis)
    h1c_store = std::make_unique<mfem::H1_FECollection>(P.order, 3);
    h1_store = std::make_unique<mfem::ParFiniteElementSpace>(&pm, h1c_store.get());
    mfem::ParFiniteElementSpace& h1 = *h1_store;
    mfem::Vector sig(max_attr);
    sig = 0.0;
    sig(pattr - 1) = 1.0;
    mfem::PWConstCoefficient sigma(sig);
    mfem::ParBilinearForm k(&h1);
    k.AddDomainIntegrator(new mfem::DiffusionIntegrator(sigma, &port_feed_rule(P.order)));
    k.Assemble();
    V_store = std::make_unique<mfem::ParGridFunction>(&h1);
    mfem::ParGridFunction& V = *V_store;
    V = 0.0;
    // port and terminal DOFs: local element faces correspond to the serial ones in element-face order
    // (the partitioned elements keep the serial vertex order); markers are OR-synchronized across ranks
    mfem::Array<int> in_port(h1.GetVSize()), term0(h1.GetVSize()), term1(h1.GetVSize());
    in_port = 0;
    term0 = 0;
    term1 = 0;
    mfem::Array<int> dofs, lf, lo;
    for (int e = 0; e < ne; ++e) {
      if (pm.GetAttribute(e) != pattr) continue;
      h1.GetElementDofs(e, dofs);
      for (int dd : dofs) in_port[dd] = 1;
      pm.GetElementFaces(e, lf, lo);
      sm.GetElementFaces(P.loc2ser[e], fcs, ori);
      for (int k = 0; k < fcs.Size(); ++k) {
        auto it = pt.terminal.find(fcs[k]);
        if (it == pt.terminal.end()) continue;
        h1.GetFaceDofs(lf[k], dofs);
        for (int dd : dofs) (it->second == 0 ? term0 : term1)[dd] = 1;
      }
    }
    h1.Synchronize(in_port);
    h1.Synchronize(term0);
    h1.Synchronize(term1);
    // terminal values: port_terminals made terminal 1 the side further along `toward` in every piece, so the
    // current J = -sigma grad V (normalized below to a positive I_pick) flows along `toward`
    const double v0 = 1.0, v1 = 0.0;
    std::vector<char> essv(h1.GetVSize(), 0);
    for (int i = 0; i < h1.GetVSize(); ++i) {
      if (term0[i]) { essv[i] = 1; V(i) = v0; }
      if (term1[i]) { essv[i] = 1; V(i) = v1; }
    }
    mfem::Array<int> ess_h1;
    for (int i = 0; i < h1.GetVSize(); ++i) {
      if (!in_port[i] || essv[i]) {
        int t = h1.GetLocalTDofNumber(i);
        if (t >= 0) ess_h1.Append(t);
      }
    }
    mfem::ParLinearForm zb(&h1);
    zb.Assemble();
    zb = 0.0;
    mfem::OperatorPtr Kop;
    mfem::Vector X, B;
    keep_eliminated_rows(k, ess_h1);  // the DOFs outside the port: zero rows (hypre on a GPU would drop them)
    k.FormLinearSystem(ess_h1, V, zb, Kop, X, B);
    mfem::HypreBoomerAMG amg(*Kop.As<mfem::HypreParMatrix>());
    amg.SetPrintLevel(0);
    mfem::HyprePCG pcg(*Kop.As<mfem::HypreParMatrix>());
    pcg.SetTol(1e-12);
    pcg.SetMaxIter(500);
    pcg.SetPrintLevel(0);
    pcg.SetPreconditioner(amg);
    pcg.Mult(B, X);
    int port_its = 0;
    pcg.GetNumIterations(port_its);
    if (port_its >= 500)
      warnings.push_back("readout port: the conduction solve reached its iteration cap." + worst_elements_note(P));
    // Refine to the rounding floor. J = -grad V enters the full-wave right-hand side, and its discrete divergence
    // at the free port nodes is this solve's residual; at low frequency the field amplifies that charge by about
    // (k h)^-2 (a 1.2e-14 relative residual gave a 12 % error in a port's electric energy at 100 Hz).
    int refine_steps = 0;
    {
      mfem::Vector r(B.Size()), dx(B.Size());
      const double nb = std::sqrt(std::max(allsum(B * B), 1e-300));
      double prev = 1e300;
      for (int step = 0; step < 4; ++step) {
        Kop->Mult(X, r);
        r -= B;
        r *= -1.0;  // r = B - K X
        const double rel = std::sqrt(allsum(r * r)) / nb;
        if (rel < 1e-17 || rel > 0.5 * prev) break;  // at the floor, or no longer improving
        prev = rel;
        dx = 0.0;
        pcg.Mult(r, dx);
        X += dx;
        ++refine_steps;
      }
    }
    mfem::Vector port_residual(B.Size());
    Kop->Mult(X, port_residual);
    port_residual -= B;
    const double port_rel = std::sqrt(allsum(port_residual * port_residual) /
                                      std::max(allsum(B * B), 1e-300));
    log(2, strcat("  port conduction solve: ", port_its, " PCG iterations, ", refine_steps,
                  " refinement steps, relative residual ", port_rel));
    if (!(std::isfinite(port_rel) && port_rel < 1e-8)) {  // port_rel is global: every rank takes this branch
      throw Error(strcat("port conduction solve failed its true-residual check (relative residual ", port_rel,
                         " after ", port_its, " PCG iterations and ", refine_steps,
                         " refinement steps; the gate is 1e-8)"),
                  path, worst_elements_note(P, "").substr(1));
    }
    k.RecoverFEMSolution(X, zb, V);
    // raw current with sigma = 1: I = int |grad V|^2 dV (unit voltage), and per piece
    double Iraw = 0;
    std::vector<double> Ik(pt.pieces, 0.0);
    mfem::Vector gV(3);
    for (int e = 0; e < ne; ++e) {
      if (pm.GetAttribute(e) != pattr) continue;
      AXF_REQUIRE(pm.GetElementBaseGeometry(e) == mfem::Geometry::TETRAHEDRON,
                  "a port region must be meshed with tetrahedra (its feed integrals share one tetrahedron rule)", "/pickup");
      mfem::ElementTransformation* T = pm.GetElementTransformation(e);
      const auto& ir = port_feed_rule(P.order);
      double Ie = 0;
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        T->SetIntPoint(&ir.IntPoint(q));
        V.GetGradient(*T, gV);
        Ie += (gV * gV) * ir.IntPoint(q).weight * T->Weight();
      }
      Iraw += Ie;
      Ik[pt.piece_of.at(P.loc2ser[e])] += Ie;
    }
    Iraw = allsum(Iraw);
    AXF_REQUIRE(Iraw > 0, "the port conduction problem carries no current", path);
    std::vector<std::string> piece_warn;
    const json pieces_json = port_pieces_json(pt, Ik, Iraw, region, piece_warn);
    for (const auto& w : piece_warn) warnings.push_back(w);
    if (pt.pieces > 1)
      log(1, strcat("distributed port '", region, "': ", pt.pieces, " pieces fed in parallel"));
    const double scale = -P.I_pick / Iraw;  // J = -sigma grad V, normalized to I_pick
    if (primary) { P.port_scale = scale; P.port_attr = pattr; }
    else { extra->scale = scale; extra->attr = pattr; }
    diagnostics = {{"region", region}, {"terminal_patches", npatch}, {"port_pieces", pt.pieces},
                   {"conduction_relative_residual", port_rel}, {"conduction_refinement_steps", refine_steps},
                   {"raw_conductance_m", Iraw},
                   {"normalization_current_A", P.I_pick}, {"toward", {toward.x,toward.y,toward.z}}};
    if (pt.pieces > 1) diagnostics["pieces"] = pieces_json;
    SourceJ Jp;
    mfem::ParGridFunction* Vp = &V;
    Jp.by_attr[pattr] = [Vp, scale](const mfem::Vector&, mfem::ElementTransformation& T, mfem::Vector& out) {
      mfem::Vector g(3);
      Vp->GetGradient(T, g);
      out(0) = scale * g(0);
      out(1) = scale * g(1);
      out(2) = scale * g(2);
    };
    mfem::ParLinearForm bl(&fes);
    bl.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(Jp, &port_feed_rule(P.order)));
    bl.Assemble();
    bl.ParallelAssemble(source);
    };
    prepare_port(d.pickup->port, d.pickup->toward, nullptr, b);
    P.extra_ports.clear();
    P.extra_ports.reserve(d.fullwave.ports.size());
    for (const auto& port : d.fullwave.ports) {
      P.extra_ports.emplace_back();
      auto& extra = P.extra_ports.back();
      extra.name = port.name;
      extra.region = port.region;
      mfem::Vector unused_source(fes.GetTrueVSize());
      spec::Vec3 direction = port.toward;
      prepare_port(port.region, direction, &extra, unused_source);
    }
    }
    // RF states: fluxoid conserved, except the azimuthal pickup loop (net current I) and the loops about the
    // axis of slit bodies (no net current), as in the axisymmetric engine
    std::vector<std::vector<MasterSpec>> ms(bodies.size());
    for (size_t bi = 0; bi < bodies.size(); ++bi) {
      const spec::Superconductor* st = state_of(bodies[bi].names);
      const bool slit = st && st->azimuthal_slit;
      const bool is_pick = az_pick && static_cast<int>(bi) == pick_body;
      int naz = 0;
      for (size_t j = 0; j < bodies[bi].lifts.size(); ++j) {
        const int gt = bodies[bi].geo_type[j];
        if ((is_pick || slit) && gt < 0)
          throw Error("conductor '" + bodies[bi].names.front() + "': its loops are not loops about the z axis (no exact "
                      "geometric cut), so the readout or slit condition cannot be assigned", "/pickup/conductor");
        if (gt == 0 && (is_pick || slit)) {
          ms[bi].push_back({false, is_pick ? P.I_pick : 0.0});  // prescribed net current (period exactly 1)
          ++naz;
        } else {
          ms[bi].push_back({true, 0.0});
        }
      }
      AXF_REQUIRE(!is_pick || naz == 1,
                  strcat("the azimuthal pickup conductor must be a closed ring about the z axis (loops around the axis: ",
                         naz, ")"), "/pickup/conductor", "For a real slit with a readout gap use a 'port' pickup.");
    }
    auto ms0 = ms;  // homogeneous states (circuit responses)
    for (auto& v : ms0)
      for (auto& x : v) x.value = 0.0;
    P.A_pick = std::make_unique<mfem::ParGridFunction>(&fes);
    json jb;
    log(1, "solving pickup mode (3D)");
    // iron: secant permeability at the static field (same convention as the axisymmetric engine)
    std::unique_ptr<NuTensorCoefficient> secant;
    std::unique_ptr<mfem::HypreParMatrix> Ksec;
    if (nonlinear && P.have_lab) {
      secant = std::make_unique<NuTensorCoefficient>(P.A_lab.get(), P.elem_bh, nu_attr, true, shell);
      mfem::ParBilinearForm as(&fes);
      auto* cc = new mfem::CurlCurlIntegrator(*secant);
      cc->SetIntRule(&irq);
      as.AddDomainIntegrator(cc);
      as.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(eps));
      as.Assemble();
      as.Finalize();
      Ksec.reset(as.ParallelAssemble());
      set_nu_mat(secant.get());
      Kcur = Ksec.get();
    }
    solve_with_bodies(bodies, P.body_rf, ms, b, *P.A_pick, jb, false);
    if (!circs.empty()) {
      // persistent circuits respond to the readout current with linkage change zero
      const int nc = static_cast<int>(circs.size());
      std::vector<std::unique_ptr<mfem::ParGridFunction>> uc(nc);
      for (int c = 0; c < nc; ++c) {
        uc[c] = std::make_unique<mfem::ParGridFunction>(&fes);
        json jtmp;
        solve_with_bodies(bodies, P.body_rf, ms0, circs[c].b, *uc[c], jtmp, false);
      }
      Eigen::MatrixXd M(nc, nc);
      Eigen::VectorXd r(nc);
      mfem::Vector U(fes.GetTrueVSize()), Uk(fes.GetTrueVSize());
      P.A_pick->GetTrueDofs(U);
      for (int c = 0; c < nc; ++c) {
        r(c) = -gdot(circs[c].b, U);
        for (int k = 0; k < nc; ++k) {
          uc[k]->GetTrueDofs(Uk);
          M(c, k) = gdot(circs[c].b, Uk);
        }
      }
      Eigen::VectorXd dI = M.fullPivLu().solve(r);
      for (int c = 0; c < nc; ++c) P.A_pick->Add(dI(c), *uc[c]);
      json pc = json::array();
      for (int c = 0; c < nc; ++c) pc.push_back({{"name", circs[c].name}, {"current_per_readout_ampere", dI(c)}});
      jb = json{{"bodies", jb}, {"circuits", pc}};
    }
    set_nu_mat(nullptr);
    Kcur = Kfull.get();
    P.have_pick = true;
    double Wp = energy(*P.A_pick, secant ? 2 : 0);
    pick_json = {{"current_A", P.I_pick}, {"L_p_H", 2 * Wp / (P.I_pick * P.I_pick)}, {"energy_J", Wp}, {"bodies", jb}};
    if (az_pick) {
      // second estimate: fluxoid of the pickup loop per ampere (its generator has period exactly 1)
      const json& jbs = jb.is_object() && jb.contains("bodies") ? jb["bodies"] : jb;
      double flux = 0;
      for (size_t j = 0; j < bodies[pick_body].lifts.size(); ++j)
        if (bodies[pick_body].geo_type[j] == 0) flux = jbs[pick_body]["generators"][j]["fluxoid_coefficient"].get<double>();
      pick_json["kind"] = "azimuthal";
      pick_json["conductor"] = d.pickup->conductor;
      pick_json["L_p_from_flux_H"] = flux / P.I_pick;
    } else {
      pick_json["kind"] = "port";
      pick_json["port"] = d.pickup->port;
      pick_json["note"] = "net_current_A of the conductor touched by the port includes the port source (not a physical current)";
      pick_json["port_feed"] = P.port_diagnostics;  // terminal patches, pieces and their current fractions, conduction
                                                    // residual (also in full-wave `ports`)
    }
  }

  lap("pickup_s");
  // ------------------------------------------------------------ post
  json R;
  R["static"] = static_json;
  R["pickup"] = pick_json;
  double Psi = 0, W = static_json["energy_J"];
  if (P.have_pick) {
    mfem::Vector Av(3), Bv(3);
    for (int e = 0; e < ne; ++e) {
      mfem::ElementTransformation* T = pm.GetElementTransformation(e);
      const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P.order + 1);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        T->SetIntPoint(&ir.IntPoint(q));
        P.A_pick->GetVectorValue(*T, ir.IntPoint(q), Av);
        P.A_lab->GetCurl(*T, Bv);
        add_bg(*T, Bv);
        Psi += (Av * Bv) * ir.IntPoint(q).weight * T->Weight();
      }
    }
    Psi = allsum(Psi) / P.I_pick;
    double Psi2 = 0;
    {
      mfem::Vector Av(3), Bv(3);
      for (int e = 0; e < ne; ++e) {
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P.order + 1);
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          T->SetIntPoint(&ir.IntPoint(q));
          P.A_lab->GetVectorValue(*T, ir.IntPoint(q), Av);
          if (P.bg && !in_shell(e)) {
            mfem::Vector x(3);
            T->Transform(ir.IntPoint(q), x);
            double bb[3], aa[3];
            P.bg_at(x, bb, aa);
            for (int k = 0; k < 3; ++k) Av(k) += aa[k];
          }
          P.A_pick->GetCurl(*T, Bv);
          Psi2 += (Av * Bv) * ir.IntPoint(q).weight * T->Weight();
        }
      }
      Psi2 = allsum(Psi2) / P.I_pick;
    }
    R["checks"]["Psi_A_pick_dot_B_lab"] = Psi;
    R["checks"]["Psi_A_lab_dot_B_pick"] = Psi2;
    R["checks"]["Psi_reciprocity_rel_diff"] = Psi != 0 ? (Psi2 - Psi) / Psi : 0.0;
    // With a background the primary form int A_pick . B_lab is ill-defined: B_bg is cut at R1 (the shell is excluded),
    // which leaves a jump of its normal component there, so the form depends on the gauge of A_pick, i.e. on the
    // per-element gauge mass and so on the mesh (0.70-0.88 of the correct value for a uniform field over meshes and
    // radii; 2/3 in the Coulomb gauge). The reciprocal form int (A_lab + A_bg) . B_pick needs nothing outside R1 for a
    // readout inside it (no readout current crosses a spherical cap outside R1, so the azimuthal average of B_pick,phi
    // that A_bg sees vanishes there), and it is the form the axisymmetric engine reports. With a background it is the
    // reported Psi; the primary form stays as a diagnostic.
    const double Psi_primary = Psi;
    if (P.bg) {
      Psi = Psi2;
      const double ratio = Psi2 != 0 ? Psi_primary / Psi2 : 0.0;
      if (!(std::fabs(ratio - 1) <= 1e-3))
        warnings.push_back(strcat("Psi with the background is the reciprocal form int (A_lab + A_bg) . B_pick; the "
                                  "primary form int A_pick . B_lab, which cuts the background at the domain sphere "
                                  "and so depends on the gauge (the mesh), is ", ratio,
                                  " of it (checks.Psi_primary_form)"));
    }
    R["checks"]["Psi_reported_form"] = P.bg ? "reciprocal: int (A_lab + A_bg) . B_pick (background present)"
                                            : "primary: int A_pick . B_lab";
    // Orientation convention (matches the axisymmetric engine): without an explicit
    // `toward`, the readout current direction is chosen so that Psi >= 0.
    const auto& tw = d.pickup->toward;
    const bool automatic = tw.x == 0 && tw.y == 0 && tw.z == 0;
    double orientation = 1.0;
    if (automatic && Psi < 0) {
      Psi = -Psi;
      orientation = -1.0;
      *P.A_pick *= -1.0;
      P.port_scale = -P.port_scale;  // the readout current runs the other way
    }
    if (P.bg)
      R["checks"]["Psi_primary_form"] = {
          {"value_SI_H_T_m2", orientation * Psi_primary},
          {"over_reported", Psi2 != 0 ? Psi_primary / Psi2 : 0.0},
          {"scope", "int A_pick . B_lab with the background cut at the domain sphere: gauge- and mesh-dependent, a "
                    "diagnostic only; oriented as the reported Psi (Psi_A_pick_dot_B_lab is before the orientation)"}};
    R["pickup"]["orientation"] = automatic ? "automatic (current direction chosen so that Psi >= 0)"
                                           : "set by pickup.toward";
    if (P.nranks > 1) P.view_A_pick = gather(*P.A_pick);
  }
  const double Lp = P.have_pick ? pick_json["L_p_H"].get<double>() : 0.0;
  const double F = Lp > 0 ? std::pow(Psi, 4) / (Lp * Lp) : 0.0;
  // extents (all non-air elements) and cold envelope
  double rmax = 0, zmin = 1e300, zmax = -1e300, crmax = 0, czmin = 1e300, czmax = -1e300;
  double iron_kg = 0;
  {
    mfem::Array<int> vv;
    for (int e = 0; e < ne; ++e) {
      if (P.elem_region[e] == "air" || in_shell(e)) continue;
      const auto* m = P.elem_mat[e];
      bool wind = std::any_of(d.windings.begin(), d.windings.end(), [&](auto& w) { return w.region == P.elem_region[e]; });
      bool cold = !m->warm && (wind || m->kind != spec::MaterialKind::Vacuum);
      pm.GetElementVertices(e, vv);
      for (int v : vv) {
        const double* x = pm.GetVertex(v);
        double r = std::hypot(x[0], x[1]);
        rmax = std::max(rmax, r);
        zmin = std::min(zmin, x[2]);
        zmax = std::max(zmax, x[2]);
        if (cold) {
          crmax = std::max(crmax, r);
          czmin = std::min(czmin, x[2]);
          czmax = std::max(czmax, x[2]);
        }
      }
      if (m->kind == spec::MaterialKind::Iron) iron_kg += m->density * pm.GetElementVolume(e);
    }
    rmax = allmax(rmax);
    zmin = allmin(zmin);
    zmax = allmax(zmax);
    crmax = allmax(crmax);
    czmin = allmin(czmin);
    czmax = allmax(czmax);
    iron_kg = allsum(iron_kg);
  }
  const double A_cold = crmax > 0 ? 2 * pi * crmax * (crmax + (czmax - czmin)) : 0.0;
  // peak fields and superconductor surface fields
  double peak = 0;
  std::map<std::string, double> by_region, sc_peak, sc_peak_regular;
  json sharp_json = json::array();
  {
    mfem::Vector Bv(3);
    for (int e = 0; e < ne; ++e) {
      if (P.body_static[e] >= 0 || in_shell(e)) continue;  // shell fields are mapped, not physical
      mfem::ElementTransformation* T = pm.GetElementTransformation(e);
      const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P.order);
      for (int q = 0; q < ir.GetNPoints(); ++q) {
        T->SetIntPoint(&ir.IntPoint(q));
        P.A_lab->GetCurl(*T, Bv);
        add_bg(*T, Bv);
        double B = Bv.Norml2();
        peak = std::max(peak, B);
        by_region[P.elem_region[e]] = std::max(by_region[P.elem_region[e]], B);
      }
    }
    peak = allmax(peak);
    {  // every region present outside the bodies and the shell, in the same order on every rank
      std::set<std::string> names;
      for (int s2 = 0; s2 < nser; ++s2)
        if (body_static_ser[s2] < 0 && !(shell.attr > 0 && P.ser_attr[s2] == shell.attr)) names.insert(ser_region[s2]);
      std::map<std::string, double> all;
      for (const auto& nm : names) all[nm] = allmax(by_region.count(nm) ? by_region[nm] : 0.0);
      by_region = all;
    }
    // faces between static bodies and other elements (gathered serial view)
    mfem::ParMesh& vm = P.vmesh();
    const mfem::ParGridFunction& vA = *P.vA_lab();
    const std::vector<int>& vbs = P.vbody_static();
    const auto& ir2 = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * P.order);
    // Sharp convex edges of the bodies (interior dihedral angle < 150 deg): the vacuum field is singular there
    // and its surface maximum grows under refinement. The "regular" peak excludes points within three face
    // sizes of such edges (as the 2D engine does near corners).
    struct SharpEdge {
      double a[3], b[3], delta, angle_deg;
      std::string body;
    };
    std::vector<SharpEdge> sharp;
    {
      struct FaceInfo {
        double n[3], c[3], size;
        int body_elem;
      };
      std::map<int, FaceInfo> bf;          // body boundary faces
      std::map<int, std::vector<int>> ef;  // edge -> body boundary faces
      mfem::Array<int> fv, fe, fo, ev;
      for (int f = 0; f < vm.GetNumFaces(); ++f) {
        int e1, e2;
        vm.GetFaceElements(f, &e1, &e2);
        if (e1 < 0 || e2 < 0) continue;
        const bool b1 = vbs[e1] >= 0, b2 = vbs[e2] >= 0;
        if (b1 == b2) continue;
        const int eb = b1 ? e1 : e2;
        vm.GetFaceVertices(f, fv);
        const double *p0 = vm.GetVertex(fv[0]), *p1 = vm.GetVertex(fv[1]), *p2 = vm.GetVertex(fv[2]);
        FaceInfo fi;
        const double u[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, w[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
        fi.n[0] = u[1] * w[2] - u[2] * w[1];
        fi.n[1] = u[2] * w[0] - u[0] * w[2];
        fi.n[2] = u[0] * w[1] - u[1] * w[0];
        const double nn = std::sqrt(fi.n[0] * fi.n[0] + fi.n[1] * fi.n[1] + fi.n[2] * fi.n[2]);
        for (int k = 0; k < 3; ++k) {
          fi.n[k] /= std::max(nn, 1e-300);
          fi.c[k] = (p0[k] + p1[k] + p2[k]) / 3;
        }
        vm.GetElementVertices(eb, ev);  // orient the normal out of the body
        double cb[3] = {0, 0, 0};
        for (int v : ev)
          for (int k = 0; k < 3; ++k) cb[k] += vm.GetVertex(v)[k] / ev.Size();
        if ((fi.c[0] - cb[0]) * fi.n[0] + (fi.c[1] - cb[1]) * fi.n[1] + (fi.c[2] - cb[2]) * fi.n[2] < 0)
          for (double& x : fi.n) x = -x;
        fi.size = std::sqrt(std::max(nn, 0.0));
        fi.body_elem = eb;
        bf[f] = fi;
        vm.GetFaceEdges(f, fe, fo);
        for (int ed : fe) ef[ed].push_back(f);
      }
      for (auto& [ed, faces] : ef) {
        if (faces.size() != 2) continue;  // junctions are not classified
        const FaceInfo &A = bf[faces[0]], &B = bf[faces[1]];
        if (vbs[A.body_elem] != vbs[B.body_elem]) continue;
        const double cosn = std::clamp(A.n[0] * B.n[0] + A.n[1] * B.n[1] + A.n[2] * B.n[2], -1.0, 1.0);
        const double turn = std::acos(cosn);
        // convex if B's centroid lies on the body side of A's plane
        const double side = (B.c[0] - A.c[0]) * A.n[0] + (B.c[1] - A.c[1]) * A.n[1] + (B.c[2] - A.c[2]) * A.n[2];
        const double theta = side < 0 ? pi - turn : pi + turn;
        if (theta >= 150.0 * pi / 180.0) continue;
        mfem::Array<int> evx;
        vm.GetEdgeVertices(ed, evx);
        SharpEdge se;
        for (int k = 0; k < 3; ++k) {
          se.a[k] = vm.GetVertex(evx[0])[k];
          se.b[k] = vm.GetVertex(evx[1])[k];
        }
        se.delta = 3 * std::max(A.size, B.size);
        se.angle_deg = theta * 180 / pi;
        se.body = ser_region[A.body_elem];
        sharp.push_back(se);
      }
    }
    auto near_sharp = [&](const std::string& body, const mfem::Vector& x) {
      for (auto& se : sharp) {
        if (se.body != body) continue;
        double t = 0, l2 = 0;
        for (int k = 0; k < 3; ++k) {
          t += (x(k) - se.a[k]) * (se.b[k] - se.a[k]);
          l2 += (se.b[k] - se.a[k]) * (se.b[k] - se.a[k]);
        }
        t = std::clamp(t / std::max(l2, 1e-300), 0.0, 1.0);
        double d2 = 0;
        for (int k = 0; k < 3; ++k) {
          const double q = x(k) - (se.a[k] + t * (se.b[k] - se.a[k]));
          d2 += q * q;
        }
        if (d2 < se.delta * se.delta) return true;
      }
      return false;
    };
    mfem::Vector xq(3);
    for (int f = 0; f < vm.GetNumFaces(); ++f) {
      int e1, e2;
      vm.GetFaceElements(f, &e1, &e2);
      if (e1 < 0 || e2 < 0) continue;
      bool b1 = vbs[e1] >= 0, b2 = vbs[e2] >= 0;
      if (b1 == b2) continue;
      mfem::FaceElementTransformations* FT = vm.GetFaceElementTransformations(f);
      std::string body = ser_region[b1 ? e1 : e2];
      for (int q = 0; q < ir2.GetNPoints(); ++q) {
        FT->SetAllIntPoints(&ir2.IntPoint(q));
        mfem::ElementTransformation& T = b1 ? *FT->Elem2 : *FT->Elem1;
        vA.GetCurl(T, Bv);
        add_bg(T, Bv);
        const double B = Bv.Norml2();
        sc_peak[body] = std::max(sc_peak[body], B);
        FT->Transform(ir2.IntPoint(q), xq);
        if (!near_sharp(body, xq)) sc_peak_regular[body] = std::max(sc_peak_regular[body], B);
      }
    }
    for (auto& [b, v] : sc_peak)
      if (!sc_peak_regular.count(b)) {
        sc_peak_regular[b] = v;
        warnings.push_back(strcat("superconductor '", b, "': every surface point lies within three face sizes of a sharp "
                                  "edge, so the regular peak equals the raw peak (", v, " T); refine its surface mesh"));
      }
    for (size_t i = 0; i < sharp.size() && i < 50; ++i)
      sharp_json.push_back({{"body", sharp[i].body}, {"a_m", {sharp[i].a[0], sharp[i].a[1], sharp[i].a[2]}},
                            {"b_m", {sharp[i].b[0], sharp[i].b[1], sharp[i].b[2]}},
                            {"interior_angle_deg", sharp[i].angle_deg}, {"excluded_radius_m", sharp[i].delta}});
    if (!sharp.empty())
      log(1, strcat("superconductor surfaces: ", sharp.size(), " sharp edge segments excluded from the regular peak"));
  }
  R["coupling"] = {{"Psi_SI_H_T_m2", Psi}, {"Psi_natural_T_m3", Psi / mu0}, {"L_p_H", Lp}, {"L_p_natural_m", Lp / mu0},
                   {"F_SI_T4_H2_m8", F}, {"F_natural_T4_m10", F / (mu0 * mu0)}, {"W_mag_J", W},
                   {"device_radius_m", rmax}, {"device_height_m", zmax - zmin},
                   {"g_factor", (Lp > 0 && W > 0) ? Psi * Psi / (Lp * W * rmax * rmax) / (mu0 * mu0) : 0.0}};
  R["fields"] = {{"B_max_T", peak}, {"by_region_T", by_region}, {"superconductor_surface_T", sc_peak},
                 {"superconductor_surface_regular_T", sc_peak_regular}, {"superconductor_sharp_edges", sharp_json}};
  // ---------------------------------------------------------------- RF surface losses of the pickup
  // As the 2D engine (model.cpp, R["pickup"]["losses"]): for every conductor or superconductor surface region,
  // Gamma = int |H_pick|^2 dA per unit readout current squared over its faces to the field region (H on the field
  // side, with that material's mu; infinity-shell faces excluded). R_eff = R_s(f) Gamma and Q = 2 pi f L_p / R_eff at
  // the scan band edges for surfaces with an `rf` block, and the T/Q budget with each surface's T_K. A
  // magnetoresistive rf block weights R_s by the local static |B| (Gamma_magnetoresistance). On the gathered serial
  // view, identical on every rank. The faces, rule, field-side mu and magnetoresistance weight are shared with the
  // full-wave surface losses (surface_losses3d.hpp).
  if (P.have_pick && Lp > 0) {
    mfem::ParMesh& vm = P.vmesh();
    const mfem::ParGridFunction& vAp = *P.vA_pick();
    const mfem::ParGridFunction* vAl = P.have_lab ? P.vA_lab() : nullptr;
    const auto& irs = surface_loss_rule(P.order);
    auto is_rfc = [&](int s2) { return ser_sc(s2) || ser_cond(s2); };
    std::map<std::string, double> gamma, gamma_mr;
    std::map<std::string, std::map<std::string, double>> gamma_by;  // surface -> field-side region -> Gamma part
    std::map<std::string, const spec::Material*> surf_mat;
    mfem::Vector Bp(3), Bs(3), Bi(3), nor(3);
    auto in_shell = [&](int s2) { return shell.attr > 0 && P.ser_attr[s2] == shell.attr; };
    for (const SurfaceLossFace& lf : surface_loss_faces(vm, is_rfc, in_shell)) {
      const int f = lf.face;
      const int ec = lf.conductor;
      const int ev = lf.field;
      const std::string& body = ser_region[ec];
      const spec::Material* cm = ser_mat[ec];
      surf_mat[body] = cm;
      const spec::Material* fm = ser_mat[ev];
      const double mu_lin = surface_loss_mu(*fm);
      // iron with a static solution: the pickup solve's secant permeability 1/nu(|B_static|), not the initial mu_r
      // (nonlinear iron excludes a background field, so B_static = curl A_lab)
      const BHLaw* law = nullptr;
      if (fm->kind == spec::MaterialKind::Iron && nonlinear && vAl) law = bh_of.at(d.find_region(ser_region[ev])->material);
      mfem::FaceElementTransformations* FT = vm.GetFaceElementTransformations(f);
      for (int q = 0; q < irs.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = irs.IntPoint(q);
        FT->SetAllIntPoints(&ip);
        mfem::ElementTransformation& T = ev == FT->Elem1No ? *FT->Elem1 : *FT->Elem2;
        vAp.GetCurl(T, Bp);
        mfem::CalcOrtho(FT->Jacobian(), nor);
        double mu = mu_lin;
        if (law) {
          vAl->GetCurl(T, Bi);
          mu = 1.0 / law->nu(Bi.Norml2());
        }
        const double dG = (Bp * Bp) / (P.I_pick * P.I_pick * mu * mu) * ip.weight * nor.Norml2();
        gamma[body] += dG;
        gamma_by[body][ser_region[ev]] += dG;
        if (!cm->rf.mr_model.empty()) {
          Bs = 0.0;
          if (vAl) vAl->GetCurl(T, Bs);
          add_bg(T, Bs);
          gamma_mr[body] += surface_loss_mr_weighted(*cm, Bs.Norml2(), dG);
        }
      }
    }
    json lj = json::object();
    double TQ_lo = 0, TQ_hi = 0;
    bool have_rf = false;
    const double f_lo = d.scan.f_min, f_hi = d.scan.f_max;
    for (auto& [b, G0] : gamma) {
      json e = {{"Gamma", G0}, {"note", "R_eff = R_s(f) * Gamma; Q = 2 pi f L_p / R_eff"},
                {"Gamma_by_field_region", gamma_by[b]}};  // where the loss sits: cavity, slit gap, port, ...
      const spec::Material* mt = surf_mat[b];
      double G = G0;
      if (!mt->rf.mr_model.empty()) {
        G = gamma_mr[b];
        e["Gamma_magnetoresistance"] = G;
        e["note"] = "R_eff = R_s(f) * Gamma_magnetoresistance (R_s weighted by the local static field); "
                    "Q = 2 pi f L_p / R_eff";
      }
      if (!mt->rf.model.empty()) {
        have_rf = true;
        const double Rl = mt->rf.Rs(f_lo) * G, Rh = mt->rf.Rs(f_hi) * G;
        const double Ql = 2 * M_PI * f_lo * Lp / Rl, Qh = 2 * M_PI * f_hi * Lp / Rh;
        e["R_eff_ohm"] = {{"f_min", Rl}, {"f_max", Rh}};
        e["Q"] = {{"f_min", Ql}, {"f_max", Qh}};
        e["T_K"] = mt->rf.T_K;
        e["T_over_Q_K"] = {{"f_min", mt->rf.T_K / Ql}, {"f_max", mt->rf.T_K / Qh}};
        TQ_lo += mt->rf.T_K / Ql;
        TQ_hi += mt->rf.T_K / Qh;
      }
      lj[b] = e;
    }
    json losses = {{"surfaces", lj}};
    if (have_rf) {
      losses["T_over_Q_total_K"] = {{"f_min", TQ_lo}, {"f_max", TQ_hi}};
      json cmp = json::array();
      for (auto& bm : d.benchmarks)
        cmp.push_back({{"benchmark", bm.name}, {"T_over_Q_K", bm.T / bm.Q},
                       {"computed_over_benchmark", {{"f_min", TQ_lo / (bm.T / bm.Q)}, {"f_max", TQ_hi / (bm.T / bm.Q)}}}});
      losses["vs_benchmarks"] = cmp;
      losses["note"] = "only surfaces with rf loss data are included; materials without an rf block are lossless here";
    }
    losses["engine"] = "3D: surface integral of the pickup field over the conductor faces";
    R["pickup"]["losses"] = losses;
    std::set<std::string> rf_ignored;
    for (auto& g : d.regions) {
      const spec::Material& gm = d.materials.at(g.material);
      if (!gm.rf.model.empty() && gm.kind != spec::MaterialKind::Superconductor &&
          gm.kind != spec::MaterialKind::Conductor)
        rf_ignored.insert(g.material);
    }
    for (auto& name : rf_ignored)
      warnings.push_back(strcat("material '", name, "' has an rf block but is not a conductor or superconductor: ",
                                "pickup.losses counts only conductor and superconductor surfaces, so it has no loss"));
  }
  if (static_json.contains("forces")) R["forces"] = static_json["forces"];
  R["geometry"] = {{"A_cold_m2", A_cold}, {"iron_mass_kg", iron_kg}};
  // field limits for the cost options (as in 2D): superconductor wall (regular peak) and winding peak
  double sc_wall = 0, wind_peak = 0;
  for (auto& [k2, v] : sc_peak_regular) sc_wall = std::max(sc_wall, v);
  for (auto& w : d.windings)
    if (by_region.count(w.region)) wind_peak = std::max(wind_peak, by_region[w.region]);
  const double sc_cap = d.cost.sc_wall_cap, b_cap = d.cost.B_peak_cap;
  auto caps_json = [&](double scale) {
    json j = json::object();
    double viol = 0;
    if (sc_cap > 0) {
      j["sc_wall_T"] = scale * sc_wall;
      j["sc_wall_cap_T"] = sc_cap;
      viol = std::max(viol, scale * sc_wall / sc_cap - 1);
    }
    if (b_cap > 0) {
      j["winding_peak_T"] = scale * wind_peak;
      j["B_peak_cap_T"] = b_cap;
      viol = std::max(viol, scale * wind_peak / b_cap - 1);
    }
    if (sc_cap > 0 || b_cap > 0) {
      j["violation"] = std::max(viol, 0.0);
      j["ok"] = viol <= 0;
    }
    return j;
  };
  const bool fixed_sources =
      std::any_of(d.superconductors.begin(), d.superconductors.end(),
                  [](auto& sc) { return sc.azimuthal.value != 0.0 || sc.poloidal.value != 0.0; }) ||
      std::any_of(d.circuits.begin(), d.circuits.end(), [](auto& c) { return c.flux_linkage != 0.0; }) ||
      static_cast<bool>(P.bg);
  json bench = json::array();
  for (auto& b : d.benchmarks) {
    auto req = scan::F_required(b, d.scan);
    json bj = {{"name", b.name}, {"Q", b.Q}, {"T_K", b.T}, {"eta_A", b.eta_A}, {"F_req_SI", req.F_SI},
               {"F_req_natural_T4_m10", req.F_nat}, {"g_min_GeV", req.g_min_GeV}};
    if (F > 0) {
      double s = std::pow(req.F_SI / F, 0.25);
      const double Wr = s * s * W;
      bj["amplitude_scale"] = s;
      bj["F_over_F_req"] = F / req.F_SI;
      bj["W_mag_at_req_J"] = Wr;
      bj["B_max_at_req_T"] = s * peak;
      json scp = json::object(), scr = json::object();
      for (auto& [k2, v] : sc_peak) scp[k2] = s * v;
      for (auto& [k2, v] : sc_peak_regular) scr[k2] = s * v;
      bj["superconductor_surface_at_req_T"] = scp;
      bj["superconductor_surface_regular_at_req_T"] = scr;
      if (sc_cap > 0 || b_cap > 0) bj["caps_at_req"] = caps_json(s);
      bj["cost"] = cost::evaluate(d.cost, Wr, A_cold, iron_kg).to_json();
      if (iron_kg == 0) {
        const double l = cost::optimal_scale(d.cost, Wr, A_cold);
        auto copt = cost::evaluate(d.cost, Wr / (l * l), A_cold * l * l, 0);
        const double fs = s * std::pow(l, -2.5);  // field scale at the cost-optimal size
        json sco = json::object();
        for (auto& [k2, v] : sc_peak_regular) sco[k2] = fs * v;
        bj["cost_optimal_size"] = {{"length_scale", l},
                                   {"device_radius_m", rmax * l},
                                   {"B_max_T", fs * peak},
                                   {"superconductor_surface_regular_T", sco},
                                   {"W_mag_J", Wr / (l * l)},
                                   {"cost", copt.to_json()}};
        if (sc_cap > 0 || b_cap > 0) bj["cost_optimal_size"]["caps"] = caps_json(fs);
      }
      if (nonlinear) {
        bj["note"] = "iron is nonlinear: amplitude scaling is a first estimate; rescale the sources and re-solve";
      } else if (fixed_sources) {
        bj["note"] = "fixed fluxes, prescribed superconductor currents or persistent-circuit linkages do not scale "
                     "with the winding currents: amplitude scaling is approximate";
      }
    }
    bench.push_back(bj);
  }
  R["benchmarks"] = bench;
  R["mesh"] = {{"elements", nser}, {"order", P.order}, {"mpi_ranks", P.nranks}, {"dofs", fes.GlobalTrueVSize()}, {"info", P.mesh_info}};
  R["solver"] = {{"pcg_iterations_total", total_its}, {"max_relative_residual", max_rel_res},
                {"mpi_ranks", P.nranks}, {"hypre_device", hypre_device_label()}, {"options", so.to_json(kappa)},
                {"solves", solve_log}};
  if (!body_constraint_solves.empty()) R["solver"]["body_constraint_solves"] = body_constraint_solves;
  R["warnings"] = warnings;
  R["design"] = {{"name", d.name}, {"parameters", d.parameters}, {"symmetry", "3d"}};
  R["version"] = kVersion;
  lap("post_s");
  timing["total_s"] = timer.seconds();
  R["timing"] = timing;
  results_ = R;
  electrostatics_analysis(o);  // 3D electrostatics (cpp/src/model3d_electro.cpp)
  mechanics_thermal_analysis(o);  // 3D mechanics and conduction (cpp/src/model3d_mech.cpp)
}

std::vector<Model3D::Sample> Model3D::sample(const std::vector<double>& x, const std::vector<double>& y,
                                             const std::vector<double>& z) const {
  AXF_REQUIRE(p_->have_lab, "solve the model before sampling");
  const Impl& P = *p_;
  const int n = static_cast<int>(x.size());
  mfem::DenseMatrix pts(3, n);
  // points beyond the shell's inner radius are physical points of the exterior: map them to the
  // computational shell (rho = R2 - R1 (R2 - R1)/rho') and transform B as a 2-form
  std::vector<double> fmap(n, 0.0), rmap(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double p[3] = {x[i], y[i], z[i]};
    if (P.sh_attr > 0) {
      double d[3] = {p[0] - P.sh_c[0], p[1] - P.sh_c[1], p[2] - P.sh_c[2]};
      const double rp = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (rp > P.sh_R1) {
        const double rc = P.sh_R2 - P.sh_R1 * (P.sh_R2 - P.sh_R1) / rp;
        for (int k = 0; k < 3; ++k) p[k] = P.sh_c[k] + d[k] * rc / rp;
        fmap[i] = rp;
        rmap[i] = rc;
      }
    }
    for (int k = 0; k < 3; ++k) pts(k, i) = p[k];
  }
  mfem::Array<int> eid;
  mfem::Array<mfem::IntegrationPoint> ips;
  P.vmesh().FindPoints(pts, eid, ips, false);
  std::vector<Sample> out(n);
  mfem::Vector Bv(3);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto phys = [&](int i, double B[3]) {
    if (fmap[i] <= 0) return;
    const double f = fmap[i], rho = rmap[i], fp = f / (P.sh_R2 - rho);
    double r[3] = {pts(0, i) - P.sh_c[0], pts(1, i) - P.sh_c[1], pts(2, i) - P.sh_c[2]};
    const double rn = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    for (double& v : r) v /= rn;
    const double br = B[0] * r[0] + B[1] * r[1] + B[2] * r[2];
    const double sr = (rho / f) * (rho / f), st = rho / (f * fp);
    for (int k = 0; k < 3; ++k) B[k] = br * r[k] * sr + (B[k] - br * r[k]) * st;
  };
  for (int i = 0; i < n; ++i) {
    if (eid[i] < 0) {
      for (int k = 0; k < 3; ++k) out[i].B[k] = out[i].pB[k] = nan;
      continue;
    }
    mfem::ElementTransformation* T = P.vmesh().GetElementTransformation(eid[i]);
    T->SetIntPoint(&ips[i]);
    P.vA_lab()->GetCurl(*T, Bv);
    for (int k = 0; k < 3; ++k) out[i].B[k] = Bv(k);
    phys(i, out[i].B);
    if (P.bg) {  // background at the physical point
      mfem::Vector xp(3);
      xp(0) = x[i];
      xp(1) = y[i];
      xp(2) = z[i];
      double bb[3];
      P.bg_at(xp, bb);
      for (int k = 0; k < 3; ++k) out[i].B[k] += bb[k];
    }
    if (P.have_pick) {
      P.vA_pick()->GetCurl(*T, Bv);
      for (int k = 0; k < 3; ++k) out[i].pB[k] = Bv(k) / P.I_pick;
      phys(i, out[i].pB);
    }
    out[i].region = P.vmesh().GetAttribute(eid[i]);
  }
  return out;
}

json scene_f32(const std::vector<double>& v, int cols);
json scene_u32(const std::vector<uint32_t>& v, int cols);

// 3D scene (docs/scene_format.md, "symmetry": "3d"): region boundary surfaces with the surface |B|
// (vacuum side) and an exact slice of the mesh by the plane y = y0 with field quantities.
json Model3D::scene(const SceneOptions& o) const {
  AXF_REQUIRE(p_->have_lab, "solve the model before building a scene");
  const Impl& P = *p_;
  mfem::ParMesh& m = P.vmesh();
  const int ne = m.GetNE();
  const int shell_attr = P.mesh_info.value("shell_attr", -1);
  auto skip_region = [&](int attr) {
    const std::string& r = P.region_of_attr.count(attr) ? P.region_of_attr.at(attr) : std::string("air");
    return r == "air" || r == "infinity_shell" || attr == shell_attr;
  };
  auto role_of = [&](const std::string& reg) -> std::string {
    const spec::Region* g = d_.find_region(reg);
    if (!g) return "vacuum";
    for (auto& w : d_.windings)
      if (w.region == reg) return "winding";
    if (d_.pickup && d_.pickup->port == reg) return "port";
    switch (d_.materials.at(g->material).kind) {
      case spec::MaterialKind::Superconductor: return "superconductor";
      case spec::MaterialKind::Iron: return "iron";
      case spec::MaterialKind::Conductor: return "conductor";
      case spec::MaterialKind::Linear: return "magnetic";
      default: return "vacuum";
    }
  };
  auto eliminated = [&](int e) { return P.vbody_static()[e] >= 0; };  // superconductor interiors: B = 0
  // field at an element-reference point
  mfem::Vector Bv(3);
  const bool rf = P.fw_scene_f > 0 && P.vE_re();
  double rfE = 0, rfB = 0;  // RF |E| and |B| at the last field_at point
  auto field_at = [&](int e, const mfem::IntegrationPoint& ip, double& B, double& pB, double Bc[3]) {
    mfem::ElementTransformation* T = m.GetElementTransformation(e);
    T->SetIntPoint(&ip);
    if (rf) {
      mfem::Vector a(3), b(3), ca(3), cb(3);
      P.vE_re()->GetVectorValue(*T, ip, a);
      P.vE_im()->GetVectorValue(*T, ip, b);
      P.vE_re()->GetCurl(*T, ca);
      P.vE_im()->GetCurl(*T, cb);
      rfE = std::sqrt(a * a + b * b);
      rfB = std::sqrt(ca * ca + cb * cb) / (2 * pi * P.fw_scene_f);
    }
    P.vA_lab()->GetCurl(*T, Bv);
    if (P.bg && !(P.sh_attr > 0 && T->Attribute == P.sh_attr)) {
      mfem::Vector xp(3);
      T->Transform(ip, xp);
      double bb[3];
      P.bg_at(xp, bb);
      for (int k = 0; k < 3; ++k) Bv(k) += bb[k];
    }
    B = Bv.Norml2();
    for (int k = 0; k < 3; ++k) Bc[k] = Bv(k);
    pB = 0;
    if (P.have_pick) {
      P.vA_pick()->GetCurl(*T, Bv);
      pB = Bv.Norml2() / P.I_pick;
    }
  };
  double bb[6] = {1e300, 1e300, 1e300, -1e300, -1e300, -1e300};
  // ---- region surfaces
  std::map<int, std::vector<int>> faces_of;  // attr -> faces (signed: side 1 or 2 encoded below)
  struct SideFace {
    int face, side;
  };
  std::map<int, std::vector<SideFace>> sf;
  for (int f = 0; f < m.GetNumFaces(); ++f) {
    int e1, e2;
    m.GetFaceElements(f, &e1, &e2);
    const int a1 = e1 >= 0 ? m.GetAttribute(e1) : -1, a2 = e2 >= 0 ? m.GetAttribute(e2) : -1;
    if (a1 == a2) continue;
    if (e1 >= 0 && !skip_region(a1)) sf[a1].push_back({f, 1});
    if (e2 >= 0 && !skip_region(a2)) sf[a2].push_back({f, 2});
  }
  json regions = json::array();
  for (auto& [attr, list] : sf) {
    const std::string name = P.region_of_attr.at(attr);
    std::map<int, int> local;
    std::vector<double> V, Bsum, cnt;
    std::vector<uint32_t> T;
    mfem::Array<int> fv;
    for (auto [f, side] : list) {
      int e1, e2;
      m.GetFaceElements(f, &e1, &e2);
      const int es = side == 1 ? e1 : e2, eo = side == 1 ? e2 : e1;
      // evaluate on the outside element unless it is absent or an eliminated superconductor
      const int ev = (eo >= 0 && !eliminated(eo)) ? eo : es;
      const int evside = ev == e1 ? 1 : 2;
      m.GetFaceVertices(f, fv);
      // outward orientation from the region element es
      const double *p0 = m.GetVertex(fv[0]), *p1 = m.GetVertex(fv[1]), *p2 = m.GetVertex(fv[2]);
      double nrm[3] = {(p1[1] - p0[1]) * (p2[2] - p0[2]) - (p1[2] - p0[2]) * (p2[1] - p0[1]),
                       (p1[2] - p0[2]) * (p2[0] - p0[0]) - (p1[0] - p0[0]) * (p2[2] - p0[2]),
                       (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])};
      mfem::Array<int> ev_v;
      m.GetElementVertices(es, ev_v);
      double c[3] = {0, 0, 0};
      for (int v : ev_v)
        for (int k = 0; k < 3; ++k) c[k] += m.GetVertex(v)[k] / ev_v.Size();
      const double dot = nrm[0] * (p0[0] - c[0]) + nrm[1] * (p0[1] - c[1]) + nrm[2] * (p0[2] - c[2]);
      int order3[3] = {0, 1, 2};
      if (dot < 0) std::swap(order3[1], order3[2]);
      mfem::FaceElementTransformations* FT = m.GetFaceElementTransformations(f);
      static const double fx[3][2] = {{0, 0}, {1, 0}, {0, 1}};
      for (int q : order3) {
        const int v = fv[q];
        auto it = local.find(v);
        int li;
        if (it == local.end()) {
          li = static_cast<int>(local.size());
          local[v] = li;
          const double* x = m.GetVertex(v);
          for (int k = 0; k < 3; ++k) {
            V.push_back(x[k]);
            bb[k] = std::min(bb[k], x[k]);
            bb[3 + k] = std::max(bb[3 + k], x[k]);
          }
          Bsum.push_back(0.0);
          cnt.push_back(0.0);
        } else {
          li = it->second;
        }
        mfem::IntegrationPoint ipf;
        ipf.Set2(fx[q][0], fx[q][1]);
        FT->SetAllIntPoints(&ipf);
        const mfem::IntegrationPoint& eip = evside == 1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
        double B, pB, Bc[3];
        field_at(ev, eip, B, pB, Bc);
        Bsum[li] += B;
        cnt[li] += 1;
        T.push_back(static_cast<uint32_t>(li));
      }
    }
    for (size_t i = 0; i < Bsum.size(); ++i) Bsum[i] /= std::max(cnt[i], 1.0);
    const spec::Region* g = d_.find_region(name);
    regions.push_back({{"name", name}, {"role", role_of(name)}, {"material", g ? g->material : "vacuum"},
                       {"surface", {{"vertices", scene_f32(V, 3)}, {"indices", scene_u32(T, 3)},
                                    {"Bmag", scene_f32(Bsum, 1)}}}});
  }
  // ---- quarter cutaway: the quadrant {x > xc, y < yc} is removed; each exposed half-plane carries a slice
  // (exact intersection with the tetrahedra), clipped to a window around the device. Superconductor
  // interiors (B = 0) are kept as material patches (quantity values NaN).
  const double xc = 0.5 * (bb[0] + bb[3]), yc = 0.5 * (bb[1] + bb[4]);
  const double Lw = std::max({bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]});
  double win[6];
  for (int k = 0; k < 3; ++k) {
    win[k] = bb[k] - 0.03 * Lw;
    win[3 + k] = bb[3 + k] + 0.03 * Lw;
  }
  struct SVtx {
    double x[3];
    double B, Bc[3], pB, rE = 0, rB = 0;  // static |B|, components, pickup |B| per A, RF |E| and |B| per A
  };
  auto make_slice = [&](int axis, double off, int haxis, double hsign, double hval) {
    // plane x[axis] = off; keep hsign * (x[haxis] - hval) >= 0 and the window box
    std::vector<double> V, qB, qBx, qBy, qBz, qpB, qrE, qrB;
    std::vector<uint32_t> T, Ra;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    mfem::Array<int> vv;
    static const double rv[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int e = 0; e < ne; ++e) {
      const int attr = m.GetAttribute(e);
      if (attr == shell_attr) continue;
      m.GetElementVertices(e, vv);
      double dv[4];
      bool pos[4];
      int npos = 0;
      for (int k = 0; k < 4; ++k) {
        dv[k] = m.GetVertex(vv[k])[axis] - off;
        npos += (pos[k] = dv[k] >= 0);
      }
      if (npos == 0 || npos == 4) continue;
      const bool sc_el = eliminated(e);
      std::vector<SVtx> poly;
      for (int a = 0; a < 4; ++a)
        for (int b = a + 1; b < 4; ++b) {
          if (pos[a] == pos[b]) continue;
          const double t = dv[a] / (dv[a] - dv[b]);
          SVtx sv{};
          const double* xa = m.GetVertex(vv[a]);
          const double* xb = m.GetVertex(vv[b]);
          for (int k = 0; k < 3; ++k) sv.x[k] = xa[k] + t * (xb[k] - xa[k]);
          if (sc_el) {
            sv.B = sv.pB = sv.rE = sv.rB = nan;
            sv.Bc[0] = sv.Bc[1] = sv.Bc[2] = nan;
          } else {
            mfem::IntegrationPoint ip;
            ip.Set3(rv[a][0] + t * (rv[b][0] - rv[a][0]), rv[a][1] + t * (rv[b][1] - rv[a][1]),
                    rv[a][2] + t * (rv[b][2] - rv[a][2]));
            field_at(e, ip, sv.B, sv.pB, sv.Bc);
            sv.rE = rfE;
            sv.rB = rfB;
          }
          poly.push_back(sv);
        }
      // order around the centroid in the plane
      const int u1 = (axis + 1) % 3, u2 = (axis + 2) % 3;
      double c1 = 0, c2 = 0;
      for (auto& p : poly) c1 += p.x[u1] / poly.size(), c2 += p.x[u2] / poly.size();
      std::sort(poly.begin(), poly.end(), [&](const SVtx& A, const SVtx& B) {
        return std::atan2(A.x[u2] - c2, A.x[u1] - c1) < std::atan2(B.x[u2] - c2, B.x[u1] - c1);
      });
      // clip against the half-plane and the window (Sutherland-Hodgman, all constraints axis aligned)
      auto clipc = [&](int ax, double val, double sg) {
        std::vector<SVtx> o2;
        for (size_t a = 0; a < poly.size(); ++a) {
          const SVtx& P = poly[a];
          const SVtx& Q = poly[(a + 1) % poly.size()];
          const double fp = sg * (P.x[ax] - val), fq = sg * (Q.x[ax] - val);
          if (fp >= 0) o2.push_back(P);
          if ((fp >= 0) != (fq >= 0)) {
            const double tt = fp / (fp - fq);
            SVtx R{};
            for (int k = 0; k < 3; ++k) R.x[k] = P.x[k] + tt * (Q.x[k] - P.x[k]);
            R.B = P.B + tt * (Q.B - P.B);
            R.pB = P.pB + tt * (Q.pB - P.pB);
            R.rE = P.rE + tt * (Q.rE - P.rE);
            R.rB = P.rB + tt * (Q.rB - P.rB);
            for (int k = 0; k < 3; ++k) R.Bc[k] = P.Bc[k] + tt * (Q.Bc[k] - P.Bc[k]);
            o2.push_back(R);
          }
        }
        poly.swap(o2);
      };
      clipc(haxis, hval, hsign);
      for (int k = 0; k < 3 && !poly.empty(); ++k) {
        if (k == axis) continue;
        clipc(k, win[k], 1.0);
        if (!poly.empty()) clipc(k, win[3 + k], -1.0);
      }
      if (poly.size() < 3) continue;
      const uint32_t base = static_cast<uint32_t>(V.size() / 3);
      for (auto& p : poly) {
        for (int k = 0; k < 3; ++k) V.push_back(p.x[k]);
        qB.push_back(p.B);
        qBx.push_back(p.Bc[0]);
        qBy.push_back(p.Bc[1]);
        qBz.push_back(p.Bc[2]);
        qpB.push_back(p.pB);
        qrE.push_back(p.rE);
        qrB.push_back(p.rB);
      }
      for (size_t k = 1; k + 1 < poly.size(); ++k) {
        T.push_back(base);
        T.push_back(base + static_cast<uint32_t>(k));
        T.push_back(base + static_cast<uint32_t>(k + 1));
        Ra.push_back(static_cast<uint32_t>(attr));
      }
    }
    auto range = [](const std::vector<double>& v) {
      double a = 1e300, b = -1e300;
      for (double x : v)
        if (std::isfinite(x)) a = std::min(a, x), b = std::max(b, x);
      return a <= b ? json::array({a, b}) : json::array({0, 0});
    };
    json n = json::array({0, 0, 0});
    n[axis] = 1;
    json q = {{"Bmag", scene_f32(qB, 1)}, {"Bx", scene_f32(qBx, 1)}, {"By", scene_f32(qBy, 1)}, {"Bz", scene_f32(qBz, 1)}};
    json ranges = {{"Bmag", range(qB)}, {"Bx", range(qBx)}, {"By", range(qBy)}, {"Bz", range(qBz)}};
    json units = {{"Bmag", "T"}, {"Bx", "T"}, {"By", "T"}, {"Bz", "T"}, {"pickup_Bmag", "T/A"}};
    if (P.have_pick) {
      q["pickup_Bmag"] = scene_f32(qpB, 1);
      ranges["pickup_Bmag"] = range(qpB);
    }
    if (rf) {
      q["rf_Emag"] = scene_f32(qrE, 1);
      q["rf_Bmag"] = scene_f32(qrB, 1);
      ranges["rf_Emag"] = range(qrE);
      ranges["rf_Bmag"] = range(qrB);
      units["rf_Emag"] = "V/(m A)";
      units["rf_Bmag"] = "T/A";
    }
    json sl = {{"plane", {{"normal", n}, {"offset", off}}}, {"vertices", scene_f32(V, 3)},
               {"indices", scene_u32(T, 3)}, {"region_attr", scene_u32(Ra, 1)}, {"quantities", q},
               {"ranges", ranges}, {"units", units}};
    if (rf) sl["rf_frequency_Hz"] = P.fw_scene_f;
    return sl;
  };
  json attr_names = json::object();
  for (auto& [a, n] : P.region_of_attr) attr_names[std::to_string(a)] = n;
  json scene = {{"schema", "axionfem.scene/1"},
                {"name", d_.name},
                {"symmetry", "3d"},
                {"units", "m"},
                {"extent3", {{"min", {bb[0], bb[1], bb[2]}}, {"max", {bb[3], bb[4], bb[5]}}}},
                {"cutaway", {{"center", {xc, yc}}, {"removed", "x > xc and y < yc"}}},
                {"attr_names", attr_names},
                {"regions", regions}};
  if (o.fields) {
    // half-plane y = yc with x >= xc, and half-plane x = xc with y <= yc
    scene["slices"] = json::array({make_slice(1, yc, 0, 1.0, xc), make_slice(0, xc, 1, -1.0, yc)});
  }
  json summary = json::object();
  for (const char* k : {"coupling", "fields", "benchmarks", "geometry", "pickup", "warnings"})
    if (results_.contains(k)) summary[k] = results_[k];
  scene["summary"] = summary;
  scene["design"] = d_.source;
  return scene;
}

void Model3D::write_paraview(const std::string& dir) const {
  const bool loaded = !p_->pmesh;  // a loaded solution has only the serial view
  mfem::ParaViewDataCollection dc("fields", loaded ? p_->view_mesh.get() : p_->pmesh.get());
  dc.SetPrefixPath(dir);
  dc.SetLevelsOfDetail(p_->order);
  dc.SetHighOrderOutput(true);
  dc.RegisterField("A_lab", const_cast<mfem::ParGridFunction*>(loaded ? p_->vA_lab() : p_->A_lab.get()));
  if (p_->have_pick)
    dc.RegisterField("A_pickup", const_cast<mfem::ParGridFunction*>(loaded ? p_->vA_pick() : p_->A_pick.get()));
  dc.SetCycle(0);
  dc.SetTime(0.0);
  dc.Save();
}

void Model3D::save(const std::string& dir) const {
  AXF_REQUIRE(p_->have_lab, "solve the model before saving it");
  const Impl& P = *p_;
  if (P.rank != 0) return;  // the gathered view is identical on every rank
  std::filesystem::create_directories(dir);
  {
    std::ofstream f(dir + "/mesh.mesh");
    f.precision(17);
    P.serial->Print(f);
  }
  // fields on the serial mesh: the view has the serial element and vertex order, so the element-wise
  // interpolation reproduces them exactly
  mfem::FiniteElementSpace sf(P.serial.get(), P.fec.get());
  std::vector<int> ident(P.serial->GetNE());
  std::iota(ident.begin(), ident.end(), 0);
  auto put = [&](const mfem::ParGridFunction* g, const std::string& name) {
    mfem::GridFunction s(&sf);
    SerialFieldCoefficient c(P.vmesh(), *g, ident);
    s.ProjectCoefficient(c);
    std::ofstream f(dir + "/" + name);
    f.precision(17);
    s.Save(f);
  };
  put(P.vA_lab(), "A_lab.gf");
  if (P.have_pick) put(P.vA_pick(), "A_pick.gf");
  json roa = json::array();
  for (auto& [a, n] : P.region_of_attr) roa.push_back({a, n});
  json st = {{"format", "axionfem.fields3d/1"}, {"design", d_.source}, {"order", P.order},
             {"have_pick", P.have_pick}, {"I_pick", P.I_pick}, {"region_of_attr", roa},
             {"body_static", P.vbody_static()}, {"mesh_info", P.mesh_info},
             {"shell", {{"attr", P.sh_attr}, {"R1", P.sh_R1}, {"R2", P.sh_R2},
                        {"center", {P.sh_c[0], P.sh_c[1], P.sh_c[2]}}}},
             {"results", results_}};
  std::ofstream(dir + "/state.json") << st.dump();
}

std::unique_ptr<Model3D> Model3D::load(const std::string& dir) {
  ensure_mpi();
  std::ifstream sfile(dir + "/state.json");
  AXF_REQUIRE(sfile.good(), "no saved 3D solution in '" + dir + "' (state.json missing)", "",
              "Save one with `axf solve design.json --save` or Results3D.save().");
  const json st = json::parse(sfile);
  AXF_REQUIRE(st.value("format", "") == "axionfem.fields3d/1", "unknown saved-solution format in '" + dir + "'");
  auto m = std::make_unique<Model3D>(spec::parse_design(st["design"]));
  Impl& P = *m->p_;
  {
    std::ifstream f(dir + "/mesh.mesh");
    P.serial = std::make_unique<mfem::Mesh>(f, 1, 0, false);  // exactly as written (no reordering)
  }
  P.order = st["order"];
  P.fec = std::make_unique<mfem::ND_FECollection>(P.order, 3);
  P.view_mesh = std::make_unique<mfem::ParMesh>(MPI_COMM_SELF, *P.serial);
  P.view_fes = std::make_unique<mfem::ParFiniteElementSpace>(P.view_mesh.get(), P.fec.get());
  std::vector<int> ident(P.serial->GetNE());
  std::iota(ident.begin(), ident.end(), 0);
  auto get = [&](const std::string& name) {
    std::ifstream f(dir + "/" + name);
    AXF_REQUIRE(f.good(), "saved solution: missing " + name);
    mfem::GridFunction g(P.serial.get(), f);
    auto v = std::make_unique<mfem::ParGridFunction>(P.view_fes.get());
    SerialFieldCoefficient c(*P.serial, g, ident);
    v->ProjectCoefficient(c);
    return v;
  };
  P.view_A_lab = get("A_lab.gf");
  P.have_lab = true;
  P.have_pick = st["have_pick"];
  if (P.have_pick) P.view_A_pick = get("A_pick.gf");
  P.I_pick = st["I_pick"];
  for (auto& e : st["region_of_attr"]) {
    P.region_of_attr[e[0].get<int>()] = e[1].get<std::string>();
    P.attr_of_region[e[1].get<std::string>()] = e[0].get<int>();
  }
  P.view_body_static = st["body_static"].get<std::vector<int>>();
  P.mesh_info = st["mesh_info"];
  P.sh_attr = st["shell"]["attr"];
  P.sh_R1 = st["shell"]["R1"];
  P.sh_R2 = st["shell"]["R2"];
  for (int i = 0; i < 3; ++i) P.sh_c[i] = st["shell"]["center"][i];
  if (!m->d_.background.type.empty()) P.bg = std::make_unique<BackgroundField>(m->d_.background);
  m->results_ = st["results"];
  return m;
}

int mpi_rank() {
  int inited = 0, rank = 0;
  MPI_Initialized(&inited);
  if (inited) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

json mesh_any(const json& doc, const SolveOptions& o) {
  auto d = spec::parse_design(doc);
  if (d.symmetry != "3d" && d.symmetry != "xsec" && !d.mesh.file.empty())  // 2D imported mesh: nothing to mesh
    return {{"ok", true}, {"symmetry", d.symmetry}, {"mesh", d.mesh.file}, {"mesher_info", d.mesh.info_file},
            {"note", "imported mesh (mesh.file): no meshing"}};
  json out = {{"ok", true}};
  if (d.symmetry == "3d") {
    Model3D m(std::move(d));
    SolveOptions om = o;
    om.mesh_only = true;
    m.run(om);
    out["symmetry"] = "3d";
    out["mesh"] = m.results()["mesh"]["file"];
    out["mesher_input"] = o.workdir + "/mesh3d_in.json";
    out["mesher_info"] = o.workdir + "/mesh3d_info.json";
    const json& minfo = m.results().at("mesh").at("info");
    if (minfo.contains("curved_elements_invalid")) out["curved_elements_invalid"] = minfo.at("curved_elements_invalid");
    if (!m.results().value("warnings", json::array()).empty()) out["warnings"] = m.results().at("warnings");
    return out;
  }
  AxiModel m(std::move(d));
  m.mesh(o);
  out["symmetry"] = "axisymmetric";
  out["mesh"] = o.workdir + "/mesh.msh";
  out["mesher_input"] = o.workdir + "/mesh_in.json";
  out["mesher_info"] = o.workdir + "/mesh_info.json";
  return out;
}

json run_any(const json& doc, const SolveOptions& o) {
  auto d = spec::parse_design(doc);
  if (d.symmetry != "3d" && d.symmetry != "xsec" && !d.mesh.file.empty()) {  // 2D full wave, imported mesh
    Fw2DModel m(std::move(d));
    m.run(o);
    return m.results();
  }
  if (d.symmetry == "3d") {
    Model3D m(std::move(d));
    m.run(o);
    return m.results();
  }
  AxiModel m(std::move(d));
  m.run(o);
  return m.results();
}

}  // namespace axf
