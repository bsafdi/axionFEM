#include "shell_map.hpp"
// 3D full-wave readout analysis (time-harmonic Maxwell, e^{-i omega t}) with complex Nedelec E fields:
//
//   curl(nu curl E) - omega^2 eps_c E = i omega J_port,     eps_c = eps (1 + i tan delta) + i sigma / omega,
//
// Superconductors and `conductor` materials are perfect conductors (E = 0 in them). The readout current I
// flows through the port region with the distribution of the MQS pickup solve (J = -sigma grad V,
// sigma = 1), so real slits and gaps are fed at their actual location: unlike the axisymmetric solver
// (uniform feed), non-axisymmetric current patterns and the self-resonances of a slit appear.
//
// Open boundary (exact): the MQS infinity shell [R1, R2) uses a real map to a finite cap with a gauge mass,
// and the DtN update subtracts the matching continuum capped-shell response (the magnetic multipoles; electric ones cost nothing there). The
// dynamic remainder is a low-rank term on the sphere rho = R1, in the orthonormal tangential vector spherical
// harmonics W (curl type: TE, magnetic multipoles; gradient type: TM, electric multipoles), from the outgoing
// solutions h_n(k rho):  int_S (n x nu curl E) . v = sum c (E, W)(v, W), with
//   c_TE = (gamma_n(kR1) - st_n) / (mu0 R1), st_n = [n+(n+1)q]/(1-q), q=cap^(-2n-1),   c_TM = -omega^2 eps0 R1 / gamma_n(kR1),   gamma_n = n - x h_{n-1}/h_n
// (gamma_n -> n as k -> 0, so both corrections vanish in the static limit and are small for n >> kR1). This keeps
// the operator as well conditioned as a closed problem: GMRES with block AMS converges as for a PEC sphere.
//
// Per frequency:  Z = -(1/I^2) int E . J dV = R - i omega L,   Psi(omega) = (1/(i omega I)) int_{rho<R1} E . B_lab dV
// (-> the MQS Psi as omega -> 0), F = |Psi|^4 / L^2, radiation resistance R (lossless media) and Q_rad. An imposed
// background adds (1/(i omega I)) int curl E . A_bg dV inside and outside R1 (axi::BackgroundExterior).
// Eigenmodes: hypre AME for the closed structure (perfectly conducting sphere at the domain radius R1).
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <optional>
#include <set>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include "axionfem/core.hpp"
#include "axionfem/fullwave.hpp"
#include "axionfem/pml.hpp"
#include "axionfem/model3d.hpp"
#include "axionfem/prescribed_source.hpp"
#include "axionfem/rational_fit.hpp"
#ifdef AXF_STRUMPACK_MPI
#include <StrumpackSparseSolverMPIDist.hpp>
#endif
#include "axionfem/sym_solver.hpp"
#include "axionfem/units.hpp"
#include "fw_error_indicator.hpp"
#include "model3d_impl.hpp"
#include "surface_losses3d.hpp"

namespace axf {

using units::mu0;
using units::pi;

namespace {

using cplx = std::complex<double>;

double fw_allsum(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return v;
}

// The full-wave AMS runs MFEM's host options on either device: hybrid Gauss-Seidel smoothing, HMIS coarsening, l1
// hybrid GS relaxation, one aggressive level. MFEM's own defaults when hypre runs on a GPU (l1-Jacobi, PMIS, l1-Jacobi,
// none) cost 2-3 times the GMRES iterations, whose orthogonalization on the host then dominates: a test model at ND2 on
// an H100 took 1347 s against 408 s (docs/validation/gpu.md). AXF_FW_AMS_RLX, AXF_FW_AMS_AMG_COARSEN, AXF_FW_AMS_AMG_RELAX
// and AXF_FW_AMS_AGG override them (solver options only). On the host with none set nothing is called: these are
// MFEM's defaults there. Returns the options in force, for fullwave.solver_settings.ams_options.
json fw_ams_overrides(mfem::HypreAMS& ams) {
  const bool overridden = std::getenv("AXF_FW_AMS_RLX") || std::getenv("AXF_FW_AMS_AMG_COARSEN") ||
                          std::getenv("AXF_FW_AMS_AMG_RELAX") || std::getenv("AXF_FW_AMS_AGG");
  const int rlx = env_int_checked("AXF_FW_AMS_RLX", 2);   // MFEM's host defaults (HypreAMS::MakeSolver)
  const int coarsen = env_int_checked("AXF_FW_AMS_AMG_COARSEN", 10);
  const int relax = env_int_checked("AXF_FW_AMS_AMG_RELAX", 8);
  const int agg = env_int_checked("AXF_FW_AMS_AGG", 1);
  json used = {{"smoother", rlx}, {"amg_coarsen", coarsen}, {"amg_relax", relax}, {"amg_coarse_relax", relax},
               {"amg_agg_levels", agg}, {"amg_theta", 0.25}};
  if (!overridden && !mfem::HypreUsingGPU()) {
    used["set_by"] = "MFEM's defaults on the host";
    return used;
  }
  used["set_by"] = overridden ? "AXF_FW_AMS_* overrides" : "axionFEM: MFEM's host options, hypre on a GPU";
  HYPRE_Solver h = ams;
  HYPRE_AMSSetSmoothingOptions(h, rlx, 1, 1.0, 1.0);
  HYPRE_AMSSetAlphaAMGOptions(h, coarsen, agg, relax, 0.25, 6, 4);
  HYPRE_AMSSetBetaAMGOptions(h, coarsen, agg, relax, 0.25, 6, 4);
  HYPRE_AMSSetAlphaAMGCoarseRelaxType(h, relax);
  HYPRE_AMSSetBetaAMGCoarseRelaxType(h, relax);
  return used;
}
// The ranks on which `failed` is true, agreed on every rank: "" when none, else "0, 3" (I/O errors are decided
// collectively, so that one rank's failure cannot hang the others at the next collective)
std::string fw_failing_ranks(bool failed) {
  int nranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  std::vector<int> all(static_cast<size_t>(nranks), 0);
  int mine = failed ? 1 : 0;
  MPI_Allgather(&mine, 1, MPI_INT, all.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::string list;
  for (int r = 0; r < nranks; ++r)
    if (all[static_cast<size_t>(r)]) list += (list.empty() ? "" : ", ") + std::to_string(r);
  return list;
}

double fw_allmax(double v) {
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return v;
}
// global maximum of v and the point x where it occurs (x from the rank holding the maximum)
double fw_allmax_at(double v, double x[3]) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  struct { double v; int r; } in{v, rank}, out{v, rank};
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  MPI_Bcast(x, 3, MPI_DOUBLE, out.r, MPI_COMM_WORLD);
  return out.v;
}
// glibc's account of its heap (the totals of malloc_info, 64-bit): {bytes the program holds (arena memory less the free
// chunks, plus separately mapped large blocks), free bytes the allocator keeps in its arenas (bins and each arena's top
// chunk, resident or not)}. {0, 0} without glibc. malloc_trim can return the free bins of every arena and the main
// arena's top, not the top of a thread's arena (fw_malloc_fix_thresholds).
std::array<double, 2> fw_heap_bytes() {
  std::array<double, 2> out = {0.0, 0.0};
#if defined(__GLIBC__)
  char* text = nullptr;
  std::size_t length = 0;
  std::FILE* stream = open_memstream(&text, &length);
  if (!stream) return out;
  malloc_info(0, stream);
  std::fclose(stream);
  const std::string info(text, length);
  std::free(text);
  // the process totals follow the last arena's block
  const std::size_t totals = info.rfind("</heap>");
  auto size_of = [&](const char* tag) {
    const std::size_t at = info.find(tag, totals == std::string::npos ? 0 : totals);
    if (at == std::string::npos) return 0.0;
    const std::size_t size = info.find("size=\"", at);
    return size == std::string::npos ? 0.0 : std::atof(info.c_str() + size + 6);
  };
  const double free_bins = size_of("<total type=\"fast\"") + size_of("<total type=\"rest\"");
  out[0] = size_of("<system type=\"current\"") - free_bins + size_of("<total type=\"mmap\"");
  out[1] = free_bins;
#endif
  return out;
}

// This rank's memory: {VmRSS, VmHWM, RssAnon, RssShmem} (Linux /proc/self/status; 0 where it cannot be read) and
// {heap held, heap free} (fw_heap_bytes), in bytes. Not collective.
std::array<double, 6> fw_local_memory() {
  double kib[4] = {0.0, 0.0, 0.0, 0.0};   // VmRSS, VmHWM, RssAnon, RssShmem
  if (std::FILE* status = std::fopen("/proc/self/status", "r")) {
    char line[256];
    const char* keys[4] = {"VmRSS:", "VmHWM:", "RssAnon:", "RssShmem:"};
    while (std::fgets(line, sizeof line, status))
      for (int k = 0; k < 4; ++k)
        if (std::strncmp(line, keys[k], std::strlen(keys[k])) == 0) kib[k] = std::atof(line + std::strlen(keys[k]));
    std::fclose(status);
  }
  const std::array<double, 2> heap = fw_heap_bytes();
  return {1024.0 * kib[0], 1024.0 * kib[1], 1024.0 * kib[2], 1024.0 * kib[3], heap[0], heap[1]};
}

// Memory of this process: VmRSS, the resident set now, its anonymous and shared-memory parts (RssAnon, RssShmem), and
// VmHWM, its peak so far, and the heap glibc holds for the program and keeps free (fw_local_memory), each as the maximum
// over the ranks, their sum and rank zero's (the owner of a gathered factor). Collective. Recorded at the start and end
// of every frequency, it separates what a sweep keeps (heap held) from what the allocator keeps (heap free, resident)
// and from memory outside malloc.
json fw_memory_record() {
  const std::array<double, 6> mine = fw_local_memory();
  const char* names[6] = {"rss", "peak_rss", "rss_anon", "rss_shmem", "heap_held", "heap_free"};
  double rank0[6] = {mine[0], mine[1], mine[2], mine[3], mine[4], mine[5]};
  MPI_Bcast(rank0, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  json record = json::object();
  for (int k = 0; k < 6; ++k) {
    record[strcat(names[k], "_max_rank_bytes")] = fw_allmax(mine[k]);
    record[strcat(names[k], "_sum_bytes")] = fw_allsum(mine[k]);
    record[strcat(names[k], "_rank0_bytes")] = rank0[k];
  }
  return record;
}

// Rank 0's resident set and heap right after a direct factorization and its trim: {VmRSS, heap held, heap free}. The
// factor objects sample it on rank 0 and broadcast it (fw_broadcast_sample); each row records it as
// memory.after_factorization_rank0. What is resident there and not held is the factorization's work memory left over.
std::array<double, 3> fw_rank0_memory_sample() {
  const std::array<double, 6> m = fw_local_memory();
  return {m[0], m[4], m[5]};
}
json fw_broadcast_sample(std::array<double, 3> sample) {   // collective; rank 0's values
  MPI_Bcast(sample.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return {{"rss_bytes", sample[0]}, {"heap_held_bytes", sample[1]}, {"heap_free_bytes", sample[2]}};
}

// The peak resident set of this process in bytes: VmHWM (Linux), else getrusage's ru_maxrss. ru_maxrss also carries
// the peak of the image that exec'd the process (a task started by a cluster launcher can inherit 2-4 GB from it), so it
// overstates a rank's own peak.
double fw_peak_rss_bytes() {
  if (std::FILE* status = std::fopen("/proc/self/status", "r")) {
    char line[256];
    double kib = -1.0;
    while (std::fgets(line, sizeof line, status))
      if (std::strncmp(line, "VmHWM:", 6) == 0) kib = std::atof(line + 6);
    std::fclose(status);
    if (kib >= 0.0) return 1024.0 * kib;
  }
  struct rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss);   // bytes on macOS
#else
  return 1024.0 * usage.ru_maxrss;               // KiB on Linux
#endif
}

// Returning a released factor's memory to the system (docs/validation/fullwave_memory.md). glibc keeps freed memory:
// free chunks in its bins until malloc_trim, and the top chunk of each arena until free() finds it above the trim
// threshold. glibc raises that threshold up to 64 MiB as large blocks are freed, and malloc_trim shrinks only the main
// arena's top. STRUMPACK's OpenMP threads allocate the fronts in their own arenas, so each thread kept up to 64 MiB of a
// freed factor resident, and the more threads per rank, the more.
//   AXF_FW_MALLOC_TRIM: factor (default) calls malloc_trim after every direct factorization (its work memory) and
//     whenever a direct factor is released; frequency also does at the start of every frequency, recording the
//     resident set before and after (a diagnostic); off never does.
//   AXF_FW_MALLOC_THRESHOLDS: fixed (default) sets glibc's trim threshold to 1 MiB and its mmap threshold to 32 MiB
//     (mallopt) before the first direct factorization of a full-wave analysis (a sweep or shift-invert eigenmodes), so
//     that free() shrinks every arena's top. When the analysis ends, also by an exception, FwMallocScope hands the
//     process trim 64 MiB and mmap 32 MiB: glibc's own dynamic maxima, its steady state after a large free, under which
//     later work in the process (numpy temporaries in a Python session, for example) is not slowed
//     by a 1 MiB trim threshold. mallopt has no getter and cannot restore the dynamic adjustment, so the process keeps
//     those values, and the record says so. `glibc` leaves glibc's thresholds alone, as does a run whose environment
//     sets them (MALLOC_TRIM_THRESHOLD_, MALLOC_MMAP_THRESHOLD_ or glibc.malloc tunables).
// Both are needed: the trim for the bins, the threshold for the thread arenas' tops. mallopt is MT-unsafe in glibc. It is
// called on the master thread outside parallel regions, with the OpenMP workers idle; a Python or MPI progress thread
// freeing a mapped block in the same few instructions could leave glibc's dynamic values in force, nothing worse.
constexpr int kFwTrimThreshold = 1 << 20;
constexpr int kFwMmapThreshold = 32 << 20;
constexpr int kFwRelaxedTrimThreshold = 64 << 20;
const char* fw_malloc_trim_mode() {
  const char* v = std::getenv("AXF_FW_MALLOC_TRIM");
  return v && *v ? v : "factor";
}
const char* fw_malloc_thresholds_mode() {
  const char* v = std::getenv("AXF_FW_MALLOC_THRESHOLDS");
  return v && *v ? v : "fixed";
}
bool fw_have_glibc_malloc() {
#if defined(__GLIBC__)
  return true;
#else
  return false;
#endif
}
// the environment sets glibc's thresholds itself
bool fw_malloc_environment() {
  const char* tunables = std::getenv("GLIBC_TUNABLES");
  return std::getenv("MALLOC_TRIM_THRESHOLD_") || std::getenv("MALLOC_MMAP_THRESHOLD_") ||
         (tunables && std::strstr(tunables, "glibc.malloc"));
}
void fw_malloc_trim() {
#if defined(__GLIBC__)
  malloc_trim(0);
#endif
}
void fw_trim_after_factor() {
  if (std::strcmp(fw_malloc_trim_mode(), "off") != 0) fw_malloc_trim();
}
// This process's thresholds (per rank).
struct FwMallocState {
  bool fixed = false;                 // 1 MiB / 32 MiB in force now
  bool fixed_this_analysis = false;   // set during the current analysis
  bool relaxed_before = false;        // an earlier analysis of this process left 64 MiB / 32 MiB in force
};
FwMallocState& fw_malloc_state() {
  static FwMallocState state;
  return state;
}
// before a direct factorization: once per analysis, unless AXF_FW_MALLOC_THRESHOLDS=glibc or the environment sets them
void fw_malloc_fix_thresholds() {
#if defined(__GLIBC__)
  FwMallocState& state = fw_malloc_state();
  if (state.fixed || std::strcmp(fw_malloc_thresholds_mode(), "fixed") != 0 || fw_malloc_environment()) return;
  const bool mmap_set = mallopt(M_MMAP_THRESHOLD, kFwMmapThreshold) == 1;
  const bool trim_set = mallopt(M_TRIM_THRESHOLD, kFwTrimThreshold) == 1;
  state.fixed = mmap_set && trim_set;
  state.fixed_this_analysis = state.fixed_this_analysis || mmap_set || trim_set;
#endif
}
// One per full-wave analysis, on every rank: the thresholds the analysis fixed are relaxed when it ends.
class FwMallocScope {
 public:
  FwMallocScope() {
    FwMallocState& state = fw_malloc_state();
    state.fixed_this_analysis = false;
    before_ = state.relaxed_before ? "trim 67108864 B, mmap 33554432 B, left by an earlier full-wave analysis of this "
                                     "process (glibc's dynamic adjustment no longer in force)"
                                   : "glibc's dynamic thresholds";
    if (fw_malloc_environment())
      before_ = "set by the environment (MALLOC_TRIM_THRESHOLD_, MALLOC_MMAP_THRESHOLD_ or GLIBC_TUNABLES)";
  }
  // runs after the analysis's factors are freed (declared first, destroyed last): their memory goes back too
  ~FwMallocScope() {
#if defined(__GLIBC__)
    FwMallocState& state = fw_malloc_state();
    if (state.fixed) {
      mallopt(M_TRIM_THRESHOLD, kFwRelaxedTrimThreshold);
      mallopt(M_MMAP_THRESHOLD, kFwMmapThreshold);
      state.fixed = false;
      state.relaxed_before = true;
    }
    fw_trim_after_factor();
#endif
  }
  FwMallocScope(const FwMallocScope&) = delete;
  FwMallocScope& operator=(const FwMallocScope&) = delete;
  // fullwave.memory's allocator settings for this analysis: what was in force for it. Collective.
  json Record(const std::string& trim_mode, const std::string& thresholds_mode) const {
    const double ranks = fw_allsum(fw_malloc_state().fixed_this_analysis ? 1.0 : 0.0);
    const std::string no_glibc = "unavailable (not glibc)";
    json record = {{"malloc_trim", fw_have_glibc_malloc() ? trim_mode : no_glibc},
                   {"malloc_thresholds_mode", fw_have_glibc_malloc() ? thresholds_mode : no_glibc},
                   {"malloc_thresholds_before", before_},
                   {"malloc_thresholds_applied_ranks", static_cast<int>(ranks)}};
    if (!fw_have_glibc_malloc())
      record["malloc_thresholds"] = no_glibc;
    else if (ranks > 0)
      record["malloc_thresholds"] = strcat("trim ", kFwTrimThreshold, " B and mmap ", kFwMmapThreshold, " B on ",
                                           ranks, " rank(s) from this analysis's first direct factorization; trim ",
                                           kFwRelaxedTrimThreshold, " B and mmap ", kFwMmapThreshold, " B after it");
    else
      record["malloc_thresholds"] = "not set by this analysis: " + before_;
    return record;
  }

 private:
  std::string before_;
};

// AXF_FW_FREQUENCY_PEAK=1: reset the kernel's peak resident set (VmHWM) to the current one at the start of every
// frequency (Linux /proc/self/clear_refs), so that each row's peak figures are that frequency's own rather than the
// running maximum of the whole run. A diagnostic: it changes what peak_rss_* mean. "" or 0: off.
bool fw_frequency_peak_requested() {
  const char* v = std::getenv("AXF_FW_FREQUENCY_PEAK");
  return v && std::strcmp(v, "1") == 0;
}
// true when the kernel accepted the reset
bool fw_reset_peak_rss() {
  std::FILE* refs = std::fopen("/proc/self/clear_refs", "w");
  if (!refs) return false;
  const bool written = std::fputs("5", refs) >= 0;
  return std::fclose(refs) == 0 && written;
}

double fw_gdot(const mfem::Vector& a, const mfem::Vector& b) { return mfem::InnerProduct(MPI_COMM_WORLD, a, b); }

cplx fw_bilinear(const mfem::Vector& ar, const mfem::Vector& ai,
                 const mfem::Vector& br, const mfem::Vector& bi) {
  return {fw_gdot(ar,br)-fw_gdot(ai,bi), fw_gdot(ar,bi)+fw_gdot(ai,br)};
}

// A basis-dependent cancellation diagnostic, not an error estimate or a
// replacement for the componentwise acceptance criterion.
double fw_absolute_overlap(const mfem::Vector& q, const mfem::Vector& er,
                           const mfem::Vector& ei) {
  double sum = 0.0;
  for (int i = 0; i < q.Size(); ++i) sum += std::abs(q(i))*std::hypot(er(i),ei(i));
  return fw_allsum(sum);
}

json fw_complex_vector_json(const Eigen::VectorXcd& v) {
  json r=json::array(), im=json::array();
  for (int i=0;i<v.size();++i) { r.push_back(v(i).real()); im.push_back(v(i).imag()); }
  return {{"re",r},{"im",im}};
}

json fw_complex_matrix_json(const Eigen::MatrixXcd& v) {
  json r = json::array(), im = json::array();
  for (int j = 0; j < v.rows(); ++j) {
    json rr = json::array(), ii = json::array();
    for (int k = 0; k < v.cols(); ++k) {
      rr.push_back(v(j,k).real()); ii.push_back(v(j,k).imag());
    }
    r.push_back(rr); im.push_back(ii);
  }
  return {{"re", r}, {"im", im}};
}

// All columns are normalized to one incoming terminal ampere. Conjugation is
// essential for energy; reciprocity elsewhere deliberately uses transpose.
Eigen::MatrixXcd fw_energy_gram(const mfem::Operator& metric,
                              const std::vector<mfem::Vector>& er,
                              const std::vector<mfem::Vector>& ei) {
  const int n = static_cast<int>(er.size());
  Eigen::MatrixXcd gram(n,n);
  mfem::Vector mr(metric.Height()), mi(metric.Height());
  for (int k = 0; k < n; ++k) {
    metric.Mult(er[k], mr); metric.Mult(ei[k], mi);
    for (int j = 0; j < n; ++j)
      gram(j,k) = cplx(fw_gdot(er[j],mr) + fw_gdot(ei[j],mi),
                      fw_gdot(er[j],mi) - fw_gdot(ei[j],mr));
  }
  return gram;
}

// Real infinity map of the shell elements (as in the MQS solve): stretch factors s_r = f', s_t = f / rho.
// fmax > 1: the map ends at f = fmax R1 (a perfect conductor there) instead of infinity, which bounds the
// coefficients (better AMS) at a relative cost (R1 / (fmax R1))^(2n+1) in the static multipole of order n.
struct ShellMapR {
  int attr = -1;
  double R1 = 0, R2 = 0, c[3] = {0, 0, 0}, fmax = 0;
  bool in(int a) const { return attr > 0 && a == attr; }
  void factors(const mfem::Vector& x, double& sr, double& st, double rh[3]) const {
    double d[3] = {x(0) - c[0], x(1) - c[1], x(2) - c[2]};
    const double rho = std::max(std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), 1e-300);
    for (int i = 0; i < 3; ++i) rh[i] = d[i] / rho;
    // f = R1 (R2 - R1) / (R2 + delta - rho) - c0 with f(R1) = R1 and f(R2) = fmax R1
    double delta = 0, c0 = 0;
    if (fmax > 1) {
      delta = detail::shell_delta(R2 - R1, fmax);
      c0 = R1 * (R2 - R1) / (R2 + delta - R1) - R1;
    }
    const double gap = std::max(R2 + delta - rho, 1e-12 * R2);
    const double f = R1 * (R2 - R1) / gap - c0;
    sr = R1 * (R2 - R1) / (gap * gap);
    st = f / rho;
  }
};

// Real spherical harmonics Y_nm (orthonormal on the unit sphere; m > 0: cos(m phi), m < 0: sin(|m| phi)).
double real_ylm(int n, int m, double th, double ph) {
  const int am = std::abs(m);
  const double x = std::cos(th), sx = std::sqrt(std::max(0.0, 1.0 - x * x));
  double pmm = 1.0;  // P_am^am (no Condon-Shortley phase)
  for (int k = 1; k <= am; ++k) pmm *= (2.0 * k - 1.0) * sx;
  double p = pmm;
  if (n > am) {
    double p0 = pmm, p1 = x * (2.0 * am + 1.0) * pmm;
    for (int l = am + 2; l <= n; ++l) {
      const double p2 = ((2.0 * l - 1.0) * x * p1 - (l + am - 1.0) * p0) / (l - am);
      p0 = p1;
      p1 = p2;
    }
    p = p1;
  }
  const double nrm = std::sqrt((2.0 * n + 1.0) / (4 * pi) * std::exp(std::lgamma(n - am + 1.0) - std::lgamma(n + am + 1.0)));
  if (m == 0) return nrm * p;
  return std::sqrt(2.0) * nrm * p * (m > 0 ? std::cos(am * ph) : std::sin(am * ph));
}

// Unit-sphere surface gradient of Y_nm at direction rh (Cartesian), by central differences in (theta, phi).
void ylm_surface_grad(int n, int m, const double rh[3], double g[3]) {
  const double th = std::acos(std::clamp(rh[2], -1.0, 1.0)), ph = std::atan2(rh[1], rh[0]);
  const double h = 1e-6;
  const double dth = (real_ylm(n, m, th + h, ph) - real_ylm(n, m, th - h, ph)) / (2 * h);
  const double st = std::max(std::sin(th), 1e-12);
  const double dph = (real_ylm(n, m, th, ph + h) - real_ylm(n, m, th, ph - h)) / (2 * h) / st;
  const double et[3] = {std::cos(th) * std::cos(ph), std::cos(th) * std::sin(ph), -std::sin(th)};
  const double ep[3] = {-std::sin(ph), std::cos(ph), 0.0};
  for (int i = 0; i < 3; ++i) g[i] = dth * et[i] + dph * ep[i];
}

// Material tensor of the full-wave operator. kind 0: nu (curl-curl), 1: eps_c (mass, times `factor`); part 0: real,
// 1: imaginary, 2: magnitude (SPD preconditioner). `scal` is the complex scalar per element. Shell elements carry
// the real infinity map for nu and only the gauge mass `gauge` (the static exterior).
// PML layers of a 3D sweep: layer index per local element (-1: none) and the frequency.
struct FwPml {
  std::vector<int> epml;
  std::vector<pml::Layer> layers;
  double omega = 0;
  bool active() const { return !layers.empty(); }
};

class FwTensor : public mfem::MatrixCoefficient {
 public:
  FwTensor(const ShellMapR& sh, const std::vector<cplx>& scal, int kind, int part, double factor, double gauge,
           const FwPml* pml = nullptr)
      : mfem::MatrixCoefficient(3), sh_(sh), scal_(scal), kind_(kind), part_(part), factor_(factor), gauge_(gauge),
        pml_(pml) {}
  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    K.SetSize(3);
    K = 0.0;
    const cplx s = scal_[T.ElementNo];
    auto pick = [&](cplx z) { return part_ == 0 ? z.real() : part_ == 1 ? z.imag() : std::abs(z); };
    if (pml_ && pml_->active() && pml_->epml[T.ElementNo] >= 0) {
      // stretched-coordinate medium: nu Lambda^{-1} (curl-curl) or eps_c Lambda (mass), times factor
      mfem::Vector x(3);
      T.Transform(ip, x);
      cplx Lt[9], Li[9];
      double La[9], Lia[9];
      pml_->layers[pml_->epml[T.ElementNo]].tensors(x.GetData(), pml_->omega, Lt, Li, La, Lia);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          const int k = 3 * a + b;
          K(a, b) = part_ == 2 ? factor_ * std::abs(s) * (kind_ == 0 ? Lia[k] : La[k])
                               : factor_ * pick(s * (kind_ == 0 ? Li[k] : Lt[k]));
        }
      return;
    }
    if (!sh_.in(T.Attribute)) {
      const double v = factor_ * pick(s);
      for (int i = 0; i < 3; ++i) K(i, i) = v;
      return;
    }
    if (kind_ == 1) {  // gauge mass only, with the sign of the physical mass term (factor): gradient fields
                       // straddling R1 then see one sign in the operator and the other in the preconditioner
                       // (ratio -1), instead of preconditioned eigenvalues spread through zero
      for (int i = 0; i < 3; ++i) K(i, i) = part_ == 1 ? 0.0 : (factor_ < 0 && part_ == 0 ? -gauge_ : gauge_);
      return;
    }
    if (part_ == 1) return;
    mfem::Vector x(3);
    T.Transform(ip, x);
    double sr, st, rh[3];
    sh_.factors(x, sr, st, rh);
    const double nu = s.real(), vr = nu * sr / (st * st), vt = nu / sr;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) K(i, j) = (i == j ? vt : 0.0) + (vr - vt) * rh[i] * rh[j];
  }

 private:
  const ShellMapR& sh_;
  const std::vector<cplx>& scal_;
  int kind_, part_;
  double factor_, gauge_;
  const FwPml* pml_;
};

// The PML layers' part of dA/dw (the exact dZ/dw): kind 0, the curl-curl term nu d(Lambda^{-1})/dw; kind 1, the mass
// term d(-w^2 eps_c Lambda)/dw = deps Lambda - w^2 eps_c dLambda/dw, with deps = d(-w^2 eps_c)/dw. Zero outside the
// layers (the volume mass there is FwTensor's, unchanged). part 0 real, 1 imaginary.
class FwPmlTensorDw : public mfem::MatrixCoefficient {
 public:
  FwPmlTensorDw(const FwPml& pml, const std::vector<cplx>& nu, const std::vector<cplx>& eps,
                const std::vector<cplx>& deps, int kind, int part)
      : mfem::MatrixCoefficient(3), pml_(pml), nu_(nu), eps_(eps), deps_(deps), kind_(kind), part_(part) {}
  void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    K.SetSize(3);
    K = 0.0;
    if (!pml_.active() || pml_.epml[T.ElementNo] < 0) return;
    mfem::Vector x(3);
    T.Transform(ip, x);
    cplx Lt[9], Li[9], dLt[9], dLi[9];
    pml_.layers[pml_.epml[T.ElementNo]].tensors(x.GetData(), pml_.omega, Lt, Li, nullptr, nullptr, dLt, dLi);
    const int e = T.ElementNo;
    const double w = pml_.omega;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const int k = 3 * a + b;
        const cplx v = kind_ == 0 ? nu_[e] * dLi[k] : deps_[e] * Lt[k] - w * w * eps_[e] * dLt[k];
        K(a, b) = part_ == 0 ? v.real() : v.imag();
      }
  }

 private:
  const FwPml& pml_;
  const std::vector<cplx>& nu_;
  const std::vector<cplx>& eps_;
  const std::vector<cplx>& deps_;
  int kind_, part_;
};

// The real symmetric factor of K - w^2 M that preconditions GMRES on the closed, lossless direct path: on rank zero
// (FwRealFactorization) or distributed over the ranks (FwDistFactorization).
class FwDirectFactor : public mfem::Solver {
 public:
  using mfem::Solver::Solver;
  // runs after the derived class has freed its factor: hand that memory back (AXF_FW_MALLOC_TRIM)
  ~FwDirectFactor() override { fw_trim_after_factor(); }
  virtual long FactorNonzeros() const = 0;
  virtual const json& Inertia() const = 0;
  virtual const json& Compression() const = 0;
  virtual json Layout() const = 0;  // where the factor lives, for direct_factorization
  // rank 0's memory right after the factorization and its trim (fw_rank0_memory_sample)
  const json& MemoryAfterFactorization() const { return memory_after_factorization_; }

 protected:
  json memory_after_factorization_;
};

// Symmetric diagonal equilibration 1/sqrt(max(|a_rr|, 0.1 max_c |a_rc|)) of one row.
static double fw_row_scale(const int* cols, const double* values, int length, int row) {
  double diagonal=0., maximum=0.;
  for(int k=0;k<length;++k) {
    maximum=std::max(maximum,std::abs(values[k]));
    if(cols[k]==row) diagonal=values[k];
  }
  AXF_REQUIRE(maximum>0 && std::isfinite(maximum),"zero/nonfinite row in direct full-wave matrix", "/fullwave");
  return 1./std::sqrt(std::max(std::abs(diagonal),0.1*maximum));
}

// Opt-in lossy (block low-rank) compression of the factor: GMRES refines against the exact distributed
// operator, so a compressed factor changes the iteration count, not the accepted answer (gates unchanged).
static SymSolver::Compression fw_compression(json& record) {
  SymSolver::Compression comp;
  if (const char* e = std::getenv("AXF_FW_STRUMPACK_COMPRESSION"); e && *e) comp.type = e;
  if (const char* e = std::getenv("AXF_FW_STRUMPACK_COMPRESSION_TOL"); e && *e) comp.rel_tol = std::stod(e);
  if (const char* e = std::getenv("AXF_FW_STRUMPACK_COMPRESSION_MIN_SEP"); e && *e) comp.min_sep_size = std::stoi(e);
  record = {{"type", comp.type}, {"rel_tol", comp.rel_tol}, {"min_sep_size", comp.min_sep_size},
            {"variables", "AXF_FW_STRUMPACK_COMPRESSION (none|blr), _TOL, _MIN_SEP"}};
  return comp;
}

// where a rejected frequency's diagnostics (and a failed solve's mesh check) are kept: retain_rejected
static const char* const kRejectedDiagnostics =
    "the rejected-frequency diagnostics (fullwave_rejected.json, or f_<i>_rejected.json beside the saved fields)";

// Collective failure check: every rank throws, with the message of the lowest failing rank.
static void fw_require_all(const std::string& failure, const std::string& what) {
  int rank=0, ranks=1;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  int mine=failure.empty() ? ranks : rank, first=ranks;
  MPI_Allreduce(&mine,&first,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
  if(first==ranks) return;
  std::string message=failure;
  int size=static_cast<int>(message.size());
  MPI_Bcast(&size,1,MPI_INT,first,MPI_COMM_WORLD);
  message.resize(size);
  if(size) MPI_Bcast(message.data(),size,MPI_CHAR,first,MPI_COMM_WORLD);
  // every rank throws the same failure: Model3D::fullwave_analysis appends the mesh check (collective)
  throw SolverFailure(what+message,"/fullwave");
}

// A reusable real sparse factorization for closed, lossless PEC problems. The
// matrix rows and RHS are gathered in true-DOF rank order. Only rank zero owns
// the factor; both complex blocks reuse it. GMRES still performs refinement
// against the original distributed operator, including validation load updates.
class FwRealFactorization : public FwDirectFactor {
 public:
  // blocks: right-hand-side blocks per Mult (2: real and imaginary parts; 1: a real solve, e.g. shift-invert)
  explicit FwRealFactorization(mfem::HypreParMatrix& matrix, int blocks = 2)
      : FwDirectFactor(blocks*matrix.Height()), local_n_(matrix.Height()), blocks_(blocks) {
    fw_malloc_fix_thresholds();
    MPI_Comm_rank(MPI_COMM_WORLD,&rank_);
    int ranks; MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    counts_.resize(ranks); offsets_.resize(ranks);
    MPI_Allgather(&local_n_,1,MPI_INT,counts_.data(),1,MPI_INT,MPI_COMM_WORLD);
    long long total = 0;
    for (int r=0;r<ranks;++r) { offsets_[r]=static_cast<int>(total); total+=counts_[r]; }
    AXF_REQUIRE(total > 0 && total <= std::numeric_limits<int>::max() &&
                total == matrix.GetGlobalNumRows() && total == matrix.GetGlobalNumCols(),
                "direct full-wave factorization needs a square matrix with 32-bit global indices", "/fullwave");
    global_n_=static_cast<int>(total);
    mfem::SparseMatrix rows;
    matrix.MergeDiagAndOffd(rows);
    const int local_nnz=rows.NumNonZeroElems();
    std::vector<int> nnz_counts(ranks), nnz_offsets(ranks), row_lengths(local_n_);
    MPI_Allgather(&local_nnz,1,MPI_INT,nnz_counts.data(),1,MPI_INT,MPI_COMM_WORLD);
    total=0;
    for(int r=0;r<ranks;++r) { nnz_offsets[r]=static_cast<int>(total); total+=nnz_counts[r]; }
    AXF_REQUIRE(total <= std::numeric_limits<int>::max(),
                "direct full-wave factorization exceeds 32-bit gathered nonzero count", "/fullwave");
    const int global_nnz=static_cast<int>(total);
    for(int r=0;r<local_n_;++r) row_lengths[r]=rows.GetI()[r+1]-rows.GetI()[r];
    std::vector<int> all_lengths(rank_==0?global_n_:0), all_cols(rank_==0?global_nnz:0);
    std::vector<double> all_values(rank_==0?global_nnz:0);
    MPI_Gatherv(row_lengths.data(),local_n_,MPI_INT,all_lengths.data(),counts_.data(),offsets_.data(),MPI_INT,0,MPI_COMM_WORLD);
    MPI_Gatherv(rows.GetJ(),local_nnz,MPI_INT,all_cols.data(),nnz_counts.data(),nnz_offsets.data(),MPI_INT,0,MPI_COMM_WORLD);
    MPI_Gatherv(rows.GetData(),local_nnz,MPI_DOUBLE,all_values.data(),nnz_counts.data(),nnz_offsets.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    const SymSolver::Compression comp=fw_compression(compression_);
    std::string failure;
    std::array<double,3> after{};
    if(rank_==0) {
      try {
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(global_nnz);
        scale_.resize(global_n_);
        int p=0;
        for(int r=0;r<global_n_;++r) {
          scale_(r)=fw_row_scale(all_cols.data()+p,all_values.data()+p,all_lengths[r],r);
          p+=all_lengths[r];
        }
        p=0;
        for(int r=0;r<global_n_;++r)
          for(int k=0;k<all_lengths[r];++k,++p)
            entries.emplace_back(r,all_cols[p],all_values[p]*scale_(r)*scale_(all_cols[p]));
        Eigen::SparseMatrix<double> scaled(global_n_,global_n_);
        scaled.setFromTriplets(entries.begin(),entries.end());
        scaled.makeCompressed();
        // the gathered copies are not needed any more; free them before the factor is allocated
        std::vector<Eigen::Triplet<double>>().swap(entries);
        std::vector<double>().swap(all_values);
        std::vector<int>().swap(all_cols);
        std::vector<int>().swap(all_lengths);
        factor_.set_backend(1);
        factor_.set_compression(comp);
        factor_.compute(scaled);
        AXF_REQUIRE(factor_.info()==Eigen::Success,"STRUMPACK full-wave factorization failed", "/fullwave");
        fw_trim_after_factor();   // the factorization's work memory (contribution blocks, copies), now free
        after=fw_rank0_memory_sample();
        factor_nonzeros_=factor_.factor_nonzeros();
        SymSolver::InertiaResult skipped;
        skipped.status = "SKIPPED_COMPRESSED_FACTOR";
        const auto inertia = comp.type == "none" ? factor_.inertia() : skipped;
        inertia_ = {{"status",inertia.status},{"accurate",inertia.accurate},
                    {"negative",inertia.accurate ? json(inertia.negative) : json(nullptr)},
                    {"zero",inertia.accurate ? json(inertia.zero) : json(nullptr)},
                    {"positive",inertia.accurate ? json(inertia.positive) : json(nullptr)},
                    {"scope","discrete factor only; a justified curl-free kernel count is required before inferring a pole bound; no continuum certification"}};
        rhs_.resize(global_n_); solution_.resize(global_n_);
      } catch(const std::exception& error) { failure=error.what(); }
    }
    int size=static_cast<int>(failure.size());
    MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);
    failure.resize(size);
    if(size) MPI_Bcast(failure.data(),size,MPI_CHAR,0,MPI_COMM_WORLD);
    if(!failure.empty()) throw SolverFailure("direct full-wave setup failed: "+failure,"/fullwave");  // every rank
    MPI_Bcast(&factor_nonzeros_,1,MPI_LONG,0,MPI_COMM_WORLD);
    memory_after_factorization_=fw_broadcast_sample(after);
    std::string inertia_text = rank_ == 0 ? inertia_.dump() : std::string();
    int inertia_size = static_cast<int>(inertia_text.size());
    MPI_Bcast(&inertia_size,1,MPI_INT,0,MPI_COMM_WORLD);
    inertia_text.resize(inertia_size);
    MPI_Bcast(inertia_text.data(),inertia_size,MPI_CHAR,0,MPI_COMM_WORLD);
    if(rank_ != 0) inertia_ = json::parse(inertia_text);
  }
  // The factor remains the unchanged unloaded operator when GMRES checks a
  // low-rank loaded operator. This is a preconditioner, not its replacement.
  void SetOperator(const mfem::Operator&) override {}
  void Mult(const mfem::Vector& b,mfem::Vector& x) const override {
    x.SetSize(blocks_*local_n_);
    for(int part=0;part<blocks_;++part) {
      MPI_Gatherv(b.GetData()+part*local_n_,local_n_,MPI_DOUBLE,rhs_.data(),counts_.data(),offsets_.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
      std::string failure;
      if(rank_==0) {
        try {
          const Eigen::Map<const Eigen::VectorXd> input(rhs_.data(),global_n_);
          const Eigen::VectorXd scaled_rhs=scale_.array()*input.array();
          const Eigen::VectorXd answer=factor_.solve(scaled_rhs);
          AXF_REQUIRE(answer.size()==global_n_ && answer.allFinite(),
                      "STRUMPACK returned a wrong-sized or nonfinite solution", "/fullwave");
          Eigen::Map<Eigen::VectorXd> output(solution_.data(),global_n_);
          output=scale_.array()*answer.array();
          AXF_REQUIRE(output.allFinite(),"nonfinite equilibrated STRUMPACK solution", "/fullwave");
        } catch(const std::exception& error) { failure=error.what(); }
      }
      int size=static_cast<int>(failure.size());
      MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);
      failure.resize(size);
      if(size) MPI_Bcast(failure.data(),size,MPI_CHAR,0,MPI_COMM_WORLD);
      if(!failure.empty()) throw SolverFailure("direct full-wave solve failed: "+failure,"/fullwave");  // every rank
      MPI_Scatterv(solution_.data(),counts_.data(),offsets_.data(),MPI_DOUBLE,x.GetData()+part*local_n_,local_n_,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
  }
  long FactorNonzeros() const override { return factor_nonzeros_; }
  const json& Inertia() const override { return inertia_; }
  const json& Compression() const override { return compression_; }
  json Layout() const override {
    return {{"backend","STRUMPACK"}, {"factor_owner_rank",0}, {"factor_ranks",1},
            {"gathered_matrix","real symmetric Maxwell block; identical factor for both complex parts"}};
  }
 private:
  int rank_=0,local_n_=0,global_n_=0,blocks_=2;
  long factor_nonzeros_=0;
  std::vector<int> counts_,offsets_;
  Eigen::VectorXd scale_;
  SymSolver factor_;
  json inertia_, compression_;
  mutable std::vector<double> rhs_,solution_;
};

#ifdef AXF_STRUMPACK_MPI
// Every call goes through the collective STRUMPACK guard (sym_solver.hpp, strumpack_guarded_collective_call/_life):
// one caller per rank, on the rank's main thread (thread number 0, which the timer vector always covers).  With the
// serialized guard each call holds the process-wide STRUMPACK lock across its collectives.  With the concurrent guard
// (8.0.x) the calls just run, and the constructor takes the task cutoff from params::num_threads, which that guard
// sets to the OpenMP thread count of the last guarded call outside a parallel region (the load-time value before
// any): the rank's own thread count in practice.
using FwDistSolver = strumpack::SparseSolverMPIDist<double,int>;
// destroys a distributed solver under the guard, also when the constructor that built it throws
struct FwDistDelete {
  void operator()(FwDistSolver* s) const { strumpack_guarded_collective_life([&] { delete s; }); }
};
using FwDistHandle = std::unique_ptr<FwDistSolver,FwDistDelete>;

// As SymSolver::Strumpack::rc_name, so that both layouts report the same status names.
static std::string fw_rc_name(strumpack::ReturnCode rc) {
  switch (rc) {
    case strumpack::ReturnCode::SUCCESS: return "SUCCESS";
    case strumpack::ReturnCode::MATRIX_NOT_SET: return "MATRIX_NOT_SET";
    case strumpack::ReturnCode::REORDERING_ERROR: return "REORDERING_ERROR";
    case strumpack::ReturnCode::ZERO_PIVOT: return "ZERO_PIVOT";
    case strumpack::ReturnCode::NO_CONVERGENCE: return "NO_CONVERGENCE";
    default: return "INACCURATE_INERTIA";
  }
}

static void fw_dist_options(FwDistSolver& s, const SymSolver::Compression& comp) {
  auto& o=s.options();
  o.set_verbose(false);
  o.set_reordering_method(strumpack::ReorderingStrategy::METIS);  // on the gathered graph (root)
  o.set_matching(strumpack::MatchingJob::NONE);                    // symmetric permutations only (inertia)
  o.set_Krylov_solver(strumpack::KrylovSolver::DIRECT);
  if(comp.type=="blr") {
    o.set_compression(strumpack::CompressionType::BLR);
    o.set_compression_rel_tol(comp.rel_tol);
    if(comp.min_sep_size>=0) o.set_compression_min_sep_size(comp.min_sep_size);
  } else {
    AXF_REQUIRE(comp.type=="none","unknown STRUMPACK compression '"+comp.type+"'","/fullwave","Use none or blr.");
    o.set_compression(strumpack::CompressionType::NONE);
  }
}

// STRUMPACK's distributed inertia, checked once (collectively) on a diagonally dominant tridiagonal matrix with
// four rows per rank, diagonal -2, +2, -2, ... and off-diagonal 0.25: its inertia is the sign count of the diagonal.
static bool fw_dist_inertia_verified() {
  static const bool verified=[]() {
    int rank=0, ranks=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    const int m=4, n=m*ranks;
    std::vector<int> dist(ranks+1), ptr{0}, col;
    std::vector<double> val;
    for(int r=0;r<=ranks;++r) dist[r]=m*r;
    for(int i=m*rank;i<m*(rank+1);++i) {
      if(i>0) {
        col.push_back(i-1);
        val.push_back(0.25);
      }
      col.push_back(i);
      val.push_back(i%2==0 ? -2.0 : 2.0);
      if(i<n-1) {
        col.push_back(i+1);
        val.push_back(0.25);
      }
      ptr.push_back(static_cast<int>(col.size()));
    }
    int ok=0;
    // built, used and destroyed in one go; every rank makes every (collective) call, and the local return codes are
    // combined afterwards
    strumpack_guarded_collective_call([&] {
      FwDistSolver s(MPI_COMM_WORLD,false);
      fw_dist_options(s,SymSolver::Compression{});
      s.set_distributed_csr_matrix(m,ptr.data(),col.data(),val.data(),dist.data(),true);
      const bool reordered=s.reorder()==strumpack::ReturnCode::SUCCESS;
      const bool factored=s.factor()==strumpack::ReturnCode::SUCCESS;
      int negative=0, zero=0, positive=0;
      const bool counted=s.inertia(negative,zero,positive)==strumpack::ReturnCode::SUCCESS;
      ok=reordered && factored && counted && negative==n/2 && zero==0 && positive==n/2;
    });
    int all=0;
    MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    return all==1;
  }();
  return verified;
}

// The same factor distributed over the ranks (STRUMPACK SparseSolverMPIDist). Each rank hands over its own rows of
// the true-DOF distribution; METIS orders the gathered graph on the root, and the multifrontal factor and the solves
// are spread over all ranks. Neither the matrix nor a right-hand side is gathered, so rank zero does not hold
// the whole factor. Same equilibration and options as FwRealFactorization; both complex parts are one 2-column solve.
class FwDistFactorization : public FwDirectFactor {
 public:
  explicit FwDistFactorization(mfem::HypreParMatrix& matrix)
      : FwDirectFactor(2*matrix.Height()), local_n_(matrix.Height()) {
    fw_malloc_fix_thresholds();
    MPI_Comm_rank(MPI_COMM_WORLD,&rank_);
    MPI_Comm_size(MPI_COMM_WORLD,&ranks_);
    std::vector<int> counts(ranks_);
    MPI_Allgather(&local_n_,1,MPI_INT,counts.data(),1,MPI_INT,MPI_COMM_WORLD);
    dist_.assign(ranks_+1,0);
    long long total=0;
    for(int r=0;r<ranks_;++r) {
      dist_[r]=static_cast<int>(total);
      total+=counts[r];
    }
    AXF_REQUIRE(total>0 && total<=std::numeric_limits<int>::max() &&
                total==matrix.GetGlobalNumRows() && total==matrix.GetGlobalNumCols(),
                "direct full-wave factorization needs a square matrix with 32-bit global indices", "/fullwave");
    dist_[ranks_]=static_cast<int>(total);
    // rows in rank order, and the same partition for the columns (the diagonal is found by global row index)
    int in_order=matrix.GetRowStarts()[0]==dist_[rank_] && matrix.GetColStarts()[0]==matrix.GetRowStarts()[0];
    int all_in_order=0;
    MPI_Allreduce(&in_order,&all_in_order,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    AXF_REQUIRE(all_in_order==1,"internal: the full-wave matrix rows or columns are not partitioned in rank order",
                "/fullwave");
    // STRUMPACK is told the pattern is symmetric (no symmetrization), so check the matrix is, as SymSolver does
    // for the rank-zero path: two probes, |A x - A^T x| <= 1e-10 (|A x| + |A^T x|) in the max norm
    for(int probe=0;probe<2;++probe) {
      mfem::Vector x(local_n_), ax(local_n_), atx(local_n_);
      x.Randomize(12345+7919*probe+rank_);
      matrix.Mult(x,ax);
      matrix.MultTranspose(x,atx);
      double local[3]={0.,0.,0.}, global[3]={0.,0.,0.};
      for(int i=0;i<local_n_;++i) {
        local[0]=std::max(local[0],std::abs(ax(i)-atx(i)));
        local[1]=std::max(local[1],std::abs(ax(i)));
        local[2]=std::max(local[2],std::abs(atx(i)));
      }
      MPI_Allreduce(local,global,3,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      AXF_REQUIRE(global[0]<=1e-10*(global[1]+global[2]),
                  "internal: the direct full-wave matrix is not symmetric to round-off","/fullwave");
    }
    const SymSolver::Compression comp=fw_compression(compression_);
    mfem::SparseMatrix rows;
    matrix.MergeDiagAndOffd(rows);   // own rows, global column indices
    rows.SortColumnIndices();
    // this STRUMPACK and METIS index with int: the global nonzero count must fit
    long long local_nnz=rows.NumNonZeroElems(), global_nnz=0;
    MPI_Allreduce(&local_nnz,&global_nnz,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
    AXF_REQUIRE(global_nnz<=std::numeric_limits<int>::max(),
                strcat("the direct full-wave matrix has ",global_nnz," nonzeros, beyond the 32-bit indices (2^31 - 1) "
                       "of the distributed STRUMPACK and METIS"),
                "/fullwave","A STRUMPACK with integer_t = int64_t and a 64-bit METIS would be needed; or use AMS.");
    const int* I=rows.GetI();
    const int* J=rows.GetJ();
    double* V=rows.GetData();
    // equilibration: own rows here, every column's scale from one all-gather (8 bytes per global row and rank)
    std::string failure;
    scale_.SetSize(local_n_);
    try {
      for(int r=0;r<local_n_;++r) scale_(r)=fw_row_scale(J+I[r],V+I[r],I[r+1]-I[r],dist_[rank_]+r);
    } catch(const std::exception& error) { failure=error.what(); }
    fw_require_all(failure,"direct full-wave setup failed: ");
    {
      std::vector<double> all_scale(dist_[ranks_]);
      MPI_Allgatherv(scale_.GetData(),local_n_,MPI_DOUBLE,all_scale.data(),counts.data(),dist_.data(),MPI_DOUBLE,
                     MPI_COMM_WORLD);
      for(int r=0;r<local_n_;++r)
        for(int k=I[r];k<I[r+1];++k) V[k]*=scale_(r)*all_scale[J[k]];
    }
    strumpack_guarded_collective_life([&] { solver_.reset(new FwDistSolver(MPI_COMM_WORLD,false)); });
    fw_dist_options(*solver_,comp);
    strumpack_guarded_collective_call([&] {
      solver_->set_distributed_csr_matrix(local_n_,I,J,V,dist_.data(),true);  // copied by STRUMPACK
    });
    rows.Clear();
    auto ordered=strumpack::ReturnCode::SUCCESS;
    strumpack_guarded_collective_call([&] { ordered=solver_->reorder(); });
    if(ordered!=strumpack::ReturnCode::SUCCESS) failure=strcat("ordering failed (STRUMPACK code ",static_cast<int>(ordered),")");
    fw_require_all(failure,"distributed STRUMPACK: ");
    auto factored=strumpack::ReturnCode::SUCCESS;
    strumpack_guarded_collective_call([&] { factored=solver_->factor(); });
    if(factored!=strumpack::ReturnCode::SUCCESS) failure=strcat("factorization failed (STRUMPACK code ",static_cast<int>(factored),")");
    fw_require_all(failure,"distributed STRUMPACK: ");
    fw_trim_after_factor();   // the factorization's work memory (contribution blocks, copies), now free
    memory_after_factorization_=fw_broadcast_sample(fw_rank0_memory_sample());
    strumpack_guarded_collective_call([&] {
      factor_nonzeros_=static_cast<long>(solver_->factor_nonzeros());   // summed over the ranks
    });
    // the same record as the rank-zero path (SymSolver::inertia): status names, null counts unless accurate
    inertia_={{"status",""},{"accurate",false},{"negative",nullptr},{"zero",nullptr},{"positive",nullptr},
              {"scope","discrete factor only; a justified curl-free kernel count is required before inferring a pole bound; no continuum certification"}};
    if(comp.type!="none") inertia_["status"]="SKIPPED_COMPRESSED_FACTOR";
    else if(!fw_dist_inertia_verified()) inertia_["status"]="BACKEND_INERTIA_SELF_CHECK_FAILED";
    else {
      int negative=-1, zero=-1, positive=-1, code=0;
      strumpack_guarded_collective_call([&] {
        code=static_cast<int>(solver_->inertia(negative,zero,positive));   // counts summed over the ranks
      });
      int worst=0;
      MPI_Allreduce(&code,&worst,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);         // the returned code is local
      const auto rc=static_cast<strumpack::ReturnCode>(worst);
      inertia_["status"]=fw_rc_name(rc);
      if(rc==strumpack::ReturnCode::SUCCESS) {
        if(negative<0 || zero<0 || positive<0 || static_cast<long long>(negative)+zero+positive!=total) {
          inertia_["status"]="INCONSISTENT_COUNTS";
        } else {
          inertia_["accurate"]=true;
          inertia_["negative"]=negative;
          inertia_["zero"]=zero;
          inertia_["positive"]=positive;
        }
      }
    }
  }
  void SetOperator(const mfem::Operator&) override {}
  void Mult(const mfem::Vector& b,mfem::Vector& x) const override {
    x.SetSize(2*local_n_);
    const int ld=std::max(local_n_,1);   // a rank without rows still passes valid (unused) storage
    rhs_.assign(2*ld,0.0);
    solution_.assign(2*ld,0.0);
    for(int part=0;part<2;++part)
      for(int r=0;r<local_n_;++r) rhs_[part*ld+r]=scale_(r)*b(part*local_n_+r);
    auto code=strumpack::ReturnCode::SUCCESS;
    strumpack_guarded_collective_call([&] { code=solver_->solve(2,rhs_.data(),ld,solution_.data(),ld); });
    int bad=code!=strumpack::ReturnCode::SUCCESS;
    for(int part=0;part<2;++part)
      for(int r=0;r<local_n_;++r) {
        x(part*local_n_+r)=scale_(r)*solution_[part*ld+r];
        if(!std::isfinite(x(part*local_n_+r))) bad=1;
      }
    int any=0;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    AXF_REQUIRE(any==0,"distributed STRUMPACK returned an error or a nonfinite solution","/fullwave");
  }
  long FactorNonzeros() const override { return factor_nonzeros_; }
  const json& Inertia() const override { return inertia_; }
  const json& Compression() const override { return compression_; }
  json Layout() const override {
    return {{"backend","STRUMPACK MPI (distributed)"}, {"factor_ranks",ranks_},
            {"distributed_matrix","real symmetric Maxwell block in the true-DOF row distribution, not gathered; "
                                  "identical factor for both complex parts"},
            {"ordering","METIS on the gathered graph (root rank)"}};
  }
 private:
  int rank_=0,ranks_=1,local_n_=0;
  long factor_nonzeros_=0;
  std::vector<int> dist_;
  mfem::Vector scale_;
  FwDistHandle solver_;
  json inertia_, compression_;
  mutable std::vector<double> rhs_,solution_;
};
#endif

// AXF_FW_STRUMPACK_DIST: auto (default: distributed when built with an MPI STRUMPACK and run on several ranks),
// 1 (distributed, also on one rank), 0 (the rank-zero factor). Checked when the analysis starts.
static std::string fw_dist_mode() {
  std::string mode="auto";
  if(const char* e=std::getenv("AXF_FW_STRUMPACK_DIST"); e && *e) mode=e;
  AXF_REQUIRE(mode=="auto" || mode=="0" || mode=="1","AXF_FW_STRUMPACK_DIST must be auto, 0 or 1","/fullwave");
#ifndef AXF_STRUMPACK_MPI
  AXF_REQUIRE(mode!="1","AXF_FW_STRUMPACK_DIST=1 needs a build with an MPI STRUMPACK (CMake AXF_STRUMPACK_PREFIX)",
              "/fullwave");
#endif
  return mode;
}

static std::unique_ptr<FwDirectFactor> fw_make_direct_factor(mfem::HypreParMatrix& matrix) {
#ifdef AXF_STRUMPACK_MPI
  const std::string mode=fw_dist_mode();
  int ranks=1;
  MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  if(mode=="1" || (mode=="auto" && ranks>1)) return std::make_unique<FwDistFactorization>(matrix);
#endif
  return std::make_unique<FwRealFactorization>(matrix);
}


// ---------------------------------------------------------------------------------------------------------
// Low-frequency-stable direct solve (AXF_FW_GAUGE=tree-cotree; closed lossless PEC, STRUMPACK path only).
//
// Why: with E gradient-dominated (k h << 1), the summed real matrix A = K - w^2 M keeps only about
// log10((k h)^2 / eps) digits of the mass term, and A x cancels in floating point because K G = 0 only in
// exact arithmetic. The true residual of the summed form then has a floor ~ eps |K| |E_grad| / |b| ~
// eps / (k h)^2 for ANY solver, and the stored matrix itself perturbs the gradient (capacitive) block at
// the same relative level.
//
// What: an exact, square, nonsingular change of basis of the free (non-PEC) edge unknowns,
//     E = S_c e + G_c phi,
// where S_c injects cotree-edge coefficients and G_c is the lowest-order discrete gradient of nodal potentials
// on the edge graph in which every PEC-connected set of nodes (connected through PEC/essential edges) is
// contracted to one node: each PEC conductor carries one potential, one node per connected component is
// grounded (the largest contracted PEC node, normally the outer wall). The spanning tree is a BFS tree of the
// contracted free-edge graph; tree edges <-> phi unknowns one to one. With K G_c = 0 used exactly,
//     P^T A P = [[K_cc - w^2 M_cc, -w^2 (M G_c)_c], [sym, -w^2 G_c^T M G_c]],
// none of whose entries involve K applied to a gradient (the potential block is negative definite; the cotree
// block is K_cc - w^2 M_cc, positive definite at low frequency for the meshes checked, not proven in general).
// The same split is used for every residual: A E = K E_c - w^2 (M E_c + (M G_c) phi), E_c = S_c e, so K never
// acts on the gradient part; the gate is the residual of the same discrete equations, evaluated stably.
// Low-rank loads U diag(c) U^T are applied exactly by the Woodbury identity in the transformed space; the
// solve is iterative refinement on the transformed residual r_y = P^T b - A_y y (no K in its potential rows)
// until it stagnates; by default (AXF_FW_TC_RESIDUAL=extended) that residual is the composed operators'
// (K, M and G_c), accumulated in long double, and the loop stops on the correction's size. The acceptance gate is the
// original-space residual evaluated in the split form above.
// Topology: with the WHOLE boundary PEC (essential), ker curl / grad is H^1(Omega, dOmega) ~ H_2(Omega), spanned by
// potentials constant on each boundary component, i.e. exactly the contracted conductor potentials here; conductor
// handles only enter H^1(Omega), outside the PEC-constrained kernel. With natural (PMC) boundary parts, relative
// cohomology H^1(Omega, Gamma_PEC) can leave harmonic fields in the cotree block (not handled). Lowest-order ND only.
// ---------------------------------------------------------------------------------------------------------
struct FwRows {  // rows gathered to rank 0 in true-DOF rank order; global column indices
  std::vector<int> ptr, col;
  std::vector<double> val;
};

FwRows fw_gather_rows(mfem::HypreParMatrix& A, const std::vector<int>& counts, const std::vector<int>& offsets,
                      int rank) {
  const int ranks = static_cast<int>(counts.size());
  mfem::SparseMatrix rows;
  A.MergeDiagAndOffd(rows);
  const int local_n = rows.Height();
  const int local_nnz = rows.NumNonZeroElems();
  std::vector<int> nnz_counts(ranks), nnz_offsets(ranks), lengths(local_n);
  MPI_Allgather(&local_nnz, 1, MPI_INT, nnz_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  long long total = 0;
  for (int r = 0; r < ranks; ++r) { nnz_offsets[r] = static_cast<int>(total); total += nnz_counts[r]; }
  AXF_REQUIRE(total <= std::numeric_limits<int>::max(), "tree-cotree gather exceeds 32-bit nonzero count", "/fullwave");
  long long global_n = 0;
  for (int c : counts) global_n += c;
  for (int r = 0; r < local_n; ++r) lengths[r] = rows.GetI()[r + 1] - rows.GetI()[r];
  FwRows g;
  std::vector<int> all_lengths(rank == 0 ? global_n : 0);
  if (rank == 0) { g.col.resize(total); g.val.resize(total); g.ptr.assign(global_n + 1, 0); }
  MPI_Gatherv(lengths.data(), local_n, MPI_INT, all_lengths.data(), counts.data(), offsets.data(), MPI_INT, 0,
              MPI_COMM_WORLD);
  MPI_Gatherv(rows.GetJ(), local_nnz, MPI_INT, g.col.data(), nnz_counts.data(), nnz_offsets.data(), MPI_INT, 0,
              MPI_COMM_WORLD);
  MPI_Gatherv(rows.GetData(), local_nnz, MPI_DOUBLE, g.val.data(), nnz_counts.data(), nnz_offsets.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);
  if (rank == 0)
    for (long long r = 0; r < global_n; ++r) g.ptr[r + 1] = g.ptr[r] + all_lengths[r];
  return g;
}

class FwTreeCotreeDirect {
 public:
  using SpMat = Eigen::SparseMatrix<double>;
  FwTreeCotreeDirect(mfem::HypreParMatrix& K, mfem::HypreParMatrix& M, mfem::HypreParMatrix& G,
                     const mfem::Array<int>& ess)
      : local_n_(K.Height()) {
    Timer timer;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    int ranks; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    counts_.resize(ranks); offsets_.resize(ranks);
    MPI_Allgather(&local_n_, 1, MPI_INT, counts_.data(), 1, MPI_INT, MPI_COMM_WORLD);
    long long total = 0;
    for (int r = 0; r < ranks; ++r) { offsets_[r] = static_cast<int>(total); total += counts_[r]; }
    AXF_REQUIRE(total > 0 && total <= std::numeric_limits<int>::max() && total == K.GetGlobalNumRows() &&
                total == M.GetGlobalNumRows() && total == G.GetGlobalNumRows() && G.Height() == local_n_,
                "tree-cotree solve needs consistent square ND operators with 32-bit indices", "/fullwave");
    n_ = static_cast<int>(total);
    const long long nv_global = G.GetGlobalNumCols();
    AXF_REQUIRE(nv_global > 0 && nv_global <= std::numeric_limits<int>::max(), "tree-cotree: bad H1 size", "/fullwave");
    const int nv = static_cast<int>(nv_global);
    std::vector<int> mark(local_n_, 0), all_mark(rank_ == 0 ? n_ : 0);
    for (int i : ess) mark[i] = 1;
    MPI_Gatherv(mark.data(), local_n_, MPI_INT, all_mark.data(), counts_.data(), offsets_.data(), MPI_INT, 0,
                MPI_COMM_WORLD);
    FwRows kr = fw_gather_rows(K, counts_, offsets_, rank_);
    FwRows mr = fw_gather_rows(M, counts_, offsets_, rank_);
    FwRows gr = fw_gather_rows(G, counts_, offsets_, rank_);
    const long long gathered_k = static_cast<long long>(kr.val.size());   // rank 0: the 32-bit gather's load
    const long long gathered_m = static_cast<long long>(mr.val.size());
    std::string failure;
    if (rank_ == 0) {
      try {
        // edge endpoints from the discrete gradient: (G phi)_r = phi[head] - phi[tail]
        std::vector<int> tail(n_), head(n_);
        for (int r = 0; r < n_; ++r) {
          int t = -1, h = -1, nz = 0;
          for (int k = gr.ptr[r]; k < gr.ptr[r + 1]; ++k) {
            const double v = gr.val[k];
            if (v == 0.0) continue;
            ++nz;
            AXF_REQUIRE(std::abs(std::abs(v) - 1.0) < 1e-12, "tree-cotree needs a +-1 lowest-order discrete gradient",
                        "/fullwave");
            (v > 0 ? h : t) = gr.col[k];
          }
          AXF_REQUIRE(nz == 2 && t >= 0 && h >= 0 && t != h, "tree-cotree: a gradient row is not an oriented edge",
                      "/fullwave");
          tail[r] = t; head[r] = h;
        }
        // 1. contract PEC-connected node sets (union over essential edges)
        std::vector<int> parent(nv);
        std::iota(parent.begin(), parent.end(), 0);
        auto find = [&](int v) { while (parent[v] != v) { parent[v] = parent[parent[v]]; v = parent[v]; } return v; };
        n_ess_ = 0;
        for (int r = 0; r < n_; ++r)
          if (all_mark[r]) {
            ++n_ess_;
            const int a = find(tail[r]), b = find(head[r]);
            if (a != b) parent[std::max(a, b)] = std::min(a, b);
          }
        std::vector<int> rep(nv), size(nv, 0);
        for (int v = 0; v < nv; ++v) { rep[v] = find(v); ++size[rep[v]]; }
        int contracted_pec = 0;
        for (int v = 0; v < nv; ++v) if (rep[v] == v && size[v] > 1) ++contracted_pec;
        // 2. free-edge adjacency of the contracted graph (self loops excluded: they are cotree)
        free_.assign(n_, -1);
        int nfree = 0;
        for (int r = 0; r < n_; ++r) if (!all_mark[r]) free_[r] = nfree++;
        std::vector<int> deg(nv + 1, 0);
        for (int r = 0; r < n_; ++r)
          if (!all_mark[r] && rep[tail[r]] != rep[head[r]]) { ++deg[rep[tail[r]] + 1]; ++deg[rep[head[r]] + 1]; }
        for (int v = 0; v < nv; ++v) deg[v + 1] += deg[v];
        std::vector<int> adj_node(deg[nv]), adj_edge(deg[nv]), fill(deg.begin(), deg.end() - 1);
        for (int r = 0; r < n_; ++r) {
          if (all_mark[r]) continue;
          const int a = rep[tail[r]], b = rep[head[r]];
          if (a == b) continue;
          adj_node[fill[a]] = b; adj_edge[fill[a]++] = r;
          adj_node[fill[b]] = a; adj_edge[fill[b]++] = r;
        }
        // 3. BFS spanning forest; root: the largest contracted node (outer PEC wall), then any unvisited rep
        std::vector<int> order;
        order.reserve(nv);
        for (int v = 0; v < nv; ++v) if (rep[v] == v) order.push_back(v);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return size[a] > size[b]; });
        std::vector<char> seen(nv, 0), tree(n_, 0);
        phi_of_.assign(nv, -1);
        phi_conductor_.clear();
        int nphi = 0;
        roots_ = 0;
        std::vector<int> queue;
        queue.reserve(nv);
        for (int root : order) {
          if (seen[root]) continue;
          ++roots_;
          if (roots_ == 1) root_size_ = size[root];
          seen[root] = 1;
          queue.clear();
          queue.push_back(root);
          for (size_t q = 0; q < queue.size(); ++q) {
            const int a = queue[q];
            for (int k = deg[a]; k < deg[a + 1]; ++k) {
              const int b = adj_node[k];
              if (seen[b]) continue;
              seen[b] = 1;
              tree[adj_edge[k]] = 1;
              phi_of_[b] = nphi++;
              phi_conductor_.push_back(size[b] > 1);   // a contracted PEC conductor, not a free vertex
              queue.push_back(b);
            }
          }
        }
        int ntree = 0;
        for (int r = 0; r < n_; ++r) ntree += tree[r];
        AXF_REQUIRE(ntree == nphi, "tree-cotree: tree edges and potentials differ", "/fullwave");
        // 4. y layout: [cotree edges in free order; potentials]
        ycot_.assign(nfree, -1);
        ncot_ = 0;
        for (int r = 0; r < n_; ++r) if (free_[r] >= 0 && !tree[r]) ycot_[free_[r]] = ncot_++;
        nphi_ = nphi;
        nfree_ = nfree;
        // 5. free-free K, M (column-major); G_c (free x phi)
        auto block = [&](const FwRows& rows) {
          std::vector<Eigen::Triplet<double>> t;
          t.reserve(rows.val.size());
          for (int r = 0; r < n_; ++r) {
            if (free_[r] < 0) continue;
            for (int k = rows.ptr[r]; k < rows.ptr[r + 1]; ++k)
              if (free_[rows.col[k]] >= 0 && rows.val[k] != 0.0) t.emplace_back(free_[r], free_[rows.col[k]], rows.val[k]);
          }
          SpMat S(nfree_, nfree_);
          S.setFromTriplets(t.begin(), t.end());
          S.makeCompressed();
          return S;
        };
        Kf_ = block(kr); kr = FwRows();
        Mf_ = block(mr); mr = FwRows();
        {
          std::vector<Eigen::Triplet<double>> t;
          t.reserve(2 * static_cast<size_t>(nfree_));
          for (int r = 0; r < n_; ++r) {
            if (free_[r] < 0) continue;
            const int ph = phi_of_[rep[head[r]]], pt = phi_of_[rep[tail[r]]];
            if (rep[head[r]] == rep[tail[r]]) continue;  // self loop on a contracted node: zero gradient
            if (ph >= 0) t.emplace_back(free_[r], ph, 1.0);
            if (pt >= 0) t.emplace_back(free_[r], pt, -1.0);
          }
          Gc_.resize(nfree_, nphi_);
          Gc_.setFromTriplets(t.begin(), t.end());
          Gc_.makeCompressed();
          Gabs_t_ = SpMat(Gc_.cwiseAbs().transpose());   // |G_c|^T, the size of each Gauss row's terms
        }
        MG_ = (Mf_ * Gc_).pruned();
        SpMat GMG = (SpMat(Gc_.transpose()) * MG_).pruned();
        GMG = (0.5 * (GMG + SpMat(GMG.transpose()))).pruned();
        // y-space blocks: Ky = [[K_cc,0],[0,0]], My = [[M_cc, MG_c],[MG_c^T, GMG]]; A_y(w) = Ky - w^2 My
        std::vector<Eigen::Triplet<double>> tk, tm;
        for (int c = 0; c < nfree_; ++c) {
          if (ycot_[c] < 0) continue;
          for (SpMat::InnerIterator it(Kf_, c); it; ++it)
            if (ycot_[it.row()] >= 0) tk.emplace_back(ycot_[it.row()], ycot_[c], it.value());
          for (SpMat::InnerIterator it(Mf_, c); it; ++it)
            if (ycot_[it.row()] >= 0) tm.emplace_back(ycot_[it.row()], ycot_[c], it.value());
        }
        for (int c = 0; c < nphi_; ++c) {
          for (SpMat::InnerIterator it(MG_, c); it; ++it)
            if (ycot_[it.row()] >= 0) {
              tm.emplace_back(ycot_[it.row()], ncot_ + c, it.value());
              tm.emplace_back(ncot_ + c, ycot_[it.row()], it.value());
            }
          for (SpMat::InnerIterator it(GMG, c); it; ++it) tm.emplace_back(ncot_ + static_cast<int>(it.row()), ncot_ + c, it.value());
        }
        Ky_.resize(nfree_, nfree_); My_.resize(nfree_, nfree_);
        Ky_.setFromTriplets(tk.begin(), tk.end()); My_.setFromTriplets(tm.begin(), tm.end());
        Ky_.makeCompressed(); My_.makeCompressed();
        topology_ = {{"global_edges", n_}, {"essential_edges", n_ess_}, {"free_edges", nfree_},
                     {"cotree_edges", ncot_}, {"tree_edges_potentials", nphi_}, {"h1_nodes", nv},
                     {"contracted_pec_nodes", contracted_pec}, {"grounded_roots", roots_},
                     {"largest_root_node_count", root_size_},
                     {"phi_block_nonzeros", static_cast<long long>(GMG.nonZeros())},
                     {"gathered_K_nonzeros", gathered_k}, {"gathered_M_nonzeros", gathered_m},
                     {"tree", "BFS spanning forest of the PEC-contracted free-edge graph"}};
      } catch (const std::exception& error) { failure = error.what(); }
    }
    fw_sync_failure(failure, "tree-cotree setup failed: ");
    topology_ = fw_bcast_json(topology_, rank_);
    topology_["setup_seconds"] = fw_allmax(timer.seconds());
  }

  // Per-frequency factorization of the symmetric, diagonally equilibrated y-space operator.
  void Factor(double w) {
    Timer timer;
    w_ = w;
    std::string failure;
    json info;
    std::array<double, 3> after{};
    if (rank_ == 0) {
      try {
        // the previous frequency's factor is not needed any more: free it, and hand its memory back
        // (AXF_FW_MALLOC_TRIM, AXF_FW_MALLOC_THRESHOLDS), before this frequency's matrices and factor are allocated
        fw_malloc_fix_thresholds();
        factor_.reset();
        fw_trim_after_factor();
        Ay_ = Ky_ - (w * w) * My_;
        Ay_.makeCompressed();
        Ay_abs_ = Ay_.cwiseAbs();   // for the componentwise backward error
        const SpMat& A = Ay_;
        scale_.resize(nfree_);
        Eigen::VectorXd diag = Eigen::VectorXd::Zero(nfree_), mx = Eigen::VectorXd::Zero(nfree_);
        for (int c = 0; c < nfree_; ++c)
          for (SpMat::InnerIterator it(A, c); it; ++it) {
            mx(it.row()) = std::max(mx(it.row()), std::abs(it.value()));
            if (it.row() == c) diag(c) = it.value();
          }
        for (int i = 0; i < nfree_; ++i) {
          AXF_REQUIRE(mx(i) > 0 && std::isfinite(mx(i)), "zero/nonfinite row in tree-cotree matrix", "/fullwave");
          scale_(i) = 1.0 / std::sqrt(std::max(std::abs(diag(i)), 0.1 * mx(i)));
        }
        SpMat S = scale_.asDiagonal() * A * scale_.asDiagonal();
        S.makeCompressed();
        factor_ = std::make_unique<SymSolver>();
        factor_->set_backend(have_strumpack() ? 1 : 0);
        factor_->compute(S);
        AXF_REQUIRE(factor_->info() == Eigen::Success, "tree-cotree factorization failed", "/fullwave");
        fw_trim_after_factor();   // the factorization's work memory (contribution blocks, copies), now free
        after = fw_rank0_memory_sample();
        info["factor_nonzeros"] = factor_->factor_nonzeros();
        info["matrix_nonzeros"] = static_cast<long long>(A.nonZeros());   // 32-bit headroom
        info["backend"] = factor_->backend();
        int pos_e = 0, neg_phi = 0;
        for (int i = 0; i < ncot_; ++i) pos_e += diag(i) > 0;
        for (int i = ncot_; i < nfree_; ++i) neg_phi += diag(i) < 0;
        info["cotree_positive_diagonal_fraction"] = ncot_ ? double(pos_e) / ncot_ : 1.0;
        info["potential_negative_diagonal_fraction"] = nphi_ ? double(neg_phi) / nphi_ : 1.0;
        const auto inertia = factor_->inertia();
        info["inertia"] = {{"status", inertia.status}, {"accurate", inertia.accurate},
                           {"negative", inertia.accurate ? json(inertia.negative) : json(nullptr)},
                           {"positive", inertia.accurate ? json(inertia.positive) : json(nullptr)},
                           {"expected_negative_below_first_resonance", nphi_},
                           // negative - n_phi > 0: discrete harmonic fields (relative cohomology) or cavity modes
                           // below w; their K E_c still cancels (see the class comment)
                           {"excess_negative_modes", inertia.accurate ? json(inertia.negative - nphi_) : json(nullptr)}};
      } catch (const std::exception& error) { failure = error.what(); }
    }
    fw_sync_failure(failure, "tree-cotree factorization failed: ", true);
    memory_after_factorization_ = fw_broadcast_sample(after);
    factor_info_ = fw_bcast_json(info, rank_);
    factor_info_["seconds"] = fw_allmax(timer.seconds());
  }

  // Solve (A + U diag(c) U^T) X = B, [re; im] true-DOF layout (HERMITIAN block form, imaginary block zero).
  // R = A X - B in split form; Xc = the curl representative S_c e (K X = K Xc exactly) when requested.
  json Solve(const mfem::Vector& B, mfem::Vector& X, mfem::Vector& R, mfem::Vector* Xc,
             const mfem::DenseMatrix* U, const std::vector<cplx>* c) const {
    Timer timer;
    const int k = U ? U->Width() : 0;
    std::vector<double> b(rank_ == 0 ? 2 * static_cast<size_t>(n_) : 0);
    for (int part = 0; part < 2; ++part)
      MPI_Gatherv(B.GetData() + part * local_n_, local_n_, MPI_DOUBLE, rank_ == 0 ? b.data() + part * n_ : nullptr,
                  counts_.data(), offsets_.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::vector<double> u(rank_ == 0 ? static_cast<size_t>(n_) * k : 0);
    {
      std::vector<double> col(local_n_);
      for (int j = 0; j < k; ++j) {
        for (int i = 0; i < local_n_; ++i) col[i] = (*U)(i, j);
        MPI_Gatherv(col.data(), local_n_, MPI_DOUBLE, rank_ == 0 ? u.data() + static_cast<size_t>(j) * n_ : nullptr,
                    counts_.data(), offsets_.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
    }
    std::vector<double> x(rank_ == 0 ? 2 * static_cast<size_t>(n_) : 0, 0.0), r(x.size(), 0.0), xc(x.size(), 0.0);
    json stats;
    std::string failure;
    if (rank_ == 0) {
      try {
        using V = Eigen::VectorXd;
        using CV = Eigen::VectorXcd;
        const double w2 = w_ * w_;
        // free parts of b; b on essential rows must vanish (sources are zeroed there)
        CV bf(nfree_);
        double bess = 0.0;
        for (int i = 0; i < n_; ++i) {
          const cplx bi(b[i], b[n_ + i]);
          if (free_[i] >= 0) bf(free_[i]) = bi; else bess = std::max(bess, std::abs(bi));
        }
        AXF_REQUIRE(bess == 0.0, "tree-cotree: right-hand side is nonzero on a PEC (essential) edge", "/fullwave");
        // The right-hand side's discrete charge q_p = (G_c^T b)_p per potential, against the size of its own terms
        // s_p = (|G_c|^T |b|)_p. A row is "uncharged" when |q_p| <= 1e-12 s_p: the rounding a charge-free source (a
        // port feed, a load, B_lab, a uniform field) leaves, 1e-16 to 1e-14 of its terms, which at low frequency still
        // enters the field like 1/w. Real charge (a terminal current at
        // a conductor, a truncated source's end faces at free vertices) is O(1) of its terms: a "charged" row.
        // Reported: the largest |q_p| over all free vertices and over the uncharged conductor rows, each divided by the
        // largest s_p over the same rows (a ratio of maxima), before and after the projection, and the charged rows of
        // each kind. AXF_FW_TC_CHARGE_PROJECTION zeroes uncharged rows of P^T b and of the load columns alike, never
        // real charge: 1 at free vertices, 2 (the default) also at conductors.
        auto uncharged = [&](double q, double s) { return q <= 1e-12 * s; };
        const V qr = Gc_.transpose() * V(bf.real());
        const V qi = Gc_.transpose() * V(bf.imag());
        const V sr = Gabs_t_ * V(bf.cwiseAbs());
        double q_free = 0.0, q_scale = 0.0, q_cond = 0.0, q_cond_scale = 0.0;
        int charged_free = 0, charged_conductors = 0;
        for (int p = 0; p < nphi_; ++p) {
          const double q = std::hypot(qr(p), qi(p));
          if (!phi_conductor_[p]) {
            q_free = std::max(q_free, q);
            q_scale = std::max(q_scale, sr(p));
            if (!uncharged(q, sr(p))) ++charged_free;
          } else if (uncharged(q, sr(p))) {
            q_cond = std::max(q_cond, q);
            q_cond_scale = std::max(q_cond_scale, sr(p));
          } else {
            ++charged_conductors;
          }
        }
        const char* proj_raw = std::getenv("AXF_FW_TC_CHARGE_PROJECTION");
        const std::string proj_env = proj_raw && *proj_raw ? proj_raw : "2";   // empty means unset, as at setup
        AXF_REQUIRE(proj_env == "0" || proj_env == "1" || proj_env == "2", "AXF_FW_TC_CHARGE_PROJECTION must be 0, 1 or 2",
                    "/fullwave");
        const int projection = std::stoi(proj_env);
        // zero the uncharged Gauss rows the projection treats in a transformed vector yv = P^T v (y = [cotree edges;
        // potentials]); absv = |v| in the original space gives each row's own terms
        auto project = [&](auto& yv, const V& absv) {
          if (projection == 0) return;
          const V s = Gabs_t_ * absv;
          for (int p = 0; p < nphi_; ++p)
            if ((!phi_conductor_[p] || projection == 2) && uncharged(std::abs(yv(ncot_ + p)), s(p))) yv(ncot_ + p) = 0.0;
        };
        const char* refine_raw = std::getenv("AXF_FW_TC_REFINE");
        const std::string refine_env = refine_raw && *refine_raw ? refine_raw : "normwise";
        AXF_REQUIRE(refine_env == "normwise" || refine_env == "componentwise",
                    "AXF_FW_TC_REFINE must be normwise or componentwise", "/fullwave");
        const bool componentwise = refine_env == "componentwise";
        Eigen::MatrixXd Uf(nfree_, k);
        for (int j = 0; j < k; ++j)
          for (int i = 0; i < n_; ++i) if (free_[i] >= 0) Uf(free_[i], j) = u[static_cast<size_t>(j) * n_ + i];
        Eigen::VectorXcd cv(k);
        for (int j = 0; j < k; ++j) cv(j) = (*c)[j];
        // maps
        auto Pt = [&](const V& g) {  // P^T g: [g_cot; G_c^T g]
          V y(nfree_);
          for (int i = 0; i < nfree_; ++i) if (ycot_[i] >= 0) y(ycot_[i]) = g(i);
          y.tail(nphi_) = Gc_.transpose() * g;
          return y;
        };
        auto curl_part = [&](const V& y) {  // S_c e
          V ec = V::Zero(nfree_);
          for (int i = 0; i < nfree_; ++i) if (ycot_[i] >= 0) ec(i) = y(ycot_[i]);
          return ec;
        };
        auto apply_A = [&](const V& y, V& E, V& Ec) {  // E = P y, returns A E in split form
          Ec = curl_part(y);
          const V phi = y.tail(nphi_);
          E = Ec + Gc_ * phi;
          return V(Kf_ * Ec - w2 * (Mf_ * Ec + MG_ * phi));
        };
        auto inv = [&](const Eigen::MatrixXd& g) {  // A_y^{-1} g (columns), equilibrated factor
          Eigen::MatrixXd s = scale_.asDiagonal() * g;
          Eigen::MatrixXd z = factor_->solve(s);
          return Eigen::MatrixXd(scale_.asDiagonal() * z);
        };
        // Woodbury pieces in y space: Vy = P^T U, Z = A_y^{-1} Vy, W = Vy^T Z
        Eigen::MatrixXd Vy(nfree_, k), Z;
        Eigen::MatrixXcd Kw;
        Eigen::PartialPivLU<Eigen::MatrixXcd> klu;
        if (k > 0) {
          for (int j = 0; j < k; ++j) {
            V vj = Pt(Uf.col(j));
            project(vj, V(Uf.col(j).cwiseAbs()));   // the load columns are port feeds too
            Vy.col(j) = vj;
          }
          Z = inv(Vy);
          const Eigen::MatrixXd W = Vy.transpose() * Z;
          Kw = Eigen::MatrixXcd::Identity(k, k) + cv.asDiagonal() * W.cast<cplx>();
          klu.compute(Kw);
        }
        auto apply_inv = [&](const CV& g) {  // (A_y + Vy C Vy^T)^{-1} g
          Eigen::MatrixXd gg(nfree_, 2);
          gg.col(0) = g.real(); gg.col(1) = g.imag();
          const Eigen::MatrixXd z0 = inv(gg);
          CV z = z0.col(0).cast<cplx>() + cplx(0, 1) * z0.col(1).cast<cplx>();
          if (k > 0) {
            const CV t = Vy.transpose().cast<cplx>() * z;
            const CV s = klu.solve(CV(cv.asDiagonal() * t));
            z -= Z.cast<cplx>() * s;
          }
          return z;
        };
        const double bnorm = bf.norm();
        AXF_REQUIRE(bnorm > 0, "tree-cotree: zero right-hand side", "/fullwave");
        // Refinement on the transformed residual r_y = P^T b - (A_y + Vy C Vy^T) y. Its potential (Gauss-law)
        // rows contain no K at all, so rounding of K E_c cannot pollute the charge balance; the weak
        // functionals (port voltages, reciprocity) are y^T r_y and therefore need exactly this residual small. By
        // default it is evaluated from K, M and G_c in long double (residual_extended, below), not from the stored A_y.
        CV by = Pt(bf.real()).cast<cplx>() + cplx(0, 1) * Pt(bf.imag()).cast<cplx>();
        project(by, V(bf.cwiseAbs()));
        double q_after = 0.0, q_cond_after = 0.0;   // the charge actually solved with, in the same rows
        for (int p = 0; p < nphi_; ++p) {
          const double q = std::abs(by(ncot_ + p));
          if (!phi_conductor_[p]) q_after = std::max(q_after, q);
          else if (uncharged(std::hypot(qr(p), qi(p)), sr(p))) q_cond_after = std::max(q_cond_after, q);
        }
        auto Ay_apply = [&](const CV& yy) {
          CV out = (Ay_ * yy.real()).cast<cplx>() + cplx(0, 1) * (Ay_ * yy.imag()).cast<cplx>();
          if (k > 0) out += Vy.cast<cplx>() * CV(cv.asDiagonal() * CV(Vy.transpose().cast<cplx>() * yy));
          return out;
        };
        // AXF_FW_TC_RESIDUAL=extended (the default): the refinement residual in long double,
        // from the operators themselves: E = S_c e + G_c phi, then [S_c^T (K S_c e - w^2 M E); -w^2 G_c^T M E] (Ky's
        // Gauss block is exactly zero), plus the projected load columns, against the projected right-hand side. In
        // double (AXF_FW_TC_RESIDUAL=double) the transformed product cancels about 5000-fold on
        // a tree-cotree test model; its residual floors near 5e-13, the corrections stall in rounding, and the
        // source reciprocity was 4e-10 against 1.4e-12 with the extended residual.
        const char* residual_raw = std::getenv("AXF_FW_TC_RESIDUAL");
        const std::string residual_env = residual_raw && *residual_raw ? residual_raw : "extended";
        AXF_REQUIRE(residual_env == "double" || residual_env == "extended",
                    "AXF_FW_TC_RESIDUAL must be double or extended", "/fullwave");
        using LD = long double;
        using LV = std::vector<LD>;
        // out += a (A x) and out += a (A^T x) for a column-major sparse A, accumulated in long double
        auto spmv_add = [](const SpMat& A, const LV& x, LD a, LV& out) {
          for (int c = 0; c < A.outerSize(); ++c) {
            const LD xc = x[c];
            if (xc == 0) continue;
            for (SpMat::InnerIterator it(A, c); it; ++it) out[it.row()] += a * static_cast<LD>(it.value()) * xc;
          }
        };
        auto spmv_t_add = [](const SpMat& A, const LV& x, LD a, LV& out) {
          for (int c = 0; c < A.outerSize(); ++c) {
            LD s = 0;
            for (SpMat::InnerIterator it(A, c); it; ++it) s += static_cast<LD>(it.value()) * x[it.row()];
            out[c] += a * s;
          }
        };
        auto residual_extended = [&](const CV& yy) {
          const LD w2l = static_cast<LD>(w_) * static_cast<LD>(w_);
          std::array<LV, 2> ay;   // A_y y per part (real, imaginary)
          std::array<LV, 2> t;    // Vy^T y per part
          for (int part = 0; part < 2; ++part) {
            const V yp = part == 0 ? V(yy.real()) : V(yy.imag());
            LV ec(nfree_, 0);
            for (int i = 0; i < nfree_; ++i) if (ycot_[i] >= 0) ec[i] = yp(ycot_[i]);
            LV phi(nphi_);
            for (int p = 0; p < nphi_; ++p) phi[p] = yp(ncot_ + p);
            LV e = ec;
            spmv_add(Gc_, phi, 1, e);
            LV ke(nfree_, 0);
            spmv_add(Kf_, ec, 1, ke);
            LV me(nfree_, 0);
            spmv_add(Mf_, e, 1, me);
            LV& out = ay[part];
            out.assign(nfree_, 0);
            for (int i = 0; i < nfree_; ++i) if (ycot_[i] >= 0) out[ycot_[i]] = ke[i] - w2l * me[i];
            LV gauss(nphi_, 0);
            spmv_t_add(Gc_, me, -w2l, gauss);
            for (int p = 0; p < nphi_; ++p) out[ncot_ + p] = gauss[p];
            t[part].assign(k, 0);
            for (int j = 0; j < k; ++j) {
              LD s = 0;
              for (int i = 0; i < nfree_; ++i) s += static_cast<LD>(Vy(i, j)) * static_cast<LD>(yp(i));
              t[part][j] = s;
            }
          }
          LV sr(k);   // C Vy^T y per load, real and imaginary parts
          LV si(k);
          for (int j = 0; j < k; ++j) {
            const LD cr = cv(j).real();
            const LD ci = cv(j).imag();
            sr[j] = cr * t[0][j] - ci * t[1][j];
            si[j] = cr * t[1][j] + ci * t[0][j];
          }
          CV r(nfree_);
          for (int i = 0; i < nfree_; ++i) {
            LD ar = ay[0][i];
            LD ai = ay[1][i];
            for (int j = 0; j < k; ++j) {
              ar += static_cast<LD>(Vy(i, j)) * sr[j];
              ai += static_cast<LD>(Vy(i, j)) * si[j];
            }
            const LD rr = static_cast<LD>(by(i).real()) - ar;
            const LD ri = static_cast<LD>(by(i).imag()) - ai;
            r(i) = cplx(static_cast<double>(rr), static_cast<double>(ri));
          }
          return r;
        };
        auto residual_y = [&](const CV& yy) {
          if (residual_env == "extended") return residual_extended(yy);
          return CV(by - Ay_apply(yy));
        };
        // Componentwise backward error |r_i| / (|A_y| |y| + |b_y|)_i (Oettli-Prager) of the curl rows, the free-vertex
        // Gauss rows and the conductor Gauss rows. A normwise-small residual can leave Gauss rows whose own terms are
        // small with a large relative error, and at low frequency that error acts like a charge.
        const Eigen::MatrixXd Vy_abs = Vy.cwiseAbs();
        const V cv_abs = cv.cwiseAbs();
        auto backward = [&](const CV& yy, const CV& r) {
          const V ya = yy.cwiseAbs();
          V den = Ay_abs_ * ya + V(by.cwiseAbs());
          if (k > 0) den += Vy_abs * V(cv_abs.asDiagonal() * V(Vy_abs.transpose() * ya));
          std::array<double, 3> w{0.0, 0.0, 0.0};
          for (int i = 0; i < nfree_; ++i) {
            if (!(den(i) > 0)) continue;   // then r_i is exactly zero
            const int block = i < ncot_ ? 0 : (phi_conductor_[i - ncot_] ? 2 : 1);
            w[block] = std::max(w[block], std::abs(r(i)) / den(i));
          }
          return w;
        };
        const CV dby = scale_.cast<cplx>().asDiagonal() * by;
        const double bynorm = dby.norm();
        AXF_REQUIRE(bynorm > 0, "tree-cotree: zero transformed right-hand side", "/fullwave");
        CV y = CV::Zero(nfree_), ry = by, res(nfree_), E(nfree_), Ec(nfree_);
        json history = json::array(), cw_history = json::array();
        double rely = 1.0, best = std::numeric_limits<double>::infinity(), rel = 1.0;
        double best_measure = std::numeric_limits<double>::infinity(), previous = best_measure;
        CV best_y = y;
        const int max_steps = std::getenv("AXF_FW_REFINE_MAX") ? std::atoi(std::getenv("AXF_FW_REFINE_MAX")) : 8;
        // normwise (double residual): stop below 1e-16 in the scaled transformed residual; componentwise (experimental,
        // double residual only): stop when every row's backward error is at rounding; with the extended residual (the
        // default): stop when the correction, the forward-error estimate, is at rounding (the residual of even the
        // correctly rounded y stays at the double floor). Each stops when its measure no longer halves.
        const bool extended = residual_env == "extended";
        const std::string refinement_stop = extended ? "relative correction (equilibrated)"
                                          : componentwise ? "componentwise backward error"
                                                          : "scaled transformed residual";
        // long double is wider than double on x86-64 Linux (64-bit mantissa); elsewhere (macOS arm64, MSVC) it is double
        const int extended_bits = std::numeric_limits<LD>::digits;
        const bool extended_wider = static_cast<LD>(1) + std::ldexp(static_cast<LD>(1), -60) != static_cast<LD>(1);
        const double eps = std::numeric_limits<double>::epsilon();
        const double tolerance = extended ? 4 * eps : componentwise ? 8 * eps : 1e-16;
        // corrections in the equilibrated unknowns z = D^-1 y (the factor solves D A_y D z = D b_y), so that neither the
        // edge nor the potential block dominates the measure
        auto equilibrated_norm = [&](const CV& v) { return CV(v.cwiseQuotient(scale_.cast<cplx>())).norm(); };
        json correction_history = json::array();
        int steps = 0;
        for (; steps < max_steps;) {
          const CV dy = apply_inv(ry);
          y += dy;
          ++steps;
          const double correction = equilibrated_norm(dy) / std::max(equilibrated_norm(y), 1e-300);
          correction_history.push_back(correction);
          // with the extended residual a correction estimates the error of the iterate it corrects: one that grew (a
          // divergence) or is not finite is undone, and the iterate before it kept
          if (extended && (!std::isfinite(correction) || (steps >= 2 && correction > previous))) {
            y -= dy;
            break;
          }
          ry = residual_y(y);
          rely = CV(scale_.cast<cplx>().asDiagonal() * ry).norm() / bynorm;
          const auto w = backward(y, ry);
          const double cw = std::max({w[0], w[1], w[2]});
          history.push_back(rely);
          cw_history.push_back(cw);
          const double measure = extended ? correction : componentwise ? cw : rely;
          if (measure < best_measure) {
            best_measure = measure;
            best = rely;
            best_y = y;
          }
          if (measure < tolerance || (steps >= 2 && measure > 0.5 * previous)) break;
          previous = measure;
        }
        ry = residual_y(best_y);
        const auto omega = backward(best_y, ry);
        const CV dry = scale_.cast<cplx>().asDiagonal() * ry;
        const CV day = scale_.cast<cplx>().asDiagonal() * Ay_apply(best_y);
        const double gauss_res = dry.tail(nphi_).norm(), curl_res = dry.head(ncot_).norm();
        // each block's scale includes its rows' own terms |A_y| |y|: under the default charge projection the Gauss rows
        // of b are zero and those of A y are rounding, so their ratio alone would be 1 by construction
        const Eigen::VectorXd terms = scale_.asDiagonal() * (Ay_abs_ * best_y.cwiseAbs());
        const double gauss_scale = std::max({day.tail(nphi_).norm(), CV(scale_.cast<cplx>().asDiagonal() * by).tail(nphi_).norm(),
                                             terms.tail(nphi_).norm()});
        const double curl_scale = std::max({day.head(ncot_).norm(), CV(scale_.cast<cplx>().asDiagonal() * by).head(ncot_).norm(),
                                            terms.head(ncot_).norm()});
        // report the best iterate and its own residual
        y = best_y;
        V Er, Ei, Ecr, Eci;
        const V ar = apply_A(y.real(), Er, Ecr), ai = apply_A(y.imag(), Ei, Eci);
        CV Ay = ar.cast<cplx>() + cplx(0, 1) * ai.cast<cplx>();
        if (k > 0) {
          const CV ue = Uf.transpose().cast<cplx>() * (Er.cast<cplx>() + cplx(0, 1) * Ei.cast<cplx>());
          Ay += Uf.cast<cplx>() * CV(cv.asDiagonal() * ue);
        }
        res = bf - Ay;
        rel = res.norm() / bnorm;
        // K_f E - w^2 M_f E with the separately assembled K and M applied to the recombined E (not the legacy
        // summed matrix, which the caller evaluates as legacy_summed_matrix_relative_residual_diagnostic)
        CV summed = (Kf_ * Er - w2 * (Mf_ * Er)).cast<cplx>() + cplx(0, 1) * (Kf_ * Ei - w2 * (Mf_ * Ei)).cast<cplx>();
        if (k > 0) {
          const CV ue = Uf.transpose().cast<cplx>() * (Er.cast<cplx>() + cplx(0, 1) * Ei.cast<cplx>());
          summed += Uf.cast<cplx>() * CV(cv.asDiagonal() * ue);
        }
        const double summed_rel = (bf - summed).norm() / bnorm;
        // Estimators of the low-frequency error: the field a Gauss-row source adds,
        // against the solution, in the energy norm of E, sqrt(dE^H M dE / E^H M E) (a gradient adds its own energy).
        // One more solve each: the right-hand side's rounding-level charge (its uncharged rows, free and conductor, as
        // computed before any projection: what projection 2 removes); what of it is left in the right-hand side solved
        // with (the output's estimated error; no solve when nothing is left or nothing was removed); and the Gauss rows
        // of the final residual.
        auto m_energy = [&](const V& er, const V& ei) { return er.dot(Mf_ * er) + ei.dot(Mf_ * ei); };
        const double e_energy = std::max(m_energy(Er, Ei), 1e-300);
        auto field_fraction = [&](const CV& g) {
          if (g.cwiseAbs().maxCoeff() == 0.0) return 0.0;
          const CV dy = apply_inv(g);
          V dEr, dEi, dEcr, dEci;
          apply_A(dy.real(), dEr, dEcr);
          apply_A(dy.imag(), dEi, dEci);
          return std::sqrt(m_energy(dEr, dEi) / e_energy);
        };
        CV g_charge = CV::Zero(nfree_), g_left = CV::Zero(nfree_), g_residual = CV::Zero(nfree_);
        for (int p = 0; p < nphi_; ++p) {
          if (uncharged(std::hypot(qr(p), qi(p)), sr(p))) {
            g_charge(ncot_ + p) = cplx(qr(p), qi(p));
            g_left(ncot_ + p) = by(ncot_ + p);
          }
          g_residual(ncot_ + p) = ry(ncot_ + p);
        }
        const double charge_fraction = field_fraction(g_charge);
        const double left_fraction = projection == 0 ? charge_fraction : field_fraction(g_left);
        const double residual_fraction = field_fraction(g_residual);
        json solve_warnings = json::array();
        if (extended && !extended_wider)
          solve_warnings.push_back(strcat("long double has ", extended_bits, " mantissa bits on this platform, so the "
                                          "extended refinement residual is no more precise than double"));
        if (left_fraction > 1e-3)
          solve_warnings.push_back(strcat("the right-hand side's rounding-level charge adds an estimated field error of ",
                                          left_fraction, " (rhs_charge_field_fraction_solved), which field outputs carry; "
                                          "AXF_FW_TC_CHARGE_PROJECTION=2 removes it"));
        for (int i = 0; i < n_; ++i) {
          const int f = free_[i];
          if (f < 0) continue;
          x[i] = Er(f); x[n_ + i] = Ei(f);
          r[i] = -res(f).real(); r[n_ + i] = -res(f).imag();  // R = A X - B
          xc[i] = Ecr(f); xc[n_ + i] = Eci(f);
        }
        stats = {{"refinement_steps", steps}, {"split_true_relative_residual", rel},
                 {"refinement_history_scaled_transformed_residual", history},
                 {"transformed_scaled_relative_residual", best},
                 {"gauss_block_relative_residual", gauss_res / std::max(gauss_scale, 1e-300)},
                 {"curl_block_relative_residual", curl_res / std::max(curl_scale, 1e-300)},
                 {"recombined_field_separate_KM_relative_residual_diagnostic", summed_rel},
                 {"curl_representative_norm_fraction",
                  std::sqrt((Ecr.squaredNorm() + Eci.squaredNorm()) / std::max(Er.squaredNorm() + Ei.squaredNorm(), 1e-300))},
                 {"rhs_free_vertex_charge_relative", q_free / std::max(q_scale, 1e-300)},
                 {"rhs_free_vertex_charge_relative_solved", q_after / std::max(q_scale, 1e-300)},
                 {"rhs_uncharged_conductor_charge_relative", q_cond / std::max(q_cond_scale, 1e-300)},
                 {"rhs_uncharged_conductor_charge_relative_solved", q_cond_after / std::max(q_cond_scale, 1e-300)},
                 {"charged_free_vertex_rows", charged_free},
                 {"charged_conductor_rows", charged_conductors},
                 {"charge_projection", projection},
                 {"rhs_charge_field_fraction", charge_fraction},
                 {"rhs_charge_field_fraction_solved", left_fraction},
                 {"gauss_residual_field_fraction", residual_fraction},
                 {"refinement", refine_env},
                 {"refinement_residual", residual_env},
                 {"refinement_residual_mantissa_bits", extended ? extended_bits : std::numeric_limits<double>::digits},
                 {"refinement_stop", refinement_stop},
                 {"refinement_history_relative_correction", correction_history},
                 {"refinement_history_componentwise_backward_error", cw_history},
                 {"componentwise_backward_error",
                  {{"curl_rows", omega[0]}, {"free_vertex_gauss_rows", omega[1]}, {"conductor_gauss_rows", omega[2]}}},
                 {"warnings", solve_warnings},
                 {"low_rank_columns", k}};
      } catch (const std::exception& error) { failure = error.what(); }
    }
    fw_sync_failure(failure, "tree-cotree solve failed: ", true);
    X.SetSize(2 * local_n_); R.SetSize(2 * local_n_);
    for (int part = 0; part < 2; ++part) {
      MPI_Scatterv(rank_ == 0 ? x.data() + part * n_ : nullptr, counts_.data(), offsets_.data(), MPI_DOUBLE,
                   X.GetData() + part * local_n_, local_n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Scatterv(rank_ == 0 ? r.data() + part * n_ : nullptr, counts_.data(), offsets_.data(), MPI_DOUBLE,
                   R.GetData() + part * local_n_, local_n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (Xc) {
      Xc->SetSize(2 * local_n_);
      for (int part = 0; part < 2; ++part)
        MPI_Scatterv(rank_ == 0 ? xc.data() + part * n_ : nullptr, counts_.data(), offsets_.data(), MPI_DOUBLE,
                     Xc->GetData() + part * local_n_, local_n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    stats = fw_bcast_json(stats, rank_);
    stats["solve_seconds"] = fw_allmax(timer.seconds());
    return stats;
  }
  const json& Topology() const { return topology_; }
  const json& FactorInfo() const { return factor_info_; }
  // rank 0's memory right after the factorization and its trim (fw_rank0_memory_sample)
  const json& MemoryAfterFactorization() const { return memory_after_factorization_; }

 private:
  // rank 0's failure, thrown on every rank; solver = true for a factorization or solve (a SolverFailure, which
  // Model3D::fullwave_analysis completes with the mesh check), false for a setup that may refuse the design
  static void fw_sync_failure(std::string& failure, const std::string& prefix, bool solver = false) {
    int size = static_cast<int>(failure.size());
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    failure.resize(size);
    if (size) MPI_Bcast(failure.data(), size, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (failure.empty()) return;
    if (solver) throw SolverFailure(prefix + failure, "/fullwave");
    throw Error(prefix + failure, "/fullwave");
  }
  static json fw_bcast_json(const json& j, int rank) {
    std::string text = rank == 0 ? j.dump() : std::string();
    int size = static_cast<int>(text.size());
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    text.resize(size);
    MPI_Bcast(text.data(), size, MPI_CHAR, 0, MPI_COMM_WORLD);
    return json::parse(text);
  }
  int rank_ = 0, local_n_ = 0, n_ = 0, n_ess_ = 0, nfree_ = 0, ncot_ = 0, nphi_ = 0, roots_ = 0, root_size_ = 0;
  double w_ = 0.0;
  std::vector<int> counts_, offsets_, free_, ycot_, phi_of_;
  std::vector<char> phi_conductor_;   // per potential: 1 for a contracted PEC conductor, 0 for a free vertex
  SpMat Kf_, Mf_, Gc_, MG_, Ky_, My_, Ay_, Gabs_t_, Ay_abs_;
  Eigen::VectorXd scale_;
  std::unique_ptr<SymSolver> factor_;
  json topology_, factor_info_, memory_after_factorization_;
};
// The full-wave operator plus the low-rank exterior correction (HERMITIAN block convention: plain complex
// multiplication), on true DOFs: y = A x + U diag(c) U^T x, U real (true DOFs x modes), c complex.
class FwOperator : public mfem::Operator {
 public:
  FwOperator(const mfem::Operator& A, const mfem::DenseMatrix& U, const std::vector<cplx>& c)
      : mfem::Operator(A.Height(), A.Width()), A_(A), U_(U), c_(c) {}
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
    A_.Mult(x, y);
    const int n = Height() / 2, nm = U_.Width();
    if (nm == 0) return;
    mfem::Vector xr(const_cast<double*>(x.GetData()), n), xi(const_cast<double*>(x.GetData()) + n, n);
    mfem::Vector yr(y.GetData(), n), yi(y.GetData() + n, n);
    std::vector<double> t(2 * nm, 0.0);
    mfem::Vector tr(t.data(), nm), ti(t.data() + nm, nm);
    if (n > 0) {
      U_.MultTranspose(xr, tr);
      U_.MultTranspose(xi, ti);
    }
    MPI_Allreduce(MPI_IN_PLACE, t.data(), 2 * nm, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    mfem::Vector sr(nm), si(nm);
    for (int j = 0; j < nm; ++j) {
      const cplx sj = c_[j] * cplx(t[j], t[nm + j]);
      sr(j) = sj.real();
      si(j) = sj.imag();
    }
    if (n > 0) {
      U_.AddMult(sr, yr);
      U_.AddMult(si, yi);
    }
  }

 private:
  const mfem::Operator& A_;
  const mfem::DenseMatrix& U_;
  const std::vector<cplx>& c_;
};

// Preconditioner of A + U C U^T from a preconditioner P of A = diag(M, s M) (s = +-1, block form on [re; im]) by the
// Woodbury identity: (A + U C U^T)^{-1} ~ P - (P U) C (I + U^T P U C)^{-1} U^T P. P U needs one application of M
// per mode (setup); the 2m x 2m dense system is the same on every rank.
class WoodburyPrec : public mfem::Solver {
 public:
  WoodburyPrec(mfem::Solver& M, double s, mfem::Solver& P, const mfem::DenseMatrix& U, const std::vector<cplx>& c)
      : mfem::Solver(P.Height()), P_(P), U_(U), s_(s), MU_(U.Height(), U.Width()) {
    const int n = U.Height(), m = U.Width();
    mfem::Vector u(n), mu(n);
    for (int j = 0; j < m; ++j) {
      for (int i = 0; i < n; ++i) u(i) = U(i, j);
      M.Mult(u, mu);
      for (int i = 0; i < n; ++i) MU_(i, j) = mu(i);
    }
    // W = U^T M U (global), K = I + diag(W, s W) Cblock, Cblock = [[Cr, -Ci], [Ci, Cr]]
    std::vector<double> W(static_cast<size_t>(m) * m, 0.0);
    for (int a = 0; a < m; ++a)
      for (int b = 0; b < m; ++b) {
        double v = 0;
        for (int i = 0; i < n; ++i) v += U(i, a) * MU_(i, b);
        W[static_cast<size_t>(a) * m + b] = v;
      }
    MPI_Allreduce(MPI_IN_PLACE, W.data(), m * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    cr_.SetSize(m);
    ci_.SetSize(m);
    for (int j = 0; j < m; ++j) {
      cr_(j) = c[j].real();
      ci_(j) = c[j].imag();
    }
    mfem::DenseMatrix K(2 * m);
    K = 0.0;
    for (int a = 0; a < m; ++a) {
      K(a, a) += 1.0;
      K(m + a, m + a) += 1.0;
      for (int b = 0; b < m; ++b) {
        const double w = W[static_cast<size_t>(a) * m + b];
        K(a, b) += w * cr_(b);
        K(a, m + b) += -w * ci_(b);
        K(m + a, b) += s * w * ci_(b);
        K(m + a, m + b) += s * w * cr_(b);
      }
    }
    lu_.Factor(K);
  }
  void SetOperator(const mfem::Operator&) override {}
  void Mult(const mfem::Vector& x, mfem::Vector& y) const override {
    P_.Mult(x, y);
    const int n = U_.Height(), m = U_.Width();
    mfem::Vector yr(y.GetData(), n), yi(y.GetData() + n, n);
    mfem::Vector t(2 * m), tr(t.GetData(), m), ti(t.GetData() + m, m);
    t = 0.0;
    if (n > 0) {
      U_.MultTranspose(yr, tr);
      U_.MultTranspose(yi, ti);
    }
    MPI_Allreduce(MPI_IN_PLACE, t.GetData(), 2 * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    lu_.Mult(t, z_);
    mfem::Vector sr(m), si(m);
    for (int j = 0; j < m; ++j) {
      sr(j) = cr_(j) * z_(j) - ci_(j) * z_(m + j);
      si(j) = ci_(j) * z_(j) + cr_(j) * z_(m + j);
    }
    if (n > 0) {
      MU_.AddMult(sr, yr, -1.0);
      MU_.AddMult(si, yi, -s_);
    }
  }

 private:
  mfem::Solver& P_;
  const mfem::DenseMatrix& U_;
  double s_;
  mfem::DenseMatrix MU_;
  mfem::Vector cr_, ci_;
  mfem::DenseMatrixInverse lu_;
  mutable mfem::Vector z_;
};

// Readout current density J = scale grad V in the port (per the stored MQS port solution), times `factor`.
class PortJ : public mfem::VectorCoefficient {
 public:
  PortJ(const mfem::ParGridFunction& V, int attr, double scale)
      : mfem::VectorCoefficient(3), V_(V), attr_(attr), scale_(scale) {}
  void Eval(mfem::Vector& J, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    J.SetSize(3);
    J = 0.0;
    if (T.Attribute != attr_) return;
    T.SetIntPoint(&ip);
    V_.GetGradient(T, J);
    J *= scale_;
  }

 private:
  const mfem::ParGridFunction& V_;
  int attr_;
  double scale_;
};

// B_lab = curl A_lab in the physical region (zero in the shell, where fields are mapped).
class LabB : public mfem::VectorCoefficient {
 public:
  // epml (optional): PML layer of each element (-1 outside); B_lab is zero inside the layers (overlap only in the
  // physical domain, as for the prescribed source)
  LabB(const mfem::ParGridFunction& A, int shell_attr, const std::vector<int>* epml = nullptr)
      : mfem::VectorCoefficient(3), A_(A), shell_(shell_attr), epml_(epml) {}
  void Eval(mfem::Vector& B, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    B.SetSize(3);
    B = 0.0;
    if (shell_ > 0 && T.Attribute == shell_) return;
    if (epml_ && !epml_->empty() && (*epml_)[T.ElementNo] >= 0) return;
    T.SetIntPoint(&ip);
    A_.GetCurl(T, B);
  }

 private:
  const mfem::ParGridFunction& A_;
  int shell_;
  const std::vector<int>* epml_;
};

// ---------------------------------------------------------------------------------------------------------
// 3D cavity eigenmodes (beta; docs/validation/cavity_3d.md)
// ---------------------------------------------------------------------------------------------------------
// Discrete gradients of H1 potentials (order p, so grad H1_p lies in ND_p) that vanish on the closed set (PEC
// elements, elements outside the cavity, PEC boundary faces): the curl kernel of the constrained Nedelec space,
// except one potential per floating conductor (constant on it), which this space cannot represent.
// project(v) computes the M-orthogonal gradient part g = G phi of v, (G^T M G) phi = G^T M v (AMG-PCG), returns
// its M-norm^2 share (the "divergence fraction") and, with apply, removes it from v.
class FwGradientProjector {
 public:
  FwGradientProjector(mfem::ParFiniteElementSpace& nd, mfem::HypreParMatrix& M, const mfem::Array<int>& ess_nd,
                      const std::vector<char>& closed_elem, mfem::Array<int>& ess_bdr, int order)
      : M_(M), ess_nd_(ess_nd), h1c_(order, 3), h1_(nd.GetParMesh(), &h1c_) {
    mfem::ParMesh& pm = *nd.GetParMesh();
    mfem::ParDiscreteLinearOperator grad(&h1_, &nd);
    grad.AddDomainInterpolator(new mfem::GradientInterpolator());
    grad.Assemble();
    grad.Finalize();
    G_.reset(grad.ParallelAssemble());
    mfem::Array<int> mark(h1_.GetVSize()), dofs;
    mark = 0;
    for (int e = 0; e < pm.GetNE(); ++e) {
      if (!closed_elem[e]) continue;
      h1_.GetElementDofs(e, dofs);
      for (int dd : dofs) mark[dd] = 1;
    }
    h1_.Synchronize(mark);
    mfem::Array<int> bt;
    h1_.GetEssentialTrueDofs(ess_bdr, bt);
    std::vector<char> t(h1_.GetTrueVSize(), 0);
    for (int i : bt) t[i] = 1;
    for (int i = 0; i < h1_.GetVSize(); ++i) {
      if (!mark[i]) continue;
      const int tt = h1_.GetLocalTDofNumber(i);
      if (tt >= 0) t[tt] = 1;
    }
    for (int i = 0; i < static_cast<int>(t.size()); ++i)
      if (t[i]) ess_h1_.Append(i);
    free_ = fw_allsum(static_cast<double>(t.size() - ess_h1_.Size()));
    S_.reset(mfem::RAP(&M_, G_.get()));
    ensure_diagonal(S_, ess_h1_);
    S_->EliminateBC(ess_h1_, mfem::Operator::DIAG_ONE);
    amg_ = std::make_unique<mfem::HypreBoomerAMG>(*S_);
    amg_->SetPrintLevel(0);
    pcg_ = std::make_unique<mfem::HyprePCG>(*S_);
    pcg_->SetTol(1e-12);
    pcg_->SetMaxIter(1000);
    pcg_->SetPrintLevel(0);
    pcg_->SetPreconditioner(*amg_);
  }
  double project(mfem::Vector& v, bool apply) {
    mfem::Vector Mv(v.Size()), rhs(G_->Width()), phi(G_->Width()), g(v.Size()), Mg(v.Size());
    M_.Mult(v, Mv);
    G_->MultTranspose(Mv, rhs);
    for (int i : ess_h1_) rhs(i) = 0.0;
    phi = 0.0;
    if (std::sqrt(fw_gdot(rhs, rhs)) > 0) pcg_->Mult(rhs, phi);
    G_->Mult(phi, g);
    for (int i : ess_nd_) g(i) = 0.0;
    M_.Mult(g, Mg);
    const double gg = fw_gdot(g, Mg), vv = fw_gdot(v, Mv);
    if (apply) v -= g;
    return vv > 0 ? gg / vv : 0.0;
  }
  double free_dofs() const { return free_; }

 private:
  mfem::HypreParMatrix& M_;
  const mfem::Array<int>& ess_nd_;
  mfem::H1_FECollection h1c_;
  mfem::ParFiniteElementSpace h1_;
  std::unique_ptr<mfem::HypreParMatrix> G_, S_;
  std::unique_ptr<mfem::HypreBoomerAMG> amg_;
  std::unique_ptr<mfem::HyprePCG> pcg_;
  mfem::Array<int> ess_h1_;
  double free_ = 0;
};

// ---------------------------------------------------------------------------------------------------------
// Gauss-law guard on the ungauged paths (AMS-GMRES, plain STRUMPACK): AXF_FW_GAUSS_GUARD = correct (default),
// estimate or off.
//
// Why: with E gradient-dominated (low frequency), A = K - w^2 M fixes the gradient (capacitive) part of E only
// through w^2 M, and a solve that passes the 1e-6 true-residual gate can still be wrong there by many times the
// field (tree-cotree test runs: AMS x13-17 at 10 Hz and 4 % at 100 Hz, plain STRUMPACK 6 % at 100 Hz, with
// nothing refused). K G = 0 exactly, so that part obeys a scalar equation without K, the discrete Gauss law
//     G_c^T (b + w^2 M E) = 0     (one row per free H1 node and one per floating conductor),
// whose residual is evaluated stably (M only).
// What: E = x + G_c dphi with -w^2 S dphi = g, S = G_c^T M G_c, g = q + w^2 G_c^T M x, q = G_c^T b with every row that
// is rounding of its own terms (|q| <= 1e-12 (|G_c|^T |b|)) zeroed, as AXF_FW_TC_CHARGE_PROJECTION=2 does on the
// gauged path. Splitting the error x - E* M-orthogonally into e_perp + G_c psi gives dphi = -psi: the correction
// removes the whole gradient part of the error and leaves e_perp, whose conditioning is that of the curl block. The
// M-norm of G_c dphi relative to that of x + G_c dphi is the estimate (field_fraction, the solved field's gradient
// error against the corrected field; field_fraction_of_solved, against x, saturates at 1 when the error dominates).
// G_c phi: phi in H1 of the field's order, zero on the grounded PEC set (the connected PEC set with the most nodes, normally the outer wall) and one constant
// on each other (floating) connected PEC set; S by AMG-PCG on the free nodes and a dense Schur complement for the
// floating potentials. Low-rank loads U diag(c) U^T (the validation loads) enter by the Woodbury identity, their Gauss
// rows G_c^T U projected like q (unprojected, their rounding charge left a 1.5e-8 loaded-field closure at 100 Hz).
// Scope: the caller applies it only where the assembled operator equals K - w^2 M on the free DOFs (closed, lossless,
// no Robin, PML or exterior; checked per frequency) and records why otherwise. The true-residual gates are unchanged
// (evaluated before the correction); the residual is then updated to r + A G_c dphi = r - w^2 M G_c dphi (+ the
// loads), exact algebra with no K applied to a gradient. The magnetic energy E^H K E still applies K to the whole
// field (the gauged path's curl representative avoids that).
// ---------------------------------------------------------------------------------------------------------
class FwGaussGuard {
 public:
  struct Reduced {        // a vector of the reduced potential space, or of its Gauss-law rows
    mfem::Vector f;       // free H1 true DOFs (zero on the PEC closure)
    Eigen::VectorXd c;    // one entry per floating PEC set (the same on every rank)
  };

  FwGaussGuard(mfem::ParFiniteElementSpace& nd, mfem::HypreParMatrix& M, const mfem::Array<int>& ess_nd,
               const std::vector<char>& closed_elem, const mfem::Array<int>& ess_bdr, int order)
      : M_(M), ess_nd_(ess_nd), h1c_(order, 3), h1_(nd.GetParMesh(), &h1c_) {
    Timer timer;
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    mfem::ParMesh& pm = *nd.GetParMesh();
    {
      mfem::ParDiscreteLinearOperator grad(&h1_, &nd);
      grad.AddDomainInterpolator(new mfem::GradientInterpolator());
      grad.Assemble();
      grad.Finalize();
      G_.reset(grad.ParallelAssemble());
    }
    const int nl = h1_.GetVSize();
    const int nt = h1_.GetTrueVSize();
    AXF_REQUIRE(h1_.GlobalTrueVSize() < std::numeric_limits<int>::max(),
                "Gauss-law guard: the H1 space exceeds 32-bit node numbers", "/fullwave");
    // 1. The PEC closure (the H1 nodes of closed elements and of essential boundary faces) and its connected sets: a
    //    local union-find over each closed element's and face's nodes, then the smallest global node number of each
    //    set exchanged over the shared nodes until no label changes.
    std::vector<int> parent(nl);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int v) {
      while (parent[v] != v) {
        parent[v] = parent[parent[v]];
        v = parent[v];
      }
      return v;
    };
    mfem::Array<int> mark(nl), dofs;
    mark = 0;
    auto join = [&]() {
      for (int k = 0; k < dofs.Size(); ++k) mark[dofs[k]] = 1;
      for (int k = 1; k < dofs.Size(); ++k) {
        const int a = find(dofs[0]);
        const int b = find(dofs[k]);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
      }
    };
    for (int e = 0; e < pm.GetNE(); ++e) {
      if (!closed_elem[e]) continue;
      h1_.GetElementDofs(e, dofs);
      join();
    }
    for (int be = 0; be < pm.GetNBE(); ++be) {
      const int attr = pm.GetBdrAttribute(be);
      if (attr < 1 || attr > ess_bdr.Size() || !ess_bdr[attr - 1]) continue;
      h1_.GetBdrElementDofs(be, dofs);
      join();
    }
    h1_.Synchronize(mark);
    const int none = std::numeric_limits<int>::max();
    mfem::Array<int> label(nl);
    for (int l = 0; l < nl; ++l) label[l] = mark[l] ? static_cast<int>(h1_.GetGlobalTDofNumber(l)) : none;
    std::vector<int> smallest(nl);
    int passes = 0;
    while (true) {
      ++passes;
      AXF_REQUIRE(passes < 100000, "Gauss-law guard: the conductor labels did not settle", "/fullwave");
      std::fill(smallest.begin(), smallest.end(), none);
      for (int l = 0; l < nl; ++l)
        if (mark[l]) smallest[find(l)] = std::min(smallest[find(l)], label[l]);
      int changed = 0;
      for (int l = 0; l < nl; ++l)
        if (mark[l] && smallest[find(l)] != label[l]) {
          label[l] = smallest[find(l)];
          changed = 1;
        }
      mfem::Array<int> exchanged(label);
      h1_.GroupComm().Reduce<int>(exchanged, mfem::GroupCommunicator::Min);
      h1_.GroupComm().Bcast(exchanged);
      for (int l = 0; l < nl; ++l)
        if (exchanged[l] != label[l]) {
          label[l] = exchanged[l];
          changed = 1;
        }
      MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (!changed) break;
    }
    // 2. The sets on true DOFs, counted over all ranks; the grounded set has the most nodes (ties: smallest label).
    std::vector<int> tlabel(nt, none);
    for (int l = 0; l < nl; ++l) {
      const int t = h1_.GetLocalTDofNumber(l);
      if (t >= 0 && mark[l]) tlabel[t] = label[l];
    }
    std::map<int, double> local_count;
    for (int t = 0; t < nt; ++t)
      if (tlabel[t] != none) local_count[tlabel[t]] += 1.0;
    std::vector<int> my_labels;
    std::vector<double> my_counts;
    for (const auto& [set, nodes] : local_count) {
      my_labels.push_back(set);
      my_counts.push_back(nodes);
    }
    const int mine = static_cast<int>(my_labels.size());
    std::vector<int> sizes(ranks), offsets(ranks, 0);
    MPI_Allgather(&mine, 1, MPI_INT, sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int r = 1; r < ranks; ++r) offsets[r] = offsets[r - 1] + sizes[r - 1];
    const int total = offsets[ranks - 1] + sizes[ranks - 1];
    std::vector<int> all_labels(total);
    std::vector<double> all_counts(total);
    MPI_Allgatherv(my_labels.data(), mine, MPI_INT, all_labels.data(), sizes.data(), offsets.data(), MPI_INT,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(my_counts.data(), mine, MPI_DOUBLE, all_counts.data(), sizes.data(), offsets.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    std::map<int, double> count;
    for (int k = 0; k < total; ++k) count[all_labels[k]] += all_counts[k];
    topology_ = {{"h1_order", order}, {"h1_true_dofs", h1_.GlobalTrueVSize()}, {"label_exchange_passes", passes},
                 {"pec_sets", count.size()}};
    if (count.empty()) {
      reason_ = "no PEC set to ground the potential";
      topology_["setup_seconds"] = fw_allmax(timer.seconds());
      return;
    }
    int grounded = count.begin()->first;
    for (const auto& [set, nodes] : count)
      if (nodes > count.at(grounded)) grounded = set;
    std::map<int, int> floating;
    json floating_nodes = json::array();
    for (const auto& [set, nodes] : count) {
      if (set == grounded) continue;
      const int k = static_cast<int>(floating.size());
      floating[set] = k;
      floating_nodes.push_back(nodes);
    }
    nc_ = static_cast<int>(floating.size());
    topology_["grounded_set_nodes"] = count.at(grounded);
    topology_["floating_set_nodes"] = floating_nodes;
    if (nc_ > kMaxFloating) {
      reason_ = strcat(nc_, " floating PEC sets (the guard takes at most ", kMaxFloating, ")");
      topology_["setup_seconds"] = fw_allmax(timer.seconds());
      return;
    }
    comp_.assign(nt, kFree);
    for (int t = 0; t < nt; ++t) {
      if (tlabel[t] == none) continue;
      comp_[t] = tlabel[t] == grounded ? kGrounded : floating.at(tlabel[t]);
      closure_.Append(t);
    }
    // 3. Consistency with the ND essential DOFs: G_c phi must vanish on every essential edge.
    {
      Reduced probe;
      probe.f.SetSize(nt);
      probe.f.Randomize(9173 + rank);
      probe.c.resize(nc_);
      for (int k = 0; k < nc_; ++k) probe.c(k) = 1.0 + 0.5 * k;
      mfem::Vector v(G_->Height());
      G_->Mult(Prolong(probe), v);
      double on_essential = 0.0;
      for (int i : ess_nd_) on_essential = std::max(on_essential, std::fabs(v(i)));
      const double ratio = fw_allmax(on_essential) / std::max(fw_allmax(v.Normlinf()), 1e-300);
      topology_["essential_edge_leak_relative"] = ratio;
      if (!(ratio <= 1e-12)) {
        reason_ = strcat("the H1 PEC closure and the ND essential DOFs differ (G phi on essential edges ", ratio, ")");
        topology_["setup_seconds"] = fw_allmax(timer.seconds());
        return;
      }
    }
    // 4. the S preconditioner and the floating potentials' Schur block, on M (SetLoss rebuilds them on M + Mi)
    json prep;
    const std::string failure = Prepare(M_, prep);
    topology_.update(prep);
    if (!failure.empty()) reason_ = failure;
    topology_["setup_seconds"] = fw_allmax(timer.seconds());
  }

  // S = G_c^T Mp G_c on the free nodes (the closure eliminated) by AMG-PCG; the floating potentials by their Schur
  // complement: s_k = (G^T Mp G chi_k) on the free nodes, Y_k = S_ff^{-1} s_k, Sigma = S_cc - s^T Y. Mp is M (lossless)
  // or M + Mi (SetLoss). Returns why it failed (empty if not); stats gets the PCG iterations and achieved residual and
  // the Schur spectrum.
  std::string Prepare(const mfem::HypreParMatrix& Mp, json& stats) {
    Mp_ = &Mp;
    const int nt = h1_.GetTrueVSize();
    S_.reset(mfem::RAP(const_cast<mfem::HypreParMatrix*>(&Mp), G_.get()));
    ensure_diagonal(S_, closure_);  // a no-op unless hypre on a GPU dropped a closure row (keep_eliminated_rows)
    S_->EliminateBC(closure_, mfem::Operator::DIAG_ONE);
    amg_ = std::make_unique<mfem::HypreBoomerAMG>(*S_);
    amg_->SetPrintLevel(0);
    pcg_ = std::make_unique<mfem::HyprePCG>(*S_);
    pcg_->SetTol(1e-13);
    pcg_->SetMaxIter(2000);
    pcg_->SetPrintLevel(0);
    pcg_->SetPreconditioner(*amg_);
    pcg_->iterative_mode = false;
    // Jacobi weights diag(S)^{-1/2}: residuals are judged in this scaled norm, which a coefficient jump (a conductive
    // region, sigma / (omega eps0) ~ 1e10 and beyond) does not dominate the way it does the plain 2-norm
    const mfem::Vector diag = stored_diagonal(*S_);   // not GetDiag (model3d_impl.hpp)
    weight_.SetSize(nt);
    for (int t = 0; t < nt; ++t) weight_(t) = diag(t) > 0.0 ? 1.0 / std::sqrt(diag(t)) : 1.0;
    Eigen::MatrixXd Scc = Eigen::MatrixXd::Zero(nc_, nc_);
    s_.assign(nc_, mfem::Vector());
    Y_.assign(nc_, mfem::Vector());
    SolveStats solves;
    for (int k = 0; k < nc_; ++k) {
      Reduced chi;
      chi.f.SetSize(nt);
      chi.f = 0.0;
      chi.c = Eigen::VectorXd::Zero(nc_);
      chi.c(k) = 1.0;
      const Reduced s = GcT(MassP(Gc(chi)));
      Scc.col(k) = s.c;
      s_[k] = s.f;
      Y_[k] = SolveFree(s_[k], solves);
    }
    weight_c_.resize(nc_);
    for (int k = 0; k < nc_; ++k) weight_c_(k) = Scc(k, k) > 0.0 ? 1.0 / std::sqrt(Scc(k, k)) : 1.0;
    stats["setup_pcg_iterations_max"] = solves.iterations;
    stats["setup_pcg_relative_residual_max"] = solves.residual;
    stats["setup_pcg_refinements_max"] = solves.refinements;
    if (!(solves.residual <= kSolveTolerance))
      return strcat("the floating potentials' PCG solves missed their tolerance (relative residual ", solves.residual, ")");
    if (nc_ > 0) {
      Eigen::MatrixXd schur = Scc;
      for (int j = 0; j < nc_; ++j)
        for (int k = 0; k < nc_; ++k) schur(j, k) -= fw_gdot(s_[j], Y_[k]);
      schur = 0.5 * (schur + schur.transpose()).eval();
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(schur);
      const double lo = spectrum.eigenvalues()(0);
      const double hi = spectrum.eigenvalues()(nc_ - 1);
      stats["schur_eigenvalue_ratio"] = lo / std::max(hi, 1e-300);
      if (!(lo > 1e-13 * hi))
        return strcat("singular floating-conductor block (eigenvalue ratio ", lo / std::max(hi, 1e-300),
                      "): a field region with no grounded PEC set");
      schur_.compute(schur);
    }
    return "";
  }

  // The loss mass at this frequency, Mi (Im eps_c; empty when lossless): M_c = M + i Mi. The guard owns it, so it lives
  // until the next SetLoss. With loss, the S preconditioner is rebuilt on the SPD M + Mi. Returns why it failed (empty
  // if not); stats as Prepare's.
  std::string SetLoss(std::unique_ptr<mfem::HypreParMatrix> Mi, json& stats) {
    Mi_owned_ = std::move(Mi);
    Mi_ = Mi_owned_.get();
    if (!Mi_) return Mp_ == &M_ ? std::string() : Prepare(M_, stats);
    Msum_.reset(mfem::Add(1.0, M_, 1.0, *Mi_));
    return Prepare(*Msum_, stats);
  }
  const mfem::HypreParMatrix* LossMass() const { return Mi_; }

  bool available() const { return reason_.empty(); }
  const std::string& reason() const { return reason_; }
  const json& Topology() const { return topology_; }

  // One correction of x = [Re; Im] (true DOFs) for A = K - w^2 (M + i Mi) (+ U diag(c) U^T on [Re; Im] as FwOperator),
  // b the right-hand side, r = A x - b (updated with x when given). apply = false only estimates. Nothing is changed,
  // and not_applied says why, when an S-solve missed its tolerance or the estimate is not finite.
  json Correct(double w, mfem::Vector& x, const mfem::Vector& b, mfem::Vector* r, bool apply,
               const mfem::DenseMatrix* U = nullptr, const std::vector<cplx>* c = nullptr) const {
    Timer timer;
    const int n = G_->Height();
    const double w2 = w * w;
    const int nm = U ? U->Width() : 0;
    SolveStats solves;
    double zeroed = 0.0;
    double charged = 0.0;
    std::array<Reduced, 2> q, g, delta;
    std::array<mfem::Vector, 2> Mx, Mcx, dx, Mdx, Mcdx;   // M x (energy norms), M_c x (Gauss rows), and the same of dx
    // the loads' Gauss rows W_j = G_c^T U_j (rounding rows zeroed, as the gauged path's projection treats its load
    // columns) and their current on x, s = c (U^T x)
    std::vector<Reduced> W(nm);
    for (int j = 0; j < nm; ++j) {
      mfem::Vector column(n);
      U->GetColumn(j, column);
      W[j] = ChargeRows(column, zeroed, charged);
    }
    const Eigen::VectorXcd s = LoadCurrents(x, U, c);
    for (int p = 0; p < 2; ++p) {
      mfem::Vector xp(x.GetData() + p * n, n);
      Mx[p] = MassTimes(xp);
    }
    MassC(Mx, x, Mcx);
    double field_terms2 = 0.0;
    for (int p = 0; p < 2; ++p) {
      mfem::Vector bp(n);
      for (int i = 0; i < n; ++i) bp(i) = b(p * n + i);
      q[p] = ChargeRows(bp, zeroed, charged);
      const Reduced m = GcT(Mcx[p]);
      g[p] = q[p];
      g[p].f.Add(w2, m.f);
      g[p].c += w2 * m.c;
      field_terms2 += FieldTerms2(Mcx[p], w2);
    }
    for (int j = 0; j < nm; ++j) {
      g[0].f.Add(-s(j).real(), W[j].f);
      g[0].c -= s(j).real() * W[j].c;
      g[1].f.Add(-s(j).imag(), W[j].f);
      g[1].c -= s(j).imag() * W[j].c;
    }
    const double before2 = Dot(g[0], g[0]) + Dot(g[1], g[1]);
    // delta = T^{-1} g, T = -w^2 S_c (+ W C W^T by Woodbury: T0^{-1} - Z C (I + W^T Z C)^{-1} W^T T0^{-1}, Z = T0^{-1} W,
    // complex when lossy)
    SolveC(g[0], g[1], delta[0], delta[1], solves);
    Scale(delta[0], -1.0 / w2);
    Scale(delta[1], -1.0 / w2);
    if (nm > 0) {
      std::vector<Reduced> Zr(nm), Zi(nm);
      Reduced zero = W[0];
      Scale(zero, 0.0);
      for (int j = 0; j < nm; ++j) {
        SolveC(W[j], zero, Zr[j], Zi[j], solves);
        Scale(Zr[j], -1.0 / w2);
        Scale(Zi[j], -1.0 / w2);
      }
      Eigen::MatrixXcd H = Eigen::MatrixXcd::Identity(nm, nm);
      Eigen::VectorXcd u(nm);
      for (int i = 0; i < nm; ++i) {
        for (int j = 0; j < nm; ++j) H(i, j) += cplx(Dot(W[i], Zr[j]), Dot(W[i], Zi[j])) * (*c)[j];
        u(i) = cplx(Dot(W[i], delta[0]), Dot(W[i], delta[1]));
      }
      const Eigen::VectorXcd v = H.fullPivLu().solve(u);
      for (int j = 0; j < nm; ++j) {
        const cplx cv = (*c)[j] * v(j);
        delta[0].f.Add(-cv.real(), Zr[j].f);
        delta[0].c -= cv.real() * Zr[j].c;
        delta[1].f.Add(-cv.imag(), Zr[j].f);
        delta[1].c -= cv.imag() * Zr[j].c;
        if (!Mi_) continue;   // Z is real without loss
        delta[0].f.Add(cv.imag(), Zi[j].f);
        delta[0].c += cv.imag() * Zi[j].c;
        delta[1].f.Add(-cv.real(), Zi[j].f);
        delta[1].c -= cv.real() * Zi[j].c;
      }
    }
    // energy norms (M) of the correction, the solved field and the corrected field (the last from x + dx itself: the
    // expanded form cancels when x is almost all gradient error)
    double dd = 0.0;
    double xx = 0.0;
    double corrected = 0.0;
    for (int p = 0; p < 2; ++p) {
      dx[p] = Gc(delta[p]);
      Mdx[p] = MassTimes(dx[p]);
      mfem::Vector xp(x.GetData() + p * n, n), xn(xp), Mxn(Mx[p]);
      xn += dx[p];
      Mxn += Mdx[p];
      dd += fw_gdot(dx[p], Mdx[p]);
      xx += fw_gdot(xp, Mx[p]);
      corrected += fw_gdot(xn, Mxn);
    }
    const mfem::Vector dx_all = Concat(dx);
    MassC(Mdx, dx_all, Mcdx);
    const double infinite = std::numeric_limits<double>::infinity();
    const bool finite = std::isfinite(dd) && std::isfinite(corrected) && std::isfinite(before2);
    const double fraction = !finite ? std::numeric_limits<double>::quiet_NaN()
                          : corrected > 0 ? std::sqrt(dd / corrected) : (dd > 0 ? infinite : 0.0);
    std::string failure;
    if (!(solves.residual <= kSolveTolerance))
      failure = strcat("the potential's S-solve missed its tolerance (relative residual ", solves.residual, ")");
    else if (!finite)
      failure = "the Gauss residual or the correction is not finite";
    else if (!std::isfinite(fraction))
      failure = "the estimate is not finite (the solved field is all gradient error)";
    const bool applied = apply && failure.empty();
    zeroed = fw_allsum(zeroed);
    charged = fw_allsum(charged);
    json out = {{"applied", applied}, {"field_fraction", fraction},
                {"field_fraction_of_solved", xx > 0 ? std::sqrt(dd / xx) : infinite},
                {"gauss_residual_before", std::sqrt(before2 / std::max(field_terms2, 1e-300))},
                {"rounding_charge_rows_zeroed", zeroed}, {"charged_rows", charged},
                {"floating_conductors", nc_}, {"loads", nm}, {"lossy", Mi_ != nullptr}};
    if (!failure.empty()) out["not_applied"] = failure;
    if (applied) {
      // E <- x + G_c delta; r <- r + A G_c delta = r - w^2 M_c G_c delta + U C U^T G_c delta (A's essential rows are
      // the identity and G_c delta vanishes there)
      const Eigen::VectorXcd ds = LoadCurrents(dx_all, U, c);
      for (int p = 0; p < 2; ++p) {
        mfem::Vector xp(x.GetData() + p * n, n);
        xp += dx[p];
        if (!r) continue;
        mfem::Vector rp(r->GetData() + p * n, n), update(Mcdx[p]);
        update *= -w2;
        for (int i : ess_nd_) update(i) = 0.0;
        rp += update;
        for (int j = 0; j < nm; ++j) {
          mfem::Vector column(n);
          U->GetColumn(j, column);
          rp.Add(p == 0 ? ds(j).real() : ds(j).imag(), column);
        }
      }
      // the Gauss residual after, from the updated x (stable: no K), against the field's own Gauss terms, and the
      // reduction the solve achieved
      const Eigen::VectorXcd s_after = s + ds;
      double after2 = 0.0;
      double field_terms_after2 = 0.0;
      for (int p = 0; p < 2; ++p) {
        mfem::Vector Mcxn(Mcx[p]);
        Mcxn += Mcdx[p];
        const Reduced m = GcT(Mcxn);
        Reduced ga = q[p];
        ga.f.Add(w2, m.f);
        ga.c += w2 * m.c;
        for (int j = 0; j < nm; ++j) {
          const double sj = p == 0 ? s_after(j).real() : s_after(j).imag();
          ga.f.Add(-sj, W[j].f);
          ga.c -= sj * W[j].c;
        }
        after2 += Dot(ga, ga);
        field_terms_after2 += FieldTerms2(Mcxn, w2);
      }
      out["gauss_residual_after"] = std::sqrt(after2 / std::max(field_terms_after2, 1e-300));
      out["gauss_residual_reduction"] = before2 > 0 ? std::sqrt(after2 / before2) : 0.0;
    }
    out["pcg_iterations_max"] = solves.iterations;
    out["pcg_relative_residual_max"] = solves.residual;
    out["pcg_refinements_max"] = solves.refinements;
    if (Mi_) {
      out["gmres_iterations"] = solves.outer;
      out["gmres_inner_pcg_residual_max"] = solves.inner;
    }
    out["seconds"] = fw_allmax(timer.seconds());
    return out;
  }

  // The curl representative x - G_c S_p^{-1} G_c^T M_p x of x = [Re; Im] (M_p = M, or M + Mi with loss), the field less
  // its M_p-orthogonal gradient part: K applied to it equals K x exactly (K G = 0), without K's rounding on a large
  // gradient part (the ungauged paths' magnetic energy and probe curls). An inexact potential changes only how much
  // gradient is left, not K x.
  mfem::Vector CurlPart(const mfem::Vector& x) const {
    const int n = G_->Height();
    mfem::Vector out(x);
    SolveStats solves;
    for (int p = 0; p < 2; ++p) {
      mfem::Vector xp(const_cast<double*>(x.GetData()) + p * n, n);
      const Reduced moment = GcT(MassP(xp));
      if (Dot(moment, moment) == 0.0) continue;
      const mfem::Vector gradient = Gc(SolveS(moment, solves));
      mfem::Vector op(out.GetData() + p * n, n);
      op -= gradient;
    }
    return out;
  }

 private:
  static constexpr int kFree = -2;
  static constexpr int kGrounded = -1;
  static constexpr int kMaxFloating = 64;
  static constexpr double kSolveTolerance = 1e-10;   // achieved ||S y - r|| / ||r|| a correction is accepted with
  static constexpr int kRefinePasses = 4;            // iterative refinements of a PCG solve (SolveFree)

  struct SolveStats {     // over the S-solves of one correction
    int iterations = 0;   // largest PCG iteration count
    double residual = 0;  // largest achieved relative residual (the free-node block; the complex S with loss)
    int outer = 0;        // GMRES iterations (complex S, with loss)
    int refinements = 0;  // largest number of iterative refinements of a PCG solve
    double inner = 0;     // largest backward error of GMRES's inner PCG solves (recorded, not gating)
  };
  struct Pair {           // a complex reduced vector, [Re; Im]
    Reduced r, i;
  };

  static bool rank0() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank == 0;
  }
  static void Scale(Reduced& v, double a) {
    v.f *= a;
    v.c *= a;
  }
  static mfem::Vector Concat(const std::array<mfem::Vector, 2>& parts) {
    const int n = parts[0].Size();
    mfem::Vector v(2 * n);
    for (int i = 0; i < n; ++i) {
      v(i) = parts[0](i);
      v(n + i) = parts[1](i);
    }
    return v;
  }
  // c_j (U^T v)_j for v = [Re; Im] (empty without loads)
  static Eigen::VectorXcd LoadCurrents(const mfem::Vector& v, const mfem::DenseMatrix* U, const std::vector<cplx>* c) {
    const int nm = U ? U->Width() : 0;
    Eigen::VectorXcd s(nm);
    if (nm == 0) return s;
    const int n = v.Size() / 2;
    std::vector<double> t(2 * nm, 0.0);
    mfem::Vector tr(t.data(), nm), ti(t.data() + nm, nm);
    mfem::Vector vr(const_cast<double*>(v.GetData()), n), vi(const_cast<double*>(v.GetData()) + n, n);
    if (n > 0) {
      U->MultTranspose(vr, tr);
      U->MultTranspose(vi, ti);
    }
    MPI_Allreduce(MPI_IN_PLACE, t.data(), 2 * nm, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int j = 0; j < nm; ++j) s(j) = (*c)[j] * cplx(t[j], t[nm + j]);
    return s;
  }
  // G_c^T v (essential rows of v dropped) with every row that is rounding of its own terms, |row| <= 1e-12 (|G_c|^T
  // |v|), set to zero. A floating set's scale sums |G|^T |v| over its nodes, which also counts free chord edges with
  // both ends on the set: a slightly looser threshold than the gauged path's contracted |G_c|. Counts the zeroed and
  // the charged (kept nonzero) rows: free nodes on every rank (summed by the caller), conductor rows on rank 0 only
  // (they are the same on every rank).
  Reduced ChargeRows(const mfem::Vector& v, double& zeroed, double& charged) const {
    const int n = v.Size();
    mfem::Vector vf(v), vabs(n), hscale(G_->Width());
    for (int i : ess_nd_) vf(i) = 0.0;
    for (int i = 0; i < n; ++i) vabs(i) = std::fabs(vf(i));
    G_->AbsMultTranspose(vabs, hscale);
    const Reduced scale = Restrict(hscale);
    Reduced q = GcT(vf);
    for (int t = 0; t < q.f.Size(); ++t) {
      if (comp_[t] != kFree || q.f(t) == 0.0) continue;
      if (std::fabs(q.f(t)) <= 1e-12 * scale.f(t)) {
        q.f(t) = 0.0;
        zeroed += 1.0;
      } else {
        charged += 1.0;
      }
    }
    for (int k = 0; k < nc_; ++k) {
      if (q.c(k) == 0.0) continue;
      const bool rounding = std::fabs(q.c(k)) <= 1e-12 * scale.c(k);
      if (rounding) q.c(k) = 0.0;
      if (rank0()) (rounding ? zeroed : charged) += 1.0;
    }
    return q;
  }
  mfem::Vector MassTimes(const mfem::Vector& v) const {
    mfem::Vector y(v.Size());
    M_.Mult(v, y);
    return y;
  }
  // the preconditioner's SPD mass M_p (M, or M + Mi with loss)
  mfem::Vector MassP(const mfem::Vector& v) const {
    mfem::Vector y(v.Size());
    Mp_->Mult(v, y);
    return y;
  }
  // M_c x = (M + i Mi)(x_r + i x_i) per part, given M x (Mx) and x = [Re; Im]: [M x_r - Mi x_i; M x_i + Mi x_r]
  void MassC(const std::array<mfem::Vector, 2>& Mx, const mfem::Vector& x, std::array<mfem::Vector, 2>& out) const {
    out[0] = Mx[0];
    out[1] = Mx[1];
    if (!Mi_) return;
    const int n = Mx[0].Size();
    mfem::Vector xr(const_cast<double*>(x.GetData()), n), xi(const_cast<double*>(x.GetData()) + n, n);
    mfem::Vector t(n);
    Mi_->Mult(xi, t);
    out[0] -= t;
    Mi_->Mult(xr, t);
    out[1] += t;
  }
  // || w^2 |G_c|^T |M x| ||^2 (free edges only): the size of the field's own terms in the Gauss rows
  double FieldTerms2(const mfem::Vector& Mx, double w2) const {
    mfem::Vector mabs(Mx.Size()), mscale(G_->Width());
    for (int i = 0; i < Mx.Size(); ++i) mabs(i) = std::fabs(Mx(i));
    for (int i : ess_nd_) mabs(i) = 0.0;
    G_->AbsMultTranspose(mabs, mscale);
    Reduced terms = Restrict(mscale);
    Scale(terms, w2);
    return Dot(terms, terms);
  }
  // the reduced rows of an H1 true-DOF vector: free nodes as they are, each floating set summed, the grounded set dropped
  Reduced Restrict(const mfem::Vector& h) const {
    Reduced r;
    r.f = h;
    r.c = Eigen::VectorXd::Zero(nc_);
    for (int t = 0; t < r.f.Size(); ++t) {
      if (comp_[t] == kFree) continue;
      if (comp_[t] >= 0) r.c(comp_[t]) += h(t);
      r.f(t) = 0.0;
    }
    if (nc_ > 0) MPI_Allreduce(MPI_IN_PLACE, r.c.data(), nc_, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return r;
  }
  // the H1 potential of a reduced vector: the floating sets' constants on their nodes, zero on the grounded set
  mfem::Vector Prolong(const Reduced& r) const {
    mfem::Vector h(r.f);
    for (int t = 0; t < h.Size(); ++t) {
      if (comp_[t] == kFree) continue;
      h(t) = comp_[t] >= 0 ? r.c(comp_[t]) : 0.0;
    }
    return h;
  }
  // G_c^T v; the essential rows of v are dropped (G_c vanishes there, so this changes only rounding in the
  // conductor sums)
  Reduced GcT(const mfem::Vector& v) const {
    mfem::Vector vf(v), h(G_->Width());
    for (int i : ess_nd_) vf(i) = 0.0;
    G_->MultTranspose(vf, h);
    return Restrict(h);
  }
  mfem::Vector Gc(const Reduced& r) const {
    mfem::Vector v(G_->Height());
    G_->Mult(Prolong(r), v);
    for (int i : ess_nd_) v(i) = 0.0;
    return v;
  }
  double Dot(const Reduced& a, const Reduced& b) const { return fw_gdot(a.f, b.f) + a.c.dot(b.c); }
  // y = S_ff^{-1} f by PCG (zero on the closure); records its iterations and achieved ||S y - f|| / ||f|| in the
  // Jacobi-weighted norm, as a backward error ||S y - f|| / (|| |S| |y| || + ||f||) (HyprePCG returns at its iteration
  // cap without an error). Under a large coefficient jump (a resistive block, sigma / (omega eps0) ~ 1e10) the potential
  // there is nearly constant and S y cancels: the residual against f alone stalls at that rounding floor (~1e-6), while
  // against the rows' own terms it reaches rounding. PCG's own stopping norm can also stop short of the weighted one:
  // iterative refinement re-solves for the remaining residual, up to kRefinePasses.
  mfem::Vector SolveFree(const mfem::Vector& f, SolveStats& solves) const {
    const int n = f.Size();
    mfem::Vector y(n), residual(n), dy(n), yabs(n), terms(n);
    y = 0.0;
    const double f2 = fw_gdot(f, f);
    if (!(f2 > 0.0)) return y;
    pcg_->Mult(f, y);
    int its = 0;
    pcg_->GetNumIterations(its);
    int total = its;
    double relative = 0.0;
    int passes = 0;
    while (true) {
      S_->Mult(y, residual);
      residual -= f;
      for (int t = 0; t < n; ++t) yabs(t) = std::fabs(y(t));
      S_->AbsMult(yabs, terms);
      double r2 = 0.0;   // in the Jacobi-weighted norm
      double t2 = 0.0;
      for (int t = 0; t < n; ++t) {
        const double w2 = weight_(t) * weight_(t);
        const double scale = terms(t) + std::fabs(f(t));
        r2 += residual(t) * residual(t) * w2;
        t2 += scale * scale * w2;
      }
      relative = std::sqrt(fw_allsum(r2) / std::max(fw_allsum(t2), 1e-300));
      if (relative <= 0.01 * kSolveTolerance || passes >= kRefinePasses) break;
      ++passes;
      residual.Neg();
      dy = 0.0;
      pcg_->Mult(residual, dy);
      pcg_->GetNumIterations(its);
      total += its;
      y += dy;
    }
    solves.iterations = std::max(solves.iterations, total);
    if (!(relative <= solves.residual)) solves.residual = relative;   // a NaN propagates (std::max would drop it)
    solves.refinements = std::max(solves.refinements, passes);
    return y;
  }
  // the Jacobi-weighted inner product of the reduced space (free nodes by diag(S)^{-1/2}, floating sets by S_cc^{-1/2})
  double DotW(const Reduced& a, const Reduced& b) const {
    double s = 0.0;
    for (int t = 0; t < a.f.Size(); ++t) s += a.f(t) * b.f(t) * weight_(t) * weight_(t);
    return fw_allsum(s) + a.c.cwiseProduct(weight_c_).dot(b.c.cwiseProduct(weight_c_));
  }
  // S^{-1} r by the Schur complement of the floating potentials
  Reduced SolveS(const Reduced& r, SolveStats& solves) const {
    Reduced x;
    x.f = SolveFree(r.f, solves);
    x.c = Eigen::VectorXd::Zero(nc_);
    if (nc_ > 0) {
      Eigen::VectorXd t = r.c;
      for (int j = 0; j < nc_; ++j) t(j) -= fw_gdot(s_[j], x.f);
      x.c = schur_.solve(t);
      for (int k = 0; k < nc_; ++k) x.f.Add(-x.c(k), Y_[k]);
    }
    return x;
  }
  // S_c y = g for S_c = G_c^T (M + i Mi) G_c, g = gr + i gi. Without loss, two real S-solves. With loss, right-
  // preconditioned restarted GMRES on the real 2x2 block form, preconditioned by S_p = G_c^T (M + Mi) G_c on each part:
  // the numerical range of S_p^{-1} S_c (Rayleigh quotients v^H S_c v / v^H S_p v) lies on the segment from 1 to i,
  // (1 + i t)/(1 + t), t >= 0, with |lambda| >= 1/sqrt(2) for any loss. GMRES is flexible (it keeps the preconditioned
  // vectors), so the inner PCG solves need not be exact and do not gate acceptance (their own backward error is recorded
  // apart). It runs in the Jacobi-weighted inner product; after each restart it takes the backward error ||g - S_c y|| /
  // (|| |S_c| |y| || + ||g||), with a bound on |S_c| |y|, and stops at 1e-12, or when the residual has failed to halve
  // over two cycles (or over one, once the backward error is below 1e-11): a rounding floor. That backward error enters
  // solves.residual.
  void SolveC(const Reduced& gr, const Reduced& gi, Reduced& yr, Reduced& yi, SolveStats& solves) const {
    if (!Mi_) {
      yr = SolveS(gr, solves);
      yi = SolveS(gi, solves);
      return;
    }
    const Pair g{gr, gi};
    Pair y{gr, gi};
    y.r.f = 0.0;
    y.r.c.setZero();
    y.i.f = 0.0;
    y.i.c.setZero();
    const double gnorm = std::sqrt(DotPair(g, g));
    if (!std::isfinite(gnorm)) {
      solves.residual = gnorm;   // fails the acceptance
      yr = y.r;
      yi = y.i;
      return;
    }
    if (!(gnorm > 0.0)) {
      yr = y.r;
      yi = y.i;
      return;
    }
    SolveStats inner;
    const int restart = 30;
    const int max_iterations = 600;
    const double tolerance = 1e-13;
    Pair residual = g;
    double beta = gnorm;
    double previous = gnorm;
    double backward = 1.0;
    int iterations = 0;
    int slow = 0;   // consecutive restart cycles that did not halve the residual
    while (iterations < max_iterations) {
      std::vector<Pair> V, Zp;
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(restart + 1, restart);
      V.push_back(residual);
      ScalePair(V[0], 1.0 / beta);
      int k = 0;
      Eigen::VectorXd coefficients;
      for (; k < restart && iterations < max_iterations; ++k, ++iterations) {
        Pair z{SolveS(V[k].r, inner), SolveS(V[k].i, inner)};
        Pair wv = ApplySc(z);
        for (int i = 0; i <= k; ++i) {
          H(i, k) = DotPair(wv, V[i]);
          AxpyPair(wv, -H(i, k), V[i]);
        }
        H(k + 1, k) = std::sqrt(DotPair(wv, wv));
        Zp.push_back(z);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(k + 2);
        rhs(0) = beta;
        const Eigen::MatrixXd Hk = H.topLeftCorner(k + 2, k + 1);
        coefficients = Hk.colPivHouseholderQr().solve(rhs);
        const double estimate = (rhs - Hk * coefficients).norm();
        if (H(k + 1, k) == 0.0 || estimate <= tolerance * gnorm) {
          ++k;
          ++iterations;
          break;
        }
        ScalePair(wv, 1.0 / H(k + 1, k));
        V.push_back(wv);
      }
      for (int j = 0; j < static_cast<int>(coefficients.size()); ++j) AxpyPair(y, coefficients(j), Zp[j]);
      residual = g;
      AxpyPair(residual, -1.0, ApplySc(y));
      beta = std::sqrt(DotPair(residual, residual));
      const Reduced terms = AbsSc(y);
      backward = beta / (std::sqrt(DotW(terms, terms)) + gnorm);
      if (beta <= tolerance * gnorm || backward <= 0.01 * kSolveTolerance) break;
      slow = beta > 0.5 * previous ? slow + 1 : 0;
      if (slow >= 2 || (slow >= 1 && backward <= 0.1 * kSolveTolerance)) break;   // a floor: the backward error decides
      previous = beta;
    }
    solves.outer += iterations;
    solves.iterations = std::max(solves.iterations, inner.iterations);
    solves.refinements = std::max(solves.refinements, inner.refinements);
    if (!(inner.residual <= solves.inner)) solves.inner = inner.residual;
    if (!(backward <= solves.residual)) solves.residual = backward;   // a NaN propagates
    yr = y.r;
    yi = y.i;
  }
  // A bound on |S_c| |y| per reduced row, |G_c|^T (|M| + |Mi|) |G_c| (|y_r| + |y_i|): the size of the rows' own terms
  Reduced AbsSc(const Pair& y) const {
    Reduced m = y.r;
    for (int t = 0; t < m.f.Size(); ++t) m.f(t) = std::fabs(y.r.f(t)) + std::fabs(y.i.f(t));
    m.c = y.r.c.cwiseAbs() + y.i.c.cwiseAbs();
    const mfem::Vector h = Prolong(m);
    mfem::Vector a(G_->Height()), ma(G_->Height()), mia(G_->Height()), out(G_->Width());
    G_->AbsMult(h, a);
    for (int i : ess_nd_) a(i) = 0.0;
    M_.AbsMult(a, ma);
    if (Mi_) {
      Mi_->AbsMult(a, mia);
      ma += mia;
    }
    G_->AbsMultTranspose(ma, out);
    return Restrict(out);
  }
  // S_c v = G_c^T (M + i Mi) G_c v
  Pair ApplySc(const Pair& v) const {
    const std::array<mfem::Vector, 2> a{Gc(v.r), Gc(v.i)};
    const std::array<mfem::Vector, 2> Ma{MassTimes(a[0]), MassTimes(a[1])};
    std::array<mfem::Vector, 2> Mca;
    MassC(Ma, Concat(a), Mca);
    return Pair{GcT(Mca[0]), GcT(Mca[1])};
  }
  // GMRES works in the Jacobi-weighted inner product (DotW), so that its residual norm is the scaled one
  double DotPair(const Pair& a, const Pair& b) const { return DotW(a.r, b.r) + DotW(a.i, b.i); }
  static void ScalePair(Pair& v, double a) {
    Scale(v.r, a);
    Scale(v.i, a);
  }
  static void AxpyPair(Pair& y, double a, const Pair& x) {
    y.r.f.Add(a, x.r.f);
    y.r.c += a * x.r.c;
    y.i.f.Add(a, x.i.f);
    y.i.c += a * x.i.c;
  }

  mfem::HypreParMatrix& M_;
  const mfem::Array<int>& ess_nd_;
  mfem::H1_FECollection h1c_;
  mfem::ParFiniteElementSpace h1_;
  std::unique_ptr<mfem::HypreParMatrix> G_, S_;
  std::unique_ptr<mfem::HypreBoomerAMG> amg_;
  std::unique_ptr<mfem::HyprePCG> pcg_;
  std::vector<int> comp_;          // per H1 true DOF: kFree, kGrounded or the floating set's index
  mfem::Array<int> closure_;       // H1 true DOFs on the PEC closure (grounded and floating)
  int nc_ = 0;                     // floating sets
  std::vector<mfem::Vector> s_, Y_;
  mfem::Vector weight_;            // diag(S)^{-1/2} on the H1 true DOFs (Jacobi weights of the residual checks)
  Eigen::VectorXd weight_c_;       // S_cc^{-1/2} of the floating sets
  Eigen::LDLT<Eigen::MatrixXd> schur_;
  const mfem::HypreParMatrix* Mp_ = nullptr;   // the preconditioner's mass: M, or Msum_ = M + Mi with loss
  const mfem::HypreParMatrix* Mi_ = nullptr;   // the loss mass at the current frequency (SetLoss), or none
  std::unique_ptr<mfem::HypreParMatrix> Mi_owned_;
  std::unique_ptr<mfem::HypreParMatrix> Msum_;
  std::string reason_;
  json topology_;
};

// A deterministic pseudo-random vector field of position only: projected start vectors are then the same field
// for any MPI partition (rank-count-independent Lanczos sequences up to round-off).
class FwHashField : public mfem::VectorCoefficient {
 public:
  explicit FwHashField(int seed) : mfem::VectorCoefficient(3), seed_(seed) {}
  void Eval(mfem::Vector& V, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
    V.SetSize(3);
    mfem::Vector x(3);
    T.Transform(ip, x);
    for (int k = 0; k < 3; ++k) {
      const double a = 1e3 * (12.9898 * x(0) + 78.233 * x(1) + 37.719 * x(2)) + 7.31 * (3 * seed_ + k + 1);
      const double h = std::sin(a) * 43758.5453;
      V(k) = h - std::floor(h) - 0.5;
    }
  }

 private:
  int seed_;
};

// Shift-invert Lanczos in the M inner product (full reorthogonalization), distributed vectors. op(x, y):
// y = (A - sigma M)^-1 M x; deflate(v): removes the discrete-gradient part (it is an eigenvector of op with
// theta = -1/sigma, lambda = 0). Returns the nev Ritz pairs nearest sigma (largest |theta|, both sides), with the
// Ritz values lambda < kernel_tol sigma (curl kernel: floating-conductor potentials, gradient round-off)
// discarded and counted. A step with beta <= 1e-10 |alpha| (an invariant subspace, e.g. a target that coincides
// with an eigenvalue: alpha ~ 1/(lambda - sigma) dwarfs the rest) continues from a fresh start vector
// M-orthogonal to the basis (block-diagonal T) instead of stopping, so nev pairs are still returned.
struct FwLanczosResult {
  std::vector<mfem::Vector> x;
  std::vector<double> lam, estimate;
  json diag;
};

FwLanczosResult fw_shift_invert_lanczos(const mfem::HypreParMatrix& M,
                                        const std::function<void(const mfem::Vector&, mfem::Vector&)>& op,
                                        const std::function<void(mfem::Vector&)>& deflate,
                                        const std::function<void(int, mfem::Vector&)>& start, double sigma, int nev,
                                        int max_steps, double global_n, int check_steps) {
  const int n = M.Height();
  const double kernel_tol = 1e-6, ritz_tol = 1e-9;
  const int m_max = static_cast<int>(std::min(global_n, static_cast<double>(
                                                            max_steps > 0 ? max_steps : std::max(6 * nev + 60, 150))));
  std::vector<mfem::Vector> Q;
  std::vector<double> alpha, beta;
  mfem::Vector Mw(n), w(n);
  int seed = 0, restarts = 0;
  auto orthonormalize = [&](mfem::Vector& v) -> double {
    for (int pass = 0; pass < 2; ++pass) {
      M.Mult(v, Mw);
      std::vector<double> c(Q.size());
      for (size_t i = 0; i < Q.size(); ++i) c[i] = mfem::InnerProduct(Q[i], Mw);
      MPI_Allreduce(MPI_IN_PLACE, c.data(), static_cast<int>(c.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      for (size_t i = 0; i < Q.size(); ++i) v.Add(-c[i], Q[i]);
    }
    M.Mult(v, Mw);
    return std::sqrt(std::max(0.0, fw_gdot(v, Mw)));
  };
  auto fresh = [&]() {
    mfem::Vector v(n), y(n);
    start(seed++, v);
    op(v, y);
    deflate(y);
    const double b = orthonormalize(y);
    AXF_REQUIRE(b > 0 && std::isfinite(b), "shift-invert Lanczos: the start vector vanished after orthogonalization",
                "/fullwave/eigensolver");
    y /= b;
    Q.push_back(y);
  };
  fresh();
  FwLanczosResult out;
  bool converged = false;
  int kernel = 0, k = 0;
  Eigen::VectorXd theta;
  Eigen::MatrixXd S;
  std::vector<int> pick;
  std::vector<double> est;
  auto ritz = [&]() {
    const int kk = static_cast<int>(alpha.size());
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(kk, kk);
    for (int i = 0; i < kk; ++i) {
      T(i, i) = alpha[i];
      if (i + 1 < kk) T(i, i + 1) = T(i + 1, i) = beta[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    theta = es.eigenvalues();
    S = es.eigenvectors();
    std::vector<int> order;
    kernel = 0;
    for (int i = 0; i < kk; ++i) {
      const double lam = sigma + 1.0 / theta(i);
      if (!std::isfinite(lam)) continue;
      if (lam < kernel_tol * sigma) { ++kernel; continue; }
      order.push_back(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return std::fabs(theta(a)) > std::fabs(theta(b)); });
    pick.assign(order.begin(), order.begin() + std::min<size_t>(order.size(), nev));
    est.clear();
    bool ok = static_cast<int>(pick.size()) == nev;
    for (int i : pick) {
      est.push_back(std::fabs(beta.back() * S(kk - 1, i)) / std::fabs(theta(i)));
      if (!(est.back() <= ritz_tol)) ok = false;
    }
    return ok;
  };
  for (int j = 0; j < m_max; ++j) {
    op(Q[j], w);
    deflate(w);
    M.Mult(w, Mw);
    const double a = fw_gdot(Q[j], Mw);
    w.Add(-a, Q[j]);
    if (j > 0) w.Add(-beta.back(), Q[j - 1]);
    const double b = orthonormalize(w);
    alpha.push_back(a);
    beta.push_back(b);
    k = j + 1;
    if (k >= std::min(m_max, nev + 10) && (k % 5 == 0 || k == m_max)) {
      converged = ritz();
      if (converged) break;
    }
    if (k == m_max) break;
    if (!(b > 1e-10 * std::fabs(a))) {  // invariant subspace: continue from a fresh start vector
      beta.back() = 0.0;
      ++restarts;
      fresh();
    } else {
      w /= b;
      Q.push_back(w);
    }
  }
  if (!converged) converged = ritz();
  const int kk = static_cast<int>(alpha.size());
  for (size_t c = 0; c < pick.size(); ++c) {
    const int i = pick[c];
    mfem::Vector y(n);
    y = 0.0;
    for (int jj = 0; jj < kk; ++jj) y.Add(S(jj, i), Q[jj]);
    out.x.push_back(std::move(y));
    out.lam.push_back(sigma + 1.0 / theta(i));
    out.estimate.push_back(est[c]);
  }
  out.diag = {{"lanczos_steps", kk}, {"restarts_after_invariant_subspace", restarts}, {"ritz_tolerance", ritz_tol},
              {"kernel_ritz_values_discarded", kernel}, {"kernel_threshold_lambda_over_sigma", kernel_tol},
              {"max_steps", m_max}, {"converged", converged}};
  // ---- completeness check (a heuristic, not a certificate; STRUMPACK gives no usable inertia here): a short
  // Lanczos run of the same operator on the M-orthogonal complement of the returned modes, from a fresh start
  // vector. An eigenvalue the main run missed inside the returned window [sigma - d, sigma + d] (d = the largest
  // |lambda_i - sigma| returned), e.g. a member of an exactly degenerate cluster whose round-off seed had not yet
  // grown, is the extreme of the complement's spectrum in |theta| and shows up within a few steps.
  if (check_steps > 0 && !out.x.empty()) {
    double d_edge = 0;
    for (double l : out.lam) d_edge = std::max(d_edge, std::fabs(l - sigma));
    std::vector<mfem::Vector> Qc;
    std::vector<double> ac, bc;
    auto orth_c = [&](mfem::Vector& v) -> double {
      for (int pass = 0; pass < 2; ++pass) {
        M.Mult(v, Mw);
        std::vector<double> c(out.x.size() + Qc.size());
        for (size_t i = 0; i < out.x.size(); ++i) c[i] = mfem::InnerProduct(out.x[i], Mw);
        for (size_t i = 0; i < Qc.size(); ++i) c[out.x.size() + i] = mfem::InnerProduct(Qc[i], Mw);
        MPI_Allreduce(MPI_IN_PLACE, c.data(), static_cast<int>(c.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (size_t i = 0; i < out.x.size(); ++i) v.Add(-c[i], out.x[i]);
        for (size_t i = 0; i < Qc.size(); ++i) v.Add(-c[out.x.size() + i], Qc[i]);
      }
      M.Mult(v, Mw);
      return std::sqrt(std::max(0.0, fw_gdot(v, Mw)));
    };
    {
      mfem::Vector v(n), y(n);
      start(1000 + seed, v);
      op(v, y);
      deflate(y);
      const double b0 = orth_c(y);
      if (b0 > 0 && std::isfinite(b0)) { y /= b0; Qc.push_back(y); }
    }
    for (int j = 0; j < check_steps && j < static_cast<int>(Qc.size()); ++j) {
      op(Qc[j], w);
      deflate(w);
      M.Mult(w, Mw);
      const double a = fw_gdot(Qc[j], Mw);
      const double b = orth_c(w);  // full reorthogonalization removes the alpha and beta terms too
      ac.push_back(a);
      bc.push_back(b);
      if (!(b > 1e-10 * std::fabs(a)) || j + 1 == check_steps) break;
      w /= b;
      Qc.push_back(w);
    }
    int missing = 0, near_partner = 0, checked = 0;
    double nearest_ratio = std::numeric_limits<double>::infinity(), worst_est = 0;
    const int kc = static_cast<int>(ac.size());
    if (kc > 0) {
      Eigen::MatrixXd T = Eigen::MatrixXd::Zero(kc, kc);
      for (int i = 0; i < kc; ++i) {
        T(i, i) = ac[i];
        if (i + 1 < kc) T(i, i + 1) = T(i + 1, i) = bc[i];
      }
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
      for (int i = 0; i < kc; ++i) {
        const double th = es.eigenvalues()(i), lam = sigma + 1.0 / th;
        if (!std::isfinite(lam) || lam < kernel_tol * sigma) continue;
        ++checked;
        const double dist = std::fabs(lam - sigma), ratio = d_edge > 0 ? dist / d_edge : 1e300;
        nearest_ratio = std::min(nearest_ratio, ratio);
        if (ratio < 1 - 1e-8) {
          ++missing;
          worst_est = std::max(worst_est, std::fabs(bc.back() * es.eigenvectors()(kc - 1, i)) / std::fabs(th));
        } else {
          for (double l : out.lam)
            if (std::fabs(lam - l) <= 1e-4 * std::fabs(l)) { ++near_partner; break; }
        }
      }
    }
    out.diag["completeness_check"] = {
        {"steps", kc}, {"window_halfwidth_k2", d_edge}, {"ritz_values_checked", checked},
        {"missing_in_window", missing}, {"near_degenerate_outside_window", near_partner},
        {"nearest_distance_over_halfwidth", std::isfinite(nearest_ratio) ? json(nearest_ratio) : json(nullptr)},
        {"missing_ritz_estimate", worst_est},
        {"note", "heuristic: a short Lanczos run on the M-orthogonal complement of the returned modes (fresh start "
                 "vector); a Ritz value with |lambda - sigma| below the returned window's half-width is a mode the "
                 "main run missed; one within 1e-4 (in k^2) of a returned mode but outside the window is the "
                 "partner of a near-degenerate pair cut by the window edge. Ritz values approach from outside the "
                 "window, so a clean check is evidence, not a proof, of completeness"}};
  }
  return out;
}

bool is_pec(const spec::Material* m) {
  return m->kind == spec::MaterialKind::Superconductor || m->kind == spec::MaterialKind::Conductor;
}

// ------------------------------------------------------------ adaptive sweep (docs/physics/adaptive_sweep.md)
// The fit's own tolerance at the samples: this fraction of the sweep's tolerance, not below the floor (the solves are
// accurate to about their 1e-10 true residual; fitting below that only fits noise).
constexpr double kAdaptiveFitFraction = 1e-3;
constexpr double kAdaptiveFitFloor = 1e-10;
// noise_bound is fit tolerance < 10 x pole_noise; within this further factor the run warns that the sweep may still
// be noise-bound next to its poles (pole_noise understates near-pole errors by up to 24 x)
constexpr double kAdaptiveNoiseMargin = 10.0;
constexpr double kAdaptiveRealAxis = 1e-12;   // |Im p| / |p| below this: a pole on the real axis (no Q)
// A pole with Q below this is classed as background, not a resonance: its half-power width exceeds a quarter of its
// frequency, so it is a trend of the band rather than a peak (a fit of a DMRadio-Core-type pickup had one at 85 MHz
// with Q 0.5)
constexpr double kAdaptiveMinQ = 2.0;
// Im p counts as loss (a resolved Q) or as growth (non_passive) only beyond this many standard errors of its
// leave-one-out jackknife (rational::classify_poles)
constexpr double kAdaptivePoleSignificance = 3.0;

double fw_json_number(const json& v) {
  return v.is_number() ? v.get<double>() : std::numeric_limits<double>::quiet_NaN();
}

// The fitted outputs of a sweep row: Z row-major (n x n), then Psi (n); dZ/df = 2 pi dZ/dw for the Z entries, used only
// where the row's dZ_domega_H exists and its scope says it is exact (otherwise that solve's Z is fitted by value).
// Nothing (with the reason) when a port's solve was not accepted or a value is not finite.
std::optional<rational::Sample> fw_adaptive_sample(const json& row, int nports, std::string& why) {
  for (const auto& port : row.value("port_solutions", json::array()))
    if (!port["solver"].value("accepted", false)) {
      why = strcat("port ", port.value("port_index", 0), ": true relative residual ",
                   fw_json_number(port["solver"].value("relative_residual", json())), " not below 1e-6");
      return std::nullopt;
    }
  const int nz = nports * nports;
  rational::Sample s;
  s.f = row["f_Hz"].get<double>();
  s.value.resize(nz + nports);
  s.derivative = Eigen::VectorXcd::Constant(nz + nports, cplx(std::numeric_limits<double>::quiet_NaN(), 0.0));
  for (int j = 0; j < nports; ++j)
    for (int k = 0; k < nports; ++k)
      s.value(j * nports + k) = cplx(fw_json_number(row["Z_matrix_re_ohm"][j][k]),
                                     fw_json_number(row["Z_matrix_im_ohm"][j][k]));
  for (int k = 0; k < nports; ++k)
    s.value(nz + k) = cplx(fw_json_number(row["Psi_vector"]["re"][k]), fw_json_number(row["Psi_vector"]["im"][k]));
  const bool exact = row.contains("dZ_domega_H") && row.value("dZ_domega_scope", std::string()).rfind("exact", 0) == 0;
  if (exact)
    for (int j = 0; j < nports; ++j)
      for (int k = 0; k < nports; ++k)
        s.derivative(j * nports + k) = 2 * pi * cplx(fw_json_number(row["dZ_domega_H"]["re"][j][k]),
                                                     fw_json_number(row["dZ_domega_H"]["im"][j][k]));
  if (!s.value.allFinite()) {
    why = "a non-finite entry of Z or Psi";
    return std::nullopt;
  }
  return s;
}

json fw_complex_json(cplx z) { return {{"re", z.real()}, {"im", z.imag()}}; }

// The `fullwave.adaptive` block: settings, outcome, history, the model (support points, weights, values; poles), and
// its dense evaluation over the band. `resonances` receives the model's poles in the band (self_resonances).
json fw_adaptive_record(const rational::AdaptiveResult& r, const spec::FullWaveSpec::Adaptive& ad,
                        const rational::AdaptiveOptions& options, const rational::FitOptions& fit_options, int nports,
                        const json& refused, double data_noise, json& resonances) {
  const int nz = nports * nports;
  const int K = nz + nports;
  const rational::Model& model = r.fit.model;
  const double width = ad.f_max_Hz - ad.f_min_Hz;
  json record;
  record["settings"] = {
      {"f_min_Hz", ad.f_min_Hz}, {"f_max_Hz", ad.f_max_Hz}, {"tolerance", ad.tolerance},
      {"max_solves", ad.max_solves}, {"initial_points", ad.initial_points}, {"spacing", ad.spacing},
      {"dense_points", ad.dense_points}, {"candidates", options.candidates},
      {"fit_tolerance", fit_options.fit_tolerance}, {"floor_fraction", fit_options.floor_fraction},
      {"error_measure", "max over outputs of |a - b| / max(|b|, floor); floor = floor_fraction x sqrt(median |Z_jj| x "
                        "median |Z_kk|) for Z[j][k] (medians over the solves), and floor_fraction x the median of the "
                        "largest |Psi| entry for Psi"},
      {"estimate", "the larger of the fit against the same solves fitted with one support point fewer and the fit "
                   "against the fit before the newest solve, on the candidate grid with each in-band pole's "
                   "neighbourhood"},
      {"stopping_rule", "estimate below the tolerance over the candidates (those within a quarter grid step of a "
                        "solved frequency excepted), each of the newest `confirmations` solves within the tolerance of "
                        "the model fitted before it, and the fit within 10 x fit_tolerance of its own solves; or, when "
                        "the estimate exceeds the tolerance only through the terms res_q / (f - q) of the comparison "
                        "model's poles q above the real axis (by at most a quarter of the band) that the model lacks, "
                        "4 solves in a row within tolerance / 100 of their predictions, a model change below the "
                        "tolerance and a consistent fit. After convergence, in-band poles on the real axis invisible "
                        "one candidate step away are removed if the model then stays within 0.1 x the tolerance of its "
                        "solves and of itself elsewhere (invisible_poles_removed)"},
      {"confirmations", options.confirmations},
      {"hermite_data", "the exact dZ/dw (dZ_domega_H, where its scope is exact) as derivative rows for every Z entry; "
                       "Psi by value"},
      {"pole_noise", data_noise}, {"noise_bound_factor", rational::kNoiseBoundFactor},
      {"pole_classes", strcat("in-band poles, with the uncertainty of p (in quadrature: the jackknife over the ",
                              "leave-one-out refits, the offset of their mean from p, and the RMS deviation over ",
                              "refits under the solves' noise, pole_noise, with a random phase unless an output's every ",
                              "solve is real or every solve imaginary): "
                              "unresolved (reproduced by fewer than half the refits, or uncertain by more than "
                              "max(|Im p|, one candidate step)), non_passive (Im p > max(", kAdaptiveRealAxis, " |p|, ",
                              kAdaptivePoleSignificance, " sigma_Im): growing in time under e^{-i omega t}), "
                              "negligible (the largest residue over the outputs, relative to the output's scale, at "
                              "one candidate step or |Im p| from the pole, below the tolerance), background (Q < ",
                              kAdaptiveMinQ, "), else resonance. Q is given when -Im p > max(", kAdaptiveRealAxis,
                              " |p|, ", kAdaptivePoleSignificance, " sigma_Im), with its uncertainty; otherwise the "
                              "pole is lossless within the fit, with Q >= Q_min = Re p / (2 max(", kAdaptiveRealAxis,
                              " |p|, |Im p| + ", kAdaptivePoleSignificance, " sigma_Im)). self_resonances lists the "
                              "resonances of a converged sweep")}};
  record["converged"] = r.converged;
  record["stop_reason"] = r.stop_reason;
  record["hermite"] = {{"used_in_final_fit", r.hermite}, {"dropped", r.hermite_note.empty() ? json(nullptr)
                                                                                         : json(r.hermite_note)}};
  record["fit_consistent"] = r.fit_consistent;
  record["invisible_poles_removed"] = r.invisible_poles_removed;
  record["sample_fit_error_before_removal"] = r.sample_error_before_removal;
  // the fit tolerance against the solves' accuracy: a noise-bound sweep's stop is unreliable
  record["noise_bound"] = rational::noise_bound(fit_options.fit_tolerance, data_noise);
  record["solves"] = r.attempted.size();
  record["fitted_solves"] = r.samples.size();
  record["solved_frequencies_Hz"] = r.attempted;
  record["refused"] = refused;
  record["estimated_error"] = r.estimated_error;
  record["last_prediction_error"] = r.last_prediction_error;
  json history = json::array();
  for (const auto& step : r.history) {
    json h = {{"solves", step.solves}, {"fitted_solves", step.samples}, {"support_points", step.support_points},
              {"sample_fit_error", step.sample_fit_error}, {"doublets_removed", step.doublets_removed},
              {"estimated_error", step.estimated_error}, {"estimated_error_at_Hz", step.estimated_error_at},
              {"model_change", step.model_change}, {"next_Hz", step.next_frequency}};
    if (step.predicted_at > 0) {
      h["prediction_error"] = step.prediction_error;   // JSON null: that solve was refused
      h["predicted_at_Hz"] = step.predicted_at;
    }
    history.push_back(h);
  }
  record["history"] = history;
  resonances = json::array();
  if (model.size() == 0) {   // no frequency gave a usable solve: nothing to evaluate
    record["model"] = nullptr;
    record["poles"] = json::array();
    record["dense"] = nullptr;
    return record;
  }

  // the model
  json names = json::array();
  for (int j = 0; j < nports; ++j)
    for (int k = 0; k < nports; ++k) names.push_back(strcat("Z[", j, "][", k, "]"));
  for (int k = 0; k < nports; ++k) names.push_back(strcat("Psi[", k, "]"));
  json weights_re = json::array();
  json weights_im = json::array();
  json values_re = json::array();
  json values_im = json::array();
  for (int j = 0; j < model.size(); ++j) {
    weights_re.push_back(model.weights(j).real());
    weights_im.push_back(model.weights(j).imag());
    json vr = json::array();
    json vi = json::array();
    for (int k = 0; k < K; ++k) {
      vr.push_back(model.values(j, k).real());
      vi.push_back(model.values(j, k).imag());
    }
    values_re.push_back(vr);
    values_im.push_back(vi);
  }
  const std::vector<cplx> poles = model.poles();
  const Eigen::MatrixXcd residues = model.residues(poles);
  json pole_list = json::array();
  for (const cplx& p : poles) pole_list.push_back({{"re_Hz", p.real()}, {"im_Hz", p.imag()}});
  record["model"] = {
      {"form", "r_k(f) = sum_j w_j F_jk / (f - f_j) / sum_j w_j / (f - f_j), f in Hz; r_k(f_j) = F_jk"},
      {"outputs", names}, {"units", "Z in ohm (Z[j][k]: voltage j for unit current k), Psi in H T m^2 = T m^2 A^-1"},
      {"support_Hz", model.support}, {"weights", {{"re", weights_re}, {"im", weights_im}}},
      {"values", {{"re", values_re}, {"im", values_im}}}, {"poles_Hz", pole_list},
      {"sample_fit_error", r.fit.sample_error}, {"doublets_removed", r.fit.doublets_removed}};

  // The model's poles in the band, classed (settings.pole_classes; rational::classify_poles), each with its
  // uncertainty from leave-one-out refits; the resonances of a converged sweep are its self_resonances. Every pole
  // stays in model.poles_Hz.
  rational::PoleOptions pole_options;
  pole_options.f_min = ad.f_min_Hz;
  pole_options.f_max = ad.f_max_Hz;
  pole_options.tolerance = ad.tolerance;
  pole_options.candidates = options.candidates;
  pole_options.log_spacing = ad.spacing == "log";
  pole_options.real_axis = kAdaptiveRealAxis;
  pole_options.significance = kAdaptivePoleSignificance;
  pole_options.min_Q = kAdaptiveMinQ;
  pole_options.noise = data_noise;
  const std::vector<rational::PoleInfo> classed = rational::classify_poles(r.fit, r.samples, r.options, pole_options);
  json pole_rows = json::array();
  for (const auto& info : classed) {
    const cplx q = info.p;
    size_t index = 0;   // the pole's index among the model's poles (for the residue of Z[0][0])
    for (size_t p = 0; p < poles.size(); ++p)
      if (std::abs(poles[p] - q) < std::abs(poles[index] - q)) index = p;
    double below = ad.f_min_Hz;
    double above = ad.f_max_Hz;
    for (const auto& s : r.samples) {
      if (s.f <= q.real()) below = std::max(below, s.f);
      if (s.f >= q.real()) above = std::min(above, s.f);
    }
    auto finite_or_null = [](double x) { return std::isfinite(x) ? json(x) : json(nullptr); };
    json entry = {{"f_Hz", q.real()}, {"pole_im_Hz", q.imag()},
                  {"pole_uncertainty_Hz", {{"re", finite_or_null(info.sigma_re)}, {"im", finite_or_null(info.sigma_im)}}},
                  {"reproduced_by_refits", {info.reproduced, info.refits}},
                  // Q only for a resolved pole: an unresolved one's loss is not determined
                  {"Q", info.lossy && info.cls != "unresolved" ? json(info.Q) : json(nullptr)},
                  {"residue_Z_ohm_Hz", fw_complex_json(residues(index, 0))},
                  {"relative_strength", info.strength / width}, {"contribution_one_step_away", info.visible},
                  {"class", info.cls}, {"between_Hz", {below, above}},
                  {"source", "pole of the adaptive model (fullwave.adaptive)"}};
    if (info.lossy && info.cls != "unresolved") entry["Q_uncertainty"] = finite_or_null(info.Q_sigma);
    if (!info.lossy && !info.non_passive && info.cls != "unresolved") {
      // the loss is not resolved: lossless within the fit, with a lower bound on Q (not for an unresolved pole, whose
      // sigma may be infinite)
      entry["lossless_within_fit"] = true;
      entry["Q_min"] = finite_or_null(info.Q_min);
    }
    pole_rows.push_back(entry);
    if (info.cls == "resonance" && r.converged) resonances.push_back(entry);
  }
  record["poles"] = pole_rows;

  // dense evaluation: the band grid, plus each narrow in-band lossy pole's peak and half-width points
  std::vector<double> fd = rational::band_grid(ad.f_min_Hz, ad.f_max_Hz, ad.dense_points, ad.spacing == "log");
  const double step = ad.spacing == "log"
                          ? std::pow(ad.f_max_Hz / ad.f_min_Hz, 1.0 / (ad.dense_points - 1)) - 1.0   // relative
                          : width / (ad.dense_points - 1);
  for (const cplx& q : poles) {
    const double half = std::fabs(q.imag());
    const double local = ad.spacing == "log" ? step * q.real() : step;
    // (a pole on the real axis has no peak to resolve, only a singularity)
    const bool lossy_pole = q.imag() < -kAdaptiveRealAxis * std::abs(q);
    if (!(q.real() > ad.f_min_Hz && q.real() < ad.f_max_Hz && lossy_pole && half < local)) continue;
    for (double c : {-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0}) {
      const double x = q.real() + c * half;
      if (x > ad.f_min_Hz && x < ad.f_max_Hz) fd.push_back(x);
    }
  }
  std::sort(fd.begin(), fd.end());
  fd.erase(std::unique(fd.begin(), fd.end()), fd.end());
  // the estimate the sweep stopped on: the degree term and the change from the model before the newest solve
  const Eigen::VectorXd estimate = rational::estimated_error(r.fit, fd, r.previous.size() > 0 ? &r.previous : nullptr);
  json Zre = json::array();
  json Zim = json::array();
  json L = json::array();
  json Psi_re = json::array();
  json Psi_im = json::array();
  json Zm_re = json::array();
  json Zm_im = json::array();
  json Pv_re = json::array();
  json Pv_im = json::array();
  json dZ_re = json::array();
  json dZ_im = json::array();
  json est = json::array();
  for (size_t i = 0; i < fd.size(); ++i) {
    const Eigen::VectorXcd v = model.value(fd[i]);
    const Eigen::VectorXcd dv = model.derivative(fd[i]) / (2 * pi);   // dZ/dw = (dZ/df) / (2 pi)
    const double w = 2 * pi * fd[i];
    Zre.push_back(v(0).real());
    Zim.push_back(v(0).imag());
    L.push_back(-v(0).imag() / w);
    Psi_re.push_back(v(nz).real());
    Psi_im.push_back(v(nz).imag());
    json zr = json::array();
    json zi = json::array();
    json dr = json::array();
    json di = json::array();
    for (int j = 0; j < nports; ++j) {
      json zr_row = json::array();
      json zi_row = json::array();
      json dr_row = json::array();
      json di_row = json::array();
      for (int k = 0; k < nports; ++k) {
        zr_row.push_back(v(j * nports + k).real());
        zi_row.push_back(v(j * nports + k).imag());
        dr_row.push_back(dv(j * nports + k).real());
        di_row.push_back(dv(j * nports + k).imag());
      }
      zr.push_back(zr_row);
      zi.push_back(zi_row);
      dr.push_back(dr_row);
      di.push_back(di_row);
    }
    Zm_re.push_back(zr);
    Zm_im.push_back(zi);
    dZ_re.push_back(dr);
    dZ_im.push_back(di);
    json pr = json::array();
    json pi_ = json::array();
    for (int k = 0; k < nports; ++k) {
      pr.push_back(v(nz + k).real());
      pi_.push_back(v(nz + k).imag());
    }
    Pv_re.push_back(pr);
    Pv_im.push_back(pi_);
    est.push_back(estimate(static_cast<int>(i)));
  }
  record["dense"] = {
      {"f_Hz", fd}, {"Z_re_ohm", Zre}, {"Z_im_ohm", Zim}, {"L_H", L}, {"Psi_re", Psi_re}, {"Psi_im", Psi_im},
      {"Z_matrix_re_ohm", Zm_re}, {"Z_matrix_im_ohm", Zm_im}, {"Psi_vector", {{"re", Pv_re}, {"im", Pv_im}}},
      {"dZ_domega_H", {{"re", dZ_re}, {"im", dZ_im}}}, {"estimated_relative_error", est},
      {"note", "the model evaluated on the band grid (dense_points, spacing) plus each narrow in-band pole's peak "
               "and half-width points; Z_re_ohm, Z_im_ohm, L_H, Psi_re, Psi_im are port 0's; dZ_domega_H is the "
               "model's derivative; estimated_relative_error is the error estimate there"}};
  return record;
}

}  // namespace

// A collective direct-solver failure (SolverFailure: rank-0 or distributed STRUMPACK, tree-cotree factorization and
// solve; thrown on every rank) is rethrown with the mesh check.
void Model3D::fullwave_analysis(const SolveOptions& o) try {
  // first, so that it is destroyed last: the allocator thresholds of the direct factorizations end with the analysis
  FwMallocScope malloc_scope;
  Impl& P = *p_;
  const auto& fw = d_.fullwave;
  Timer timer;
  struct LogRestore {  // only rank 0 logs
    int v;
    ~LogRestore() { log_level() = v; }
  } restore{log_level()};
  const int verbosity = log_level();  // the same on every rank (collective diagnostics depend on it)
  if (P.rank != 0) log_level() = 0;
  json& R = results_;
  json& warnings = R["warnings"];
  json out;
  out["status"] = "provisional";
  const std::string solver_backend = std::getenv("AXF_FW_SOLVER") ? std::getenv("AXF_FW_SOLVER") : "ams";
  AXF_REQUIRE(solver_backend=="ams" || solver_backend=="strumpack",
              "AXF_FW_SOLVER must be ams or strumpack", "/fullwave");
  const bool use_direct = solver_backend=="strumpack";
  AXF_REQUIRE(!use_direct || (have_strumpack() && fw.boundary=="pec"),
              "STRUMPACK full-wave option needs a STRUMPACK-enabled build and closed PEC boundary", "/fullwave");
  fw_dist_mode();   // a bad AXF_FW_STRUMPACK_DIST fails here with AXF_FW_SOLVER, not after assembly
  out["solver_path"] = use_direct ? "STRUMPACK real sparse LU reused across RHS with GMRES refinement"
                                  : "AMS-GMRES";
  // Low-frequency breakdown fix (opt-in): see FwTreeCotreeDirect.
  const std::string gauge_mode = std::getenv("AXF_FW_GAUGE") ? std::getenv("AXF_FW_GAUGE") : "none";
  AXF_REQUIRE(gauge_mode == "none" || gauge_mode == "tree-cotree", "AXF_FW_GAUGE must be none or tree-cotree",
              "/fullwave");
  const bool tree_cotree = gauge_mode == "tree-cotree";
  AXF_REQUIRE(!tree_cotree || use_direct, "AXF_FW_GAUGE=tree-cotree requires AXF_FW_SOLVER=strumpack", "/fullwave");
  // the gauged path's experimental options (FwTreeCotreeDirect::Solve): checked here, before assembly, refused without
  // the gauge rather than silently ignored, and named in solver_path
  const std::string tc_projection_env = std::getenv("AXF_FW_TC_CHARGE_PROJECTION") ? std::getenv("AXF_FW_TC_CHARGE_PROJECTION") : "";
  const std::string tc_projection = tc_projection_env.empty() ? (tree_cotree ? "2" : "0") : tc_projection_env;
  const std::string tc_refine_env = std::getenv("AXF_FW_TC_REFINE") ? std::getenv("AXF_FW_TC_REFINE") : "";
  const std::string tc_refine = tc_refine_env.empty() ? "normwise" : tc_refine_env;   // empty means unset
  AXF_REQUIRE(tc_projection == "0" || tc_projection == "1" || tc_projection == "2",
              "AXF_FW_TC_CHARGE_PROJECTION must be 0, 1 or 2", "/fullwave");
  AXF_REQUIRE(tc_refine == "normwise" || tc_refine == "componentwise", "AXF_FW_TC_REFINE must be normwise or componentwise",
              "/fullwave");
  AXF_REQUIRE(tree_cotree || (tc_projection == "0" && tc_refine == "normwise"),
              "AXF_FW_TC_CHARGE_PROJECTION and AXF_FW_TC_REFINE apply to AXF_FW_GAUGE=tree-cotree only", "/fullwave");
  // the refinement residual: extended (default) or double; the extended residual's loop
  // stops on the correction, so it has no componentwise variant
  const std::string tc_residual_env = std::getenv("AXF_FW_TC_RESIDUAL") ? std::getenv("AXF_FW_TC_RESIDUAL") : "";
  const std::string tc_residual = tc_residual_env.empty() ? "extended" : tc_residual_env;   // empty means unset
  AXF_REQUIRE(tc_residual == "extended" || tc_residual == "double", "AXF_FW_TC_RESIDUAL must be extended or double",
              "/fullwave");
  AXF_REQUIRE(tree_cotree || tc_residual_env.empty(), "AXF_FW_TC_RESIDUAL applies to AXF_FW_GAUGE=tree-cotree only",
              "/fullwave");
  AXF_REQUIRE(tc_refine == "normwise" || tc_residual == "double",
              "AXF_FW_TC_REFINE=componentwise needs AXF_FW_TC_RESIDUAL=double (with the extended residual the "
              "refinement stops on the correction)", "/fullwave");
  // Gauss-law guard on the ungauged paths (FwGaussGuard): correct (default), estimate (report only) or off
  const std::string guard_env = std::getenv("AXF_FW_GAUSS_GUARD") ? std::getenv("AXF_FW_GAUSS_GUARD") : "";
  const std::string guard_mode = guard_env.empty() ? (tree_cotree ? "off" : "correct") : guard_env;
  AXF_REQUIRE(guard_mode == "correct" || guard_mode == "estimate" || guard_mode == "off",
              "AXF_FW_GAUSS_GUARD must be correct, estimate or off", "/fullwave");
  AXF_REQUIRE(!tree_cotree || guard_mode == "off",
              "AXF_FW_GAUSS_GUARD applies to the ungauged paths (the gauged path solves the Gauss law exactly)",
              "/fullwave");
  // recovery error indicator of the port fields per frequency and port (FwErrorIndicator): on (default) or off
  const std::string indicator_env = std::getenv("AXF_FW_ERROR_INDICATOR") ? std::getenv("AXF_FW_ERROR_INDICATOR") : "";
  const std::string indicator_mode = indicator_env.empty() ? "on" : indicator_env;
  AXF_REQUIRE(indicator_mode == "on" || indicator_mode == "off", "AXF_FW_ERROR_INDICATOR must be on or off",
              "/fullwave");
  fw_indicator_gradient_tolerance();   // $AXF_FW_INDICATOR_TOL, checked before the sweep
  // how a released factor's memory goes back to the system (fw_malloc_trim_mode, fw_malloc_thresholds_mode), and the
  // per-frequency peak diagnostic (fw_frequency_peak_requested)
  const std::string trim_mode = fw_malloc_trim_mode();
  AXF_REQUIRE(trim_mode == "factor" || trim_mode == "frequency" || trim_mode == "off",
              "AXF_FW_MALLOC_TRIM must be factor, frequency or off", "/fullwave");
  const std::string thresholds_mode = fw_malloc_thresholds_mode();
  AXF_REQUIRE(thresholds_mode == "fixed" || thresholds_mode == "glibc",
              "AXF_FW_MALLOC_THRESHOLDS must be fixed or glibc", "/fullwave");
  const char* peak_env = std::getenv("AXF_FW_FREQUENCY_PEAK");
  AXF_REQUIRE(!peak_env || !*peak_env || std::strcmp(peak_env, "0") == 0 || std::strcmp(peak_env, "1") == 0,
              "AXF_FW_FREQUENCY_PEAK must be 0 or 1", "/fullwave");
  if (tree_cotree)
    out["solver_path"] = "tree-cotree gauged STRUMPACK LU (E = S_c e + G_c phi, K G = 0 exact), Woodbury loads, "
                         "iterative refinement on the transformed residual (" + tc_refine + ", " + tc_residual +
                         " residual); charge projection " + tc_projection + "; gate on the split-form original-space "
                         "residual";
  mfem::ParMesh& pm = *P.pmesh;
  mfem::ParFiniteElementSpace& fes = *P.fes;
  const int ne = pm.GetNE();
  const double eps0 = 1.0 / (mu0 * units::c * units::c);
  const bool port = d_.pickup && d_.pickup->kind == spec::Pickup::Kind::Port;
  AXF_REQUIRE(fw.frequencies.empty() || port,
              "3D full-wave sweeps need a port pickup: the readout current must enter the structure through a gap",
              "/pickup", "Revolve with axionfem.convert.to_3d(design, pickup=\"slit\"), or add a port region bridging "
              "a gap of the readout conductor.");
  const bool surface_port = static_cast<bool>(P.port_surface_J);
  AXF_REQUIRE(fw.frequencies.empty() || P.port_V || surface_port, "internal: the port solution is missing");
  AXF_REQUIRE(fw.boundary == "radiation" || fw.boundary == "pec", "fullwave.boundary must be radiation or pec");
  // an imposed background is the static field of the sweep together with the reaction field in A_lab; a prescribed
  // fullwave.source replaces B_lab altogether, so the two cannot be combined
  AXF_REQUIRE(fw.frequencies.empty() || !P.bg || fw.source.is_null(),
              "a 3D full-wave sweep cannot combine an imposed background with a prescribed fullwave.source",
              "/fullwave/source", "Give the static field once: as `background` (the solver adds the reaction of "
              "superconductors and magnetic materials) or as fullwave.source (the applied field alone).");
  const bool radiation = fw.boundary == "radiation";
  AXF_REQUIRE(!radiation || P.sh_attr > 0, "internal: the radiation condition needs the infinity shell");
  ShellMapR sh;
  if (radiation) {
    // the shell maps [R1, R2) onto [R1, fmax R1] with a perfect conductor there: mildly anisotropic coefficients
    // (AMS converges as for a closed problem); the exterior correction below restores the exact exterior
    sh.fmax = 3.0;
    if (const char* v = std::getenv("AXF_FW_SHELL_FMAX")) sh.fmax = std::atof(v);  // experiments
    sh.attr = P.sh_attr;
    sh.R1 = P.sh_R1;
    sh.R2 = P.sh_R2;
    for (int i = 0; i < 3; ++i) sh.c[i] = P.sh_c[i];
  }
  double gauge = 0;  // gauge mass of the static shell, as in the MQS solve
  {
    mfem::Vector mn, mx;
    P.serial->GetBoundingBox(mn, mx);
    double L = 0;
    for (int i = 0; i < 3; ++i) L = std::max(L, mx(i) - mn(i));
    gauge = 1e-8 / (mu0 * L * L);
  }
  // device radius for k R (outer radius of all regions about the z axis, as coupling.device_radius_m)
  const double Rdev = R["coupling"].value("device_radius_m", P.sh_R1 / 2);
  // (with a background the MQS Psi is its reciprocal form int (A_lab + A_bg) . B_pick, the limit of Psi(omega))
  const double Psi_mqs = R["coupling"].value("Psi_SI_H_T_m2", 0.0), L_mqs = R["coupling"].value("L_p_H", 0.0);

  // essential true DOFs: perfect conductors (all their DOFs), the outer sphere, and for "pec" or eigenmodes the
  // whole shell (closed sphere at R1)
  // keep (cavity designs): elements outside the cavity component are closed off like perfect conductors
  bool impedance_as_pec = false;  // eigenmodes: impedance walls are PEC (losses not evaluated in 3D)
  // the elements essential() closes, and its essential boundary attributes (shared with the Gauss-law guard)
  auto closed_element = [&](int e, bool close_at_R1, const std::vector<char>* keep) {
    const bool sh = P.sh_attr > 0 && pm.GetAttribute(e) == P.sh_attr;
    return is_pec(P.elem_mat[e]) || (close_at_R1 && sh) || (keep && !(*keep)[e]);
  };
  auto essential_bdr = [&]() {
    mfem::Array<int> bdr(pm.bdr_attributes.Size() ? pm.bdr_attributes.Max() : 0);
    bdr = 0;
    if (bdr.Size() >= axi::kOuterAttr) bdr[axi::kOuterAttr - 1] = 1;
    if (P.mesh_info.contains("boundary_groups"))
      for (const auto& group : P.mesh_info["boundary_groups"]) {
        const int attr = group.at("attr").get<int>();
        AXF_REQUIRE(attr > 0 && attr <= bdr.Size(), "named boundary attribute is absent from the mesh",
                    "/mesh/boundary_groups");
        const std::string gty = group.at("type").get<std::string>();
        if (gty == "pec" || (impedance_as_pec && gty == "impedance")) bdr[attr - 1] = 1;
        // PMC is the natural n x H = 0 boundary of this H(curl) weak form.
      }
    return bdr;
  };
  auto essential = [&](bool close_at_R1, mfem::Array<int>& ess, const std::vector<char>* keep = nullptr) {
    mfem::Array<int> mark(fes.GetVSize()), dofs;
    mark = 0;
    for (int e = 0; e < ne; ++e) {
      if (!closed_element(e, close_at_R1, keep)) continue;
      fes.GetElementDofs(e, dofs);
      for (int dd : dofs) mark[dd >= 0 ? dd : -1 - dd] = 1;
    }
    fes.Synchronize(mark);
    mfem::Array<int> bdr = essential_bdr(), ob;
    fes.GetEssentialTrueDofs(bdr, ob);
    std::vector<char> t(fes.GetTrueVSize(), 0);
    for (int i : ob) t[i] = 1;
    for (int i = 0; i < fes.GetVSize(); ++i) {
      if (!mark[i]) continue;
      const int tt = fes.GetLocalTDofNumber(i);
      if (tt >= 0) t[tt] = 1;
    }
    ess.SetSize(0);
    for (int i = 0; i < static_cast<int>(t.size()); ++i)
      if (t[i]) ess.Append(i);
  };
  // per-element complex material scalars
  std::vector<cplx> nu_e(ne), eps_e(ne);
  auto set_eps = [&](double w) {
    for (int e = 0; e < ne; ++e) {
      const spec::Material* m = P.elem_mat[e];
      const double er = m->eps_r > 0 ? m->eps_r : 1.0;
      eps_e[e] = eps0 * er * cplx(1.0, m->tan_delta) + cplx(0.0, w > 0 ? m->sigma_S_m / w : 0.0);
    }
  };
  for (int e = 0; e < ne; ++e) {
    const spec::Material* m = P.elem_mat[e];
    const double mur = (m->kind == spec::MaterialKind::Linear || m->kind == spec::MaterialKind::Iron) ? m->mu_r : 1.0;
    nu_e[e] = 1.0 / (mu0 * mur);
  }
  bool lossy = false;
  for (int e = 0; e < ne; ++e)
    if (!is_pec(P.elem_mat[e]) && (P.elem_mat[e]->tan_delta > 0 || P.elem_mat[e]->sigma_S_m > 0)) lossy = true;
  const int lossy_ranks = static_cast<int>(fw_allsum(lossy ? 1.0 : 0.0));
  lossy = lossy_ranks > 0;  // the loss integral below is a collective, even on loss-free ranks
  out["bulk_loss_ranks"] = lossy_ranks;
  if (std::any_of(P.elem_mat.begin(), P.elem_mat.end(), [](auto* m) { return m->kind == spec::MaterialKind::Iron; }))
    warnings.push_back("3D full wave: iron uses its linear permeability mu_r (no static bias)");

  // ------------------------------------------------------------ Robin boundary groups and PML
  // Boundary groups of an imported mesh may be impedance (Leontovich wall, beta = -i omega / Z_s), sheet (thin
  // layer with continuous E_t on an interior boundary, beta = -i omega Y_s, Y_s = -(2i/eta_m) tan(k_m d/2)) or
  // scattering (first-order absorbing, beta = -i omega / eta of the adjacent medium; cylindrical waves add
  // nu/(2 rho) (a a^T - phi phi^T)). The term beta oint E_t . v_t enters the operator (the same coefficients as the
  // 2D engine, fw2d.cpp). PML layers: stretched-coordinate tensors in FwTensor.
  struct FwRobin {
    int attr = 0;
    std::string name, type, wave = "plane";
    const spec::Material* mat = nullptr;
    bool has_Zs = false;
    cplx Zs = 0;
    double thickness = 0, center[3] = {0, 0, 0}, axis[3] = {0, 0, 1};
  };
  std::vector<FwRobin> robin;
  for (const auto& g : P.mesh_info.value("boundary_groups", json::array())) {
    const std::string ty = g.value("type", std::string("pec"));
    const std::string where = "/mesh/info_file boundary_groups '" + g.value("name", std::string()) + "'";
    AXF_REQUIRE(ty == "pec" || ty == "pmc" || ty == "port" || ty == "impedance" || ty == "sheet" || ty == "scattering",
                "3D boundary group type '" + ty + "' is not one of pec, pmc, port, impedance, sheet, scattering", where);
    if (ty != "impedance" && ty != "sheet" && ty != "scattering") continue;
    FwRobin b;
    b.attr = g.at("attr").get<int>();
    b.name = g.value("name", strcat("attr_", b.attr));
    b.type = ty;
    if (ty == "impedance" || ty == "sheet") {
      if (g.contains("Zs_ohm")) {
        AXF_REQUIRE(ty == "impedance", "Zs_ohm is for impedance groups", where);
        b.has_Zs = true;
        b.Zs = cplx(g.at("Zs_ohm").at(0).get<double>(), g.at("Zs_ohm").at(1).get<double>());
        AXF_REQUIRE(b.Zs.real() >= 0 && std::abs(b.Zs) > 0, "Zs_ohm must have Re >= 0 (e^{-i omega t})", where);
      } else {
        const std::string mn = g.value("material", std::string());
        AXF_REQUIRE(d_.materials.count(mn), "impedance/sheet group needs a design material ('material') or Zs_ohm", where);
        b.mat = &d_.materials.at(mn);
        AXF_REQUIRE(!is_pec(b.mat), "the material of an impedance/sheet group must be linear (finite sigma)", where);
      }
      if (ty == "sheet") {
        b.thickness = g.value("thickness_m", 0.0);
        AXF_REQUIRE(b.thickness > 0, "a sheet group needs thickness_m > 0", where);
      }
    } else {
      b.wave = g.value("wave", std::string("plane"));
      AXF_REQUIRE(b.wave == "plane" || b.wave == "spherical" || b.wave == "cylindrical",
                  "scattering wave must be plane, spherical or cylindrical", where);
      if (g.contains("center")) for (int i = 0; i < 3; ++i) b.center[i] = g.at("center").at(i).get<double>();
      if (g.contains("axis")) {
        double n2 = 0;
        for (int i = 0; i < 3; ++i) { b.axis[i] = g.at("axis").at(i).get<double>(); n2 += b.axis[i] * b.axis[i]; }
        AXF_REQUIRE(n2 > 0, "scattering axis must be nonzero", where);
        for (int i = 0; i < 3; ++i) b.axis[i] /= std::sqrt(n2);
      }
    }
    robin.push_back(b);
  }
  FwPml fwpml;
  if (!fw.pml.empty()) {
    AXF_REQUIRE(!radiation, "fullwave.pml needs fullwave.boundary \"pec\"", "/fullwave/pml");
    fwpml.epml.assign(ne, -1);
    std::map<int, int> layer_of_attr;
    for (size_t i = 0; i < fw.pml.size(); ++i) {
      pml::Layer L = pml::from_json(fw.pml[i], 3, strcat("/fullwave/pml/", i));
      std::string nm;
      for (const auto& r : fw.pml[i].at("regions")) {
        const std::string rn = r.get<std::string>();
        AXF_REQUIRE(P.attr_of_region.count(rn), "PML region '" + rn + "' has no elements", strcat("/fullwave/pml/", i));
        const int a = P.attr_of_region.at(rn);
        AXF_REQUIRE(!layer_of_attr.count(a), "region '" + rn + "' is in two PML layers", strcat("/fullwave/pml/", i));
        layer_of_attr[a] = static_cast<int>(i);
        nm += (nm.empty() ? "" : ",") + rn;
      }
      L.name = nm;
      fwpml.layers.push_back(L);
    }
    for (int e = 0; e < ne; ++e) {
      auto it = layer_of_attr.find(pm.GetAttribute(e));
      if (it != layer_of_attr.end()) fwpml.epml[e] = it->second;
    }
    json lsum = json::array();
    double bb_lo[3], bb_hi[3];
    {
      mfem::Vector mn, mx;
      P.serial->GetBoundingBox(mn, mx);
      for (int i = 0; i < 3; ++i) { bb_lo[i] = mn(i); bb_hi[i] = mx(i); }
    }
    double size = 0;
    for (int i = 0; i < 3; ++i) size = std::max(size, bb_hi[i] - bb_lo[i]);
    for (size_t i = 0; i < fwpml.layers.size(); ++i) {
      pml::Extents phys, reg;
      mfem::Array<int> vs;
      for (int e = 0; e < ne; ++e) {
        if (is_pec(P.elem_mat[e]) || (P.sh_attr > 0 && pm.GetAttribute(e) == P.sh_attr)) continue;
        if (fwpml.epml[e] >= 0 && fwpml.epml[e] != static_cast<int>(i)) continue;
        pm.GetElementVertices(e, vs);
        for (int v : vs) {
          double q[3];
          fwpml.layers[i].local(pm.GetVertex(v), q);
          (fwpml.epml[e] < 0 ? phys : reg).add(q);
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, phys.lo, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, reg.lo, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, phys.hi, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, reg.hi, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      pml::configure(fwpml.layers[i], phys, reg, 1e-7 * size);
      lsum.push_back(fwpml.layers[i].summary());
    }
    out["pml"] = lsum;
    AXF_REQUIRE(fw.modes == 0, "3D eigenmodes with PML layers are not supported (complex modes)", "/fullwave/modes");
  }
  if (!robin.empty()) {
    json rs = json::array();
    for (const auto& b : robin) rs.push_back({{"name", b.name}, {"attr", b.attr}, {"type", b.type}, {"wave", b.wave}});
    out["robin_boundaries"] = rs;
  }
  // adjacent element of each local boundary element (scattering: the medium of the field side)
  std::vector<int> bdr_elem(pm.GetNBE(), -1);
  for (int be = 0; be < pm.GetNBE(); ++be) {
    int el, inf;
    pm.GetBdrElementAdjacentElement(be, el, inf);
    bdr_elem[be] = el;
  }
  if (!robin.empty()) {
    // placement (serial mesh, identical on every rank): impedance / scattering on the exterior of the field domain,
    // sheets with field on both sides, no Robin group next to a conductor (PEC) element
    mfem::Mesh& sm = *P.serial;
    auto ser_pec = [&](int e) {
      auto it = P.region_of_attr.find(sm.GetAttribute(e));
      const spec::Region* g = it == P.region_of_attr.end() ? nullptr : d_.find_region(it->second);
      return g && is_pec(&d_.materials.at(g->material));
    };
    for (int be = 0; be < sm.GetNBE(); ++be) {
      const int a = sm.GetBdrAttribute(be);
      const FwRobin* b = nullptr;
      for (const auto& r : robin)
        if (r.attr == a) b = &r;
      if (!b) continue;
      int e1, e2;
      sm.GetFaceElements(sm.GetBdrElementFaceIndex(be), &e1, &e2);
      const bool c1 = ser_pec(e1), c2 = e2 >= 0 && ser_pec(e2);
      const bool interior = e2 >= 0 && !c1 && !c2;
      const std::string where = "/mesh/info_file boundary_groups '" + b->name + "'";
      if (b->type != "sheet")
        AXF_REQUIRE(!interior, "a " + b->type + " boundary must be exterior (field on one side only); group '" + b->name +
                    "' has field elements on both sides", where);
      else
        AXF_REQUIRE(interior, "a sheet boundary must have field elements on both sides; group '" + b->name + "'", where);
      AXF_REQUIRE(!c1 && !c2, "a " + b->type + " boundary next to a conductor (PEC) element: group '" + b->name + "'",
                  where, "The conductor element's E = 0 would override the Robin condition; remove the element or the group.");
    }
    if (!fwpml.epml.empty()) {  // Robin terms are not stretched inside a PML layer
      std::vector<int> touched(robin.size(), 0);
      for (int be = 0; be < pm.GetNBE(); ++be)
        for (size_t k = 0; k < robin.size(); ++k)
          if (robin[k].attr == pm.GetBdrAttribute(be) && bdr_elem[be] >= 0 && fwpml.epml[bdr_elem[be]] >= 0) touched[k] = 1;
      MPI_Allreduce(MPI_IN_PLACE, touched.data(), static_cast<int>(touched.size()), MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      for (size_t k = 0; k < robin.size(); ++k)
        if (touched[k])
          warnings.push_back("3D full wave: the Robin boundary '" + robin[k].name + "' runs through a PML layer, where "
                             "its term is not stretched (a small spurious reflection; make it PEC inside the layer)");
    }
  }
  // Robin coefficient tensor at a boundary point: part 0 real, 1 imaginary, 2 magnitude (preconditioner); with
  // `derivative`, d beta / dw instead (parts 0 and 1), the boundary's part of dA/dw for the exact dZ/dw
  class RobinCoef : public mfem::MatrixCoefficient {
   public:
    RobinCoef(const FwRobin& b, const std::vector<int>& bdr_elem, const std::vector<const spec::Material*>& mat,
              double w, int part, bool derivative = false)
        : mfem::MatrixCoefficient(3), b_(b), bdr_elem_(bdr_elem), mat_(mat), w_(w), part_(part),
          derivative_(derivative) {}
    static cplx epsc(const spec::Material* m, double w) {
      const double er = m->eps_r > 0 ? m->eps_r : 1.0;
      return cplx(1.0 / (mu0 * units::c * units::c) * er, 1.0 / (mu0 * units::c * units::c) * er * m->tan_delta) +
             cplx(0.0, m->sigma_S_m / w);
    }
    static double muof(const spec::Material* m) {
      const bool lin = m->kind == spec::MaterialKind::Linear || m->kind == spec::MaterialKind::Iron;
      return mu0 * (lin && m->mu_r > 0 ? m->mu_r : 1.0);
    }
    // d beta / dw. With y = sqrt(eps_c / mu) and d eps_c / dw = -i sigma / w^2:
    //   impedance with a fixed Z_s: beta = -i w / Z_s;  with a material: beta = -i w y (Z_s = 1 / y);
    //   scattering: beta = -i w y of the adjacent medium (its cylindrical correction does not depend on w);
    //   sheet: beta = -2 w y tan(k d / 2), k = w sqrt(mu eps_c).
    cplx beta_derivative(mfem::ElementTransformation& T) const {
      if (b_.type == "impedance" && b_.has_Zs) return cplx(0.0, -1.0) / b_.Zs;
      const spec::Material* m = b_.type == "scattering" ? mat_[bdr_elem_[T.ElementNo]] : b_.mat;
      const double mu = muof(m);
      const cplx ec = epsc(m, w_);
      const cplx dec(0.0, -m->sigma_S_m / (w_ * w_));
      const cplx y = std::sqrt(ec / mu);
      const cplx dy = dec / (2.0 * mu * y);
      if (b_.type != "sheet") return cplx(0.0, -1.0) * (y + w_ * dy);
      const cplx s = std::sqrt(mu * ec);
      const cplx k = w_ * s;
      const cplx dk = s + w_ * mu * dec / (2.0 * s);
      const cplx t = std::tan(0.5 * k * b_.thickness);
      const cplx dt = (1.0 + t * t) * 0.5 * b_.thickness * dk;
      return -2.0 * (y * t + w_ * dy * t + w_ * y * dt);
    }
    void Eval(mfem::DenseMatrix& K, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
      K.SetSize(3);
      K = 0.0;
      if (derivative_) {
        const cplx db = beta_derivative(T);
        for (int i = 0; i < 3; ++i) K(i, i) = part_ == 0 ? db.real() : db.imag();
        return;
      }
      cplx beta;
      double c = 0.0;  // cylindrical correction nu/(2 rho)
      double a[3] = {0, 0, 0}, ph[3] = {0, 0, 0};
      if (b_.type == "impedance") {
        const cplx Zs = b_.has_Zs ? b_.Zs : std::sqrt(muof(b_.mat) / epsc(b_.mat, w_));
        beta = cplx(0, -w_) / Zs;
      } else if (b_.type == "sheet") {
        const cplx ec = epsc(b_.mat, w_);
        const double mu = muof(b_.mat);
        const cplx k = w_ * std::sqrt(mu * ec), eta = std::sqrt(mu / ec);
        beta = cplx(0, -w_) * (cplx(0, -2.0) / eta * std::tan(0.5 * k * b_.thickness));
      } else {
        const int e = bdr_elem_[T.ElementNo];
        const spec::Material* m = mat_[e];
        const double mu = muof(m);
        beta = cplx(0, -w_) * std::sqrt(epsc(m, w_) / mu);
        if (b_.wave == "cylindrical") {
          mfem::Vector x(3);
          T.Transform(ip, x);
          double d[3], t = 0;
          for (int i = 0; i < 3; ++i) { d[i] = x(i) - b_.center[i]; t += d[i] * b_.axis[i]; }
          double rv[3], rho = 0;
          for (int i = 0; i < 3; ++i) { rv[i] = d[i] - t * b_.axis[i]; rho += rv[i] * rv[i]; }
          rho = std::sqrt(rho);
          if (rho > 0) {
            for (int i = 0; i < 3; ++i) a[i] = b_.axis[i];
            ph[0] = a[1] * rv[2] - a[2] * rv[1];
            ph[1] = a[2] * rv[0] - a[0] * rv[2];
            ph[2] = a[0] * rv[1] - a[1] * rv[0];
            for (int i = 0; i < 3; ++i) ph[i] /= rho;
            c = 1.0 / (mu * 2.0 * rho);
          }
        }
      }
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          const cplx v = (i == j ? beta : cplx(0.0)) + c * (a[i] * a[j] - ph[i] * ph[j]);
          K(i, j) = part_ == 0 ? v.real() : part_ == 1 ? v.imag() : (i == j ? std::abs(beta) + std::fabs(c) : 0.0);
        }
    }

   private:
    const FwRobin& b_;
    const std::vector<int>& bdr_elem_;
    const std::vector<const spec::Material*>& mat_;
    double w_;
    int part_;
    bool derivative_;
  };
  // boundary markers (persistent: MFEM keeps a pointer)
  std::vector<mfem::Array<int>> robin_marker(robin.size());
  {
    const int nb = pm.bdr_attributes.Size() ? pm.bdr_attributes.Max() : 0;
    for (size_t k = 0; k < robin.size(); ++k) {
      AXF_REQUIRE(robin[k].attr > 0 && robin[k].attr <= nb && pm.bdr_attributes.Find(robin[k].attr) >= 0,
                  strcat("boundary group '", robin[k].name, "' (attribute ", robin[k].attr, ") is absent from the mesh"),
                  "/mesh/info_file");
      robin_marker[k].SetSize(nb);
      robin_marker[k] = 0;
      robin_marker[k][robin[k].attr - 1] = 1;
    }
  }
  const bool open_bc = !robin.empty() || fwpml.active();

  // ------------------------------------------------------------ frequency sweep
  if (!fw.frequencies.empty()) {
    Timer port_source_timer;
    mfem::Array<int> ess;
    essential(!radiation, ess);
    const double I = P.I_pick;
    const int nports = static_cast<int>(P.extra_ports.size()) + 1;
    std::vector<mfem::Vector> Jports(nports);
    json port_info = json::array();
    for (int k = 0; k < nports; ++k) {
      if (k == 0 && surface_port) {
        Jports[k] = *P.port_surface_J;  // surface (lumped) port: int_S K . N dS, assembled in prepare_surface_port
      } else {
        auto* V = k == 0 ? P.port_V.get() : P.extra_ports[k-1].V.get();
        const int attr = k == 0 ? P.port_attr : P.extra_ports[k-1].attr;
        const double scale = k == 0 ? P.port_scale : P.extra_ports[k-1].scale;
        PortJ Jc(*V, attr, scale);
        mfem::ParLinearForm l(&fes);
        l.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(Jc, &port_feed_rule(P.order)));
        l.Assemble();
        Jports[k].SetSize(fes.GetTrueVSize());
        l.ParallelAssemble(Jports[k]);
      }
      // Homogeneous tangential PEC conditions: retain exactly the driven free-DOF source.
      for (int i : ess) Jports[k](i) = 0.0;
      auto info = k == 0 ? P.port_diagnostics : P.extra_ports[k-1].diagnostics;
      info["name"] = k == 0 ? "pickup" : P.extra_ports[k-1].name;
      info["index"] = k;
      info["current_A"] = I;
      port_info.push_back(info);
    }
    Eigen::MatrixXd gram(nports, nports);
    for (int j = 0; j < nports; ++j) for (int k = 0; k < nports; ++k)
      gram(j,k) = fw_gdot(Jports[j], Jports[k]);
    for (int j = 0; j < nports; ++j)
      AXF_REQUIRE(gram(j,j) > 0, "a current port has no free finite-element source DOFs", "/fullwave/ports");
    Eigen::VectorXd norms = gram.diagonal().cwiseSqrt();
    for (int j = 0; j < nports; ++j) for (int k = 0; k < nports; ++k)
      gram(j,k) /= norms(j)*norms(k);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> gev(gram);
    AXF_REQUIRE(gev.info() == Eigen::Success && gev.eigenvalues()(0) > 1e-10,
                "current-port source vectors are linearly dependent after PEC elimination", "/fullwave/ports");
    out["ports"] = port_info;
    out["port_source_gram_min_eigenvalue"] = gev.eigenvalues()(0);
    out["phasor_convention"] = "exp(-i omega t); passive inductance Z=-i omega L";
    out["matrix_convention"] = "Z[j][k] is terminal voltage j for unit incoming current k; no symmetrization";
    out["port_source_assembly_seconds"] = fw_allmax(port_source_timer.seconds());
    Timer magnetic_source_timer;
    mfem::Vector Bt(fes.GetTrueVSize());
    LabB Bc(*P.A_lab, P.sh_attr, &fwpml.epml);
    std::unique_ptr<PrescribedMagneticSource> prescribed;
    if (!fw.source.is_null()) {
      prescribed = std::make_unique<PrescribedMagneticSource>(PrescribedMagneticSource::from_json(fw.source));
      out["source"] = prescribed->summary();
      if (radiation) {
        // The overlap outside R1 is oint Phi_0 E_r dA with B0 = -grad Phi_0 there: valid only if every source
        // current lies inside R1 and the currents close (an open filament's field is not curl-free outside).
        AXF_REQUIRE(!prescribed->is_uniform(), "a uniform prescribed source cannot be used with a radiation boundary",
                    "/fullwave/source");
        const double rmax = prescribed->max_distance_m({sh.c[0], sh.c[1], sh.c[2]});
        const double imbalance = prescribed->max_relative_node_imbalance(1e-9 * sh.R1);
        out["source"]["exterior"] = {{"max_current_distance_from_domain_center_m", rmax}, {"R1_m", sh.R1},
                                     {"max_relative_node_current_imbalance", imbalance}};
        AXF_REQUIRE(rmax < sh.R1, strcat("prescribed source currents reach ", rmax, " m from the domain centre, outside "
                    "the physical domain sphere R1 = ", sh.R1, " m"), "/fullwave/source",
                    "Increase mesh.domain_radius so that all source segments lie inside it (with margin: the exterior "
                    "multipole series converges as (r_source/R1)^n).");
        AXF_REQUIRE(imbalance <= 1e-6, strcat("prescribed source segments do not form closed circuits (relative node "
                    "current imbalance ", imbalance, ")"), "/fullwave/source",
                    "With a radiation boundary the source field outside R1 must be curl-free; close every filament "
                    "(segment endpoints must coincide to 1e-9 R1 with balanced currents).");
      }
      if (!prescribed->is_uniform()) {
        // Near-singular sampling: B0 ~ 1/d at distance d from a filament, so Gaussian quadrature of
        // int N_i . B0 dV does not converge in elements a filament passes through (with a cancelling overlap,
        // e.g. a TM port, Psi then moves by O(1) with source_quadrature_order). Count the source-domain
        // elements a filament meets (straight-sided test on the element vertices).
        // Also report the closest approach (filament to element centroid, in element sizes): elements next to a
        // filament fail in the same way when that ratio is << 1, even if no filament crosses them.
        long long near = 0, counted = 0, close = 0, unchecked = 0;
        double near_volume = 0, volume = 0, min_ratio = 1e300;
        const auto fils = prescribed->filaments();
        mfem::Array<int> vtx;
        for (int e = 0; e < ne; ++e) {
          if (is_pec(P.elem_mat[e]) || pm.GetAttribute(e) == P.sh_attr) continue;
          if (pm.GetElementBaseGeometry(e) != mfem::Geometry::TETRAHEDRON) { ++unchecked; continue; }
          pm.GetElementVertices(e, vtx);
          std::array<PrescribedMagneticSource::Point, 4> tet;
          for (int i = 0; i < 4; ++i) for (int k = 0; k < 3; ++k) tet[i][k] = pm.GetVertex(vtx[i])[k];
          const double vol = pm.GetElementVolume(e);
          ++counted; volume += vol;
          const auto pr = PrescribedMagneticSource::proximity(fils, tet, 2.0);
          if (pr.meets) { ++near; near_volume += vol; }
          if (pr.distance_over_size < 0.5) ++close;
          min_ratio = std::min(min_ratio, pr.distance_over_size);
        }
        MPI_Allreduce(MPI_IN_PLACE, &min_ratio, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        const double near_all = fw_allsum(static_cast<double>(near)), counted_all = fw_allsum(static_cast<double>(counted));
        out["source"]["filament_proximity"] = {
            {"source_domain_elements_met_by_filament", near_all}, {"source_domain_elements", counted_all},
            {"met_by_filament_volume_m3", fw_allsum(near_volume)}, {"source_domain_volume_m3", fw_allsum(volume)},
            {"elements_with_centroid_within_half_size", fw_allsum(static_cast<double>(close))},
            {"min_centroid_distance_over_element_size", min_ratio < 2.0 ? json(min_ratio) : json(">= 2")},
            {"non_tetrahedral_elements_not_checked", fw_allsum(static_cast<double>(unchecked))},
            {"criterion", "non-PEC, non-shell tetrahedron (straight-sided, from its vertices) that a filament meets; "
                          "size = largest bounding-box edge"}};
        if (near_all > 0)
          warnings.push_back(strcat("3D full wave: ", near_all, " source-domain (non-PEC) elements are crossed or touched by a "
                                    "prescribed source filament, where the 1/d Biot-Savart field defeats Gaussian quadrature of the "
                                    "source overlap; run filaments inside PEC conductors (e.g. the superconducting winding "
                                    "itself) or demonstrate convergence in fullwave.source_quadrature_order"));
      }
    }
    class PrescribedB : public mfem::VectorCoefficient {
     public:
      PrescribedB(const PrescribedMagneticSource* source, const Impl& state, const std::set<int>& attrs,
                  const std::vector<int>* epml = nullptr)
          : mfem::VectorCoefficient(3), source_(source), state_(state), attrs_(attrs), epml_(epml) {}
      void Eval(mfem::Vector& v, mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip) override {
        v.SetSize(3); v = 0.0;
        if (is_pec(state_.elem_mat[T.ElementNo]) || T.Attribute == state_.sh_attr) return;
        if (epml_ && !epml_->empty() && (*epml_)[T.ElementNo] >= 0) return;  // no overlap in the stretched layers
        if (!attrs_.empty() && !attrs_.count(T.Attribute)) return;  // fullwave.source_regions
        mfem::Vector x(3); T.Transform(ip,x);
        const auto b = source_->field_T({x(0),x(1),x(2)});
        for (int k = 0; k < 3; ++k) v(k) = b[k];
      }
     private:
      const PrescribedMagneticSource* source_;
      const Impl& state_;
      const std::set<int>& attrs_;
      const std::vector<int>* epml_;
    };
    std::set<int> source_attrs;  // fullwave.source_regions: B0 vanishes outside these regions
    for (const auto& name : fw.source_regions) source_attrs.insert(P.attr_of_region.at(name));
    if (!fw.source_regions.empty()) out["source_regions"] = fw.source_regions;
    PrescribedB prescribed_B(prescribed.get(), P, source_attrs, &fwpml.epml);
    mfem::Vector Bbg;  // an imposed background's part of Bt (int curl N_i . A_bg dV), reported separately
    {
      mfem::ParLinearForm l(&fes);
      mfem::VectorCoefficient& field = prescribed ? static_cast<mfem::VectorCoefficient&>(prescribed_B)
                                                 : static_cast<mfem::VectorCoefficient&>(Bc);
      const mfem::IntegrationRule* rule = nullptr;
      if (fw.source_quadrature_order > 0) {
        for (int e = 0; e < ne; ++e)
          AXF_REQUIRE(pm.GetElementBaseGeometry(e) == mfem::Geometry::TETRAHEDRON,
                      "explicit source quadrature currently requires tetrahedra", "/fullwave/source_quadrature_order");
        rule = &mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, fw.source_quadrature_order);
      }
      l.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(field, rule));
      l.Assemble();
      l.ParallelAssemble(Bt);
      if (P.bg) {
        // Imposed background: A_lab holds only the reaction field, so the background's own field enters here, in the
        // reciprocal form int curl E . A_bg dV with A_bg = (psi / r) phi_hat. For B_bg = curl A_bg it equals
        // int E . B_bg dV up to surface terms: zero on perfect conductors and on a closed sphere (E_t = 0); on R1 of a
        // radiating boundary the exterior term below continues the integral. Unlike E . B_bg it does not see the
        // gradient (electrostatic) part of E, so a background cut off at its grid edge or at R1 adds no
        // 1/omega^2 artifact, and as omega -> 0 it tends to int A_bg . B_pick (checks.Psi_A_lab_dot_B_pick).
        class BackgroundA : public mfem::VectorCoefficient {
         public:
          BackgroundA(const Impl& state, const std::vector<int>* epml)
              : mfem::VectorCoefficient(3), state_(state), epml_(epml) {}
          void Eval(mfem::Vector& v, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
            v.SetSize(3);
            v = 0.0;
            if (is_pec(state_.elem_mat[T.ElementNo]) || T.Attribute == state_.sh_attr) return;
            if (epml_ && !epml_->empty() && (*epml_)[T.ElementNo] >= 0) return;
            mfem::Vector x(3);
            T.Transform(ip, x);
            double b[3];
            double a[3];
            state_.bg_at(x, b, a);
            for (int k = 0; k < 3; ++k) v(k) = a[k];
          }

         private:
          const Impl& state_;
          const std::vector<int>* epml_;
        };
        BackgroundA abg(P, &fwpml.epml);
        mfem::ParLinearForm lb(&fes);
        auto* curl_form = new mfem::VectorFEDomainLFCurlIntegrator(abg);
        if (rule) curl_form->SetIntRule(rule);
        lb.AddDomainIntegrator(curl_form);
        lb.Assemble();
        Bbg.SetSize(fes.GetTrueVSize());
        lb.ParallelAssemble(Bbg);
        for (int i : ess) Bbg(i) = 0.0;
        Bt += Bbg;
      }
    }
    out["source_quadrature_order"] = fw.source_quadrature_order;
    for (int i : ess) Bt(i) = 0.0;
    if (P.bg) {
      // what the sweep sees of the background: the form of its term and, for a grid, the share of the physical region
      // outside the grid (where it is zero) and how large A_bg still is on the grid's edge
      const auto& bgs = d_.background;
      json binfo = {{"type", bgs.type},
                    {"form", "Psi part int curl E . A_bg dV / (i omega I) with A_bg = (psi / r) phi_hat over the physical "
                             "region (perfect conductors, the shell and PML layers excluded), plus the exterior term "
                             "with a radiating boundary (exterior_term_background)"}};
      if (bgs.type == "grid") {
        double outside = 0;
        double volume = 0;
        mfem::Vector c(3);
        for (int e = 0; e < ne; ++e) {
          if (is_pec(P.elem_mat[e]) || pm.GetAttribute(e) == P.sh_attr) continue;
          if (fwpml.active() && fwpml.epml[e] >= 0) continue;
          pm.GetElementCenter(e, c);
          const double vol = pm.GetElementVolume(e);
          volume += vol;
          if (!P.bg->inside(std::hypot(c(0), c(1)), c(2))) outside += vol;
        }
        outside = fw_allsum(outside);
        volume = fw_allsum(volume);
        double a_max = 0;
        double a_edge = 0;
        const size_t nr = bgs.r.size();
        const size_t nz = bgs.z.size();
        for (size_t j = 0; j < nz; ++j)
          for (size_t i = 1; i < nr; ++i) {
            const double a = std::fabs(P.bg->psi(bgs.r[i], bgs.z[j]) / bgs.r[i]);
            a_max = std::max(a_max, a);
            if (i == nr - 1 || j == 0 || j == nz - 1) a_edge = std::max(a_edge, a);
          }
        const double out_frac = volume > 0 ? outside / volume : 0.0;
        const double edge_frac = a_max > 0 ? a_edge / a_max : 0.0;
        binfo["physical_volume_outside_grid_fraction"] = out_frac;
        binfo["grid_edge_A_over_max"] = edge_frac;
        if (out_frac > 1e-6 && edge_frac > 1e-3)
          warnings.push_back(strcat("3D full wave: ", out_frac, " of the physical region lies outside the background "
                                    "grid, whose edge still carries A_bg = psi/r at ", edge_frac, " of its maximum: Psi "
                                    "sees the background cut off there; extend the grid over the domain sphere"));
      }
      out["background"] = binfo;
      if (!radiation && fwpml.active()) {
        // the layers absorb, so the domain is open, but neither the background nor B_lab is coupled in or beyond them
        out["exterior_term_background"] = "none (PML: the background beyond the layers is not coupled)";
        warnings.push_back("3D full wave: with PML layers the background is coupled only in the physical region, not "
                           "in the layers or beyond them, where the open problem would also see it");
      } else if (!radiation) {
        out["exterior_term_background"] = "none (closed: a perfectly conducting sphere at R1, outside which E = 0)";
      }
    }
    out["magnetic_source_assembly_seconds"] = fw_allmax(magnetic_source_timer.seconds());
    // Reintegrate alternative source/FE quadratures against exactly the same
    // retained Maxwell fields. This separates source integration from RF
    // spatial convergence and costs no additional Maxwell right-hand sides.
    std::vector<mfem::Vector> source_control_q;
    std::vector<std::string> source_control_names;
    if (!fw.source_controls.empty()) {
      out["source_controls"] = json::array();
      for (const auto& control : fw.source_controls) {
        Timer source_control_timer;
        const auto alternative = PrescribedMagneticSource::from_json(control.at("source"));
        PrescribedB coefficient(&alternative,P,source_attrs,&fwpml.epml);
        const int order = control.value("quadrature_order",0);
        const mfem::IntegrationRule* rule = nullptr;
        if (order > 0) {
          for (int e = 0; e < ne; ++e)
            AXF_REQUIRE(pm.GetElementBaseGeometry(e) == mfem::Geometry::TETRAHEDRON,
                        "explicit source-control quadrature requires tetrahedra", "/fullwave/source_controls");
          rule = &mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON,order);
        }
        mfem::ParLinearForm form(&fes);
        form.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(coefficient,rule));
        form.Assemble();
        source_control_q.emplace_back(fes.GetTrueVSize());
        form.ParallelAssemble(source_control_q.back());
        for (int i : ess) source_control_q.back()(i) = 0.0;
        source_control_names.push_back(control.at("name").get<std::string>());
        out["source_controls"].push_back({{"name",source_control_names.back()}, {"source",alternative.summary()},
            {"quadrature_order",order}, {"assembly_seconds",fw_allmax(source_control_timer.seconds())},
            {"scope","same RF mesh/field columns; vacuum includes any restored transparent-coil pack volume"}});
      }
    }

    // Fixed-geometry, nondispersive PEC energy operators. Their quadratures
    // match the Maxwell operators, so the discrete reactance-slope identity
    // does not need an additional Maxwell solve. Radiation energy is excluded:
    // its shell and frequency-dependent DtN require separate stored energy.
    Timer energy_setup_timer;
    std::unique_ptr<mfem::HypreParMatrix> energy_K, energy_M;
    if (!radiation && !open_bc) {
      set_eps(0.0);
      ShellMapR none;
      FwTensor kc(none, nu_e, 0, 0, 1.0, 0.0), mc(none, eps_e, 1, 0, 1.0, 0.0);
      mfem::ParBilinearForm k(&fes), m(&fes);
      k.AddDomainIntegrator(new mfem::CurlCurlIntegrator(kc));
      m.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mc));
      k.Assemble(); k.Finalize(); m.Assemble(); m.Finalize();
      energy_K.reset(k.ParallelAssemble()); energy_M.reset(m.ParallelAssemble());
      out["energy_convention"] = "G=(E^dagger M E+E^dagger K E/omega^2)/4; U=I^dagger G I; field columns per incoming A";
      out["energy_derivative_identity_valid"] = !lossy && !open_bc;
      out["energy_scope"] = "meshed RF domain only; add omitted component, transmission-line and input-coil energy";
    } else {
      out["energy_derivative_identity_valid"] = false;
      out["energy_scope"] = "not exported for radiation, Robin (impedance/sheet/scattering) or PML boundaries";
    }
    out["energy_operator_setup_seconds"] = fw_allmax(energy_setup_timer.seconds());
    std::unique_ptr<FwTreeCotreeDirect> gauged;
    mfem::Vector grad_probe;  // tree-cotree diagnostic: a pure discrete gradient with zero boundary potential
    if (tree_cotree) {
      AXF_REQUIRE(energy_K && energy_M, "tree-cotree gauge needs the closed-PEC K and M operators", "/fullwave");
      AXF_REQUIRE(fes.FEColl()->GetOrder() == 1, "tree-cotree gauge supports lowest-order Nedelec only", "/fullwave");
      // Topology guard: the conductor-potential contraction spans the whole curl kernel only when every boundary
      // part is PEC (see FwTreeCotreeDirect). Natural (PMC) parts can leave relative-cohomology fields.
      if (P.mesh_info.contains("boundary_groups"))
        for (const auto& group : P.mesh_info["boundary_groups"])
          AXF_REQUIRE(group.at("type").get<std::string>() == "pec",
                      "AXF_FW_GAUGE=tree-cotree requires every boundary group to be PEC (found " +
                      group.at("type").get<std::string>() + ")", "/fullwave",
                      "Relative cohomology with PMC boundary parts is not handled; use the default path.");
      mfem::H1_FECollection h1c(1, pm.Dimension());
      mfem::ParFiniteElementSpace h1(&pm, &h1c);
      mfem::ParDiscreteLinearOperator grad(&h1, &fes);
      grad.AddDomainInterpolator(new mfem::GradientInterpolator());
      grad.Assemble();
      grad.Finalize();
      std::unique_ptr<mfem::HypreParMatrix> Gm(grad.ParallelAssemble());
      gauged = std::make_unique<FwTreeCotreeDirect>(*energy_K, *energy_M, *Gm, ess);
      {
        // A random discrete gradient of the reduced system: phi = 0 on every vertex of an essential (PEC) edge, so
        // that G phi vanishes on the eliminated edges. (Zeroing phi only on mesh-boundary vertices left G phi
        // nonzero on the eliminated PEC edges, and the check measured that, not rounding.)
        // A vertex touches an essential edge iff (|G|^T m) > 0 there, m the essential-edge indicator.
        mfem::HypreParMatrix Gabs(*Gm);
        {
          Gabs.HostReadWrite();   // on the host (a flag for hypre's unified memory: no synchronization)
          sync_hypre_device();    // hypre's copy of Gm (a device-to-device cudaMemcpy) finished before the host writes
          hypre_ParCSRMatrix* hg = Gabs;
          for (hypre_CSRMatrix* blk : {hypre_ParCSRMatrixDiag(hg), hypre_ParCSRMatrixOffd(hg)}) {
            HYPRE_Real* a = hypre_CSRMatrixData(blk);
            for (HYPRE_Int k = 0; k < hypre_CSRMatrixNumNonzeros(blk); ++k) a[k] = std::fabs(a[k]);
          }
          Gabs.HypreRead();   // back in hypre's memory for MultTranspose (a no-op on the host)
        }
        mfem::Vector m(fes.GetTrueVSize()), touched(h1.GetTrueVSize());
        m = 0.0;
        for (int i : ess) m(i) = 1.0;
        Gabs.MultTranspose(m, touched);
        mfem::Vector phi(h1.GetTrueVSize());
        phi.Randomize(4321 + P.rank);
        for (int i = 0; i < phi.Size(); ++i)
          if (touched(i) > 0) phi(i) = 0.0;
        grad_probe.SetSize(fes.GetTrueVSize());
        Gm->Mult(phi, grad_probe);
      }
      out["tree_cotree"] = gauged->Topology();
    }
    // Gauss-law guard (FwGaussGuard): set up once from the same closed elements and essential boundaries as ess;
    // applied per frequency where the assembled operator is K - w^2 M on the free DOFs
    std::unique_ptr<FwGaussGuard> guard;
    json guard_record = {{"mode", guard_mode}};
    if (guard_mode != "off") {
      if (!energy_K || !energy_M) {
        guard_record["not_applied"] = "open boundary (radiation, Robin or PML): no closed K and M";
      } else {
        guard_record["lossy"] = lossy;   // the loss mass is assembled per frequency (SetLoss)
        std::vector<char> closed(ne, 0);
        for (int e = 0; e < ne; ++e) closed[e] = closed_element(e, !radiation, nullptr);
        guard = std::make_unique<FwGaussGuard>(fes, *energy_M, ess, closed, essential_bdr(), P.order);
        guard_record["setup"] = guard->Topology();
        if (!guard->available()) {
          guard_record["not_applied"] = guard->reason();
          guard.reset();
        }
      }
      // the potential space spans the curl kernel when the PEC closure carries all of it; natural (PMC) boundary
      // parts on a handle can add harmonic fields the guard neither estimates nor corrects (the gauged path refuses
      // PMC)
      bool natural = false;
      for (const auto& group : P.mesh_info.value("boundary_groups", json::array()))
        natural = natural || group.value("type", std::string("pec")) == "pmc";
      if (guard && natural) {
        guard_record["harmonic_fields_note"] = "PMC boundary groups: harmonic fields are not covered";
        warnings.push_back("3D full wave: the model has PMC boundary groups; the Gauss-law guard does not cover the "
                           "harmonic fields such boundary parts can add (docs/physics/fullwave_gauss_guard.md)");
      }
    }
    out["gauss_law_guard"] = guard_record;
    // frequencies where the guard was asked for but not applied and the field is quasi-static (k R below this): the
    // ungauged paths' gradient error grows like (k h)^-2 there; one warning after the sweep
    constexpr double kGuardWarnKR = 1e-2;
    constexpr double kGuardFloorWarn = 1e-8;   // a lossy correction's floor_estimate above this is warned about
    std::vector<double> unguarded_frequencies;
    std::string unguarded_reason;

    std::filesystem::path field_dir;
    if (!fw.field_export_directory.empty()) {
      field_dir = fw.field_export_directory;
      std::filesystem::create_directories(field_dir);
      const auto mesh_file = field_dir / strcat("mesh_rank_", P.rank, ".mesh");
      std::ofstream mesh_stream(mesh_file);
      AXF_REQUIRE(bool(mesh_stream), "cannot write fullwave field-export mesh", "/fullwave/field_export_directory");
      mesh_stream << std::setprecision(17);
      pm.Print(mesh_stream);
      out["field_export"] = {{"directory", field_dir.string()}, {"mpi_ranks", P.nranks},
                             {"mesh_pattern", "mesh_rank_{rank}.mesh"},
                             {"format", "MFEM local-partition mesh and GridFunction files; real/imaginary physical E in V/m"},
                             {"normalization", "each port basis column is per incoming terminal ampere; no image scaling"}};
    }
    mfem::Array<int> probe_elements;
    mfem::Array<mfem::IntegrationPoint> probe_ips;
    const int nprobes = static_cast<int>(fw.field_probes.size());
    if (nprobes > 0) {
      mfem::DenseMatrix points(3,nprobes);
      json coordinates = json::array();
      for (int j = 0; j < nprobes; ++j) {
        const auto& p = fw.field_probes[j];
        points(0,j) = p.x; points(1,j) = p.y; points(2,j) = p.z;
        coordinates.push_back({p.x,p.y,p.z});
      }
      pm.FindPoints(points,probe_elements,probe_ips,false);
      std::vector<int> valid(nprobes,0);
      for (int j = 0; j < nprobes; ++j) {
        const int e = probe_elements[j];
        if (e >= 0 && !is_pec(P.elem_mat[e]) && !(P.sh_attr > 0 && pm.GetAttribute(e) == P.sh_attr)) valid[j] = 1;
        else probe_elements[j] = -1;
      }
      MPI_Allreduce(MPI_IN_PLACE,valid.data(),nprobes,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      out["field_probes"] = {{"points_m",coordinates}, {"valid",valid},
          {"normalization","E in V/m/A, B in T/A; unit incoming port current; B=curl E/(i omega); no image scaling"},
          {"note","invalid/outside/PEC points carry zero samples and must be excluded explicitly, never treated as physical zeros"}};
    }
    // ------------------------------------------------------------------ conductor surface quadrature
    // Quadrature points on the PEC boundary FACES of the mesh (the conductor surfaces themselves, since
    // `remove_pec` cuts the conductor volumes out and leaves their boundary).  For each point we record the
    // position, the unit normal pointing OUT of the conductor INTO the field region, the area weight, and the
    // boundary attribute; the per-port B at the same points is written in the drive loop below.  This is the
    // exact surface integral of the sheet current K = n x H: no point location, no probe offset, no reweighting.
    // Selection: the PEC groups the full-wave operator treats as PEC (a group without
    // "type" is PEC, as in the surface losses and the cavity code), minus symmetry planes ("axis"). The outer boundary
    // kOuterAttr is kept: in the rf_only models that have PEC groups it is the closed PEC sphere or the
    // mesh.domain_solid wall, a real conductor (consumers filter by attr). Faces whose field side is an infinity-shell
    // element (a guard: no rf_only mesh has a shell), a PML element or a PEC volume element (a meshed conductor) are
    // skipped and counted. Conductors kept as volumes (mesh.remove_pec not set) and impedance/sheet walls are not
    // exported.
    struct SurfPt { int elem; mfem::IntegrationPoint eip; double x[3], n[3], dA; int attr; };
    std::vector<SurfPt> surf_pts;
    std::set<int> surf_attrs;                       // PEC boundary attributes that are conductors, not symmetry planes
    if (!fw.surface_quadrature_file.empty()) {
      if (P.mesh_info.contains("boundary_groups"))
        for (const auto& group : P.mesh_info["boundary_groups"]) {
          if (group.value("type", std::string("pec")) != "pec") continue;
          if (group.contains("axis")) continue;
          surf_attrs.insert(group.at("attr").get<int>());
        }
      AXF_REQUIRE(!surf_attrs.empty(),
                  "surface_quadrature_file needs conductor PEC boundary groups in the mesh info (other than "
                  "symmetry planes)",
                  "/fullwave/surface_quadrature_file",
                  "Remove the conductor volumes from the mesh (region remove_pec: true) so that their surfaces are "
                  "PEC boundary faces, or import a mesh with PEC boundary groups.");
      // the parent directory, created and checked at setup on every rank, before any solve
      {
        const std::filesystem::path parent = std::filesystem::path(fw.surface_quadrature_file).parent_path();
        std::error_code ec;
        if (!parent.empty()) std::filesystem::create_directories(parent, ec);
        const bool ok = parent.empty() || std::filesystem::is_directory(parent, ec);
        const std::string no_dir = fw_failing_ranks(!ok);
        AXF_REQUIRE(no_dir.empty(),
                    strcat("cannot create the directory of fullwave.surface_quadrature_file (", parent.string(),
                           "; ranks ", no_dir, ")"),
                    "/fullwave/surface_quadrature_file");
        // writability: each rank writes and removes a probe file
        const std::string probe = strcat(fw.surface_quadrature_file, ".probe.rank", P.rank);
        bool writable = false;
        {
          std::ofstream ps(probe, std::ios::binary);
          if (ps) ps << "probe";
          ps.close();
          writable = static_cast<bool>(ps);
        }
        std::filesystem::remove(probe, ec);
        const std::string not_writable = fw_failing_ranks(!writable);
        AXF_REQUIRE(not_writable.empty(),
                    strcat("cannot write files with the prefix fullwave.surface_quadrature_file (", fw.surface_quadrature_file,
                           "; ranks ", not_writable, ")"),
                    "/fullwave/surface_quadrature_file", "Check that the directory exists and is writable.");
      }
      long skipped_interior = 0;
      long skipped_shell = 0;
      long skipped_pml = 0;
      long skipped_pec_side = 0;
      mfem::Vector nor(3), xq(3), cen(3);
      for (int b = 0; b < pm.GetNBE(); ++b) {
        if (!surf_attrs.count(pm.GetBdrAttribute(b))) continue;
        mfem::FaceElementTransformations* ftr = pm.GetBdrFaceTransformations(b);
        if (!ftr) {   // MFEM gives none for a boundary element on an interior, shared or nonconforming face
          ++skipped_interior;
          continue;
        }
        if (P.sh_attr > 0 && pm.GetAttribute(ftr->Elem1No) == P.sh_attr) {
          ++skipped_shell;
          continue;
        }
        if (!fwpml.epml.empty() && fwpml.epml[ftr->Elem1No] >= 0) {
          ++skipped_pml;
          continue;
        }
        if (is_pec(P.elem_mat[ftr->Elem1No])) {
          ++skipped_pec_side;
          continue;
        }
        // centroid of the adjacent volume element, to fix the normal's sign without trusting the mesh's
        // boundary-element orientation
        mfem::ElementTransformation* Te = pm.GetElementTransformation(ftr->Elem1No);
        Te->Transform(mfem::Geometries.GetCenter(pm.GetElementBaseGeometry(ftr->Elem1No)), cen);
        const mfem::IntegrationRule& ir =
            mfem::IntRules.Get(ftr->GetGeometryType(), std::max(fw.surface_quadrature_order, 0));
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(q);
          ftr->SetAllIntPoints(&ip);
          mfem::CalcOrtho(ftr->Jacobian(), nor);     // |nor| is the surface Jacobian
          const double jac = nor.Norml2();
          if (!(jac > 0)) continue;
          ftr->Transform(ip, xq);
          double out = 0;                            // (face point - element centroid) . nor  > 0 : nor leaves the domain
          for (int k = 0; k < 3; ++k) out += (xq(k) - cen(k)) * nor(k);
          const double sgn = out >= 0 ? -1.0 : +1.0; // flip: we want the normal INTO the field region
          SurfPt sp;
          sp.elem = ftr->Elem1No;
          sp.eip = ftr->GetElement1IntPoint();
          sp.dA = ip.weight * jac;
          sp.attr = pm.GetBdrAttribute(b);
          for (int k = 0; k < 3; ++k) { sp.x[k] = xq(k); sp.n[k] = sgn * nor(k) / jac; }
          surf_pts.push_back(sp);
        }
      }
      const double ntot = fw_allsum(static_cast<double>(surf_pts.size()));
      double atot = 0;
      for (const auto& sp : surf_pts) atot += sp.dA;
      atot = fw_allsum(atot);
      const double nskip_interior = fw_allsum(static_cast<double>(skipped_interior));
      const double nskip_shell = fw_allsum(static_cast<double>(skipped_shell));
      const double nskip_pml = fw_allsum(static_cast<double>(skipped_pml));
      const double nskip_pec = fw_allsum(static_cast<double>(skipped_pec_side));
      out["surface_quadrature"] = {{"file_pattern", fw.surface_quadrature_file + ".f{frequency_index}.rank{rank}.bin"},
                                   {"order", fw.surface_quadrature_order},
                                   {"attributes", std::vector<int>(surf_attrs.begin(), surf_attrs.end())},
                                   {"points", static_cast<long long>(ntot)},
                                   {"area_m2", atot},
                                   {"skipped_faces_interior", static_cast<long long>(nskip_interior)},
                                   {"skipped_faces_shell", static_cast<long long>(nskip_shell)},
                                   {"skipped_faces_pml", static_cast<long long>(nskip_pml)},
                                   {"skipped_faces_pec_side", static_cast<long long>(nskip_pec)},
                                   {"note", "one file per frequency row and rank; the union over ranks is the "
                                            "surface. Under mesh.symmetry only the meshed part is written and B "
                                            "is not mirrored (no image scaling)."},
                                   {"normalization", "B in T per incoming terminal ampere, per port basis column; "
                                                     "n is the unit normal out of the conductor into the field "
                                                     "region; dA is the quadrature weight in m^2"},
                                   {"layout", "f64 npts, f64 nports, then npts x (x,y,z,nx,ny,nz,dA,attr), then "
                                              "nports x npts x (Bre, Bim)"}};
      log(1, strcat("surface quadrature: ", static_cast<long long>(ntot), " points on ", surf_attrs.size(),
                    " PEC boundary groups, ", atot, " m^2"));
      // per attribute, and the cases where nothing useful can be exported
      json per_attr = json::object();
      double device_points = 0;
      for (int at : surf_attrs) {
        double cnt = 0;
        for (const auto& sp : surf_pts)
          if (sp.attr == at) cnt += 1;
        cnt = fw_allsum(cnt);
        per_attr[std::to_string(at)] = static_cast<long long>(cnt);
        if (at != axi::kOuterAttr) device_points += cnt;
      }
      out["surface_quadrature"]["points_per_attribute"] = per_attr;
      AXF_REQUIRE(ntot > 0, "fullwave.surface_quadrature_file: no conductor face has a field side in this mesh, so "
                            "nothing would be exported",
                  "/fullwave/surface_quadrature_file",
                  "The export needs conductors removed from the mesh (mesh.remove_pec, rf_only) or an imported mesh "
                  "with PEC boundary groups; conductors meshed as volumes have no boundary faces.");
      if (device_points == 0)
        warnings.push_back("fullwave.surface_quadrature_file: only the outer boundary (attribute 901) is exported; "
                           "the device's conductors are volume elements (set mesh.remove_pec to export their "
                           "surfaces)");
      if (nskip_interior > 0)
        warnings.push_back(strcat("fullwave.surface_quadrature_file: ", static_cast<long long>(nskip_interior),
                                  " PEC boundary faces are interior, shared or nonconforming faces (a two-sided "
                                  "PEC sheet) and are not exported"));
    }

    // exterior correction: projections U(:, j) = int_{rho = R1} N_i . W_j dS on the orthonormal tangential
    // vector spherical harmonics W_j (n = 1..N; curl type = TE, gradient type = TM), true DOFs x modes
    std::vector<int> mode_n;
    std::vector<char> mode_te;
    std::vector<int> tm0_col;  // columns of U of the TM modes with m = 0, n = 1..N (an imposed background's exterior)
    mfem::DenseMatrix U;
    struct ExtPt {
      int e;
      mfem::IntegrationPoint eip;
      double dS, rh[3], x[3], rho;
      double n[3], dOmega;  // unit facet normal out of the physical domain; solid angle dS (n.rhat)/rho^2
      std::vector<double> Y;
    };
    std::vector<ExtPt> ext_pts;   // R1 sphere points (physical side) for the exterior Psi term
    std::vector<double> ext_phi;  // Phi_nm = R1 beta_nm / (n + 1) of the static field
    axi::BackgroundExterior bg_ext;   // an imposed grid background outside R1 (moments of A_bg)
    std::vector<double> bg_sliver;    // its correction for the slivers between the R1 facets and the sphere
    if (radiation) {
      const double fmax = *std::max_element(fw.frequencies.begin(), fw.frequencies.end());
      int N = std::clamp(static_cast<int>(std::ceil(2 * pi * fmax / units::c * sh.R1)) + 10, 4,
                         std::max(4, std::min(fw.dtn_modes, 40)));
      if (const char* v = std::getenv("AXF_FW_NMODES")) N = std::atoi(v);  // experiments (0: static exterior only)
      std::vector<int> mode_m;
      for (int n = 1; n <= N; ++n)
        for (int m = -n; m <= n; ++m)
          for (int te = 1; te >= 0; --te) {
            mode_n.push_back(n);
            mode_m.push_back(m);
            mode_te.push_back(static_cast<char>(te));
            if (m == 0 && te == 0) tm0_col.push_back(static_cast<int>(mode_n.size()) - 1);
          }
      const int nm = static_cast<int>(mode_n.size());
      mfem::DenseMatrix Uloc(fes.GetVSize(), nm);
      Uloc = 0.0;
      pm.ExchangeFaceNbrData();
      const auto& irf = mfem::IntRules.Get(mfem::Geometry::TRIANGLE, 2 * P.order + 4);
      mfem::DenseMatrix vshape;
      mfem::Array<int> vd;
      mfem::Vector nor(3), x(3);
      std::vector<double> W(3 * nm);
      double area_R1 = 0;  // check: 4 pi R1^2
      int nfaces_local = 0, nfaces_shared = 0;
      mfem::DofTransformation dt;
      mfem::DenseMatrix EV;
      mfem::Vector evc;
      auto add_face = [&](mfem::FaceElementTransformations* FT, int e, bool side1) {
        const mfem::FiniteElement* fe = fes.GetFE(e);
        fes.GetElementVDofs(e, vd, dt);
        vshape.SetSize(fe->GetDof(), 3);
        EV.SetSize(fe->GetDof(), nm);
        EV = 0.0;
        mfem::ElementTransformation* Te = pm.GetElementTransformation(e);
        for (int q = 0; q < irf.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = irf.IntPoint(q);
          FT->SetAllIntPoints(&ip);
          mfem::CalcOrtho(FT->Jacobian(), nor);
          FT->Transform(ip, x);
          const double dS = ip.weight * nor.Norml2();
          area_R1 += dS;
          const mfem::IntegrationPoint& eip = side1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
          Te->SetIntPoint(&eip);
          fe->CalcVShape(*Te, vshape);
          double d[3] = {x(0) - sh.c[0], x(1) - sh.c[1], x(2) - sh.c[2]};
          const double rho = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
          const double rh[3] = {d[0] / rho, d[1] / rho, d[2] / rho};
          for (int j = 0; j < nm; ++j) {
            double g[3];
            ylm_surface_grad(mode_n[j], mode_m[j], rh, g);
            const double sc = 1.0 / (std::sqrt(mode_n[j] * (mode_n[j] + 1.0)) * sh.R1);
            if (mode_te[j]) {  // r x grad Y
              W[3 * j] = sc * (rh[1] * g[2] - rh[2] * g[1]);
              W[3 * j + 1] = sc * (rh[2] * g[0] - rh[0] * g[2]);
              W[3 * j + 2] = sc * (rh[0] * g[1] - rh[1] * g[0]);
            } else {
              for (int i = 0; i < 3; ++i) W[3 * j + i] = sc * g[i];
            }
          }
          for (int k = 0; k < vd.Size(); ++k)
            for (int j = 0; j < nm; ++j)
              EV(k, j) += dS * (vshape(k, 0) * W[3 * j] + vshape(k, 1) * W[3 * j + 1] + vshape(k, 2) * W[3 * j + 2]);
        }
        // element vector (dual) -> global orientation of the face DOFs of order >= 2 Nedelec tetrahedra, then the
        // edge signs (as LinearForm::Assemble does)
        evc.SetSize(vd.Size());
        for (int j = 0; j < nm; ++j) {
          for (int k = 0; k < vd.Size(); ++k) evc(k) = EV(k, j);
          if (!dt.IsIdentity()) dt.TransformDual(evc);
          for (int k = 0; k < vd.Size(); ++k) {
            const int row = vd[k] >= 0 ? vd[k] : -1 - vd[k];
            Uloc(row, j) += vd[k] >= 0 ? evc(k) : -evc(k);
          }
        }
      };
      for (int f = 0; f < pm.GetNumFaces(); ++f) {  // faces between the shell and the physical region
        int e1, e2;
        pm.GetFaceElements(f, &e1, &e2);
        if (e1 < 0 || e2 < 0) continue;
        const bool s1 = sh.in(pm.GetAttribute(e1)), s2 = sh.in(pm.GetAttribute(e2));
        if (s1 == s2) continue;
        add_face(pm.GetFaceElementTransformations(f), s1 ? e2 : e1, !s1);
        ++nfaces_local;
      }
      for (int sf = 0; sf < pm.GetNSharedFaces(); ++sf) {  // across ranks: counted on the physical side
        mfem::FaceElementTransformations* FT = pm.GetSharedFaceTransformations(sf);
        if (sh.in(FT->Elem1->Attribute) || !sh.in(FT->Elem2->Attribute)) continue;
        add_face(FT, FT->Elem1No, true);
        ++nfaces_shared;
      }
      U.SetSize(fes.GetTrueVSize(), nm);
      mfem::Vector col(fes.GetVSize()), tcol(fes.GetTrueVSize());
      for (int j = 0; j < nm; ++j) {
        for (int i = 0; i < fes.GetVSize(); ++i) col(i) = Uloc(i, j);
        fes.GetProlongationMatrix()->MultTranspose(col, tcol);
        for (int i = 0; i < tcol.Size(); ++i) U(i, j) = tcol(i);
      }
      for (int i : ess)
        for (int j = 0; j < nm; ++j) U(i, j) = 0.0;
      if (std::getenv("AXF_FW_UTEST")) {  // self-test: E = sin(theta) phi_hat has only the TE n = 1, m = 0 component
        mfem::VectorFunctionCoefficient Et(3, [&](const mfem::Vector& xx, mfem::Vector& v) {
          const double d0 = xx(0) - sh.c[0], d1 = xx(1) - sh.c[1], d2 = xx(2) - sh.c[2];
          const double r = std::max(std::sqrt(d0 * d0 + d1 * d1 + d2 * d2), 1e-12);
          v.SetSize(3);
          v(0) = -d1 / r;
          v(1) = d0 / r;
          v(2) = 0.0;
        });
        mfem::ParGridFunction g(&fes);
        g.ProjectCoefficient(Et);
        mfem::Vector gt(fes.GetTrueVSize()), proj(nm);
        g.GetTrueDofs(gt);
        proj = 0.0;
        if (U.Height() > 0) U.MultTranspose(gt, proj);
        MPI_Allreduce(MPI_IN_PLACE, proj.GetData(), nm, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // direct evaluation on the same faces: FE field vs analytic field, projected on W(1,0,TE) = -sqrt(3/8pi) sin(th) phi/R1
        double dir_fe = 0, dir_an = 0, err2 = 0, nrm2 = 0;
        {
          mfem::Vector val(3), xq(3), nq(3);
          auto direct = [&](mfem::FaceElementTransformations* FT, int e, bool side1) {
            mfem::ElementTransformation* Te = pm.GetElementTransformation(e);
            for (int q = 0; q < irf.GetNPoints(); ++q) {
              const mfem::IntegrationPoint& ip = irf.IntPoint(q);
              FT->SetAllIntPoints(&ip);
              mfem::CalcOrtho(FT->Jacobian(), nq);
              FT->Transform(ip, xq);
              const double dS = ip.weight * nq.Norml2();
              const mfem::IntegrationPoint& eip = side1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
              Te->SetIntPoint(&eip);
              g.GetVectorValue(*Te, eip, val);
              mfem::Vector xe(3);
              Te->Transform(eip, xe);
              const double d0 = xq(0) - sh.c[0], d1 = xq(1) - sh.c[1], d2 = xq(2) - sh.c[2];
              const double r = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2), rxy = std::sqrt(d0 * d0 + d1 * d1);
              const double sth = rxy / r, phx = rxy > 0 ? -d1 / rxy : 0, phy = rxy > 0 ? d0 / rxy : 0;
              const double Wx = -std::sqrt(3.0 / (8 * pi)) * sth * phx / sh.R1, Wy = -std::sqrt(3.0 / (8 * pi)) * sth * phy / sh.R1;
              const double ax = -d1 / r, ay = d0 / r;
              dir_fe += dS * (val(0) * Wx + val(1) * Wy);
              dir_an += dS * (ax * Wx + ay * Wy);
              err2 += dS * ((val(0) - ax) * (val(0) - ax) + (val(1) - ay) * (val(1) - ay) + val(2) * val(2));
              nrm2 += dS * (ax * ax + ay * ay);
              if (q == 0 && e % 997 == 0)
                log(1, strcat("    pt x=", xq(0), ",", xq(1), ",", xq(2), " elem x=", xe(0), ",", xe(1), ",", xe(2), " fe=", val(0), ",",
                              val(1), ",", val(2), " an=", ax, ",", ay));
            }
          };
          for (int f = 0; f < pm.GetNumFaces(); ++f) {
            int e1, e2;
            pm.GetFaceElements(f, &e1, &e2);
            if (e1 < 0 || e2 < 0) continue;
            const bool s1 = sh.in(pm.GetAttribute(e1)), s2 = sh.in(pm.GetAttribute(e2));
            if (s1 == s2) continue;
            direct(pm.GetFaceElementTransformations(f), s1 ? e2 : e1, !s1);
          }
        }
        log(1, strcat("  U self-test (direct): FE ", fw_allsum(dir_fe), ", analytic ", fw_allsum(dir_an), ", rel L2 error ",
                      std::sqrt(fw_allsum(err2) / std::max(fw_allsum(nrm2), 1e-300))));
        log(1, strcat("  U self-test: TE(1,0) projection ", proj(2), " (exact ", -std::sqrt(3.0 / (8 * pi)) * 8 * pi / 3 * sh.R1,
                      "), others max ", std::max(std::fabs(proj(0)), std::fabs(proj(4))), ", I = ", P.I_pick, ", R1 = ", sh.R1,
                      ", area ", fw_allsum(area_R1), " (exact ", 4 * pi * sh.R1 * sh.R1, "), faces local ",
                      fw_allsum(nfaces_local), " shared ", fw_allsum(nfaces_shared)));
      }
      log(2, strcat("  3D full wave: exterior map with ", nm, " vector spherical harmonics (n <= ", N, ")"));
      out["exterior_modes"] = nm;
      // exterior part of the reciprocity integral: outside R1, B_lab = -grad Phi and div E = 0, so
      //   int_{rho > R1} E . B_lab dV = oint_{R1} Phi E_r dA,  Phi = sum_nm R1 beta_nm / (n + 1) Y_nm (R1/rho)^{n+1},
      // beta_nm = (1/R1^2) oint B_r Y_nm dA (the static multipoles; the far-surface term vanishes). Points on the
      // physical side of the R1 sphere. With an imposed background, A_lab holds only the reaction (and winding) field,
      // which decays outside R1 like any other; the background itself, which need not, has its own term below.
      {
        // multipole order of the static exterior field: the DtN order N, raised for a prescribed source so that
        // (r_source/R1)^Ns < 1e-6 (only scalar Y_nm are needed, so this is cheap); capped at 40
        int Ns = N;
        if (prescribed) {
          const double q = prescribed->max_distance_m({sh.c[0], sh.c[1], sh.c[2]}) / sh.R1;
          if (q > 0 && q < 1) Ns = std::clamp(static_cast<int>(std::ceil(std::log(1e-6) / std::log(q))), N, 40);
        }
        const int NY = (Ns + 1) * (Ns + 1) - 1;  // n = 1..Ns, m = -n..n
        mfem::Vector Bv(3), xq(3), nq(3);
        double ext_br2 = 0;  // oint B_r^2 dA, for the Parseval check of the truncated multipole series
        auto collect = [&](mfem::FaceElementTransformations* FT, int e, bool side1) {
          mfem::ElementTransformation* Te = pm.GetElementTransformation(e);
          for (int q = 0; q < irf.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& ip = irf.IntPoint(q);
            FT->SetAllIntPoints(&ip);
            mfem::CalcOrtho(FT->Jacobian(), nq);
            FT->Transform(ip, xq);
            ExtPt pt;
            pt.e = e;
            pt.eip = side1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
            pt.dS = ip.weight * nq.Norml2();
            const double d[3] = {xq(0) - sh.c[0], xq(1) - sh.c[1], xq(2) - sh.c[2]};
            const double rho = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            for (int i = 0; i < 3; ++i) pt.rh[i] = d[i] / rho, pt.x[i] = xq(i);
            pt.rho = rho;
            // The R1 interface is faceted (inscribed, rho <= R1): the exterior identity is applied exactly on the
            // facets (outward facet normal, Phi0 at the actual rho), and the sphere multipoles are projected
            // radially (solid-angle weight, B_r rescaled from rho to R1 per order n).
            const double nn = nq.Norml2();
            double ndr = 0;
            for (int i = 0; i < 3; ++i) pt.n[i] = nq(i) / nn, ndr += pt.n[i] * pt.rh[i];
            if (ndr < 0) {
              for (double& v : pt.n) v = -v;
              ndr = -ndr;
            }
            pt.dOmega = pt.dS * ndr / (rho * rho);
            const double th = std::acos(std::clamp(pt.rh[2], -1.0, 1.0)), ph = std::atan2(pt.rh[1], pt.rh[0]);
            pt.Y.resize(NY);
            int k = 0;
            for (int n = 1; n <= Ns; ++n)
              for (int m = -n; m <= n; ++m) pt.Y[k++] = real_ylm(n, m, th, ph);
            (void)Te;
            ext_pts.push_back(std::move(pt));
          }
        };
        ext_phi.assign(NY, 0.0);
        std::vector<double> ext_parseval(NY, 0.0);  // oint B_r Y_nm dA on the facets, for the Parseval check only
        for (int f = 0; f < pm.GetNumFaces(); ++f) {
          int e1, e2;
          pm.GetFaceElements(f, &e1, &e2);
          if (e1 < 0 || e2 < 0) continue;
          const bool s1 = sh.in(pm.GetAttribute(e1)), s2 = sh.in(pm.GetAttribute(e2));
          if (s1 == s2) continue;
          collect(pm.GetFaceElementTransformations(f), s1 ? e2 : e1, !s1);
        }
        for (int sf = 0; sf < pm.GetNSharedFaces(); ++sf) {
          mfem::FaceElementTransformations* FT = pm.GetSharedFaceTransformations(sf);
          if (sh.in(FT->Elem1->Attribute) || !sh.in(FT->Elem2->Attribute)) continue;
          collect(FT, FT->Elem1No, true);
        }
        // The R1 interface is faceted (inscribed): the source must stay inside the actual innermost R1 point, with
        // margin, or it would pierce shell elements where the volume overlap is zero. Checked collectively before
        // any Biot-Savart evaluation on R1 (an on-filament throw on one rank would hang the others).
        double rho_min = std::numeric_limits<double>::infinity();
        for (const auto& pt : ext_pts) rho_min = std::min(rho_min, pt.rho);
        MPI_Allreduce(MPI_IN_PLACE, &rho_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (prescribed) {
          const double rmax = prescribed->max_distance_m({sh.c[0], sh.c[1], sh.c[2]});
          out["source"]["exterior"]["R1_faceted_min_radius_m"] = rho_min;
          AXF_REQUIRE(rmax < 0.95 * rho_min, strcat("prescribed source currents reach ", rmax, " m from the domain "
                      "centre; the radiation boundary needs them inside 0.95 x the innermost R1 surface point (",
                      rho_min, " m)"), "/fullwave/source",
                      "Increase mesh.domain_radius (the multipole series converges as (r_source/R1)^n).");
        }
        for (const auto& pt : ext_pts) {
          if (prescribed) {  // the overlap field: the prescribed vacuum B0 (as in the volume term), not B_lab
            const auto b = prescribed->field_T({pt.x[0], pt.x[1], pt.x[2]});
            for (int i = 0; i < 3; ++i) Bv(i) = b[i];
          } else {
            mfem::ElementTransformation* Te = pm.GetElementTransformation(pt.e);
            Te->SetIntPoint(&pt.eip);
            P.A_lab->GetCurl(*Te, Bv);
          }
          const double Br = Bv(0) * pt.rh[0] + Bv(1) * pt.rh[1] + Bv(2) * pt.rh[2];
          ext_br2 += pt.dS * Br * Br;
          // beta_nm = oint_{R1} B_r(R1) Y_nm dOmega; the n-th order term of B_r scales as (R1/rho)^(n+2)
          const double up = pt.rho / sh.R1;
          double scale = up * up;
          for (int n = 1, j = 0; n <= Ns; ++n) {
            scale *= up;
            for (int m = -n; m <= n; ++m, ++j) {
              ext_phi[j] += pt.dOmega * Br * scale * pt.Y[j];
              ext_parseval[j] += pt.dS * Br * pt.Y[j];
            }
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, ext_phi.data(), NY, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, ext_parseval.data(), NY, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        ext_br2 = fw_allsum(ext_br2);
        {  // Parseval on the facet quadrature: oint B_r^2 dA = sum (oint B_r Y_nm dA)^2 / R1^2
          double captured = 0;
          for (double v : ext_parseval) captured += v * v / (sh.R1 * sh.R1);
          const double frac = ext_br2 > 0 ? 1.0 - captured / ext_br2 : 0.0;
          const std::string field_name = prescribed ? "prescribed source B0"
                                         : P.bg ? "static B_lab = curl A_lab (reaction and windings; the background "
                                                  "has its own exterior term)"
                                                : "static B_lab";
          out["exterior_source_multipoles"] = {
              {"field", field_name}, {"n_max", Ns},
              {"oint_Br2_dA", ext_br2}, {"unrepresented_fraction", frac},
              {"scope", "signed fraction of oint B_r^2 on R1 outside n <= n_max (truncation plus R1 quadrature; "
                        "negative = quadrature over-capture); the exterior Psi term uses only n <= n_max"}};
          if (prescribed && std::fabs(frac) > 1e-3)
            warnings.push_back(strcat("3D full wave: the exterior multipoles (n <= ", Ns, ") of the prescribed source "
                                      "miss a fraction ", frac, " of oint B_r^2 on R1; enlarge mesh.domain_radius "
                                      "or refine the R1 surface"));
        }
        int k = 0;  // ext_phi = beta_nm -> Phi coefficient R1 beta_nm / (n + 1)
        for (int n = 1; n <= Ns; ++n)
          for (int m = -n; m <= n; ++m) ext_phi[k++] *= sh.R1 / (n + 1.0);
        out["exterior_term"] = NY > 0 ? "included" : "omitted (no multipoles)";
      }
      if (P.bg) {
        // the imposed background's exterior term (axi::BackgroundExterior). A grid is zero outside its box, so the
        // integral is finite; a uniform field fills all space and its exterior integral diverges (A_phi ~ rho), so it
        // is omitted as in the axisymmetric engine
        const auto& bgs = d_.background;
        json& binfo = out["background"];
        if (bgs.type != "grid") {
          out["exterior_term_background"] = "omitted (uniform background: the exterior integral diverges)";
          warnings.push_back("3D full wave: the exterior reciprocity term of Psi(omega) is omitted for the uniform "
                             "background, whose field fills all space, so Psi(omega) depends on the domain radius above "
                             "the MQS limit: between R1 and 1.5 R1 the axisymmetric engine's Psi moved by 0.7-44 % at "
                             "kR = 0.4 and by up to 159 % at kR = 0.8 in docs/validation/axi_background_exterior.md; "
                             "vary the domain radius to see how much, or give the magnet's field as a grid to include "
                             "the term");
        } else if (tm0_col.empty()) {
          out["exterior_term_background"] = "omitted (no exterior modes)";
          warnings.push_back("3D full wave: the exterior reciprocity term of Psi(omega) is omitted for the background "
                             "(no exterior modes, AXF_FW_NMODES = 0)");
        } else {
          // A guard only: generated meshes are centred at the origin, and imported meshes, mesh.symmetry and
          // mesh.domain_solid need rf_only, which a background sweep refuses, so no design reaches it today.
          AXF_REQUIRE(std::hypot(sh.c[0], sh.c[1]) <= 1e-9 * sh.R1,
                      strcat("the domain centre (", sh.c[0], ", ", sh.c[1], ", ", sh.c[2], ") is off the z axis, the "
                             "axis of the background field"), "/background",
                      "The exterior term of an axisymmetric background needs the domain sphere centred on the z axis.");
          const int Nb = mode_n[tm0_col.back()];
          const double r_max = bgs.r.back();
          const double z_lo = bgs.z.front();
          const double z_hi = bgs.z.back();
          const int panels = axi::background_exterior_panels();
          bg_ext = axi::background_exterior(*P.bg, r_max, z_lo, z_hi, sh.R1, sh.c[2], Nb, panels);
          // quadrature check: the k -> 0 moments again with twice the panels (to order 12)
          double gmax = 0;
          const double quad_change = axi::background_exterior_quadrature_check(*P.bg, r_max, z_lo, z_hi, sh.R1, sh.c[2],
                                                                               panels, bg_ext, &gmax);
          // the shell slivers between the faceted R1 interface and the sphere: neither the finite-element volume nor
          // the quadrature (from R1 outwards) covers them; first order in the facet sag (R1 - rho), at rho = R1
          bg_sliver.assign(Nb + 1, 0.0);
          std::vector<double> dy;
          for (const auto& pt : ext_pts) {
            const double r = sh.R1 * std::hypot(pt.rh[0], pt.rh[1]);
            const double z = sh.c[2] + sh.R1 * pt.rh[2];
            if (!(r > 0)) continue;
            const double A = P.bg->psi(r, z) / r;
            axi::dtheta_y0(pt.rh[2], Nb, dy);
            for (int n = 1; n <= Nb; ++n) bg_sliver[n] += pt.dOmega * (sh.R1 - pt.rho) * sh.R1 * sh.R1 * A * dy[n];
          }
          MPI_Allreduce(MPI_IN_PLACE, bg_sliver.data(), Nb + 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          double sliver = 0;
          for (int n = 1; n <= Nb; ++n) sliver = std::max(sliver, std::fabs(bg_sliver[n]));
          binfo["exterior"] = {
              {"n_max", Nb}, {"radial_nodes", bg_ext.rho.size()}, {"panels", panels},
              {"grid_max_distance_from_domain_center_m",
               std::max(std::hypot(r_max, z_hi - sh.c[2]), std::hypot(r_max, z_lo - sh.c[2]))},
              {"continuation_radius_m", bg_ext.rho_s},
              {"continuation_mismatch", bg_ext.tail_mismatch}, {"continuation_samples", bg_ext.tail_samples},
              {"quadrature_relative_change_doubled_panels", quad_change},
              {"sliver_correction_over_moments", gmax > 0 ? sliver / gmax : 0.0},
              {"scope", "int_{rho > R1} curl E . A_bg dV from the TM m = 0 exterior multipoles (n <= n_max): the grid "
                        "up to the sphere inscribed in its box (continuation_radius_m), its exterior multipoles "
                        "beyond (continuation_mismatch: relative rms against the grid where the box reaches farther; "
                        "0 samples: none to compare), or, with no such sphere outside R1, the grid over its box; "
                        "quadrature check on the k -> 0 moments to order 12; first-order correction for the slivers "
                        "between the R1 facets and the sphere (sliver_correction_over_moments)"}};
          out["exterior_term_background"] =
              bg_ext.rho.empty() ? "none (the grid ends inside the domain sphere, where its field is cut off: "
                                   "grid_edge_A_over_max)"
              : bg_ext.rho_s > 0 ? "included (grid, then its exterior multipoles)"
                                 : "included (grid only: cut at its box)";
          const double edge_frac = binfo.value("grid_edge_A_over_max", 0.0);
          if (bg_ext.rho.empty() && edge_frac > 1e-3)
            warnings.push_back(strcat("3D full wave: the background grid ends inside the domain sphere (its farthest "
                                      "point lies at rho = ", std::max(std::hypot(r_max, z_hi - sh.c[2]),
                                                                        std::hypot(r_max, z_lo - sh.c[2])),
                                      " m, R1 = ", sh.R1, " m) while A_bg on its edge is still ", edge_frac,
                                      " of its maximum: the magnet's field beyond the grid is coupled neither inside "
                                      "the domain nor outside it; extend the grid beyond the domain sphere, and far "
                                      "enough that its inscribed sphere encloses the magnet"));
          if (!(quad_change <= 1e-4))
            warnings.push_back(strcat("3D full wave: the background's exterior quadrature changes by ", quad_change,
                                      " with doubled panels; set AXF_FW_BG_PANELS above ", panels));
          if (bg_ext.rho_s > 0 && !(bg_ext.tail_mismatch <= 1e-2))
            warnings.push_back(strcat("3D full wave: beyond the sphere rho = ", bg_ext.rho_s, " m inscribed in the "
                                      "background grid, its exterior multipoles differ from the grid by ",
                                      bg_ext.tail_mismatch, " (rms): the magnet's sources may reach that sphere; "
                                      "extend the grid so that its inscribed sphere encloses them"));
          if (!bg_ext.rho.empty() && bg_ext.rho_s == 0)
            warnings.push_back("3D full wave: the background grid's inscribed sphere lies inside R1, so the exterior "
                               "term uses the grid only up to its box, where the magnet's field is cut off; extend the "
                               "grid beyond the domain sphere in every direction");
        }
      }
    }
    // RF field kept for scenes: at the sweep frequency nearest to fullwave.scene_frequency (default the first)
    double moment2 = 0;
    // ext_pts is rank-local: interior-only ranks must contribute zero to every collective.
    if (radiation && P.A_pick) {
      double moment[3] = {0,0,0}; mfem::Vector B(3);
      for (auto& pt : ext_pts) {
        auto* T = pm.GetElementTransformation(pt.e); T->SetIntPoint(&pt.eip);
        P.A_pick->GetCurl(*T,B);
        double br = B(0)*pt.rh[0]+B(1)*pt.rh[1]+B(2)*pt.rh[2];
        for (int j=0;j<3;++j) moment[j] += pt.dS*br*pt.rh[j]*3*sh.R1/(2*mu0*I);
      }
      MPI_Allreduce(MPI_IN_PLACE,moment,3,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
      out["dipole_trace_ranks"] = static_cast<int>(fw_allsum(ext_pts.empty() ? 0.0 : 1.0));
      out["mqs_magnetic_moment_per_A_m2"] = {moment[0],moment[1],moment[2]};
      for (double m : moment) moment2 += m*m;
    }
    double f_scene = fw.frequencies.front();
    if (fw.scene_frequency > 0)
      for (double f : fw.frequencies)
        if (std::fabs(f - fw.scene_frequency) < std::fabs(f_scene - fw.scene_frequency)) f_scene = f;
    const auto conv = mfem::ComplexOperator::HERMITIAN;
    // recovery error indicator of every port field (docs/physics/fullwave_error_indicator.md)
    std::unique_ptr<FwErrorIndicator> indicator;
    if (indicator_mode == "on") {
      std::vector<char> in_pml;
      if (fwpml.active())
        for (int e = 0; e < ne; ++e) in_pml.push_back(fwpml.epml[e] >= 0 ? 1 : 0);
      FwErrorIndicator::Terminations terminations;
      terminations.radiating_shell = radiation;
      terminations.pml = fwpml.active();
      for (const auto& b : robin) terminations.robin_groups.push_back(b.name + " (" + b.type + ")");
      indicator = std::make_unique<FwErrorIndicator>(P, d_, in_pml, terminations);
      out["error_indicator"] = indicator->Describe();
    }
    // perturbative surface losses of the conductor surfaces per frequency and port (FwSurfaceLosses,
    // docs/physics/fullwave_surface_losses.md); the magnetoresistance weight needs the static solve
    std::unique_ptr<FwSurfaceLosses> surface_losses =
        std::make_unique<FwSurfaceLosses>(P, d_, fwpml.epml, !d_.fullwave.rf_only);
    if (!surface_losses->reported()) surface_losses.reset();
    json sweep = json::array();
    bool capacitive = false;
    int total_its = 0;
    out["setup_before_frequencies_seconds"] = fw_allmax(timer.seconds());
    int frequency_index = 0;
    const bool malloc_trim_each = trim_mode == "frequency";
    const bool frequency_peak = fw_frequency_peak_requested();
    // set when a validation gate refused the frequency (retain_rejected): the adaptive sweep records the frequency and
    // moves on, a fixed list stops there
    bool frequency_rejected = false;
    // The per-frequency body, shared by the fixed list and the adaptive sweep: solve every port at f and append the row
    // to `sweep` (or retain the rejected row and throw). Each row carries its memory record (fw_memory_record at the
    // start and end of the frequency).
    auto solve_frequency = [&](double f) {
      Timer frequency_timer;
      json memory = {{"frequency_start", fw_memory_record()}};   // what the setup and earlier frequencies left
      if (malloc_trim_each) {
        fw_malloc_trim();
        memory["frequency_start_after_malloc_trim"] = fw_memory_record();
      }
      memory["peak_scope"] = "the run so far";
      if (frequency_peak) {
        const double failed = fw_allsum(fw_reset_peak_rss() ? 0.0 : 1.0);
        memory["peak_scope"] = failed == 0.0 ? std::string("this frequency (AXF_FW_FREQUENCY_PEAK=1)")
                                             : strcat("the run so far: AXF_FW_FREQUENCY_PEAK=1, but the reset failed on ",
                                                      failed, " rank(s) (/proc/self/clear_refs)");
      }
      const double w = 2 * pi * f;
      set_eps(w);
      fwpml.omega = w;
      FwTensor nr(sh, nu_e, 0, 0, 1.0, gauge, &fwpml), ni(sh, nu_e, 0, 1, 1.0, gauge, &fwpml),
          na(sh, nu_e, 0, 2, 1.0, gauge, &fwpml);
      FwTensor mr(sh, eps_e, 1, 0, -w * w, gauge, &fwpml), mi(sh, eps_e, 1, 1, -w * w, gauge, &fwpml),
          ma(sh, eps_e, 1, 2, w * w, gauge, &fwpml);
      std::vector<std::unique_ptr<RobinCoef>> robin_coef;
      for (const auto& b : robin)
        for (int part = 0; part < 3; ++part)
          robin_coef.push_back(std::make_unique<RobinCoef>(b, bdr_elem, P.elem_mat, w, part));
      // exterior correction coefficients at this frequency
      std::vector<cplx> cmode(mode_n.size());
      if (!mode_n.empty()) {
        const int N = *std::max_element(mode_n.begin(), mode_n.end());
        const auto gam = axi::hankel_log_derivative(w / units::c * sh.R1, N);
        for (size_t j = 0; j < mode_n.size(); ++j) {
          const int n = mode_n[j];
          // static TE response of the shell: exterior to infinity (n) or capped at F = fmax R1 with E_t = 0 there,
          // (n + (n+1) q) / (1 - q), q = (R1/F)^(2n+1); the static TM response is zero either way
          double st = n;
          if (sh.fmax > 1) {
            const double q = std::pow(1.0 / sh.fmax, 2.0 * n + 1.0);
            st = (n + (n + 1.0) * q) / (1.0 - q);
          }
          cmode[j] = axi::fw_exterior_coefficient(mode_te[j], st, w, sh.R1, gam[n]);
        }
      }
      mfem::ParSesquilinearForm a(&fes, conv);
      a.AddDomainIntegrator(new mfem::CurlCurlIntegrator(nr), new mfem::CurlCurlIntegrator(ni));
      a.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mr), new mfem::VectorFEMassIntegrator(mi));
      for (size_t k = 0; k < robin.size(); ++k)
        a.AddBoundaryIntegrator(new mfem::VectorFEMassIntegrator(*robin_coef[3 * k]),
                                new mfem::VectorFEMassIntegrator(*robin_coef[3 * k + 1]), robin_marker[k]);
      a.Assemble();
      // a lossless model's imaginary part holds only zeros: on a GPU hypre's triple product drops all of them, and the
      // elimination in FormLinearSystem then dereferences an empty matrix (keep_eliminated_rows, model3d_impl.hpp)
      keep_eliminated_rows(a.real(), ess);
      keep_eliminated_rows(a.imag(), ess);
      // right-hand side i omega J; only shapes FormLinearSystem: B is reset from Jports for every drive below
      mfem::ParComplexLinearForm b(&fes, conv);
      std::unique_ptr<PortJ> Jw;
      if (!surface_port) {
        Jw = std::make_unique<PortJ>(*P.port_V, P.port_attr, P.port_scale * w);
        b.AddDomainIntegrator(nullptr, new mfem::VectorFEDomainLFIntegrator(*Jw, &port_feed_rule(P.order)));
      }
      b.Assemble();
      mfem::ParComplexGridFunction x(&fes);
      x = 0.0;
      mfem::OperatorPtr Ah;
      mfem::Vector X, B;
      a.FormLinearSystem(ess, x, b, Ah, X, B);
      // AMS needs a separate SPD block. Do not assemble or retain that
      // unused hierarchy for the root-owned direct path: it can dominate
      // memory on the finely resolved plate-stack mesh.
      mfem::ParBilinearForm pc(&fes);
      mfem::OperatorPtr PCop;
      std::unique_ptr<mfem::HypreAMS> ams;
      json ams_options;   // the AMS options in force (fw_ams_overrides), recorded in solver_settings
      std::unique_ptr<mfem::ScaledOperator> ams_i;
      // second block: +M makes the preconditioner complex-linear (an approximation of A^{-1} up to the sign of the
      // gradient fields), which the Woodbury update needs; -M is the ex25p choice (same eigenvalue clusters)
      const bool woodbury = U.Width() > 0 && !(std::getenv("AXF_FW_WOODBURY") && std::atoi(std::getenv("AXF_FW_WOODBURY")) == 0);
      const double psign = woodbury ? 1.0 : (conv == mfem::ComplexOperator::HERMITIAN ? -1.0 : 1.0);
      mfem::Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = fes.GetTrueVSize();
      offsets[2] = fes.GetTrueVSize();
      offsets.PartialSum();
      mfem::BlockDiagonalPreconditioner prec(offsets);
      // Woodbury update with the exterior coefficients as the SPD preconditioner sees them: AMS approximates
      // +A^{-1} on curl fields (TE traces) and -A^{-1} on gradient fields (TM traces), so the TM coefficients
      // enter with the opposite sign
      std::unique_ptr<WoodburyPrec> wprec;
      std::vector<cplx> cprec(cmode.size());
      for (size_t j = 0; j < cmode.size(); ++j) cprec[j] = mode_te[j] ? cmode[j] : -cmode[j];
      if (!use_direct) {
        pc.AddDomainIntegrator(new mfem::CurlCurlIntegrator(na));
        pc.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(ma));
        for (size_t k = 0; k < robin.size(); ++k)
          pc.AddBoundaryIntegrator(new mfem::VectorFEMassIntegrator(*robin_coef[3 * k + 2]), robin_marker[k]);
        pc.Assemble();
        pc.FormSystemMatrix(ess, PCop);
        ams = std::make_unique<mfem::HypreAMS>(*PCop.As<mfem::HypreParMatrix>(), &fes);
        ams->SetPrintLevel(0);
        ams_options = fw_ams_overrides(*ams);
        ams_i = std::make_unique<mfem::ScaledOperator>(ams.get(), psign);
        prec.SetDiagonalBlock(0, ams.get());
        prec.SetDiagonalBlock(1, ams_i.get());
        if (woodbury) wprec = std::make_unique<WoodburyPrec>(*ams, 1.0, prec, U, cprec);
      }
      mfem::GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetPrintLevel(log_level() >= 3 && p_->rank == 0 ? 1 : 0);
      // Restart 600: lossy ND2 solves within ~1% of a self-resonance stalled at restart 300 (12,000-iteration
      // cap, residual 5e-5..0.17) and converged in 485-934 iterations at 600. The Krylov
      // basis costs at most (kdim + 1) x 2 x (global true DOFs) x 8 bytes summed over ranks; MFEM allocates the
      // basis vectors lazily, so solves converging in fewer than kdim iterations pay only for those they use.
      const int gmres_kdim = std::getenv("AXF_FW_KDIM") ? std::atoi(std::getenv("AXF_FW_KDIM")) : 600;
      const int gmres_max_iter = std::getenv("AXF_FW_MAX_ITER") ? std::atoi(std::getenv("AXF_FW_MAX_ITER")) : 3000;
      const double gmres_rtol = std::getenv("AXF_FW_RTOL") ? std::atof(std::getenv("AXF_FW_RTOL")) : 1e-10;
      AXF_REQUIRE(gmres_kdim > 0 && gmres_max_iter > 0 && std::isfinite(gmres_rtol) && gmres_rtol > 0,
                  "AXF_FW_KDIM, AXF_FW_MAX_ITER and AXF_FW_RTOL must be positive", "/fullwave");
      gmres.SetKDim(gmres_kdim);
      gmres.SetMaxIter(gmres_max_iter);
      gmres.SetRelTol(gmres_rtol);
      gmres.SetAbsTol(0.0);
      // basis (kdim+1 vectors) plus GMRES work vectors r, w; per Mult call (per right-hand side, freed after)
      const double krylov_bytes = (gmres_kdim + 3.0) * 2.0 * static_cast<double>(fes.GlobalTrueVSize()) * 8.0;
      out["solver_settings"] = {{"gmres_restart",gmres_kdim}, {"gmres_max_iterations",gmres_max_iter},
          {"requested_preconditioned_relative_tolerance",gmres_rtol}, {"true_relative_residual_gate",1e-6},
          {"gmres_krylov_basis_bytes_upper_bound_all_ranks",krylov_bytes},
          {"overrides","AXF_FW_KDIM, AXF_FW_MAX_ITER, AXF_FW_RTOL"}};
      if (!ams_options.is_null()) out["solver_settings"]["ams_options"] = ams_options;
      if (frequency_index == 0)
        log(2, strcat("  3D full wave: GMRES restart ", gmres_kdim, ", Krylov basis <= ", krylov_bytes / 1e9,
                      " GB summed over ranks (allocated as used; lower AXF_FW_KDIM if memory-bound)"));
      // imaginary parts of the Robin boundary operators per group (absorbed power, -Im(E^H A_g E)/(2 omega))
      std::vector<std::unique_ptr<mfem::HypreParMatrix>> robin_im;
      for (size_t k = 0; k < robin.size(); ++k) {
        mfem::ParBilinearForm bf(&fes);
        bf.AddBoundaryIntegrator(new mfem::VectorFEMassIntegrator(*robin_coef[3 * k + 1]), robin_marker[k]);
        bf.Assemble();
        bf.Finalize();
        robin_im.emplace_back(bf.ParallelAssemble());
      }
      FwOperator Afull(*Ah, U, cmode);
      gmres.SetOperator(Afull);
      std::unique_ptr<mfem::ComplexHypreParMatrix> direct_matrix;
      std::unique_ptr<FwDirectFactor> direct_preconditioner;
      double factorization_seconds=0.0;
      double tree_cotree_consistency=std::numeric_limits<double>::quiet_NaN();
      json tree_cotree_gradient_check;
      if (use_direct) {
        AXF_REQUIRE(U.Width()==0,"STRUMPACK option currently requires no exterior low-rank update", "/fullwave");
        auto* complex_operator=dynamic_cast<mfem::ComplexOperator*>(Ah.Ptr());
        AXF_REQUIRE(complex_operator && complex_operator->hasRealPart(),
                    "STRUMPACK option requires an assembled real Maxwell block", "/fullwave");
        direct_matrix.reset(complex_operator->AsComplexHypreParMatrix());
        double imaginary_maximum=0.0;
        if(direct_matrix->hasImagPart()) {
          mfem::SparseMatrix imaginary_rows;
          direct_matrix->imag().MergeDiagAndOffd(imaginary_rows);
          for(int k=0;k<imaginary_rows.NumNonZeroElems();++k)
            imaginary_maximum=std::max(imaginary_maximum,std::abs(imaginary_rows.GetData()[k]));
        }
        AXF_REQUIRE(fw_allmax(imaginary_maximum)==0.0,
                    "STRUMPACK real-block option cannot drop material loss or complex coefficients", "/fullwave",
                    "Use AMS for a complex material operator; reactive validation loads remain supported.");
        Timer factorization_timer;
        if (tree_cotree) {
          // The split form assumes the assembled operator is exactly K - w^2 M on free DOFs: check it.
          const int nt = fes.GetTrueVSize();
          mfem::Vector v(2*nt), av(2*nt), kv(nt), mv(nt), vr(v.GetData(),nt);
          v.Randomize(1234 + P.rank);
          for (int i = nt; i < 2*nt; ++i) v(i) = 0.0;
          for (int i : ess) v(i) = 0.0;
          Afull.Mult(v, av);
          energy_K->Mult(vr, kv);
          energy_M->Mult(vr, mv);
          std::vector<char> is_ess(nt, 0);
          for (int i : ess) is_ess[i] = 1;
          double d2 = 0.0, a2 = 0.0;
          for (int i = 0; i < nt; ++i) {
            if (is_ess[i]) continue;
            const double split = kv(i) - w*w*mv(i);
            d2 += (av(i)-split)*(av(i)-split); a2 += split*split;
          }
          const double consistency = std::sqrt(fw_allsum(d2)/std::max(fw_allsum(a2),1e-300));
          AXF_REQUIRE(consistency < 1e-11, strcat("tree-cotree: assembled operator differs from K - w^2 M (relative ",
                      consistency, ")"), "/fullwave");
          gauged->Factor(w);
          tree_cotree_consistency = consistency;
          // Gradient-block accuracy of the two representations on a pure gradient g = G phi (phi = 0 on the
          // boundary): exact discrete A g = -w^2 M g. legacy: the summed assembled matrix (the path without the
          // tree-cotree split);
          // separate_K: |K g| from the separately assembled K (rounding the tree-cotree path never forms).
          {
            mfem::Vector gv(2*nt), agv(2*nt), kg(nt), mg(nt);
            gv = 0.0;
            for (int i = 0; i < nt; ++i) gv(i) = grad_probe(i);
            Afull.Mult(gv, agv);
            energy_K->Mult(grad_probe, kg);
            energy_M->Mult(grad_probe, mg);
            double l2 = 0.0, k2 = 0.0, m2 = 0.0;
            for (int i = 0; i < nt; ++i) {
              if (is_ess[i]) continue;
              const double wm = w*w*mg(i);
              l2 += (agv(i)+wm)*(agv(i)+wm); k2 += kg(i)*kg(i); m2 += wm*wm;
            }
            l2 = fw_allsum(l2); k2 = fw_allsum(k2); m2 = fw_allsum(m2);
            tree_cotree_gradient_check = {
                {"legacy_summed_matrix_relative_error_on_gradients", std::sqrt(l2/std::max(m2,1e-300))},
                {"separate_K_on_gradients_relative_to_w2_M", std::sqrt(k2/std::max(m2,1e-300))},
                {"scope","random pure gradient of the reduced system, phi=0 on every vertex of an essential (PEC) edge; exact value 0 for both"}};
          }
        } else {
          direct_preconditioner=fw_make_direct_factor(direct_matrix->real());
          gmres.SetPreconditioner(*direct_preconditioner);
        }
        factorization_seconds=fw_allmax(factorization_timer.seconds());
      } else if (wprec) gmres.SetPreconditioner(*wprec);
      else gmres.SetPreconditioner(prec);
      // Gauss-law guard: only where the assembled operator is K - w^2 M on the free DOFs with a zero imaginary block
      // (closed and lossless). A random vector checks the operator as the gauged path does; it cannot see the mass
      // block at low frequency (w^2 |M v| / |K v| ~ (k h)^2), so the mass coefficients are compared directly: every
      // element the essential DOFs do not close must have the permittivity energy_M was built with, eps0 eps_r.
      bool guard_on = false;
      json guard_frequency = json::object();
      if (guard) {
        // Re eps_c of every element the essential DOFs do not close must be the eps0 eps_r energy_M was built with;
        // Im eps_c must vanish unless the model is lossy, when the guard's loss mass carries it. The loss contrast,
        // max Im eps_c / Re eps_c, bounds how well a conductive region's potential is resolved
        // (docs/physics/fullwave_gauss_guard.md).
        double eps_mismatch = 0.0;
        double loss_contrast = 0.0;
        for (int e = 0; e < ne; ++e) {
          if (closed_element(e, !radiation, nullptr)) continue;
          const double er = P.elem_mat[e]->eps_r > 0 ? P.elem_mat[e]->eps_r : 1.0;
          const double imag = lossy ? 0.0 : eps_e[e].imag();
          eps_mismatch = std::max(eps_mismatch, std::hypot(eps_e[e].real() - eps0 * er, imag) / (eps0 * er));
          loss_contrast = std::max(loss_contrast, eps_e[e].imag() / eps_e[e].real());
        }
        eps_mismatch = fw_allmax(eps_mismatch);
        loss_contrast = fw_allmax(loss_contrast);
        json loss_preconditioner;
        std::string loss_failure;
        if (lossy) {
          ShellMapR no_shell;
          FwTensor loss(no_shell, eps_e, 1, 1, 1.0, 0.0);
          mfem::ParBilinearForm form(&fes);
          form.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(loss));
          form.Assemble();
          form.Finalize();
          std::unique_ptr<mfem::HypreParMatrix> Mi(form.ParallelAssemble());
          loss_failure = guard->SetLoss(std::move(Mi), loss_preconditioner);
        }
        const int nt = fes.GetTrueVSize();
        mfem::Vector v(2*nt), av(2*nt), kv(nt), mv(nt), miv(nt), vr(v.GetData(),nt);
        v.Randomize(2468 + P.rank);
        for (int i = nt; i < 2*nt; ++i) v(i) = 0.0;
        for (int i : ess) v(i) = 0.0;
        Afull.Mult(v, av);
        energy_K->Mult(vr, kv);
        energy_M->Mult(vr, mv);
        miv = 0.0;
        if (guard->LossMass()) guard->LossMass()->Mult(vr, miv);
        std::vector<char> is_ess(nt, 0);
        for (int i : ess) is_ess[i] = 1;
        // the real block against K - w^2 M; the imaginary block against -w^2 Mi (zero without loss: then part of the
        // real block's comparison). The imaginary block is a pure mass, so its check holds at any frequency.
        double d2 = 0.0, a2 = 0.0, di2 = 0.0, ai2 = 0.0;
        for (int i = 0; i < nt; ++i) {
          if (is_ess[i]) continue;
          const double split = kv(i) - w*w*mv(i);
          const double loss_split = -w*w*miv(i);
          d2 += (av(i)-split)*(av(i)-split);
          a2 += split*split;
          di2 += (av(nt+i)-loss_split)*(av(nt+i)-loss_split);
          ai2 += loss_split*loss_split;
        }
        d2 = fw_allsum(d2);
        a2 = fw_allsum(a2);
        di2 = fw_allsum(di2);
        ai2 = fw_allsum(ai2);
        const double consistency = lossy ? std::sqrt(d2/std::max(a2,1e-300)) : std::sqrt((d2+di2)/std::max(a2,1e-300));
        const double loss_consistency = lossy ? std::sqrt(di2/std::max(ai2,1e-300)) : 0.0;
        guard_on = consistency < 1e-11 && loss_consistency < 1e-11 && eps_mismatch == 0.0 && loss_failure.empty();
        guard_frequency = {{"mode",guard_mode}, {"applied",guard_on && guard_mode == "correct"},
                           {"operator_consistency_relative",consistency}, {"mass_coefficient_mismatch",eps_mismatch},
                           {"lossy",lossy}};
        if (lossy) {
          guard_frequency["loss_mass_consistency_relative"] = loss_consistency;
          guard_frequency["loss_preconditioner"] = loss_preconditioner;
          guard_frequency["loss_contrast"] = loss_contrast;
        }
        if (!guard_on)
          guard_frequency["not_applied"] = !loss_failure.empty() ? loss_failure
              : std::string("the assembled operator is not K - w^2 (M + i Mi) on the free DOFs (complex coefficients "
                            "beyond the material loss)");
      }
      if (guard_mode != "off" && !guard_on && w / units::c * Rdev < kGuardWarnKR) {
        unguarded_frequencies.push_back(f);
        unguarded_reason = guard ? guard_frequency["not_applied"].get<std::string>()
                                 : guard_record["not_applied"].get<std::string>();
      }
      // one solve's correction (or estimate): a second pass when the first removed more than the field itself (the
      // PCG tolerance is relative to the Gauss residual, which grows with the error); warnings name the solve
      auto apply_guard = [&](mfem::Vector& xv, const mfem::Vector& bv, mfem::Vector* rv, const mfem::DenseMatrix* Uv,
                             const std::vector<cplx>* cv, const std::string& what) {
        const bool correct = guard_mode == "correct";
        json stats = guard->Correct(w, xv, bv, rv, correct, Uv, cv);
        const double fraction = stats["field_fraction"].get<double>();
        if (stats["applied"].get<bool>() && fraction > 1.0) {
          stats["second_pass"] = guard->Correct(w, xv, bv, rv, true, Uv, cv);
          if (stats["second_pass"].contains("not_applied"))
            stats["not_applied"] = "second pass: " + stats["second_pass"]["not_applied"].get<std::string>();
        }
        const std::string where = strcat("3D full wave at f = ", f, " Hz, ", what, ": ");
        // With loss, the backward error does not bound the potential of a strongly conducting region (its constant mode
        // is held only by the O(eps0) coupling through its boundary): an estimate-only pass on the corrected field
        // measures what is left there, a fresh rounding realization of that mode (floor_estimate).
        if (stats["applied"].get<bool>() && stats["lossy"].get<bool>()) {
          const json check = guard->Correct(w, xv, bv, nullptr, false, Uv, cv);
          const double floor = check["field_fraction"].get<double>();
          stats["floor_estimate"] = floor;
          if (!(floor <= kGuardFloorWarn))
            warnings.push_back(where + strcat("the Gauss-law correction is resolved only to about ", floor, " of the field "
                                              "(energy norm), near conductive regions (loss contrast ",
                                              guard_frequency.value("loss_contrast", 0.0), "): field values there carry "
                                              "that error; good conductors are better modelled as PEC or impedance walls"));
        }
        if (stats.contains("not_applied"))
          warnings.push_back(where + "the Gauss-law guard was not applied (" + stats["not_applied"].get<std::string>() +
                             "); the field outputs carry an unchecked gradient error (estimate " +
                             strcat(fraction) + ")");
        else if (!(fraction <= 1e-3) && correct)
          warnings.push_back(where + strcat("the Gauss-law guard removed a gradient-field error of ", fraction,
                                            " of the field (energy norm) from the solver's field; the curl part's "
                                            "error is not estimated"));
        else if (!(fraction <= 1e-3))
          warnings.push_back(where + strcat("the Gauss-law guard estimates a gradient-field error of ", fraction,
                                            " of the field (energy norm), NOT corrected (AXF_FW_GAUSS_GUARD=estimate)"));
        return stats;
      };
      const double solver_setup_seconds = fw_allmax(frequency_timer.seconds());
      const double unknown = std::numeric_limits<double>::quiet_NaN();
      Eigen::MatrixXcd Zmatrix = Eigen::MatrixXcd::Constant(nports,nports,cplx(unknown,unknown));
      Eigen::VectorXcd Psivec = Eigen::VectorXcd::Constant(nports,cplx(unknown,unknown));
      std::vector<mfem::Vector> fields_re(nports), fields_im(nports);
      // tree-cotree: curl representatives S_c e (K E = K E_c exactly), for magnetic energy and probe curls
      std::vector<mfem::Vector> curl_re(nports), curl_im(nports);
      std::vector<mfem::Vector> residuals_re(nports), residuals_im(nports);
      json field_files = json::array();
      json probe_fields = json::array();
      std::vector<double> surf_B;   // nports x npts x 6, filled per drive below
      auto save_field = [&](const mfem::Vector& re, const mfem::Vector& im, const std::string& name) {
        if (field_dir.empty()) return;
        const std::string prefix = strcat("f_", frequency_index, "_", name, "_rank_", P.rank);
        mfem::ParGridFunction part(&fes);
        for (int imag = 0; imag < 2; ++imag) {
          part.SetFromTrueDofs(imag ? im : re);
          const auto path = field_dir / (prefix + (imag ? "_im.gf" : "_re.gf"));
          bool written = false;
          {
            std::ofstream stream(path);
            if (stream) {
              stream << std::setprecision(17);
              part.Save(stream);
              stream.close();
            }
            written = static_cast<bool>(stream);
          }
          // agreed on every rank: a rank-local decision would let one rank's failure hang the others at the next
          // collective
          const std::string failing = fw_failing_ranks(!written);
          AXF_REQUIRE(failing.empty(), strcat("cannot write fullwave field column ", prefix, " (ranks ", failing, ")"),
                      "/fullwave/field_export_directory");
        }
      };
      json port_solutions = json::array();
      json primary_row;
      if (guard) primary_row["gauss_law_guard"] = guard_frequency;
      if(tree_cotree) primary_row["direct_factorization"] = {
          {"backend","STRUMPACK tree-cotree gauged"}, {"factorizations_per_frequency",1}, {"factor_owner_rank",0},
          {"factor",gauged->FactorInfo()}, {"seconds",factorization_seconds},
          {"operator_consistency_relative",tree_cotree_consistency},
          {"gradient_block_check",tree_cotree_gradient_check},
          {"equilibration","symmetric diagonal scaling of the gauged operator"},
          {"reuse","all port, direct source and loaded right-hand sides (loads by exact Woodbury)"}};
      else if(use_direct) {
        primary_row["direct_factorization"] = direct_preconditioner->Layout();
        primary_row["direct_factorization"].update(json{
            {"factorizations_per_frequency",1},
            {"equilibration","symmetric diagonal scaling; original distributed operator retained for true residual"},
            {"factor_nonzeros",direct_preconditioner->FactorNonzeros()}, {"seconds",factorization_seconds},
            {"inertia",direct_preconditioner->Inertia()}, {"compression",direct_preconditioner->Compression()},
            {"reuse","all port, direct source and low-rank loaded validation right-hand sides"},
            {"compatibility_AMS_setup_included_in_timing",false}, {"unused_AMS_setup_skipped",true}});
      }
      double port_solves_seconds = 0.0;
      auto retain_rejected = [&](const std::string& gate, const std::string& reason) {
        // Every caller makes the same decision on all ranks. Complete all
        // reductions before rank zero writes; a diagnostic I/O error must not
        // conceal the original failed acceptance gate.
        const int completed = static_cast<int>(port_solutions.size());
        frequency_rejected = true;
        primary_row["f_Hz"] = f;
        primary_row["status"] = "provisional_rejected";
        primary_row["validation_accepted"] = false;
        primary_row["rejection"] = {{"gate",gate},{"reason",reason}};
        primary_row["completed_port_columns"] = completed;
        primary_row["uncomputed_matrix_entries"] = "JSON null; no zero fill or extrapolation";
        const json zm = fw_complex_matrix_json(Zmatrix);
        primary_row["Z_matrix_re_ohm"] = zm["re"];
        primary_row["Z_matrix_im_ohm"] = zm["im"];
        primary_row["Psi_vector"] = fw_complex_vector_json(Psivec);
        primary_row["port_solutions"] = port_solutions;
        primary_row["field_basis_files"] = field_files;
        if (nprobes > 0) primary_row["field_probe_port_fields"] = probe_fields;
        if (!primary_row.contains("source_control_Psi_vectors") && completed > 0) {
          json controls = json::object();
          for (size_t j = 0; j < source_control_q.size(); ++j) {
            Eigen::VectorXcd psi = Eigen::VectorXcd::Constant(nports,cplx(unknown,unknown));
            for (int k = 0; k < completed; ++k)
              psi(k) = cplx(fw_gdot(source_control_q[j],fields_re[k]),
                             fw_gdot(source_control_q[j],fields_im[k]))/cplx(0,w);
            controls[source_control_names[j]] = fw_complex_vector_json(psi);
          }
          primary_row["source_control_Psi_vectors"] = controls;
        }
        if (energy_M && completed > 0 && !primary_row.contains("energy_total_J_per_A2")) {
          const std::vector<mfem::Vector> er(fields_re.begin(),fields_re.begin()+completed);
          const std::vector<mfem::Vector> ei(fields_im.begin(),fields_im.begin()+completed);
          const Eigen::MatrixXcd electric = fw_energy_gram(*energy_M,er,ei)*0.25;
          const std::vector<mfem::Vector> cr(curl_re.begin(),curl_re.begin()+completed);
          const std::vector<mfem::Vector> ci(curl_im.begin(),curl_im.begin()+completed);
          const Eigen::MatrixXcd magnetic = (tree_cotree || guard_on)
                                                ? Eigen::MatrixXcd(fw_energy_gram(*energy_K,cr,ci)/(4*w*w))
                                                : Eigen::MatrixXcd(fw_energy_gram(*energy_K,er,ei)/(4*w*w));
          primary_row["energy_electric_J_per_A2"] = fw_complex_matrix_json(electric);
          primary_row["energy_magnetic_J_per_A2"] = fw_complex_matrix_json(magnetic);
          primary_row["energy_total_J_per_A2"] = fw_complex_matrix_json(electric+magnetic);
          primary_row["energy_port_indices"] = json::array();
          for (int j = 0; j < completed; ++j) primary_row["energy_port_indices"].push_back(j);
        }
        primary_row["solver_setup_seconds"] = solver_setup_seconds;
        primary_row["port_solves_seconds"] = port_solves_seconds;
        primary_row["frequency_total_seconds"] = fw_allmax(frequency_timer.seconds());
        const double peak_bytes = fw_peak_rss_bytes();
        primary_row["peak_rss_max_rank_bytes"] = fw_allmax(peak_bytes);
        primary_row["peak_rss_sum_rank_bytes"] = fw_allsum(peak_bytes);
        memory["frequency_end"] = fw_memory_record();   // at the rejection
        primary_row["memory"] = memory;
        out["memory"] = malloc_scope.Record(trim_mode, thresholds_mode);
        out["status"] = "provisional_rejected";
        out["validation_accepted"] = false;
        out["sweep"] = sweep; // only previously accepted frequencies
        out["rejected_frequency"] = primary_row;
        out["gmres_iterations_total"] = total_its;
        out["seconds"] = fw_allmax(timer.seconds());
        R["fullwave"] = out;
        // the mesh check of a failed solve (worst_elements_note, called before this on the failure paths)
        if (P.mesh_info.contains("worst_elements") && R.contains("mesh"))
          R["mesh"]["info"]["worst_elements"] = P.mesh_info["worst_elements"];
        if (P.rank == 0) {
          json diagnostic = R;
          diagnostic["status"] = "provisional_rejected";
          diagnostic["diagnostic_only"] = true;
          diagnostic["primary_row"] = primary_row;
          const auto path = field_dir.empty() ? std::filesystem::path(o.workdir)/"fullwave_rejected.json"
                                               : field_dir/strcat("f_",frequency_index,"_rejected.json");
          try {
            std::filesystem::create_directories(path.parent_path());
            std::ofstream stream(path);
            stream << std::setprecision(17) << diagnostic.dump(2) << '\n';
            stream.flush();
            if (stream) log(1,strcat("Rejected provisional full-wave diagnostics retained at ",path.string()));
            else std::cerr << "[axf] failed to write rejected diagnostics; payload follows\n" << diagnostic.dump(2) << '\n';
          } catch (const std::exception& error) {
            std::cerr << "[axf] rejected-diagnostic I/O: " << error.what() << "\n" << diagnostic.dump(2) << '\n';
          }
        }
      };
      auto require_validation = [&](bool passed, const std::string& gate, const std::string& reason,
                                    const std::string& where = "/fullwave", const std::string& hint = "") {
        if (!passed) retain_rejected(gate,reason);
        AXF_REQUIRE(passed,reason,where,hint);
      };
      for (int drive = 0; drive < nports; ++drive) {
      const auto& Jt = Jports[drive];
      B = 0.0; X = 0.0;
      for (int i = 0; i < fes.GetTrueVSize(); ++i) B(fes.GetTrueVSize()+i) = w*Jt(i);
      Timer rhs_timer;
      json gauged_stats;
      mfem::Vector Rv(B.Size()), Xcurl;
      if (tree_cotree) {
        gauged_stats = gauged->Solve(B, X, Rv, &Xcurl, nullptr, nullptr);
        for (const auto& message : gauged_stats["warnings"])
          warnings.push_back(strcat("3D full wave at f = ", f, " Hz, port ", drive, ": ", message.get<std::string>()));
      }
      else gmres.Mult(B, X);
      const double rhs_seconds = fw_allmax(rhs_timer.seconds());
      port_solves_seconds += rhs_seconds;
      const int its = tree_cotree ? gauged_stats["refinement_steps"].get<int>() : gmres.GetNumIterations();
      total_its += its;
      // true residual (tree-cotree: split form, K never applied to the gradient part)
      double legacy_rel = std::numeric_limits<double>::quiet_NaN();
      if (!tree_cotree) {
        Afull.Mult(X, Rv);
        Rv -= B;
      } else {  // diagnostic only: the legacy assembled (summed) matrix applied to the recombined field
        mfem::Vector legacy(B.Size());
        Afull.Mult(X, legacy);
        legacy -= B;
        legacy_rel = std::sqrt(fw_gdot(legacy, legacy) / std::max(fw_gdot(B, B), 1e-300));
      }
      const double residual_norm = std::sqrt(fw_gdot(Rv,Rv));
      const double rhs_norm = std::sqrt(fw_gdot(B,B));
      const double rel = residual_norm/std::max(rhs_norm,1e-150);
      if (!(rel < 1e-6))
        warnings.push_back(strcat("3D full wave at f = ", f, " Hz: GMRES did not converge (", its,
                                  " iterations, relative residual ", rel, "); results are NOT reliable.",
                                  worst_elements_note(P)));
      // Gauss-law guard, after the residual gate's residual: the gradient part of the error, corrected or estimated
      json guard_stats;
      if (guard_on) {
        guard_stats = apply_guard(X, B, &Rv, nullptr, nullptr, strcat("port ", drive));
        Xcurl = guard->CurlPart(X);   // magnetic energy and probe curls without K's rounding on the gradient part
      }
      const bool curl_representative = tree_cotree || guard_on;
      mfem::Vector xr(X.GetData(), fes.GetTrueVSize()), xi(X.GetData()+fes.GetTrueVSize(), fes.GetTrueVSize());
      mfem::Vector rr(Rv.GetData(),fes.GetTrueVSize()), ri(Rv.GetData()+fes.GetTrueVSize(),fes.GetTrueVSize());
      residuals_re[drive] = rr; residuals_re[drive] /= I;
      residuals_im[drive] = ri; residuals_im[drive] /= I;
      x.real().SetFromTrueDofs(xr);
      x.imag().SetFromTrueDofs(xi);
      if (drive == 0 && f == f_scene) {
        P.fw_scene_f = f;
        P.E_re = std::make_unique<mfem::ParGridFunction>(&fes);
        P.E_im = std::make_unique<mfem::ParGridFunction>(&fes);
        *P.E_re = x.real();
        *P.E_im = x.imag();
        *P.E_re /= I;  // per readout ampere
        *P.E_im /= I;
        if (P.nranks > 1) {
          P.view_E_re = P.gather(*P.E_re);
          P.view_E_im = P.gather(*P.E_im);
        }
        out["scene_frequency_Hz"] = f;
      }
      mfem::Vector Er(fes.GetTrueVSize()), Ei(fes.GetTrueVSize());
      x.real().GetTrueDofs(Er);
      x.imag().GetTrueDofs(Ei);
      fields_re[drive] = Er; fields_re[drive] /= I;
      fields_im[drive] = Ei; fields_im[drive] /= I;
      if (curl_representative) {
        curl_re[drive].SetSize(fes.GetTrueVSize()); curl_im[drive].SetSize(fes.GetTrueVSize());
        for (int i = 0; i < fes.GetTrueVSize(); ++i) {
          curl_re[drive](i) = Xcurl(i)/I; curl_im[drive](i) = Xcurl(fes.GetTrueVSize()+i)/I;
        }
      }
      save_field(fields_re[drive], fields_im[drive], strcat("port_", drive));
      if (tree_cotree) save_field(curl_re[drive], curl_im[drive], strcat("port_", drive, "_curl_representative"));
      if (!field_dir.empty()) field_files.push_back(strcat("f_", frequency_index, "_port_", drive, "_rank_{rank}_{re,im}.gf"));
      if (!surf_pts.empty() || !fw.surface_quadrature_file.empty()) {
        std::unique_ptr<mfem::ParComplexGridFunction> scurl;
        if (curl_representative) {  // as the probes: the curl part, without K's rounding on the gradient part
          scurl = std::make_unique<mfem::ParComplexGridFunction>(&fes);
          mfem::Vector cr0(Xcurl.GetData(), fes.GetTrueVSize()), ci0(Xcurl.GetData()+fes.GetTrueVSize(), fes.GetTrueVSize());
          scurl->real().SetFromTrueDofs(cr0); scurl->imag().SetFromTrueDofs(ci0);
        }
        mfem::Vector cr(3), ci(3);
        surf_B.resize(static_cast<size_t>(nports) * surf_pts.size() * 6, 0.0);
        const size_t base = static_cast<size_t>(drive) * surf_pts.size() * 6;
        for (size_t j = 0; j < surf_pts.size(); ++j) {
          auto* T = pm.GetElementTransformation(surf_pts[j].elem);
          T->SetIntPoint(&surf_pts[j].eip);
          if (scurl) { scurl->real().GetCurl(*T,cr); scurl->imag().GetCurl(*T,ci); }
          else { x.real().GetCurl(*T,cr); x.imag().GetCurl(*T,ci); }
          for (int k = 0; k < 3; ++k) {
            surf_B[base + 6*j + k]     =  ci(k)/(w*I);     // Re B = curl E_im / omega
            surf_B[base + 6*j + 3 + k] = -cr(k)/(w*I);     // Im B = -curl E_re / omega
          }
        }
      }
      if (nprobes > 0) {
        std::unique_ptr<mfem::ParComplexGridFunction> curl_gf;
        if (curl_representative) {  // curl of the curl representative: no cancellation against the gradient part
          curl_gf = std::make_unique<mfem::ParComplexGridFunction>(&fes);
          mfem::Vector cr0(Xcurl.GetData(), fes.GetTrueVSize()), ci0(Xcurl.GetData()+fes.GetTrueVSize(), fes.GetTrueVSize());
          curl_gf->real().SetFromTrueDofs(cr0); curl_gf->imag().SetFromTrueDofs(ci0);
        }
        std::vector<double> values(static_cast<size_t>(nprobes)*12,0.0);
        mfem::Vector er(3), ei(3), cr(3), ci(3);
        for (int j = 0; j < nprobes; ++j) {
          if (probe_elements[j] < 0) continue;
          auto* T = pm.GetElementTransformation(probe_elements[j]);
          T->SetIntPoint(&probe_ips[j]);
          x.real().GetVectorValue(*T,probe_ips[j],er); x.imag().GetVectorValue(*T,probe_ips[j],ei);
          if (curl_gf) { curl_gf->real().GetCurl(*T,cr); curl_gf->imag().GetCurl(*T,ci); }
          else { x.real().GetCurl(*T,cr); x.imag().GetCurl(*T,ci); }
          for (int k = 0; k < 3; ++k) {
            values[12*j+k] = er(k)/I; values[12*j+3+k] = ei(k)/I;
            values[12*j+6+k] = ci(k)/(w*I); values[12*j+9+k] = -cr(k)/(w*I);
          }
        }
        MPI_Allreduce(MPI_IN_PLACE,values.data(),static_cast<int>(values.size()),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        json erj = json::array(), eij = json::array(), brj = json::array(), bij = json::array();
        for (int j = 0; j < nprobes; ++j) {
          erj.push_back({values[12*j],values[12*j+1],values[12*j+2]});
          eij.push_back({values[12*j+3],values[12*j+4],values[12*j+5]});
          brj.push_back({values[12*j+6],values[12*j+7],values[12*j+8]});
          bij.push_back({values[12*j+9],values[12*j+10],values[12*j+11]});
        }
        probe_fields.push_back({{"port_index",drive}, {"E_V_m_per_A",{{"re",erj},{"im",eij}}},
                                {"B_T_per_A",{{"re",brj},{"im",bij}}}});
      }
      if (verbosity >= 3 && U.Width() > 0) {  // exterior diagnostics: mode amplitudes and their impedance share
        const int nm = U.Width();
        std::vector<double> t(2 * nm, 0.0);
        mfem::Vector tr(t.data(), nm), ti(t.data() + nm, nm);
        if (U.Height() > 0) {
          U.MultTranspose(Er, tr);
          U.MultTranspose(Ei, ti);
        }
        MPI_Allreduce(MPI_IN_PLACE, t.data(), 2 * nm, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cplx dZ = 0;
        for (int j = 0; j < nm; ++j) {
          const cplx e(t[j], t[nm + j]);
          dZ += cplx(0, -1.0 / (w * I * I)) * cmode[j] * e * e;
          if (std::abs(e) > 1e-3 * std::sqrt(fw_gdot(Er, Er) + fw_gdot(Ei, Ei)) || j < 6)
            log(3, strcat("    mode ", j, " n=", mode_n[j], mode_te[j] ? " TE" : " TM", " e=", e.real(), "+i", e.imag(),
                          " c=", cmode[j].real(), "+i", cmode[j].imag()));
        }
        log(3, strcat("    exterior correction share of Z: ", dZ.real(), " + i ", dZ.imag(), " ohm"));
      }
      const cplx EJ(fw_gdot(Jt, Er), fw_gdot(Jt, Ei)), EB(fw_gdot(Bt, Er), fw_gdot(Bt, Ei));
      const cplx Z = -EJ / (I * I);
      for (int measured = 0; measured < nports; ++measured)
        Zmatrix(measured,drive) = -cplx(fw_gdot(Jports[measured],Er),fw_gdot(Jports[measured],Ei))/(I*I);
      cplx Psi = EB / (cplx(0.0, w) * I);
      cplx Psi_ext = 0;
      double P_poynting = std::numeric_limits<double>::quiet_NaN();
      if (!ext_phi.empty()) {  // oint Phi0 E.n dA over the faceted R1 interface (n out of the physical domain)
        const int NY = static_cast<int>(ext_phi.size());
        int Ns = 0;
        while ((Ns + 2) * (Ns + 2) - 1 <= NY) ++Ns;
        std::vector<double> er(2 * NY + 1, 0.0);  // last entry: Poynting flux
        mfem::Vector vr(3), vi(3), cr(3), ci(3);
        for (auto& pt : ext_pts) {
          mfem::ElementTransformation* Te = pm.GetElementTransformation(pt.e);
          Te->SetIntPoint(&pt.eip);
          x.real().GetVectorValue(*Te, pt.eip, vr);
          x.imag().GetVectorValue(*Te, pt.eip, vi);
          const double Err = vr(0) * pt.n[0] + vr(1) * pt.n[1] + vr(2) * pt.n[2];
          const double Eri = vi(0) * pt.n[0] + vi(1) * pt.n[1] + vi(2) * pt.n[2];
          const double down = sh.R1 / pt.rho;  // Phi0 at the facet point: order n decays as (R1/rho)^(n+1)
          double scale = down;
          for (int n = 1, j = 0; n <= Ns; ++n) {
            scale *= down;
            for (int m = -n; m <= n; ++m, ++j) {
              er[j] += pt.dS * scale * Err * pt.Y[j];
              er[NY + j] += pt.dS * scale * Eri * pt.Y[j];
            }
          }
          // independent radiated-power check from the physical-side FE field on R1 (vacuum outside, so this is
          // the power leaving through the exterior): 1/2 Re(E x B*) . rhat / mu0 with B = curl E / (i omega),
          // i.e. B_re = curl E_im / omega, B_im = -curl E_re / omega
          x.real().GetCurl(*Te, cr);
          x.imag().GetCurl(*Te, ci);
          double s_r = 0;
          for (int i = 0; i < 3; ++i) {
            const int i1 = (i + 1) % 3, i2 = (i + 2) % 3;
            const double br1 = ci(i1) / w, br2 = ci(i2) / w, bi1 = -cr(i1) / w, bi2 = -cr(i2) / w;
            s_r += pt.n[i] * (vr(i1) * br2 - vr(i2) * br1 + vi(i1) * bi2 - vi(i2) * bi1);
          }
          er[2 * NY] += pt.dS * s_r / (2 * mu0);
        }
        MPI_Allreduce(MPI_IN_PLACE, er.data(), 2 * NY + 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        P_poynting = er[2 * NY];
        cplx s_ext = 0;
        for (int j = 0; j < NY; ++j) s_ext += ext_phi[j] * cplx(er[j], er[NY + j]);
        Psi_ext = s_ext / (cplx(0.0, w) * I);
        Psi += Psi_ext;
      }
      // an imposed background: its volume part is in EB (Bbg is part of Bt); its exterior part comes from the
      // TM m = 0 trace projections t_n = (E, W_TM,n0) (axi::BackgroundExterior)
      cplx Psi_bg_volume = 0;
      cplx Psi_bg_ext = 0;
      double bg_last_order = 0;
      if (P.bg) Psi_bg_volume = cplx(fw_gdot(Bbg, Er), fw_gdot(Bbg, Ei)) / (cplx(0.0, w) * I);
      if (bg_ext.N > 0 && !bg_ext.rho.empty()) {
        const int Nb = bg_ext.N;
        std::vector<double> t(2 * Nb, 0.0);
        for (int c = 0; c < Nb; ++c) {
          const double* col = U.GetColumn(tm0_col[c]);
          for (int i = 0; i < U.Height(); ++i) {
            t[c] += col[i] * Er(i);
            t[Nb + c] += col[i] * Ei(i);
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, t.data(), 2 * Nb, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        const double k = w / units::c;
        const auto gam = axi::hankel_log_derivative(k * sh.R1, Nb);
        const auto G = bg_ext.G(k);
        cplx term = 0;
        cplx previous = 0;
        for (int n = 1; n <= Nb; ++n) {
          const cplx tn(t[n - 1], t[Nb + n - 1]);
          previous = term;
          term = k * k * tn * (G[n] + bg_sliver[n]) / (gam[n] * std::sqrt(n * (n + 1.0)));
          Psi_bg_ext += term;
        }
        // the last two orders: a magnet symmetric about the domain centre has no even orders
        bg_last_order = (std::abs(term) + std::abs(previous)) / std::max(std::abs(Psi_bg_ext), 1e-300);
        Psi_bg_ext /= cplx(0.0, w) * I;
        Psi += Psi_bg_ext;
      }
      Psivec(drive) = Psi;
      // dissipation in lossy media: 1/2 omega int Im(eps_c) |E|^2 (physical region)
      double Ploss = 0;
      if (lossy) {
        const auto& ir = mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * P.order + 1);
        mfem::Vector vr(3), vi(3);
        for (int e = 0; e < ne; ++e) {
          if (eps_e[e].imag() <= 0 || is_pec(P.elem_mat[e]) || sh.in(pm.GetAttribute(e)) ||
              (fwpml.active() && fwpml.epml[e] >= 0)) continue;
          mfem::ElementTransformation* T = pm.GetElementTransformation(e);
          for (int q = 0; q < ir.GetNPoints(); ++q) {
            T->SetIntPoint(&ir.IntPoint(q));
            x.real().GetVectorValue(*T, ir.IntPoint(q), vr);
            x.imag().GetVectorValue(*T, ir.IntPoint(q), vi);
            Ploss += 0.5 * w * eps_e[e].imag() * (vr * vr + vi * vi) * ir.IntPoint(q).weight * T->Weight();
          }
        }
        Ploss = fw_allsum(Ploss);
      }
      json robin_power = json::object();
      double Probin = 0;
      for (size_t k = 0; k < robin.size(); ++k) {
        mfem::Vector t1(Er.Size()), t2(Ei.Size());
        robin_im[k]->Mult(Er, t1);
        robin_im[k]->Mult(Ei, t2);
        const double pg = -(fw_gdot(Er, t1) + fw_gdot(Ei, t2)) / (2 * w);
        robin_power[robin[k].name] = robin_power.value(robin[k].name, 0.0) + pg / (I * I);
        Probin += pg;
      }
      const double L = -Z.imag() / w, Rr = Z.real();
      json row = {{"f_Hz", f},
                  {"k_R", w / units::c * Rdev},
                  {"Psi_re", Psi.real()},
                  {"Psi_im", Psi.imag()},
                  {"Psi_abs", std::abs(Psi)},
                  {"Z_re_ohm", Z.real()},
                  {"Z_im_ohm", Z.imag()},
                  {"L_H", L},
                  {"solver", {{"gmres_iterations", its}, {"relative_residual", rel}, {"solve_seconds", rhs_seconds},
                      {"true_residual_norm",residual_norm}, {"rhs_norm",rhs_norm},
                      {"gmres_reported_converged",tree_cotree ? json(nullptr) : json(gmres.GetConverged())},
                      {"preconditioned_initial_norm",tree_cotree ? json(nullptr) : json(gmres.GetInitialNorm())},
                      {"preconditioned_final_norm",tree_cotree ? json(nullptr) : json(gmres.GetFinalNorm())},
                      {"accepted",std::isfinite(rel) && rel < 1e-6}}}};
      if (tree_cotree) {
        row["solver"]["tree_cotree"] = gauged_stats;
        row["solver"]["tree_cotree"]["legacy_summed_matrix_relative_residual_diagnostic"] = legacy_rel;
      }
      if (!guard_stats.is_null()) row["solver"]["gauss_law_guard"] = guard_stats;
      const double source_abs_sum = fw_absolute_overlap(Bt,fields_re[drive],fields_im[drive]);
      row["source_overlap_cancellation"] = {{"absolute_true_dof_contribution_sum_V",source_abs_sum},
          {"net_overlap_abs_V",std::abs(EB/I)}, {"cancellation_factor",source_abs_sum/std::max(std::abs(EB/I),1e-300)},
          {"scope","volume (FE-domain) overlap only, excluding Psi_exterior and Psi_background.exterior; "
                    "basis-dependent absolute DOF sum; diagnostic only, not an error estimate"}};
      if (P.bg) {  // the background's parts of Psi (both included in Psi)
        // exterior: null where no exterior term is computed (uniform, closed, PML, no modes, a grid inside R1)
        const bool bg_term = bg_ext.N > 0 && !bg_ext.rho.empty();
        row["Psi_background"] = {{"volume", {{"re", Psi_bg_volume.real()}, {"im", Psi_bg_volume.imag()}}},
                                 {"exterior", bg_term ? json{{"re", Psi_bg_ext.real()}, {"im", Psi_bg_ext.imag()}}
                                                      : json(nullptr)}};
        if (bg_term) row["Psi_background"]["exterior_last_order_fraction"] = bg_last_order;
      }
      if (!ext_phi.empty()) {
        row["Psi_exterior"] = {{"re", Psi_ext.real()}, {"im", Psi_ext.imag()}};
        // power through R1 from the FE field itself (not from Re Z): a check of the port power balance
        row["P_poynting_R1_W"] = P_poynting;
        row["R_rad_poynting_ohm"] = 2 * P_poynting / (I * I);
      }
      if (!prescribed && drive == 0 && Psi_mqs != 0.0) row["Psi_over_MQS"] = std::abs(Psi) / std::fabs(Psi_mqs);
      if (drive == 0 && L_mqs > 0) row["L_over_MQS"] = L / L_mqs;
      const double Prad = 0.5 * Rr * I * I - Ploss;
      if (open_bc) {
        // power balance: input = bulk + Robin boundaries (+ PML absorption, the remainder)
        row["boundary_power_W_per_A2"] = robin_power;
        row["R_boundary_ohm"] = 2 * Probin / (I * I);
        if (fwpml.active()) row["pml_absorbed_W_per_A2"] = (0.5 * Rr * I * I - Ploss - Probin) / (I * I);
        else
          row["power_balance_rel"] = std::fabs(0.5 * Rr * I * I - Ploss - Probin) /
                                     std::max(std::fabs(0.5 * Rr * I * I), 1e-300);
      }
      if (radiation) {
        row["R_rad_ohm"] = 2 * Prad / (I * I);
        if (drive == 0) row["R_static_dipole_ohm"] = mu0*units::c*std::pow(w/units::c,4)*moment2/(6*pi);
      }
      if (lossy) {
        row["R_bulk_ohm"] = 2 * Ploss / (I * I);
        if (L > 0 && Ploss > 0) row["Q_bulk"] = w * L * I * I / (2 * Ploss);
      }
      // the residual can move Re Z by up to about rel |X|: in a dissipating model a Re Z below kReZResolution rel |X|
      // is not resolved by the solve, nor are R_rad_ohm and Q_rad, which come from it (R_bulk_ohm and R_boundary_ohm
      // are integrals of the field; surface_losses3d.hpp)
      if ((radiation || lossy || open_bc) && std::isfinite(rel) &&
          std::max(Rr, 0.0) < kReZResolution * rel * std::fabs(Z.imag()))
        row["Re_Z_unresolved"] = {
            {"Re_Z_over_abs_X", Rr / std::max(std::fabs(Z.imag()), 1e-300)},
            {"true_relative_residual", rel},
            {"affects", {"Z_re_ohm", "R_rad_ohm", "Q_rad"}},
            {"reason", strcat("Re Z is below ", kReZResolution, " times the solve's true relative residual times |X|, "
                              "the amount by which the residual can move it")}};
      if (L > 0) {
        row["F_SI_T4_H2_m8"] = std::pow(std::abs(Psi), 4) / (L * L);
        if (radiation && Prad > 0) row["Q_rad"] = w * L * I * I / (2 * Prad);
      } else {
        if (drive == 0) capacitive = true;
      }
      if (indicator) {
        const json estimate = indicator->Evaluate(fields_re[drive], fields_im[drive],
                                                  curl_representative ? &curl_re[drive] : nullptr,
                                                  curl_representative ? &curl_im[drive] : nullptr, w, eps_e, nu_e, Z,
                                                  row["solver"]["accepted"].get<bool>());
        row["error_indicator"] = estimate;
        if (!field_dir.empty())
          indicator->Save((field_dir / strcat("f_", frequency_index, "_port_", drive, "_error_indicator_rank_", P.rank,
                                              ".gf")).string());
        // JSON null (a withheld or non-finite value) reads as NaN in the log
        auto number = [&](const char* key) {
          const json& v = estimate.value(key, json());
          return v.is_number() ? v.get<double>() : std::numeric_limits<double>::quiet_NaN();
        };
        std::string where;
        if (estimate.contains("largest_element"))
          where = strcat("; largest element in '", estimate["largest_element"]["region"].get<std::string>(), "' (",
                         estimate["largest_element"]["share"].dump(), " of eta^2)");
        if (estimate.contains("withheld"))
          where = strcat(" (withheld: ", estimate["withheld"].get<std::string>(), ")");
        log(1, strcat("3D full wave port ", drive, " f = ", f, " Hz: error indicator ", number("estimated_relative_error"),
                      " (magnetic ", number("magnetic_relative_error"), ", electric ",
                      number("electric_relative_error"), ")", where));
      }
      if (drive == 0) primary_row.update(row);
      row["port_index"] = drive;
      port_solutions.push_back(row);
      log(1, strcat("3D full wave port ", drive, " f = ", f, " Hz: |Psi|/Psi_MQS = ", row.value("Psi_over_MQS", 0.0), ", L/L_MQS = ",
                    row.value("L_over_MQS", 0.0), ", R = ", Rr, " ohm (", its, " GMRES iterations; true residual ",rel,
                    "; solve ",rhs_seconds," s)"));
      if (nports > 1 || prescribed || fw.source_check || !fw.validation_shunt_R_ohm.empty() || !fw.validation_loads.is_null()) {
        const bool converged = std::isfinite(rel) && rel < 1e-6;
        require_validation(converged,strcat("port_",drive,"_true_residual"),
                    "multiport full-wave right-hand side did not converge; refusing an invalid Z matrix",
                    "/fullwave", strcat("Refine the solver/preconditioner before accepting these frequencies.",
                                        converged ? std::string() : worst_elements_note(P, kRejectedDiagnostics)));
      }
      }
      auto complex_vector_json = fw_complex_vector_json;
      json zr=json::array(), zi=json::array();
      for (int j=0;j<nports;++j) {
        json rr=json::array(), ii=json::array();
        for (int k=0;k<nports;++k) { rr.push_back(Zmatrix(j,k).real()); ii.push_back(Zmatrix(j,k).imag()); }
        zr.push_back(rr); zi.push_back(ii);
      }
      primary_row["Z_matrix_re_ohm"] = zr;
      primary_row["Z_matrix_im_ohm"] = zi;
      primary_row["Psi_vector"] = complex_vector_json(Psivec);
      if (!source_control_q.empty()) {
        json controls = json::object();
        for (size_t j = 0; j < source_control_q.size(); ++j) {
          Eigen::VectorXcd psi(nports);
          for (int k = 0; k < nports; ++k)
            psi(k) = cplx(fw_gdot(source_control_q[j],fields_re[k]),fw_gdot(source_control_q[j],fields_im[k]))/cplx(0,w);
          controls[source_control_names[j]] = complex_vector_json(psi);
        }
        primary_row["source_control_Psi_vectors"] = controls;
      }
      primary_row["port_solutions"] = port_solutions;
      if (!field_dir.empty()) primary_row["field_basis_files"] = field_files;
      if (nprobes > 0) primary_row["field_probe_port_fields"] = probe_fields;
      Timer energy_postprocess_timer;
      Eigen::MatrixXcd energy_G;
      if (energy_M) {
        const Eigen::MatrixXcd electric = fw_energy_gram(*energy_M,fields_re,fields_im)*0.25;
        const Eigen::MatrixXcd magnetic = (tree_cotree || guard_on)
                                              ? Eigen::MatrixXcd(fw_energy_gram(*energy_K,curl_re,curl_im)/(4*w*w))
                                              : Eigen::MatrixXcd(fw_energy_gram(*energy_K,fields_re,fields_im)/(4*w*w));
        if (tree_cotree) primary_row["energy_magnetic_scope"] = "E_c^dagger K E_c/(4 w^2), E_c = cotree curl representative (K G = 0 exact)";
        if (guard_on)
          primary_row["energy_magnetic_scope"] = "E_perp^dagger K E_perp/(4 w^2), E_perp = E - G_c S^-1 G_c^T M E "
                                                 "(Gauss-law guard; K G = 0 exact)";
        energy_G = electric + magnetic;
        primary_row["energy_electric_J_per_A2"] = fw_complex_matrix_json(electric);
        primary_row["energy_magnetic_J_per_A2"] = fw_complex_matrix_json(magnetic);
        primary_row["energy_total_J_per_A2"] = fw_complex_matrix_json(energy_G);
        primary_row["energy_hermiticity_relative_error"] =
            (energy_G-energy_G.adjoint()).norm()/std::max(energy_G.norm(),1e-300);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(energy_G);
        require_validation(es.info() == Eigen::Success,"energy_finiteness", "non-finite energy Gram matrix");
        primary_row["energy_min_eigenvalue_J_per_A2"] = es.eigenvalues()(0);
        if (!lossy) primary_row["d_ImZ_domega_from_energy_H"] = fw_complex_matrix_json(-4.0*energy_G);
      }
      double dz_seconds = 0.0;
      {
        // Exact dZ/dw from the stored port fields, by complex-symmetric reciprocity (no extra solve):
        //   Z_jk = -J_j^T x_k / I^2 with A(w) x_k = i w J_k and A^T = A   =>   dZ_jk/dw = Z_kj / w - (i / w) e_j^T A'(w) e_k,
        // e = x / I. A' = M[-2 w eps0 eps_r (1 + i tan delta) - i sigma]
        // on the physical elements (nu, the shell's static infinity map and its gauge mass do not depend on w)
        // + U diag(dc/dw) U^T for the radiating exterior (axi::fw_exterior_coefficient_dw). Valid with losses and
        // with the radiating exterior; exact up to the solve residual. Unlike the energy identity it gives the full
        // complex matrix (Re dZ/dw included).
        // Robin boundary groups add d beta/dw on their faces (RobinCoef, derivative), and PML layers d/dw of their
        // stretched tensors, nu dLambda^{-1}/dw and deps Lambda - w^2 eps_c dLambda/dw (FwPmlTensorDw); leaving either
        // out makes dZ/dw wrong for designs with those boundaries.
        Timer dz_timer;
        int missing = 0;   // decided on every rank before any collective call
        for (int k = 0; k < nports; ++k)
          if (fields_re[k].Size() != fes.GetTrueVSize() || fields_im[k].Size() != fes.GetTrueVSize()) missing = 1;
        MPI_Allreduce(MPI_IN_PLACE, &missing, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (!missing) {
          std::vector<cplx> deps(ne);
          int volume_loss = 0;
          for (int e = 0; e < ne; ++e) {
            const spec::Material* m = P.elem_mat[e];
            const double er = m->eps_r > 0 ? m->eps_r : 1.0;
            deps[e] = -2.0 * w * eps0 * er * cplx(1.0, m->tan_delta) - cplx(0.0, m->sigma_S_m);
            if (deps[e].imag() != 0.0) volume_loss = 1;
          }
          MPI_Allreduce(MPI_IN_PLACE, &volume_loss, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
          // in PML layers the whole mass derivative is FwPmlTensorDw's: the isotropic term is zeroed there
          std::vector<cplx> deps_outside_pml = deps;
          if (fwpml.active())
            for (int e = 0; e < ne; ++e)
              if (fwpml.epml[e] >= 0) deps_outside_pml[e] = 0.0;
          const bool frequency_dependent_terms = fwpml.active() || !robin.empty();
          FwTensor dmr(sh, deps_outside_pml, 1, 0, 1.0, 0.0), dmi(sh, deps_outside_pml, 1, 1, 1.0, 0.0);
          FwPmlTensorDw pml_mr(fwpml, nu_e, eps_e, deps, 1, 0), pml_mi(fwpml, nu_e, eps_e, deps, 1, 1);
          FwPmlTensorDw pml_kr(fwpml, nu_e, eps_e, deps, 0, 0), pml_ki(fwpml, nu_e, eps_e, deps, 0, 1);
          std::vector<std::unique_ptr<RobinCoef>> robin_dw;
          for (const auto& b : robin)
            for (int part = 0; part < 2; ++part)
              robin_dw.push_back(std::make_unique<RobinCoef>(b, bdr_elem, P.elem_mat, w, part, true));
          // A' (real or imaginary part): the volume mass derivative, plus the PML layers and the Robin groups
          auto derivative_form = [&](int part) {
            mfem::ParBilinearForm form(&fes);
            form.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(part == 0 ? dmr : dmi));
            if (fwpml.active()) {
              form.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(part == 0 ? pml_mr : pml_mi));
              form.AddDomainIntegrator(new mfem::CurlCurlIntegrator(part == 0 ? pml_kr : pml_ki));
            }
            for (size_t k = 0; k < robin.size(); ++k)
              form.AddBoundaryIntegrator(new mfem::VectorFEMassIntegrator(*robin_dw[2 * k + part]), robin_marker[k]);
            form.Assemble();
            form.Finalize();
            return std::unique_ptr<mfem::HypreParMatrix>(form.ParallelAssemble());
          };
          std::unique_ptr<mfem::HypreParMatrix> Dre, Dim;
          Dre = derivative_form(0);
          // Im A' (dielectric and conduction loss, the PML's complex stretch, the Robin groups); otherwise zero
          if (volume_loss || frequency_dependent_terms) Dim = derivative_form(1);
          std::vector<cplx> dcmode(mode_n.size());
          if (!mode_n.empty()) {
            const double xr = w / units::c * sh.R1;
            const int N = *std::max_element(mode_n.begin(), mode_n.end());
            const auto gam = axi::hankel_log_derivative(xr, N);
            const auto dgam = axi::hankel_log_derivative_dx(xr, N);
            for (size_t j = 0; j < mode_n.size(); ++j)
              dcmode[j] = axi::fw_exterior_coefficient_dw(mode_te[j], w, sh.R1, gam[mode_n[j]], dgam[mode_n[j]]);
          }
          const int nm = U.Width();
          std::vector<std::vector<cplx>> tmodes(nports);
          std::vector<mfem::Vector> dyr(nports), dyi(nports);
          for (int k = 0; k < nports; ++k) {
            dyr[k].SetSize(fes.GetTrueVSize());
            dyi[k].SetSize(fes.GetTrueVSize());
            Dre->Mult(fields_re[k], dyr[k]);
            Dre->Mult(fields_im[k], dyi[k]);
            if (Dim) {
              mfem::Vector a(fes.GetTrueVSize()), b(fes.GetTrueVSize());
              Dim->Mult(fields_im[k], a);
              dyr[k] -= a;                                 // Re(A' e) = Dre er - Dim ei
              Dim->Mult(fields_re[k], b);
              dyi[k] += b;                                 // Im(A' e) = Dre ei + Dim er
            }
            if (nm > 0) {
              std::vector<double> t(2 * nm, 0.0);
              mfem::Vector tr(t.data(), nm), ti(t.data() + nm, nm);
              if (U.Height() > 0) {
                U.MultTranspose(fields_re[k], tr);
                U.MultTranspose(fields_im[k], ti);
              }
              MPI_Allreduce(MPI_IN_PLACE, t.data(), 2 * nm, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
              tmodes[k].resize(nm);
              for (int m = 0; m < nm; ++m) tmodes[k][m] = cplx(t[m], t[nm + m]);
            }
          }
          Eigen::MatrixXcd dZ(nports, nports);
          Eigen::MatrixXcd dZte = Eigen::MatrixXcd::Zero(nports, nports);
          Eigen::MatrixXcd dZtm = Eigen::MatrixXcd::Zero(nports, nports);
          const cplx mi_w(0.0, -1.0 / w);
          for (int j = 0; j < nports; ++j)
            for (int k = 0; k < nports; ++k) {
              const cplx q = fw_bilinear(fields_re[j], fields_im[j], dyr[k], dyi[k]);
              cplx qte = 0.0, qtm = 0.0;
              for (int m = 0; m < nm; ++m) {
                const cplx t = dcmode[m] * tmodes[j][m] * tmodes[k][m];
                if (mode_te[m]) qte += t;
                else qtm += t;
              }
              dZ(j, k) = Zmatrix(k, j) / w + mi_w * (q + qte + qtm);
              dZte(j, k) = mi_w * qte;
              dZtm(j, k) = mi_w * qtm;
            }
          dz_seconds = dz_timer.seconds();
          primary_row["dZ_domega_H"] = fw_complex_matrix_json(dZ);
          primary_row["dZ_domega_scope"] =
              "exact per frequency from the port fields (reciprocity): dZ/dw = Z^T/w - (i/w) e^T A'(w) e; volume mass "
              "and loss terms plus the radiating exterior; X = -Im Z, so dX/dw = -Im dZ/dw; exact up to the solve residual";
          // the scope names only the w-dependent terms this design has (the Robin group types present, the PML)
          std::string extra_terms;
          if (!robin.empty()) {
            std::vector<std::string> types;
            for (const auto& b : robin)
              if (std::find(types.begin(), types.end(), b.type) == types.end()) types.push_back(b.type);
            std::string listed;
            for (size_t k = 0; k < types.size(); ++k) listed += (k ? ", " : "") + types[k];
            extra_terms = "d beta/dw of the Robin boundary groups (" + listed + ")";
          }
          if (fwpml.active()) {
            if (!extra_terms.empty()) extra_terms += " and ";
            extra_terms += "d/dw of the PML layers' stretched tensors";
          }
          if (frequency_dependent_terms)
            primary_row["dZ_domega_scope"] = primary_row["dZ_domega_scope"].get<std::string>() + "; A' also holds " +
                                             extra_terms;
          if (nm > 0) {   // the exterior's share of dZ/dw, per mode family (included in dZ_domega_H)
            primary_row["dZ_domega_exterior_TE_H"] = fw_complex_matrix_json(dZte);
            primary_row["dZ_domega_exterior_TM_H"] = fw_complex_matrix_json(dZtm);
          }
          primary_row["dZ_domega_seconds"] = fw_allmax(dz_seconds);
          if (!lossy && energy_M) {  // closed lossless: the energy identity gives d(Im Z)/dw = Re(-4 G)
            double num = 0, den = 0;
            for (int j = 0; j < nports; ++j)
              for (int k = 0; k < nports; ++k) {
                num = std::max(num, std::abs(dZ(j, k).imag() - (-4.0 * energy_G(j, k)).real()));
                den = std::max(den, std::abs((-4.0 * energy_G(j, k)).real()));
              }
            primary_row["dZ_domega_vs_energy_identity_relative"] = num / std::max(den, 1e-300);
          }
        }
      }
      if (surface_losses && surface_losses->active()) {
        // the port fields' curl representatives where the solve has them (curl E is the same, without K's rounding
        // on the gradient part)
        std::vector<const mfem::Vector*> loss_re(nports), loss_im(nports);
        for (int k = 0; k < nports; ++k) {
          const bool curl = (tree_cotree || guard_on) && curl_re[k].Size() == fes.GetTrueVSize();
          loss_re[k] = curl ? &curl_re[k] : &fields_re[k];
          loss_im[k] = curl ? &curl_im[k] : &fields_im[k];
        }
        surface_losses->Evaluate(f, loss_re, loss_im, Zmatrix, primary_row);
      }
      if (!fw.field_currents.is_null()) {
        Eigen::VectorXcd currents(nports);
        mfem::Vector cr(fes.GetTrueVSize()), ci(fes.GetTrueVSize()); cr = 0.0; ci = 0.0;
        for (int j = 0; j < nports; ++j) {
          const double ir = fw.field_currents.at("re").at(j).get<double>();
          const double ii = fw.field_currents.at("im").at(j).get<double>();
          currents(j) = cplx(ir,ii);
          cr.Add(ir,fields_re[j]); cr.Add(-ii,fields_im[j]);
          ci.Add(ii,fields_re[j]); ci.Add(ir,fields_im[j]);
        }
        save_field(cr,ci,"combined");
        primary_row["combined_field_currents_A"] = complex_vector_json(currents);
        primary_row["combined_field_scope"] = "linear combination of port-drive fields; no independent impressed-source field included";
        if (energy_M) primary_row["combined_field_energy_J"] = std::real((currents.adjoint()*energy_G*currents)(0,0));
        if (f == f_scene) {
          P.E_re->SetFromTrueDofs(cr); P.E_im->SetFromTrueDofs(ci);
          if (P.nranks > 1) { P.view_E_re = P.gather(*P.E_re); P.view_E_im = P.gather(*P.E_im); }
          out["scene_field"] = "combined physical-current field specified by fullwave.field_currents";
        }
      }
      primary_row["energy_postprocess_seconds"] = fw_allmax(energy_postprocess_timer.seconds() - dz_seconds);
      primary_row["reciprocity_relative_error"] = (Zmatrix-Zmatrix.transpose()).norm()/std::max(Zmatrix.norm(),1e-300);
      const Eigen::MatrixXcd resistance = (Zmatrix+Zmatrix.adjoint())*0.5;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> resistance_spectrum(resistance);
      require_validation(resistance_spectrum.info() == Eigen::Success,"resistance_finiteness",
                  "non-finite multiport resistance matrix");
      primary_row["resistance_min_eigenvalue_ohm"] = resistance_spectrum.eigenvalues()(0);
      primary_row["resistance_max_eigenvalue_ohm"] = resistance_spectrum.eigenvalues()(nports-1);
      primary_row["operator_assemblies"] = 1;
      primary_row["preconditioner_setups"] = 1;
      primary_row["port_rhs_solves"] = nports;
      primary_row["solver_setup_seconds"] = solver_setup_seconds;
      primary_row["port_solves_seconds"] = port_solves_seconds;
      auto validation_solve = [&](mfem::Operator& op, const mfem::Vector& rhs, const std::string& label,
                                  mfem::Vector* saved_field = nullptr, mfem::Vector* saved_residual = nullptr,
                                  const mfem::DenseMatrix* low_U = nullptr, const std::vector<cplx>* low_c = nullptr) {
        mfem::Vector sol(rhs.Size()); sol=0.0;
        Timer validation_timer;
        mfem::Vector residual(rhs.Size());
        json gauged_stats;
        if (tree_cotree) {
          gauged_stats = gauged->Solve(rhs, sol, residual, nullptr, low_U, low_c);
          for (const auto& message : gauged_stats["warnings"])
            warnings.push_back(strcat("3D full wave at f = ", f, " Hz, ", label, ": ", message.get<std::string>()));
          mfem::Vector legacy(rhs.Size());
          op.Mult(sol, legacy);
          legacy -= rhs;
          gauged_stats["legacy_summed_matrix_relative_residual_diagnostic"] =
              std::sqrt(fw_gdot(legacy, legacy) / std::max(fw_gdot(rhs, rhs), 1e-300));
        }
        else { gmres.SetOperator(op); gmres.Mult(rhs,sol); op.Mult(sol,residual); residual-=rhs; }
        const int validation_its = tree_cotree ? gauged_stats["refinement_steps"].get<int>() : gmres.GetNumIterations();
        const double residual_norm=std::sqrt(fw_gdot(residual,residual));
        const double rhs_norm=std::sqrt(fw_gdot(rhs,rhs));
        const double rel=residual_norm/std::max(rhs_norm,1e-150);
        // Gauss-law guard, after the gate's residual (the loads by Woodbury); residual updated with the field
        json guard_stats;
        if (guard_on) guard_stats = apply_guard(sol, rhs, &residual, low_U, low_c, label);
        const double seconds=fw_allmax(validation_timer.seconds());
        mfem::Vector re(sol.GetData(),fes.GetTrueVSize()), im(sol.GetData()+fes.GetTrueVSize(),fes.GetTrueVSize());
        Eigen::VectorXcd voltage(nports);
        json abs_sum=json::array(), cancellation=json::array();
        for (int j=0;j<nports;++j) {
          voltage(j)=-cplx(fw_gdot(Jports[j],re),fw_gdot(Jports[j],im))/I;
          const double scale=fw_absolute_overlap(Jports[j],re,im)/std::abs(I);
          abs_sum.push_back(scale);
          cancellation.push_back(scale/std::max(std::abs(voltage(j)),1e-300));
        }
        primary_row[label+"_voltage_cancellation"]={{"absolute_true_dof_contribution_sum_V",abs_sum},
            {"cancellation_factor",cancellation}, {"scope","basis-dependent absolute DOF sum; diagnostic only"}};
        primary_row[label+"_solver"]={{"relative_residual",rel},{"gmres_iterations",validation_its},
            {"true_residual_norm",residual_norm}, {"rhs_norm",rhs_norm}, {"solve_seconds",seconds},
            {"gmres_reported_converged",tree_cotree ? json(nullptr) : json(gmres.GetConverged())},
            {"preconditioned_initial_norm",tree_cotree ? json(nullptr) : json(gmres.GetInitialNorm())},
            {"preconditioned_final_norm",tree_cotree ? json(nullptr) : json(gmres.GetFinalNorm())},
            {"accepted",std::isfinite(rel) && rel<1e-6}};
        if (tree_cotree) primary_row[label+"_solver"]["tree_cotree"] = gauged_stats;
        if (!guard_stats.is_null()) primary_row[label+"_solver"]["gauss_law_guard"] = guard_stats;
        primary_row[label+"_voltage"]=complex_vector_json(voltage);
        if (saved_field) *saved_field = sol;
        if (saved_residual) *saved_residual = residual;
        save_field(re,im,label);
        if (!field_dir.empty()) primary_row[label+"_field_files"]=
            strcat("f_",frequency_index,"_",label,"_rank_{rank}_{re,im}.gf");
        total_its += validation_its;
        log(1,strcat("3D full wave ",label," f = ",f," Hz: ",validation_its,
                    tree_cotree ? " refinement steps (tree-cotree)" : " GMRES iterations",
                    ", true residual ",rel,", solve ",seconds," s; accepted = ",
                    std::isfinite(rel) && rel<1e-6));
        return voltage;
      };
      auto compare_voltages = [&](const Eigen::VectorXcd& direct, const Eigen::VectorXcd& predicted,
                                  const std::string& label) {
        // The vector norm can hide a weak readout. Every port is checked, with
        // an explicit absolute floor for symmetry-forbidden/null transfers.
        const double rtol = 1e-5;
        const double atol = 1e-12*std::max(predicted.cwiseAbs().maxCoeff(),1e-300);
        json absolute = json::array(), relative = json::array(), tolerance_ratio = json::array();
        double worst = 0.0;
        for (int j = 0; j < nports; ++j) {
          const double delta = std::abs(direct(j)-predicted(j));
          const double allowance = rtol*std::abs(predicted(j)) + atol;
          absolute.push_back(delta);
          relative.push_back(delta/std::max(std::abs(predicted(j)),atol));
          tolerance_ratio.push_back(delta/std::max(allowance,1e-300));
          worst = std::max(worst,delta/std::max(allowance,1e-300));
        }
        const bool passed = direct.allFinite() && predicted.allFinite() && std::isfinite(worst) && worst <= 1.0;
        primary_row[label+"_componentwise"] = {{"absolute_error_V",absolute}, {"relative_error",relative},
            {"tolerance_ratio",tolerance_ratio}, {"relative_tolerance",rtol},
            {"absolute_tolerance_V",atol}, {"passed",passed}};
        return passed;
      };
      if (fw.source_check) {
        mfem::Vector rhs(2*fes.GetTrueVSize()); rhs=0.0;
        for (int i=0;i<fes.GetTrueVSize();++i) rhs(fes.GetTrueVSize()+i)=w*Bt(i);
        mfem::Vector source_field, source_residual;
        const auto voltage=validation_solve(Afull,rhs,"direct_source",&source_field,&source_residual);
        // J_a=B0 is a unit source coefficient. Lorentz reciprocity gives V_a=-i omega Psi.
        const Eigen::VectorXcd predicted=cplx(0,-w)*Psivec;
        primary_row["direct_source_voltage"] = complex_vector_json(voltage);
        primary_row["reciprocal_source_voltage"] = complex_vector_json(predicted);
        primary_row["source_reciprocity_relative_error"]=(voltage-predicted).norm()/std::max(predicted.norm(),1e-300);
        const bool agreement = compare_voltages(voltage,predicted,"source_reciprocity");
        if (!radiation) {
          // A^T=A, A e_j=iw b_j+r_j, A s=iw q+r_s. No complex
          // conjugation: V_direct-V_recip=(s^T r_j-e_j^T r_s)/(iw).
          // This separates weak-overlap cancellation from actual solve error.
          const int n=fes.GetTrueVSize();
          mfem::Vector sr(source_field.GetData(),n), si(source_field.GetData()+n,n);
          mfem::Vector rr(source_residual.GetData(),n), ri(source_residual.GetData()+n,n);
          Eigen::VectorXcd port_term(nports), source_term(nports), residual_prediction(nports);
          json absolute_bound=json::array();
          for (int j=0;j<nports;++j) {
            port_term(j)=fw_bilinear(sr,si,residuals_re[j],residuals_im[j])/cplx(0,w);
            source_term(j)=-fw_bilinear(fields_re[j],fields_im[j],rr,ri)/cplx(0,w);
            residual_prediction(j)=port_term(j)+source_term(j);
            double scale=0.0;
            for (int k=0;k<n;++k)
              scale += std::hypot(sr(k),si(k))*std::hypot(residuals_re[j](k),residuals_im[j](k))
                     + std::hypot(fields_re[j](k),fields_im[j](k))*std::hypot(rr(k),ri(k));
            absolute_bound.push_back(fw_allsum(scale)/w);
          }
          const Eigen::VectorXcd defect=voltage-predicted;
          primary_row["source_residual_reciprocity_identity"]={
              {"convention","r=A E-rhs; transpose, not adjoint; defect=(s^T r_port-E_port^T r_source)/(i omega)"},
              {"observed_voltage_defect_V",complex_vector_json(defect)},
              {"port_residual_contribution_V",complex_vector_json(port_term)},
              {"source_residual_contribution_V",complex_vector_json(source_term)},
              {"predicted_voltage_defect_V",complex_vector_json(residual_prediction)},
              {"identity_remainder_V",complex_vector_json(defect-residual_prediction)},
              {"absolute_true_dof_contribution_bound_V",absolute_bound},
              {"scope","diagnostic identity only; neither residual nor componentwise acceptance is relaxed"}};
        }
        {
          const bool accepted = primary_row["direct_source_solver"]["accepted"].get<bool>();
          require_validation(accepted,"direct_source_true_residual",
                      "direct validation FEM solve failed: direct_source", "/fullwave",
                      accepted ? std::string() : worst_elements_note(P, kRejectedDiagnostics).substr(1));
        }
        require_validation(agreement,"source_reciprocity_componentwise",
                    "componentwise FEM/circuit voltage agreement failed: source_reciprocity",
                    "/fullwave", "Tighten the true residual and inspect weak/null port transfers.");
      }
      if (!fw.validation_shunt_R_ohm.empty() || !fw.validation_loads.is_null()) {
        mfem::DenseMatrix Uload(fes.GetTrueVSize(),nports);
        std::vector<cplx> cload(nports);
        Eigen::VectorXcd admittance(nports);
        for (int j=0;j<nports;++j) {
          if (!fw.validation_loads.is_null()) {
            const auto& load = fw.validation_loads.at(j);
            const double r = load.value("R_ohm",0.0), c = load.value("C_F",0.0), l = load.value("L_H",0.0);
            const cplx z(r,(c > 0 ? 1.0/(w*c) : 0.0)-w*l);
            if (r == 0 && c == 0 && l == 0) admittance(j) = 0.0;
            else {
              require_validation(std::abs(z) > 0,"load_series_singularity", "ideal validation load is singular at its series resonance",
                          "/fullwave/validation_loads", "Add a positive series resistance or avoid this exact frequency.");
              admittance(j) = 1.0/z;
            }
          } else admittance(j)=1.0/fw.validation_shunt_R_ohm[j];
          cload[j]=cplx(0,-w)*admittance(j);
          for (int i=0;i<fes.GetTrueVSize();++i) Uload(i,j)=Jports[j](i)/I;
        }
        // Insert the lumped current feedback into the FE operator itself: A-iw B Y B^T.
        FwOperator loaded(Afull,Uload,cload);
        mfem::Vector rhs(2*fes.GetTrueVSize()); rhs=0.0;
        for (int i=0;i<fes.GetTrueVSize();++i) rhs(fes.GetTrueVSize()+i)=w*Jports[0](i)/I;
        mfem::Vector loaded_field;
        const auto voltage=validation_solve(loaded,rhs,"direct_loaded",&loaded_field,nullptr,&Uload,&cload);
        Eigen::MatrixXcd closure=Eigen::MatrixXcd::Identity(nports,nports)+Zmatrix*admittance.asDiagonal();
        Eigen::FullPivLU<Eigen::MatrixXcd> lu(closure);
        require_validation(lu.isInvertible(),"load_circuit_singularity",
                           "validation circuit closure is singular","/fullwave/validation_loads");
        const Eigen::VectorXcd predicted=lu.solve(Zmatrix.col(0));
        primary_row["direct_loaded_voltage"] = complex_vector_json(voltage);
        primary_row["circuit_loaded_voltage"] = complex_vector_json(predicted);
        primary_row["loaded_closure_relative_error"]=(voltage-predicted).norm()/std::max(predicted.norm(),1e-300);
        primary_row["validation_load_admittance_S"] = complex_vector_json(admittance);
        const bool agreement = compare_voltages(voltage,predicted,"loaded_closure");
        // The same incoming-current solution must reconstruct the directly
        // loaded FEM field, not only terminal voltages. Export it for checking
        // field enhancement in the physical inter-case vacuum gaps.
        Eigen::VectorXcd currents = -admittance.cwiseProduct(predicted);
        currents(0) += 1.0;
        mfem::Vector combined(2*fes.GetTrueVSize()); combined = 0.0;
        mfem::Vector cre(combined.GetData(),fes.GetTrueVSize());
        mfem::Vector cim(combined.GetData()+fes.GetTrueVSize(),fes.GetTrueVSize());
        for (int j = 0; j < nports; ++j) {
          cre.Add(currents(j).real(),fields_re[j]); cre.Add(-currents(j).imag(),fields_im[j]);
          cim.Add(currents(j).imag(),fields_re[j]); cim.Add(currents(j).real(),fields_im[j]);
        }
        save_field(cre,cim,"circuit_loaded");
        loaded_field -= combined;
        const double field_rel = std::sqrt(fw_gdot(loaded_field,loaded_field)/std::max(fw_gdot(combined,combined),1e-300));
        primary_row["loaded_field_closure_relative_error"] = field_rel;
        primary_row["loaded_field_currents_A"] = complex_vector_json(currents);
        primary_row["loaded_closure_condition_number"] = closure.norm()*lu.inverse().norm();
        if (energy_M) primary_row["loaded_field_energy_J"] = std::real((currents.adjoint()*energy_G*currents)(0,0));
        {
          const bool accepted = primary_row["direct_loaded_solver"]["accepted"].get<bool>();
          require_validation(accepted,"direct_loaded_true_residual",
                             "direct validation FEM solve failed: direct_loaded", "/fullwave",
                             accepted ? std::string() : worst_elements_note(P, kRejectedDiagnostics).substr(1));
        }
        require_validation(agreement,"loaded_closure_componentwise",
                    "componentwise FEM/circuit voltage agreement failed: loaded_closure",
                    "/fullwave", "Tighten the true residual and inspect weak/null port transfers.");
        require_validation(std::isfinite(field_rel) && field_rel < 1e-5,"loaded_field_closure",
                    "direct loaded FEM field and port-basis reconstruction disagree", "/fullwave/validation_loads");
      }
      if (!fw.surface_quadrature_file.empty()) {
        // Written here, after the last refusal gate, so that a refused row leaves no file behind.
        // One file per rank: each boundary face belongs to exactly one rank, so the union is the
        // surface. One file per frequency row: the rows of a sweep would otherwise overwrite one another's file.
        const std::string fn = strcat(fw.surface_quadrature_file, ".f", frequency_index, ".rank", P.rank, ".bin");
        std::ofstream fb(fn, std::ios::binary);
        if (fb) {
          const double hdr[2] = {static_cast<double>(surf_pts.size()), static_cast<double>(nports)};
          fb.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
          for (const auto& sp : surf_pts) {
            const double rec[8] = {sp.x[0], sp.x[1], sp.x[2], sp.n[0], sp.n[1], sp.n[2], sp.dA,
                                   static_cast<double>(sp.attr)};
            fb.write(reinterpret_cast<const char*>(rec), sizeof(rec));
          }
          if (!surf_B.empty())
            fb.write(reinterpret_cast<const char*>(surf_B.data()),
                     static_cast<std::streamsize>(surf_B.size() * sizeof(double)));
          fb.close();
        }
        // agreed on every rank before anyone raises: a rank-local failure would hang the others. On a failure
        // every rank removes its own file of the row, so no partial row stays on disk
        const std::string failing = fw_failing_ranks(!fb);
        if (!failing.empty()) {
          std::error_code ec;
          std::filesystem::remove(fn, ec);
        }
        AXF_REQUIRE(failing.empty(),
                    strcat("cannot write the surface-quadrature file ", fw.surface_quadrature_file, ".f", frequency_index,
                           ".rank<r>.bin on rank(s) ", failing, "; the other ranks' files of this row were removed"),
                    "/fullwave/surface_quadrature_file");
        primary_row["surface_quadrature_file"] = fn;
      }
      primary_row["frequency_total_seconds"] = fw_allmax(frequency_timer.seconds());
      const double peak_bytes = fw_peak_rss_bytes();
      primary_row["peak_rss_max_rank_bytes"] = fw_allmax(peak_bytes);
      primary_row["peak_rss_sum_rank_bytes"] = fw_allsum(peak_bytes);
      memory["frequency_end"] = fw_memory_record();
      if (tree_cotree) memory["after_factorization_rank0"] = gauged->MemoryAfterFactorization();
      else if (direct_preconditioner) memory["after_factorization_rank0"] = direct_preconditioner->MemoryAfterFactorization();
      primary_row["memory"] = memory;
      primary_row["status"] = "provisional";
      primary_row["validation_accepted"] = true;
      sweep.push_back(primary_row);
      ++frequency_index;
    };
    std::vector<double> solved_frequencies = fw.frequencies;   // in solve order (the field-export index)
    json adaptive_resonances;
    if (!fw.adaptive.enabled) {
      for (double f : fw.frequencies) solve_frequency(f);
    } else {
      // Adaptive sweep (docs/physics/adaptive_sweep.md): fw.frequencies holds the initial points. Fitted outputs: every
      // entry of Z, with the exact dZ/dw as Hermite data, and of the Psi vector, over one common denominator. The
      // scene stays at the initial point nearest to scene_frequency.
      const auto& ad = fw.adaptive;
      const int nz = nports * nports;
      rational::FitOptions fit_options;
      for (int k = 0; k < nz + nports; ++k) {
        fit_options.group.push_back(k < nz ? 0 : 1);
        fit_options.hermite.push_back(k < nz ? 1 : 0);
        // Z[j][l] is measured against its ports' own sizes (sqrt(Z_jj Z_ll)), so a port of small |Z| keeps a
        // relative tolerance; Psi against the group's largest entry
        const int j = k / nports;
        const int l = k % nports;
        if (k < nz) fit_options.floor_pair.push_back({j * nports + j, l * nports + l});
        else fit_options.floor_pair.push_back({-1, -1});
      }
      fit_options.fit_tolerance = std::max(kAdaptiveFitFraction * ad.tolerance, kAdaptiveFitFloor);
      rational::AdaptiveOptions options;
      options.f_min = ad.f_min_Hz;
      options.f_max = ad.f_max_Hz;
      options.tolerance = ad.tolerance;
      options.max_solves = ad.max_solves;
      options.initial = fw.frequencies;
      options.log_spacing = ad.spacing == "log";
      options.candidates = std::max(2000, 4 * ad.dense_points);
      options.agree = [](std::vector<double>& decision) {   // every rank takes rank 0's decision
        MPI_Bcast(decision.data(), static_cast<int>(decision.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      };
      json refused = json::array();   // attempted frequencies without a usable solve
      auto solve_adaptive = [&](double f) -> std::optional<rational::Sample> {
        frequency_rejected = false;
        try {
          solve_frequency(f);
        } catch (const Error& error) {
          if (!frequency_rejected) throw;
          // a validation gate refused this frequency: keep its diagnostics, restore the sweep's status, move on. The
          // rejected-row file retain_rejected wrote (the whole result, status provisional_rejected) is renamed, so that
          // a sweep that goes on does not leave a file that reads as its own failure
          const std::filesystem::path written = field_dir.empty()
                                                    ? std::filesystem::path(o.workdir) / "fullwave_rejected.json"
                                                    : field_dir / strcat("f_", frequency_index, "_rejected.json");
          const std::filesystem::path kept_as = written.parent_path() /
                                                strcat("fullwave_adaptive_refused_f_", frequency_index, ".json");
          if (P.rank == 0) {
            std::error_code ignored;
            std::filesystem::rename(written, kept_as, ignored);
          }
          // the gate's mesh check names the file retain_rejected wrote; here it is kept_as, and results.json carries
          // mesh.info.worst_elements too
          std::string what = error.what();
          const std::string written_ptr = kRejectedDiagnostics;
          if (const auto at = what.find(written_ptr); at != std::string::npos)
            what.replace(at, written_ptr.size(),
                         strcat("the results and in ", kept_as.string(), " (fullwave.adaptive.refused[].diagnostics_file)"));
          refused.push_back({{"f_Hz", f}, {"reason", strcat("rejected: ", what)},
                             {"rejected_row", out.value("rejected_frequency", json())},
                             {"diagnostics_file", kept_as.string()}});
          out.erase("rejected_frequency");
          out.erase("validation_accepted");
          out["status"] = "provisional";
          warnings.push_back(strcat("3D full wave adaptive sweep: f = ", f, " Hz was refused (", what,
                                    "); it is left out of the fit (fullwave.adaptive.refused)"));
          ++frequency_index;
          return std::nullopt;
        }
        std::string why;
        std::optional<rational::Sample> sample = fw_adaptive_sample(sweep.back(), nports, why);
        if (!sample) {
          refused.push_back({{"f_Hz", f}, {"reason", why}});
          warnings.push_back(strcat("3D full wave adaptive sweep: f = ", f, " Hz is left out of the fit (", why, ")"));
        }
        return sample;
      };
      const rational::AdaptiveResult result = rational::adaptive_sweep(options, fit_options, solve_adaptive);
      solved_frequencies = result.attempted;
      std::stable_sort(sweep.begin(), sweep.end(), [](const json& a, const json& b) {
        return a["f_Hz"].get<double>() < b["f_Hz"].get<double>();
      });
      // the solves' relative accuracy, for the poles' uncertainty: the largest relative residual of the fitted port
      // solves, within [1e-12, 1e-6] (the acceptance gate)
      double data_noise = 1e-12;
      for (const auto& row : sweep)
        for (const auto& port : row.value("port_solutions", json::array())) {
          const json solver = port.value("solver", json::object());
          if (!solver.value("accepted", false)) continue;
          const double residual = solver.value("relative_residual", 0.0);
          if (std::isfinite(residual)) data_noise = std::max(data_noise, residual);
        }
      data_noise = std::min(data_noise, 1e-6);
      out["adaptive"] = fw_adaptive_record(result, ad, options, fit_options, nports, refused, data_noise,
                                           adaptive_resonances);
      if (result.samples.empty())
        warnings.push_back(strcat("3D full wave adaptive sweep: no frequency gave a usable solve (",
                                  result.attempted.size(), " attempted, all refused or failed; "
                                  "fullwave.adaptive.refused): there is no model, dense curve or resonance"));
      else if (!result.converged)
        warnings.push_back(strcat("3D full wave adaptive sweep: stopped (", result.stop_reason, ") after ",
                                  result.attempted.size(), " solves with an estimated error of ",
                                  result.estimated_error, " (newest solve against its prediction ",
                                  result.last_prediction_error, ") against the tolerance ", ad.tolerance,
                                  "; raise fullwave.adaptive.max_solves or the tolerance. Its poles are not reported "
                                  "as self_resonances (fullwave.adaptive.poles lists them, unconfirmed)"));
      if (!result.hermite_note.empty())
        warnings.push_back("3D full wave adaptive sweep: derivative (Hermite) rows " + result.hermite_note +
                           "; the sweep went on fitting values only");
      if (!result.samples.empty() && rational::noise_bound(fit_options.fit_tolerance, data_noise))
        warnings.push_back(strcat("3D full wave adaptive sweep: noise-bound. The fit tolerance ",
                                  fit_options.fit_tolerance, " (", kAdaptiveFitFraction, " x the tolerance) is below ",
                                  rational::kNoiseBoundFactor, " x the solves' relative residual (pole_noise ",
                                  data_noise, "), so the fit follows the solves' errors and its stop, converged or "
                                  "not, is unreliable: in the trials up to 23 % of such sweeps stopped with errors up to "
                                  "40 x the tolerance (docs/physics/adaptive_sweep.md, Limits). Raise "
                                  "fullwave.adaptive.tolerance to at least ",
                                  rational::kNoiseBoundFactor * data_noise / kAdaptiveFitFraction,
                                  " or tighten the solves (AXF_FW_RTOL)"));
      else if (!result.samples.empty() &&
               rational::noise_bound(fit_options.fit_tolerance, kAdaptiveNoiseMargin * data_noise))
        // pole_noise is a proxy: solves next to a pole had errors up to 24 x it (the measured FEM runs), so a sweep
        // whose fit tolerance is 10-100 x pole_noise can be noise-bound next to its poles without the flag
        warnings.push_back(strcat("3D full wave adaptive sweep: the fit tolerance ", fit_options.fit_tolerance,
                                  " is within ", kAdaptiveNoiseMargin * rational::kNoiseBoundFactor, " x the solves' ",
                                  "relative residual (pole_noise ", data_noise, "). Solves next to a pole can be up to "
                                  "24 x less accurate than their residual, so the sweep may be noise-bound there "
                                  "although noise_bound is false; in the trials 1-6 % of such sweeps stopped wrong "
                                  "(docs/physics/adaptive_sweep.md, Limits)"));
      if (!result.samples.empty() && !result.fit_consistent)
        warnings.push_back(strcat("3D full wave adaptive sweep: the final fit misses its own solves by ",
                                  result.fit.sample_error, " (fit tolerance ", fit_options.fit_tolerance,
                                  "): the solves are noisier than the fit tolerance or inconsistent"));
      log(1, strcat("3D full wave adaptive sweep: ", result.stop_reason, " after ", result.attempted.size(),
                    " solves (", result.failed.size(), " left out), ", result.fit.model.size(),
                    " support points, estimated error ", result.estimated_error));
    }
    out["memory"]["after_sweep"] = fw_memory_record();
    out["memory"]["per_frequency"] = "sweep[].memory: at the start and end of every frequency (fw_memory_record), and "
                                     "rank 0 right after a direct factorization";
    if (!unguarded_frequencies.empty() && !tree_cotree)
      warnings.push_back(strcat("3D full wave: the Gauss-law guard was not applied (", unguarded_reason, ") at ",
                                unguarded_frequencies.size(), " quasi-static frequencies (k R < ", kGuardWarnKR,
                                ", lowest ", unguarded_frequencies.front(), " Hz): the ungauged field outputs there "
                                "(energies, probes, exports, scenes, and Z of ports between distinct conductors) can "
                                "carry an unchecked gradient error"));
    if (capacitive)
      warnings.push_back("3D full wave: the readout is capacitive (L(omega) <= 0) at some frequencies, i.e. above its "
                         "first self-resonance; F and Q_rad are omitted there");
    {
      int unresolved = 0;
      double lowest = 0;
      for (const auto& row : sweep)
        for (const auto& port : row.value("port_solutions", json::array()))
          if (port.contains("Re_Z_unresolved")) {
            if (unresolved == 0 || row["f_Hz"].get<double>() < lowest) lowest = row["f_Hz"].get<double>();
            ++unresolved;
          }
      if (unresolved > 0)
        warnings.push_back(strcat("3D full wave: Re Z is below ", kReZResolution, " times the solve's true relative "
                                  "residual times |X| in ", unresolved, " port rows (lowest ", lowest, " Hz): Re Z, "
                                  "R_rad_ohm and Q_rad are not resolved there (Re_Z_unresolved in the rows); a tighter "
                                  "GMRES tolerance (AXF_FW_RTOL) lowers the residual where double precision allows"));
    }
    // self-resonances: sign changes of L(omega) from + to - (the readout's first resonance); in an adaptive sweep the
    // poles of the fitted model in the band, with frequency and Q
    json res = json::array();
    for (size_t i = 1; i < sweep.size() && !fw.adaptive.enabled; ++i) {
      const double l0 = sweep[i - 1]["L_H"], l1 = sweep[i]["L_H"];
      if (l0 > 0 && l1 <= 0) res.push_back({{"between_Hz", {sweep[i - 1]["f_Hz"], sweep[i]["f_Hz"]}}});
    }
    if (fw.adaptive.enabled) {
      res = adaptive_resonances;
      // an unconverged sweep lists none, which means "not determined", not "none"
      out["self_resonances_determined"] = out["adaptive"].value("converged", false);
    }
    out["self_resonances"] = res;
    if (indicator) {   // the largest estimate over the sweep's frequencies and ports, and the evaluations' cost
      // withheld (failed solve) and non-finite estimates (JSON null) are left out of the maximum
      double largest = std::numeric_limits<double>::quiet_NaN();
      double evaluation_seconds = 0.0;
      int left_out = 0;
      json at = nullptr;
      for (const auto& row : sweep)
        for (const auto& port : row["port_solutions"]) {
          const json& estimate_json = port["error_indicator"];
          evaluation_seconds += estimate_json["seconds"].get<double>();
          const json& value = estimate_json.value("estimated_relative_error", json());
          if (!value.is_number() || !std::isfinite(value.get<double>())) {
            left_out += 1;
            continue;
          }
          const double estimate = value.get<double>();
          if (!std::isnan(largest) && estimate < largest) continue;
          largest = estimate;
          at = {{"f_Hz", port["f_Hz"]}, {"port_index", port["port_index"]}};
        }
      out["error_indicator"]["max_estimated_relative_error"] = largest;   // null when every estimate was left out
      out["error_indicator"]["max_at"] = at;
      if (left_out > 0) out["error_indicator"]["estimates_left_out_of_max"] = left_out;
      out["error_indicator"]["evaluation_seconds"] = evaluation_seconds;
      if (!field_dir.empty())
        out["error_indicator"]["files"] = "f_{frequency_index}_port_{port}_error_indicator_rank_{rank}.gf: eta_e^2 / "
                                          "||E||^2 per element (MFEM L2 order 0 on mesh_rank_{rank}.mesh)";
    }
    if (surface_losses) out["surface_losses"] = surface_losses->Summary(sweep, fw.adaptive.enabled, warnings);
    out["sweep"] = sweep;
    if (!field_dir.empty() && P.rank == 0) {
      std::ofstream manifest(field_dir/"manifest.json");
      AXF_REQUIRE(bool(manifest), "cannot write field-export manifest", "/fullwave/field_export_directory");
      manifest << std::setprecision(17) << json{{"status","provisional"}, {"field_export",out["field_export"]},
                       {"ports",port_info}, {"frequencies_Hz",solved_frequencies},
                       {"files_per_port", "f_{frequency_index}_port_{port}_rank_{rank}_{re,im}.gf"},
                       {"combined_files", "f_{frequency_index}_{combined,direct_loaded,circuit_loaded}_rank_{rank}_{re,im}.gf; only when requested"},
                       {"error_indicator_files", indicator ? out["error_indicator"]["files"] : json(nullptr)}}.dump(2);
    }
    out["gmres_iterations_total"] = total_its;
    out["family"] = "3D (port feed)";
    out["note"] = "Psi integrates E . B_lab over the domain sphere plus the exterior term oint Phi_lab E_r dA "
                  "(Psi_exterior); an imposed background adds int curl E . A_bg dV over the domain sphere and outside "
                  "it (Psi_background); exterior: exact static map plus vector-spherical-harmonic radiation correction "
                  "(\"radiation\") or a perfect conductor at R1 (\"pec\")";
  }

  // ------------------------------------------------------------ eigenmodes (closed PEC structure)
  // AME (lowest modes) or shift-invert Lanczos about target_f_Hz; with a "cavity" section (beta)
  // the problem is restricted to the cavity component and every mode gets a form factor, losses and overlaps.
  if (fw.modes > 0) {
    Timer eig_timer;
    const auto& cav = d_.cavity3d;
    const auto& eopt = fw.eigensolver;
    const bool shift_invert = eopt.method == "shift_invert" || (eopt.method.empty() && fw.target_f_Hz > 0);
    AXF_REQUIRE(!shift_invert || have_strumpack(), "the 3D shift-invert eigensolver needs a STRUMPACK-enabled build",
                "/fullwave/eigensolver", "Rebuild with -DAXF_USE_STRUMPACK=ON, or use the AME eigensolver (lowest modes).");
    mfem::Mesh& sm = *P.serial;
    static const spec::Material vacuum_m = spec::builtin_materials().at("vacuum");
    auto ser_mat = [&](int s2) -> const spec::Material* {
      auto it = P.region_of_attr.find(P.ser_attr[s2]);
      const spec::Region* g = it == P.region_of_attr.end() ? nullptr : d_.find_region(it->second);
      return g ? &d_.materials.at(g->material) : &vacuum_m;
    };
    auto ser_region = [&](int s2) {
      auto it = P.region_of_attr.find(P.ser_attr[s2]);
      return it == P.region_of_attr.end() ? std::string("air") : it->second;
    };
    // boundary groups: attr -> (name, type, symmetry plane?, conductor region)
    struct BGroup { std::string name, type, conductor; bool symmetry = false; };
    std::map<int, BGroup> bgroups;
    if (P.mesh_info.contains("boundary_groups"))
      for (const auto& g : P.mesh_info["boundary_groups"]) {
        BGroup b;
        b.name = g.value("name", std::string("boundary"));
        b.type = g.value("type", std::string("pec"));
        b.conductor = g.value("conductor", std::string());
        b.symmetry = g.contains("axis");
        bgroups[g.at("attr").get<int>()] = b;
      }
    if (!bgroups.count(axi::kOuterAttr)) bgroups[axi::kOuterAttr] = {"outer", "pec", "", false};
    // ---- cavity component (serial mesh, identical on every rank)
    const int nser = sm.GetNE();
    std::vector<char> ser_cav(nser, 1), in_cav(ne, 1);
    json cav_json;
    double V_cav = 0;
    if (cav.enabled) {
      mfem::DenseMatrix pt(3, 1);
      pt(0, 0) = cav.interior_point.x; pt(1, 0) = cav.interior_point.y; pt(2, 0) = cav.interior_point.z;
      mfem::Array<int> ids;
      mfem::Array<mfem::IntegrationPoint> ips;
      sm.FindPoints(pt, ids, ips, false);
      const int s0 = ids.Size() ? ids[0] : -1;
      AXF_REQUIRE(s0 >= 0, strcat("cavity.interior_point (", cav.interior_point.x, ", ", cav.interior_point.y, ", ",
                                  cav.interior_point.z, ") lies outside the meshed domain (or inside a removed PEC "
                                  "conductor, or in a part cut away by mesh.symmetry)"),
                  "/cavity/interior_point", "Give a point inside the cavity volume (strictly inside, not on a wall).");
      AXF_REQUIRE(!is_pec(ser_mat(s0)), strcat("cavity.interior_point lies on or inside the conductor region '",
                                               ser_region(s0), "'"),
                  "/cavity/interior_point", "Move the point into the cavity, away from walls.");
      std::vector<int> comp(nser, -1);
      mfem::Array<int> fcs, ori;
      int ncomp = 0, cav_comp = -1;
      for (int s2 = 0; s2 < nser; ++s2) {
        if (comp[s2] >= 0 || is_pec(ser_mat(s2))) continue;
        std::vector<int> st{s2};
        comp[s2] = ncomp;
        while (!st.empty()) {
          const int e = st.back();
          st.pop_back();
          sm.GetElementFaces(e, fcs, ori);
          for (int f : fcs) {
            int e1, e2;
            sm.GetFaceElements(f, &e1, &e2);
            const int o = e1 == e ? e2 : e1;
            if (o >= 0 && comp[o] < 0 && !is_pec(ser_mat(o))) { comp[o] = ncomp; st.push_back(o); }
          }
        }
        if (comp[s0] == ncomp) cav_comp = ncomp;
        ++ncomp;
      }
      std::set<std::string> regs, touched;
      for (int s2 = 0; s2 < nser; ++s2) {
        ser_cav[s2] = comp[s2] == cav_comp;
        if (ser_cav[s2]) regs.insert(ser_region(s2));
      }
      std::map<std::string, int> sheets;  // PEC boundary elements with cavity elements on both sides
      for (int i = 0; i < sm.GetNBE(); ++i) {
        int e1, e2;
        sm.GetFaceElements(sm.GetBdrElementFaceIndex(i), &e1, &e2);
        const int a = sm.GetBdrAttribute(i);
        const std::string gname = bgroups.count(a) ? bgroups[a].name : strcat("attr_", a);
        if ((e1 >= 0 && ser_cav[e1]) || (e2 >= 0 && ser_cav[e2])) touched.insert(gname);
        const bool pec_group = bgroups.count(a) && bgroups[a].type == "pec" && !bgroups[a].symmetry;
        if (pec_group && e1 >= 0 && e2 >= 0 && ser_cav[e1] && ser_cav[e2]) ++sheets[gname];
      }
      if (!sheets.empty()) {
        std::string lst;
        for (auto& [g, c] : sheets) lst += (lst.empty() ? "" : ", ") + g + " (" + std::to_string(c) + " faces)";
        // As in the axisymmetric cavity: on a zero-thickness sheet the weak reaction is the sum of both sides' currents
        // and the pointwise curl sees one side only, so the wall loss there would be wrong
        AXF_REQUIRE(!cav.losses, strcat("cavity: PEC boundary faces with the cavity on both sides (a zero-thickness "
                                        "conducting sheet, e.g. from mesh.file): ", lst, ". Their wall loss is not "
                                        "supported"),
                    "/cavity/losses", "Model the sheet as a thin conductor region (or a PEC hole of finite thickness), "
                    "or set \"cavity\": {\"losses\": false} for frequencies and form factors only.");
        warnings.push_back(strcat("cavity: PEC boundary faces with the cavity on both sides (zero-thickness sheets): ",
                                  lst, "; they act as n x E = 0 on both faces (losses are off)"));
      }
      for (int e = 0; e < ne; ++e) {  // volume with the quadrature used for int |B0|^2 below (curved elements)
        in_cav[e] = ser_cav[P.loc2ser[e]];
        if (!in_cav[e]) continue;
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        const auto& vr = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P.order + 2);
        for (int q = 0; q < vr.GetNPoints(); ++q) {
          T->SetIntPoint(&vr.IntPoint(q));
          V_cav += vr.IntPoint(q).weight * T->Weight();
        }
      }
      V_cav = fw_allsum(V_cav);
      json sym = json::array();
      for (const auto& p : d_.mesh.symmetry) sym.push_back(p);
      cav_json = {{"interior_point", {cav.interior_point.x, cav.interior_point.y, cav.interior_point.z}},
                  {"regions", std::vector<std::string>(regs.begin(), regs.end())},
                  {"excluded_components", ncomp - 1}, {"volume_m3", V_cav},
                  {"boundaries", std::vector<std::string>(touched.begin(), touched.end())}, {"symmetry", sym},
                  {"note", "the cavity is the connected non-conductor component containing interior_point; other "
                           "non-conductor components are closed off (perfect conductor). Under mesh.symmetry, "
                           "volume_m3 is the meshed (reduced) volume; C, G and Q are ratios and need no image factor"}};
      if (!sheets.empty()) cav_json["interior_sheets"] = sheets;
    }
    std::vector<char> closed(ne, 0);  // PEC, outside the cavity, or the (closed-off) infinity shell: all DOFs essential
    for (int e = 0; e < ne; ++e)
      closed[e] = is_pec(P.elem_mat[e]) || !in_cav[e] || (P.sh_attr > 0 && pm.GetAttribute(e) == P.sh_attr);
    mfem::Array<int> ess;
    for (const auto& b : robin)
      AXF_REQUIRE(b.type == "impedance", "3D eigenmodes with a " + b.type + " boundary ('" + b.name +
                  "') are not supported (complex modes)", "/fullwave/modes",
                  "Remove fullwave.modes, or close the model with PEC/PMC/impedance walls.");
    if (!robin.empty()) {
      impedance_as_pec = true;
      warnings.push_back("3D eigenmodes: impedance boundaries are perfect conductors in the eigenproblem; their wall "
                         "losses are not evaluated (use the 2D engine's perturbative Q, or a driven sweep)");
    }
    essential(true, ess, &in_cav);
    impedance_as_pec = false;
    mfem::Array<int> ess_bdr(pm.bdr_attributes.Size() ? pm.bdr_attributes.Max() : 0);
    ess_bdr = 0;
    for (auto& [attr, g] : bgroups)  // impedance walls are PEC in the eigenproblem (as in essential() above)
      if ((g.type == "pec" || g.type == "impedance") && attr <= ess_bdr.Size()) ess_bdr[attr - 1] = 1;
    set_eps(0.0);
    std::vector<cplx> one_eps(eps_e);
    ShellMapR none;  // the shell is closed off
    // scaled to k^2 = omega^2 mu0 eps0 (O(1) per m^2): hypre's LOBPCG tolerance is absolute, and the SI
    // eigenvalues omega^2 ~ 1e17 never meet it (it then returns unconverged vectors)
    FwTensor nr(none, nu_e, 0, 0, mu0, 0.0), mr(none, one_eps, 1, 0, 1.0 / eps0, 0.0);
    mfem::ParBilinearForm a(&fes), m(&fes);
    a.AddDomainIntegrator(new mfem::CurlCurlIntegrator(nr));
    a.Assemble();
    a.Finalize();
    m.AddDomainIntegrator(new mfem::VectorFEMassIntegrator(mr));
    m.Assemble();
    m.Finalize();
    std::unique_ptr<mfem::HypreParMatrix> A(a.ParallelAssemble()), M(m.ParallelAssemble());
    std::unique_ptr<mfem::HypreParMatrix> A0, M0;  // unconstrained: wall reaction (consistent surface current)
    if (cav.enabled && cav.losses) { A0.reset(a.ParallelAssemble()); M0.reset(m.ParallelAssemble()); }
    ensure_diagonal(A, ess);
    A->EliminateBC(ess, mfem::Operator::DIAG_ONE);
    M->EliminateBC(ess, mfem::Operator::DIAG_ZERO);
    const double global_n = static_cast<double>(fes.GlobalTrueVSize());
    FwGradientProjector gproj(fes, *M, ess, closed, ess_bdr, P.order);
    std::vector<mfem::Vector> X;  // eigenvectors (true DOFs)
    std::vector<double> lam;      // k^2 [1/m^2]
    json solver;
    if (!shift_invert) {
      mfem::HypreAMS ams(*A, &fes);
      ams.SetPrintLevel(0);
      ams.SetSingularProblem();
      mfem::HypreAME ame(MPI_COMM_WORLD);
      ame.SetNumModes(fw.modes);
      ame.SetPreconditioner(ams);
      ame.SetMaxIter(eopt.max_iter);
      ame.SetTol(eopt.tol);
      ame.SetPrintLevel(0);
      ame.SetMassMatrix(*M);
      ame.SetOperator(*A);
      ame.Solve();
      mfem::Array<double> l;
      ame.GetEigenvalues(l);
      for (int k = 0; k < l.Size(); ++k) {
        X.emplace_back(ame.GetEigenvector(k));
        lam.push_back(l[k]);
      }
      solver = {{"method", "ame"}, {"max_iter", eopt.max_iter}, {"tol", eopt.tol},
                {"note", "hypre AME (LOBPCG with AMS); returns the lowest modes. Floating conductors add static "
                         "lambda ~ 0 fields that AME does not deflate: use shift_invert there"}};
    } else {
      const double kt = 2 * pi * fw.target_f_Hz / units::c, sigma = kt * kt;
      Timer fac_timer;
      std::unique_ptr<mfem::HypreParMatrix> Sh(mfem::Add(1.0, *A, -sigma, *M));
      FwRealFactorization fac(*Sh, 1);
      const double fac_s = fw_allmax(fac_timer.seconds());
      mfem::Vector Mx(A->Height());
      int nsolve = 0;
      auto op = [&](const mfem::Vector& x, mfem::Vector& y) {
        M->Mult(x, Mx);
        fac.Mult(Mx, y);
        for (int i : ess) y(i) = 0.0;
        ++nsolve;
      };
      auto deflate = [&](mfem::Vector& v) { gproj.project(v, true); };
      auto start = [&](int seed, mfem::Vector& v) {
        FwHashField hf(seed);
        mfem::ParGridFunction g(&fes);
        g.ProjectCoefficient(hf);
        v.SetSize(fes.GetTrueVSize());
        g.GetTrueDofs(v);
        for (int i : ess) v(i) = 0.0;
      };
      Timer lz_timer;
      auto res = fw_shift_invert_lanczos(*M, op, deflate, start, sigma, fw.modes, eopt.max_steps, global_n,
                                         eopt.completeness_steps);
      X = std::move(res.x);
      lam = res.lam;
      solver = res.diag;
      solver["method"] = "shift_invert";
      solver["target_f_Hz"] = fw.target_f_Hz;
      solver["shift_k2_per_m2"] = sigma;
      solver["factorization_seconds"] = fac_s;
      solver["lanczos_seconds"] = fw_allmax(lz_timer.seconds());
      solver["solves"] = nsolve;
      solver["factor_nonzeros"] = fac.FactorNonzeros();
      json inertia = fac.Inertia();
      solver["inertia"] = inertia;
      solver["h1_free_dofs"] = gproj.free_dofs();
      if (inertia.value("accurate", false) && inertia["negative"].is_number())
        solver["eigenvalues_below_target_plus_floating_conductors"] =
            inertia["negative"].get<double>() - gproj.free_dofs();
      solver["ritz_estimates"] = res.estimate;
      solver["note"] = "STRUMPACK LU of A - sigma M (rank 0), M-inner-product Lanczos with full reorthogonalization; "
                       "Lanczos vectors are deflated against discrete gradients; lambda ~ 0 Ritz values (curl "
                       "kernel, e.g. floating conductors) are discarded";
    }
    // ---- per-mode diagnostics (both solvers): normalization int eps_r |E|^2 = 1, true residual, Rayleigh
    // ratio, divergence fraction, M-orthogonality
    json list = json::array();
    double max_residual = 0, max_overlap = 0;
    bool finite_modes = true;
    {
      std::vector<std::pair<double, int>> ord;
      for (size_t k = 0; k < lam.size(); ++k) ord.push_back({lam[k], static_cast<int>(k)});
      std::sort(ord.begin(), ord.end());
      std::vector<mfem::Vector> Xs;
      std::vector<double> ls;
      for (auto& [l, k] : ord) { Xs.push_back(X[k]); ls.push_back(l); }
      X.swap(Xs);
      lam.swap(ls);
    }
    const int nmodes = static_cast<int>(lam.size());
    std::vector<mfem::Vector> MX(nmodes);
    std::vector<double> rel(nmodes, 0), rayleigh(nmodes, 0), divf(nmodes, 0);
    for (int k = 0; k < nmodes; ++k) {
      mfem::Vector& u = X[k];
      mfem::Vector au(u.Size()), mu(u.Size()), residual(u.Size());
      M->Mult(u, mu);
      const double mn = fw_gdot(u, mu);
      if (mn > 0 && std::isfinite(mn)) { u /= std::sqrt(mn); mu /= std::sqrt(mn); }
      A->Mult(u, au);
      residual = au;
      residual.Add(-lam[k], mu);
      rel[k] = std::sqrt(fw_gdot(residual, residual) / std::max(fw_gdot(au, au), 1e-300));
      rayleigh[k] = fw_gdot(u, au) / (lam[k] * std::max(fw_gdot(u, mu), 1e-300));
      mfem::Vector tmp(u);
      divf[k] = gproj.project(tmp, false);
      MX[k] = mu;
    }
    for (int k = 0; k < nmodes; ++k)
      for (int j = 0; j < k; ++j) max_overlap = std::max(max_overlap, std::fabs(fw_gdot(X[j], MX[k])));
    // mirror images of the meshed part under mesh.symmetry: (flip_x, flip_y, flip_z, sign of E), sign = prod over
    // the reflected planes of +1 (PMC: E(Rx) = R E(x)) or -1 (PEC: E(Rx) = -R E(x)); det R_g = prod of the flips
    std::vector<std::array<double, 4>> images{{1.0, 1.0, 1.0, 1.0}};
    for (const auto& pl : d_.mesh.symmetry) {
      const std::string ax = pl.value("axis", std::string("x"));
      const int a3 = ax == "x" ? 0 : ax == "y" ? 1 : 2;
      const double sg = pl.value("type", std::string("pec")) == "pmc" ? 1.0 : -1.0;
      const size_t n0 = images.size();
      for (size_t i0 = 0; i0 < n0; ++i0) {
        auto g = images[i0];
        g[a3] = -g[a3];
        g[3] *= sg;
        images.push_back(g);
      }
    }
    // ---- cavity: static field B0 (applied source, the field of the magnet as if the walls were absent)
    std::unique_ptr<PrescribedMagneticSource> b0src;
    mfem::Vector b0form;
    double B_ref = 0, B2 = 0;
    if (cav.enabled) {
      if (!fw.source.is_null()) b0src = std::make_unique<PrescribedMagneticSource>(PrescribedMagneticSource::from_json(fw.source));
      // B_ref: |B0| at cavity.form_factor.B_ref_point if given, else the cavity volume rms of B0 (C = C_rms)
      if (cav.B_ref_point) cav_json["B_ref_point"] = {cav.B_ref_point->x, cav.B_ref_point->y, cav.B_ref_point->z};
      if (b0src) {
        if (cav.B_ref_point) {
          const auto b = b0src->field_T({cav.B_ref_point->x, cav.B_ref_point->y, cav.B_ref_point->z});
          B_ref = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        }
        // Under mesh.symmetry the cavity is the union of the images R_g of the meshed part; the field of a mode
        // there is E(R_g x) = s_g R_g E(x) (s = +1 per PMC plane, -1 per PEC plane), so the full-cavity overlap is
        // int_part E . B_eff with B_eff = (1/n) sum_g s_g R_g B0(R_g x) (per image), and int |B0|^2 likewise.
        class CavB0 : public mfem::VectorCoefficient {
         public:
          CavB0(const PrescribedMagneticSource& s, const std::vector<char>& in, const std::vector<std::array<double, 4>>& img)
              : mfem::VectorCoefficient(3), s_(s), in_(in), img_(img) {}
          void Eval(mfem::Vector& v, mfem::ElementTransformation& T, const mfem::IntegrationPoint& ip) override {
            v.SetSize(3);
            v = 0.0;
            last_sq_ = 0.0;
            if (!in_[T.ElementNo]) return;
            mfem::Vector x(3);
            T.Transform(ip, x);
            for (const auto& g : img_) {  // g = (flip_x, flip_y, flip_z, sign)
              const auto b = s_.field_T({g[0] * x(0), g[1] * x(1), g[2] * x(2)});
              for (int k = 0; k < 3; ++k) v(k) += g[3] * g[k] * b[k] / img_.size();
              last_sq_ += (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) / img_.size();
            }
          }
          double last_sq_ = 0.0;  // mean_g |B0(R_g x)|^2 at the last evaluated point
         private:
          const PrescribedMagneticSource& s_;
          const std::vector<char>& in_;
          const std::vector<std::array<double, 4>>& img_;
        };
        std::vector<char> bulk(ne);
        for (int e = 0; e < ne; ++e) bulk[e] = in_cav[e] && !is_pec(P.elem_mat[e]);
        CavB0 cbb(*b0src, bulk, images);
        mfem::ParLinearForm l(&fes);
        l.AddDomainIntegrator(new mfem::VectorFEDomainLFIntegrator(cbb));
        l.Assemble();
        b0form.SetSize(fes.GetTrueVSize());
        l.ParallelAssemble(b0form);
        for (int i : ess) b0form(i) = 0.0;
        mfem::Vector bv(3);
        for (int e = 0; e < ne; ++e) {
          if (!bulk[e]) continue;
          mfem::ElementTransformation* T = pm.GetElementTransformation(e);
          const auto& ir = mfem::IntRules.Get(pm.GetElementBaseGeometry(e), 2 * P.order + 2);
          for (int q = 0; q < ir.GetNPoints(); ++q) {
            T->SetIntPoint(&ir.IntPoint(q));
            cbb.Eval(bv, *T, ir.IntPoint(q));
            B2 += cbb.last_sq_ * ir.IntPoint(q).weight * T->Weight();
          }
        }
        B2 = fw_allsum(B2);
        cav_json["source"] = b0src->summary();
      }
      const double Brms = V_cav > 0 ? std::sqrt(B2 / V_cav) : 0.0;
      if (!cav.B_ref_point) B_ref = Brms;
      cav_json["B_ref"] = cav.B_ref_point ? "point" : "volume_rms";
      cav_json["B_ref_T"] = B_ref;
      cav_json["B0_rms_T"] = Brms;
      if (cav.B_ref_point && Brms > 0) cav_json["B_ref_over_B0_rms"] = B_ref / Brms;
      if (!b0src)
        warnings.push_back("cavity: no fullwave.source (the applied static field B0), so the form factor C is not reported");
      else if (!(B_ref > 1e-6 * std::max(Brms, 1e-300)))
        warnings.push_back("cavity: the static field B0 is (almost) zero at the reference point, so C is not reported "
                           "(move cavity.form_factor.B_ref_point); C_rms is");
      else if (cav.B_ref_point && std::fabs(B_ref / Brms - 1) > 0.01)
        warnings.push_back(strcat("cavity: B0 at the reference point (", B_ref, " T) differs from its cavity rms (", Brms,
                                  " T) by more than 1 %: C depends on cavity.form_factor.B_ref_point; C_rms does not"));
    }
    std::map<std::string, const spec::Material*> body_mat;
    auto body_of_bdr = [&](int attr, std::string& name) -> bool {  // false: not a lossy wall (symmetry, PMC)
      auto it = bgroups.find(attr);
      if (it == bgroups.end()) { name = strcat("attr_", attr); return true; }
      if (it->second.symmetry || it->second.type != "pec") return false;
      if (attr == axi::kOuterAttr) {
        name = "outer";
        body_mat[name] = cav.wall_material.empty() ? nullptr : &d_.materials.at(cav.wall_material);
      } else if (!it->second.conductor.empty()) {
        name = it->second.conductor;
        const spec::Region* g = d_.find_region(name);
        body_mat[name] = g ? &d_.materials.at(g->material) : nullptr;
      } else {
        name = it->second.name;
        if (!body_mat.count(name)) body_mat[name] = nullptr;
      }
      return true;
    };
    // interior faces between a cavity element and a retained PEC element (no consistent-flux recovery there)
    struct WallFace { int e, lf; std::string body; bool boundary; };
    std::vector<WallFace> wall_faces;
    if (cav.enabled && cav.losses) {
      for (int be = 0; be < pm.GetNBE(); ++be) {
        std::string nm2;
        if (!body_of_bdr(pm.GetBdrAttribute(be), nm2)) continue;
        int e, info;
        pm.GetBdrElementAdjacentElement(be, e, info);
        if (e < 0 || !in_cav[e] || is_pec(P.elem_mat[e])) continue;
        wall_faces.push_back({e, be, nm2, true});
      }
      mfem::Array<int> lf, lo, fcs, ori;
      for (int e = 0; e < ne; ++e) {
        if (!in_cav[e] || is_pec(P.elem_mat[e])) continue;
        pm.GetElementFaces(e, lf, lo);
        sm.GetElementFaces(P.loc2ser[e], fcs, ori);
        for (int k = 0; k < fcs.Size() && k < lf.Size(); ++k) {
          int e1, e2;
          sm.GetFaceElements(fcs[k], &e1, &e2);
          const int o = e1 == P.loc2ser[e] ? e2 : e1;
          if (o < 0 || !is_pec(ser_mat(o))) continue;
          const std::string nm2 = ser_region(o);
          body_mat[nm2] = ser_mat(o);
          wall_faces.push_back({e, lf[k], nm2, false});
        }
      }
    }
    int interior_pec_faces = 0;
    for (auto& f : wall_faces) interior_pec_faces += f.boundary ? 0 : 1;
    interior_pec_faces = static_cast<int>(fw_allsum(interior_pec_faces));
    if (cav.enabled && cav.losses && interior_pec_faces > 0)
      warnings.push_back(strcat("cavity: ", interior_pec_faces, " wall faces border retained conductor volumes "
                                "(mesh.remove_pec not set): their surface current is recovered from the weak reaction "
                                "as on PEC holes, but the closed-off conductor DOFs make the model larger; prefer "
                                "mesh.remove_pec"));
    // ---- cavity: wall surface current (consistent flux) operator on all PEC boundary faces and on the faces
    // between the cavity and retained conductor volumes (u = 0 inside them, so the weak reaction on their surface
    // DOFs is the cavity side's flux alone)
    std::unique_ptr<mfem::HypreParMatrix> Ms;
    std::unique_ptr<mfem::HypreDiagScale> ms_prec;
    std::unique_ptr<mfem::HyprePCG> ms_pcg;
    std::vector<char> on_wall;
    mfem::Array<int> wall_marker;
    if (cav.enabled && cav.losses) {
      wall_marker = ess_bdr;
      mfem::ParBilinearForm ms(&fes);
      ms.AddBoundaryIntegrator(new mfem::VectorFEMassIntegrator(), wall_marker);
      ms.Assemble();
      {  // tangential trace mass on interior wall faces, from the cavity-side element
        mfem::Array<int> vd;
        mfem::DenseMatrix shp, el;
        mfem::Vector nor(3), tv(3);
        for (const auto& wf : wall_faces) {
          if (wf.boundary) continue;
          mfem::FaceElementTransformations* ft = pm.GetFaceElementTransformations(wf.lf, 1 | 2 | 4 | 8 | 16);
          const bool side1 = ft->Elem1No == wf.e;
          const mfem::FiniteElement& fe = *fes.GetFE(wf.e);
          mfem::DofTransformation* dtr = fes.GetElementVDofs(wf.e, vd);  // ND p > 1 on tets: face-DOF orientation
          const int nd = fe.GetDof();
          shp.SetSize(nd, 3);
          el.SetSize(nd, nd);
          el = 0.0;
          const auto& fir = mfem::IntRules.Get(ft->GetGeometryType(), 2 * P.order + 2);
          for (int q = 0; q < fir.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& ip = fir.IntPoint(q);
            ft->Face->SetIntPoint(&ip);
            mfem::IntegrationPoint eip;
            mfem::ElementTransformation* Te;
            if (side1) { ft->Loc1.Transform(ip, eip); Te = ft->Elem1; }
            else { ft->Loc2.Transform(ip, eip); Te = ft->Elem2; }
            Te->SetIntPoint(&eip);
            mfem::CalcOrtho(ft->Face->Jacobian(), nor);
            const double dA = ip.weight * nor.Norml2();
            nor /= nor.Norml2();
            fe.CalcVShape(*Te, shp);
            for (int i = 0; i < nd; ++i) {  // tangential part of each shape function
              double sn = 0;
              for (int c = 0; c < 3; ++c) sn += shp(i, c) * nor(c);
              for (int c = 0; c < 3; ++c) shp(i, c) -= sn * nor(c);
            }
            mfem::AddMult_a_AAt(dA, shp, el);
          }
          if (dtr) dtr->TransformDual(el);
          // DOFs off this face have a zero tangential trace on it, up to round-off (~1e-32 diagonals that would
          // make the boundary system singular): keep the face DOFs only
          double dmax = 0;
          for (int i = 0; i < nd; ++i) dmax = std::max(dmax, el(i, i));
          for (int i = 0; i < nd; ++i)
            if (!(el(i, i) > 1e-10 * dmax))
              for (int j = 0; j < nd; ++j) el(i, j) = el(j, i) = 0.0;
          ms.SpMat().AddSubMatrix(vd, vd, el, 1);
        }
      }
      ms.Finalize();
      Ms.reset(ms.ParallelAssemble());
      // the wall DOFs: a nonzero stored diagonal. Not GetDiag: on a GPU the off-wall rows are empty, and GetDiag would
      // read the next wall row's diagonal for them (stored_diagonal, model3d_impl.hpp)
      const mfem::Vector dg = stored_diagonal(*Ms);
      on_wall.assign(dg.Size(), 0);
      mfem::Array<int> off;
      for (int i = 0; i < dg.Size(); ++i) {
        if (dg(i) > 0) on_wall[i] = 1;
        else off.Append(i);
      }
      ensure_diagonal(Ms, off);  // off the wall Ms holds only zeros, which hypre on a GPU drops
      Ms->EliminateBC(off, mfem::Operator::DIAG_ONE);
      ms_prec = std::make_unique<mfem::HypreDiagScale>(*Ms);
      ms_pcg = std::make_unique<mfem::HyprePCG>(*Ms);
      ms_pcg->SetTol(1e-13);
      ms_pcg->SetMaxIter(2000);
      ms_pcg->SetPrintLevel(0);
      ms_pcg->SetPreconditioner(*ms_prec);
    }
    // ---- cavity: consistent-flux (charge) recovery of the normal E on the wall. For an H1 hat
    // function phi_i of a boundary node, int eps_r E.grad(phi_i) dV = oint eps_r E_n phi_i dA (the discrete
    // eigenvector is weakly divergence-free: the sum vanishes at interior nodes), so b = G^T (M0 u) with the discrete
    // gradient G (H1_p -> ND_p, exact) and the boundary H1 mass matrix give eps_r E_n as an H1 field on the cavity
    // boundary. The pointwise Nedelec normal component is the poorly controlled one on flat wall elements; this
    // integral estimate is not (docs/srf/README.md, "Wall peaks"). Built only with cavity.wall_samples (its cost is an
    // extra H1 space and a boundary solve per mode). The boundary mass covers the whole boundary: a node's reaction is
    // the flux through every boundary face around it, so a node where the wall meets another boundary is projected
    // consistently (the field there is the boundary E_n, not the wall's alone).
    std::unique_ptr<mfem::H1_FECollection> en_fec;
    std::unique_ptr<mfem::ParFiniteElementSpace> en_fes;
    std::unique_ptr<mfem::HypreParMatrix> en_G, en_M;
    std::unique_ptr<mfem::HypreDiagScale> en_prec;
    std::unique_ptr<mfem::HyprePCG> en_pcg;
    std::vector<char> en_on;
    const bool en_flux = cav.enabled && cav.losses && cav.wall_samples;
    if (en_flux) {
      en_fec = std::make_unique<mfem::H1_FECollection>(P.order, pm.Dimension());
      en_fes = std::make_unique<mfem::ParFiniteElementSpace>(&pm, en_fec.get());
      mfem::ParDiscreteLinearOperator grad(en_fes.get(), &fes);
      grad.AddDomainInterpolator(new mfem::GradientInterpolator());
      grad.Assemble();
      grad.Finalize();
      en_G.reset(grad.ParallelAssemble());
      mfem::ParBilinearForm hm(en_fes.get());
      mfem::Array<int> all_bdr(pm.bdr_attributes.Size() ? pm.bdr_attributes.Max() : 0);
      all_bdr = 1;  // the whole cavity boundary (a node's reaction collects the flux through every face around it)
      hm.AddBoundaryIntegrator(new mfem::MassIntegrator(), all_bdr);
      hm.Assemble();
      {  // interior wall faces (retained conductor volumes): the face mass from the cavity-side element
        mfem::Array<int> vd;
        mfem::Vector shp;
        mfem::DenseMatrix el;
        mfem::Vector nor(3);
        for (const auto& wf : wall_faces) {
          if (wf.boundary) continue;
          mfem::FaceElementTransformations* ft = pm.GetFaceElementTransformations(wf.lf, 1 | 2 | 4 | 8 | 16);
          const bool side1 = ft->Elem1No == wf.e;
          const mfem::FiniteElement& fe = *en_fes->GetFE(wf.e);
          en_fes->GetElementVDofs(wf.e, vd);
          const int nd = fe.GetDof();
          shp.SetSize(nd);
          el.SetSize(nd, nd);
          el = 0.0;
          const auto& fir = mfem::IntRules.Get(ft->GetGeometryType(), 2 * P.order + 2);
          for (int q = 0; q < fir.GetNPoints(); ++q) {
            const mfem::IntegrationPoint& ip = fir.IntPoint(q);
            ft->Face->SetIntPoint(&ip);
            mfem::IntegrationPoint eip;
            if (side1) ft->Loc1.Transform(ip, eip);
            else ft->Loc2.Transform(ip, eip);
            mfem::CalcOrtho(ft->Face->Jacobian(), nor);
            fe.CalcShape(eip, shp);
            mfem::AddMult_a_VVt(ip.weight * nor.Norml2(), shp, el);
          }
          double dmax = 0;
          for (int i = 0; i < nd; ++i) dmax = std::max(dmax, el(i, i));
          for (int i = 0; i < nd; ++i)
            if (!(el(i, i) > 1e-10 * dmax))
              for (int j = 0; j < nd; ++j) el(i, j) = el(j, i) = 0.0;
          hm.SpMat().AddSubMatrix(vd, vd, el, 1);
        }
      }
      hm.Finalize();
      en_M.reset(hm.ParallelAssemble());
      // the boundary nodes: a positive stored diagonal. Not GetDiag: on a GPU the off-boundary rows are empty, and
      // GetDiag would read the next boundary row's diagonal for them, as it did for the wall-current solve above
      // (stored_diagonal, model3d_impl.hpp; docs/validation/gpu.md)
      const mfem::Vector dg = stored_diagonal(*en_M);
      double dmax = 0;
      for (int i = 0; i < dg.Size(); ++i) dmax = std::max(dmax, dg(i));
      dmax = fw_allmax(dmax);
      en_on.assign(dg.Size(), 0);
      mfem::Array<int> off;
      for (int i = 0; i < dg.Size(); ++i) {
        if (dg(i) > 1e-12 * dmax) en_on[i] = 1;
        else off.Append(i);
      }
      ensure_diagonal(en_M, off);  // off the boundary en_M holds only zeros, which hypre on a GPU drops
      en_M->EliminateBC(off, mfem::Operator::DIAG_ONE);
      en_prec = std::make_unique<mfem::HypreDiagScale>(*en_M);
      en_pcg = std::make_unique<mfem::HyprePCG>(*en_M);
      en_pcg->SetTol(1e-13);
      en_pcg->SetMaxIter(2000);
      en_pcg->SetPrintLevel(0);
      en_pcg->SetPreconditioner(*en_prec);
    }
    std::set<std::string> body_names;
    for (auto& f : wall_faces) body_names.insert(f.body);
    {  // the same body list on every rank (collective sums below)
      std::string all;
      for (auto& b : body_names) all += b + '\n';
      int len = static_cast<int>(all.size());
      std::vector<int> lens(P.nranks);
      MPI_Allgather(&len, 1, MPI_INT, lens.data(), 1, MPI_INT, MPI_COMM_WORLD);
      std::vector<int> offs(P.nranks, 0);
      for (int r = 1; r < P.nranks; ++r) offs[r] = offs[r - 1] + lens[r - 1];
      std::string buf(offs.back() + lens.back(), '\0');
      MPI_Allgatherv(all.data(), len, MPI_CHAR, buf.data(), lens.data(), offs.data(), MPI_CHAR, MPI_COMM_WORLD);
      std::string cur;
      for (char c : buf) {
        if (c == '\n') { if (!cur.empty()) body_names.insert(cur); cur.clear(); } else cur += c;
      }
      for (auto& b : body_names)
        if (!body_mat.count(b)) {
          if (b == "outer") body_mat[b] = cav.wall_material.empty() ? nullptr : &d_.materials.at(cav.wall_material);
          else { const spec::Region* g = d_.find_region(b); body_mat[b] = g ? &d_.materials.at(g->material) : nullptr; }
        }
    }
    // ---- region names in the same order on every rank
    std::set<std::string> names;
    for (auto& [a2, nm] : P.region_of_attr) names.insert(nm);
    const auto& ir = mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 2 * P.order);
    std::map<std::string, const spec::Material*> reg_mat;
    for (auto& nm : names) {
      const spec::Region* g = d_.find_region(nm);
      reg_mat[nm] = g ? &d_.materials.at(g->material) : &vacuum_m;
    }
    bool rs_missing_any = false, q0_omitted = false;
    std::map<std::string, std::string> rs_warn;  // RFLoss::warnings by material/key (text of the last mode)
    std::set<std::string> flux_flow_walls;       // walls whose R_s has a flux_flow term
    std::set<std::string> ff_nodir;              // flux_flow walls with points where B0 = 0 (orientation factor 1)
    std::set<std::string> lossless_bodies;
    bool wall_solve_failed = false;  // a wall-current PCG solve missed its residual (not merely G = 0)
    std::vector<std::string> wall_current_failures;  // modes whose wall-current solve gave no usable G
    std::map<std::string, double> g_disagree;  // body -> worst |G_flux/G_pointwise - 1| over the modes
    constexpr double kGDisagreementTol = 0.05;
    // eigenmode field probes (fullwave.field_probes): E and B = curl E/(i omega) of each mode at the points (E
    // normalized as everywhere here, int eps_r |E|^2 = 1; the phase factor 1/i of B is dropped); points outside the
    // cavity (closed/PEC elements, or not found) are flagged invalid and carry zeros
    json cavity_probe_info;
    const int nprobe = static_cast<int>(fw.field_probes.size());
    mfem::Array<int> pr_el;
    mfem::Array<mfem::IntegrationPoint> pr_ip;
    std::vector<int> pr_valid(nprobe, 0);
    if (nprobe > 0) {
      mfem::DenseMatrix pts(3, nprobe);
      for (int j = 0; j < nprobe; ++j) {
        pts(0, j) = fw.field_probes[j].x;
        pts(1, j) = fw.field_probes[j].y;
        pts(2, j) = fw.field_probes[j].z;
      }
      pm.FindPoints(pts, pr_el, pr_ip, false);
      for (int j = 0; j < nprobe; ++j) {
        const int e = pr_el[j];
        if (e >= 0 && !closed[e] && !(P.sh_attr > 0 && pm.GetAttribute(e) == P.sh_attr)) pr_valid[j] = 1;
        else pr_el[j] = -1;
      }
      // a point on a partition boundary may be found by several ranks: keep the lowest rank's
      std::vector<int> owner(nprobe, std::numeric_limits<int>::max());
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      for (int j = 0; j < nprobe; ++j) if (pr_el[j] >= 0) owner[j] = rank;
      MPI_Allreduce(MPI_IN_PLACE, owner.data(), nprobe, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      for (int j = 0; j < nprobe; ++j) if (owner[j] != rank) pr_el[j] = -1;
      MPI_Allreduce(MPI_IN_PLACE, pr_valid.data(), nprobe, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      json cp = json::array();
      for (int j = 0; j < nprobe; ++j)
        cp.push_back({fw.field_probes[j].x, fw.field_probes[j].y, fw.field_probes[j].z});
      cavity_probe_info = {{"points_m", cp}, {"valid", pr_valid},
                           {"note", "per mode: probes.E (int eps_r |E|^2 dV = 1) and probes.B = curl E/omega (the "
                                    "factor 1/i dropped); invalid points (outside the cavity) carry zeros"}};
    }
    for (int k = 0; k < nmodes; ++k) {
      const mfem::Vector& u = X[k];
      if (!(lam[k] > 0) || !std::isfinite(lam[k])) finite_modes = false;
      finite_modes = finite_modes && std::isfinite(rel[k]);
      max_residual = std::max(max_residual, rel[k]);
      const double om = units::c * std::sqrt(std::max(lam[k], 0.0));  // lam = k^2
      mfem::ParGridFunction xk(&fes);
      xk.SetFromTrueDofs(u);
      // electric energy by region (eps-weighted)
      std::map<std::string, double> en;
      double tot = 0;
      double e2_all = 0, e2_phi = 0;  // int eps_r |E|^2 and int eps_r |E_phi|^2 (phi about the z axis)
      mfem::Vector v(3), xq3(3);
      for (int e = 0; e < ne; ++e) {
        if (closed[e] || (P.sh_attr > 0 && pm.GetAttribute(e) == P.sh_attr)) continue;
        mfem::ElementTransformation* T = pm.GetElementTransformation(e);
        double s = 0;
        for (int q = 0; q < ir.GetNPoints(); ++q) {
          T->SetIntPoint(&ir.IntPoint(q));
          xk.GetVectorValue(*T, ir.IntPoint(q), v);
          const double w = eps_e[e].real() * ir.IntPoint(q).weight * T->Weight();
          s += w * (v * v);
          T->Transform(ir.IntPoint(q), xq3);
          const double rho = std::hypot(xq3(0), xq3(1));
          if (rho > 0) {
            const double ep = (-xq3(1) * v(0) + xq3(0) * v(1)) / rho;
            e2_phi += w * ep * ep;
          }
          e2_all += w * (v * v);
        }
        en[P.elem_region[e]] += s;
      }
      std::vector<std::pair<double, std::string>> fr;
      std::map<std::string, double> en_all;
      for (auto& nm2 : names) {
        const double x2 = fw_allsum(en.count(nm2) ? en[nm2] : 0.0);
        en_all[nm2] = x2;
        tot += x2;
        if (x2 > 0) fr.push_back({x2, nm2});
      }
      std::sort(fr.rbegin(), fr.rend());
      json frac = json::object();
      for (size_t i = 0; i < fr.size() && i < 3; ++i) frac[fr[i].second] = fr[i].first / std::max(tot, 1e-300);
      json row = {{"relative_residual", rel[k]}, {"residual_true", rel[k]}, {"mass_norm", 1.0},
                  {"converged", std::isfinite(rel[k]) && rel[k] < 1e-4}, {"f_Hz", om / (2 * pi)},
                  {"k_R", om / units::c * Rdev}, {"energy_fraction", frac}, {"rayleigh_ratio", rayleigh[k]},
                  {"divergence_fraction", divf[k]}};
      if (nprobe > 0) {
        std::vector<double> pe(3 * nprobe, 0.0), pb(3 * nprobe, 0.0);
        mfem::Vector ev(3), cv(3);
        for (int j = 0; j < nprobe; ++j) {
          if (pr_el[j] < 0) continue;
          mfem::ElementTransformation* T = pm.GetElementTransformation(pr_el[j]);
          T->SetIntPoint(&pr_ip[j]);
          xk.GetVectorValue(*T, pr_ip[j], ev);
          xk.GetCurl(*T, cv);
          for (int c = 0; c < 3; ++c) {
            pe[3 * j + c] = ev(c);
            pb[3 * j + c] = om > 0 ? cv(c) / om : 0.0;
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, pe.data(), 3 * nprobe, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, pb.data(), 3 * nprobe, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        json je = json::array(), jb = json::array();
        for (int j = 0; j < nprobe; ++j) {
          je.push_back({pe[3 * j], pe[3 * j + 1], pe[3 * j + 2]});
          jb.push_back({pb[3 * j], pb[3 * j + 1], pb[3 * j + 2]});
        }
        row["probes"] = {{"E", je}, {"B", jb}};
      }
      if (cav.enabled) {
        // share of the electric energy in the azimuthal component about the z axis (through the origin): 0 for
        // TM0np-like (m = 0, E in the meridional plane), 1 for TE0np-like modes of an axisymmetric cavity on z
        {
          const double a2 = fw_allsum(e2_all), p2 = fw_allsum(e2_phi);
          if (a2 > 0) row["E_azimuthal_fraction"] = p2 / a2;
        }
        // ---- form factor C = |int E.B0 dV|^2 / (B_ref^2 V int eps_r |E|^2 dV)  (u: int eps_r |E|^2 = 1)
        json ff = {{"V_m3", V_cav}, {"B_ref_T", B_ref},
                   {"note", "C = |int E.B0 dV|^2 / (B_ref^2 V int eps_r |E|^2 dV), V = cavity volume (conductors "
                            "excluded); C_rms uses int |B0|^2 dV in place of B_ref^2 V; B0 = fullwave.source"}};
        if (b0src) {
          const double OEB = fw_gdot(b0form, u);
          ff["overlap_EB_T_m32"] = std::fabs(OEB);
          const double Brms = V_cav > 0 ? std::sqrt(B2 / V_cav) : 0.0;
          if (B_ref > 1e-6 * std::max(Brms, 1e-300)) ff["C"] = OEB * OEB / (B_ref * B_ref * V_cav);
          if (B2 > 0) ff["C_rms"] = OEB * OEB / B2;
        }
        row["form_factor"] = ff;
        if (cav.losses) {
          json lj = json::object();
          // dielectric: Q_d^-1 = sum tan_eff W_e,region / W_e
          double inv_Qd = 0;
          json dl = json::object();
          for (auto& [rg, we] : en_all) {
            if (!(we > 0)) continue;
            const spec::Material* mt = reg_mat.at(rg);
            const double er = mt->eps_r > 0 ? mt->eps_r : 1.0;
            const double td = mt->tan_delta + (om > 0 ? mt->sigma_S_m / (om * eps0 * er) : 0.0);
            if (!(td > 0)) continue;
            const double fr_e = we / std::max(tot, 1e-300);
            inv_Qd += td * fr_e;
            dl[rg] = {{"tan_delta_eff", td}, {"electric_energy_fraction", fr_e}, {"Q", 1.0 / (td * fr_e)}};
          }
          // walls: G_b = omega mu0 int (1/mu_r)|curl E|^2 / oint_b |n x curl E / mu_r|^2  (= Q_b R_s)
          // surface current k = n x (curl E/mu_r) from the weak reaction (A0 - lam M0) u on the PEC boundary DOFs
          mfem::Vector r(u.Size()), t(u.Size()), kt(u.Size());
          A0->Mult(u, r);
          M0->Mult(u, t);
          const double uKu = fw_gdot(u, r);
          r.Add(-lam[k], t);
          for (int i = 0; i < r.Size(); ++i)
            if (!on_wall[i]) r(i) = 0.0;
          kt = 0.0;
          ms_pcg->Mult(r, kt);
          int ms_its = 0;
          ms_pcg->GetNumIterations(ms_its);
          double ms_rel = 0.0;
          {  // the true residual of the wall-current solve: a failed solve must not pass silently
            mfem::Vector ms_res(kt.Size());
            Ms->Mult(kt, ms_res);
            ms_res -= r;
            const double rn = std::sqrt(fw_gdot(r, r));
            ms_rel = rn > 0 ? std::sqrt(fw_gdot(ms_res, ms_res)) / rn : 0.0;
          }
          lj["wall_current_solve"] = {{"pcg_iterations", ms_its}, {"relative_residual", ms_rel}};
          mfem::ParGridFunction kg(&fes);
          kg.SetFromTrueDofs(kt);
          // consistent-flux normal E on the cavity boundary: eps_r E_n = M_bdr^-1 G^T M0 u (t = M0 u above)
          std::unique_ptr<mfem::ParGridFunction> engp;
          bool en_converged = true;
          if (en_flux) {
            engp = std::make_unique<mfem::ParGridFunction>(en_fes.get());
            mfem::ParGridFunction& eng = *engp;
            mfem::Vector bq(en_G->Width()), qn(en_G->Width());
            en_G->MultTranspose(t, bq);
            for (int i = 0; i < bq.Size(); ++i)
              if (!en_on[i]) bq(i) = 0.0;
            qn = 0.0;
            en_pcg->Mult(bq, qn);
            int en_it = 0;
            en_pcg->GetNumIterations(en_it);
            en_converged = en_it < 2000;  // the PCG's SetMaxIter
            eng.SetFromTrueDofs(qn);
          }
          std::map<std::string, double> gam, gam_pt, gam_mr;
          // flux_flow walls: |H_t|^2-weighted vortex R_s with the local |B0| and orientation s = 1 - (J.B0)^2/|B0|^2
          // (J along the consistent-flux surface current k), as in the 2D cavity (docs/physics/surface_resistance.md)
          std::map<std::string, double> gam_ff, gam_ff_perp, gam_ff_s, ff_nodir_n;
          mfem::Vector cu(3), kv(3), nor(3), ew(3);
          // peak wall fields over the wall quadrature points: |E| (normal on a PEC wall), |curl E|_t / omega
          // (pointwise; = |B_t| in vacuum) and the consistent-flux |k_t| / omega
          double pk_E = 0, pk_B = 0, pk_Bk = 0, pk_Ef = 0, pk_Bf = 0, pk_En = 0, pk_Enf = 0, pk_Bkf = 0;
          double pk_E_x[3] = {0, 0, 0}, pk_B_x[3] = {0, 0, 0}, pk_En_x[3] = {0, 0, 0}, pk_Bk_x[3] = {0, 0, 0};
          mfem::Vector xpk(3);
          std::vector<double> wsamp;  // cavity.wall_samples: x, y, z, dA, |E|, |E_n| flux, |curl E|_t, |k_t| per point
          for (const auto& wf : wall_faces) {
            mfem::FaceElementTransformations* ft =
                wf.boundary ? pm.GetBdrFaceTransformations(wf.lf) : pm.GetFaceElementTransformations(wf.lf, 1 | 2 | 4 | 8 | 16);
            if (!ft) continue;
            const bool side1 = ft->Elem1No == wf.e;
            if (!side1 && wf.boundary) continue;
            const double mur = 1.0 / (mu0 * nu_e[wf.e].real());
            const spec::Material* bm = body_mat[wf.body];
            const bool mr_on = bm && !bm->rf.mr_model.empty() && b0src;
            const bool ff_on = bm && bm->rf.flux_flow;
            const auto& fir = mfem::IntRules.Get(ft->GetGeometryType(), 2 * P.order + 2);
            const double er_e = std::max(one_eps[wf.e].real() / eps0, 1e-300);
            double fEn2 = 0, fBk2 = 0;
            double fA = 0, fE2 = 0, fB2 = 0;  // face means of |E|^2 and |curl E|_t^2 (peak estimate robust to
                                              // single quadrature-point outliers)
            for (int q = 0; q < fir.GetNPoints(); ++q) {
              const mfem::IntegrationPoint& ip = fir.IntPoint(q);
              ft->Face->SetIntPoint(&ip);
              mfem::IntegrationPoint eip;
              mfem::ElementTransformation* Te;
              if (side1) { ft->Loc1.Transform(ip, eip); Te = ft->Elem1; }
              else { ft->Loc2.Transform(ip, eip); Te = ft->Elem2; }
              Te->SetIntPoint(&eip);
              mfem::CalcOrtho(ft->Face->Jacobian(), nor);
              const double dA = ip.weight * nor.Norml2();
              nor /= nor.Norml2();
              xk.GetCurl(*Te, cu);
              cu /= mur;
              const double cn = cu * nor;
              const double ht_pt = std::max(0.0, (cu * cu) - cn * cn);
              kg.GetVectorValue(*Te, eip, kv);
              const double kn = kv * nor;
              const double ht = std::max(0.0, (kv * kv) - kn * kn);
              xk.GetVectorValue(*Te, eip, ew);
              const double en = ew.Norml2(), bt = std::sqrt(ht_pt) * mur;
              if (en > pk_E) {
                pk_E = en;
                ft->Face->Transform(ip, xpk);
                for (int c = 0; c < 3; ++c) pk_E_x[c] = xpk(c);
              }
              if (bt > pk_B) {
                pk_B = bt;
                ft->Face->Transform(ip, xpk);
                for (int c = 0; c < 3; ++c) pk_B_x[c] = xpk(c);
              }
              const double bk = std::sqrt(ht) * mur;
              if (bk > pk_Bk) {
                pk_Bk = bk;
                ft->Face->Transform(ip, xpk);
                for (int c = 0; c < 3; ++c) pk_Bk_x[c] = xpk(c);
              }
              const double enf = engp ? std::fabs(engp->GetValue(*Te, eip)) / er_e : 0.0;
              if (enf > pk_En) {
                pk_En = enf;
                ft->Face->Transform(ip, xpk);
                for (int c = 0; c < 3; ++c) pk_En_x[c] = xpk(c);
              }
              fEn2 += dA * enf * enf;
              fBk2 += dA * bk * bk;
              if (cav.wall_samples) {
                ft->Face->Transform(ip, xpk);
                wsamp.insert(wsamp.end(), {xpk(0), xpk(1), xpk(2), dA, en, enf, bt, bk});
              }
              fA += dA;
              fE2 += dA * en * en;
              fB2 += dA * bt * bt;
              gam_pt[wf.body] += dA * ht_pt;
              gam[wf.body] += dA * ht;
              if (mr_on || ff_on) {
                mfem::Vector xq(3);
                ft->Face->Transform(ip, xq);
                std::array<double, 3> b{0.0, 0.0, 0.0};
                if (b0src) {
                  const auto bb = b0src->field_T({xq(0), xq(1), xq(2)});
                  b = {bb[0], bb[1], bb[2]};
                }
                const double Bm = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
                if (mr_on) gam_mr[wf.body] += dA * ht * bm->rf.Rs_factor(Bm);
                if (ff_on) {
                  const double Bv = bm->rf.ff_B_local() ? Bm : std::fabs(bm->rf.ff_B);
                  AXF_REQUIRE(Bv < bm->rf.ff_Bc2,
                              strcat("flux_flow wall '", wf.body, "': static field ", Bv, " T at (", xq(0), ", ", xq(1),
                                     ", ", xq(2), ") is above Bc2_T = ", bm->rf.ff_Bc2, " T (the wall would be normal)"),
                              "/materials/" + bm->name + "/rf/flux_flow", "Use a normal-metal rf model for this wall.");
                  // tangential surface current k_t = k - (k.n) n: the RF current direction
                  double jt[3];
                  for (int c = 0; c < 3; ++c) jt[c] = kv(c) - kn * nor(c);
                  const double jm = std::sqrt(jt[0] * jt[0] + jt[1] * jt[1] + jt[2] * jt[2]);
                  double sf;
                  if (bm->rf.ff_orientation >= 0) {
                    sf = bm->rf.ff_orientation;
                  } else if (Bm > 1e-9 * std::max(B_ref, 1e-300) && Bm > 0 && jm > 0) {
                    const double jb = (jt[0] * b[0] + jt[1] * b[1] + jt[2] * b[2]) / (jm * Bm);
                    sf = std::clamp(1 - jb * jb, 0.0, 1.0);
                  } else {
                    sf = 1;  // no static field (or no current) at this point: direction undefined, upper bound
                    if (!(Bm > 1e-9 * std::max(B_ref, 1e-300) && Bm > 0)) ff_nodir_n[wf.body] += 1.0;
                  }
                  const double wh = dA * ht;
                  gam_ff[wf.body] += wh * bm->rf.Rs_flux_flow(om / (2 * pi), Bv, sf);
                  gam_ff_perp[wf.body] += wh * bm->rf.Rs_flux_flow(om / (2 * pi), Bv, 1.0);
                  gam_ff_s[wf.body] += wh * sf;
                }
              }
            }
            if (fA > 0) {
              pk_Ef = std::max(pk_Ef, std::sqrt(fE2 / fA));
              pk_Bf = std::max(pk_Bf, std::sqrt(fB2 / fA));
              pk_Enf = std::max(pk_Enf, std::sqrt(fEn2 / fA));
              pk_Bkf = std::max(pk_Bkf, std::sqrt(fBk2 / fA));
            }
          }
          double inv_Qw = 0, gsum = 0, TQ = 0;
          bool any_rf = false, rs_missing = false;
          std::vector<std::string> lossless;
          json bj = json::object();
          for (auto& b : body_names) {
            const double G = fw_allsum(gam.count(b) ? gam[b] : 0.0);
            const double Gp = fw_allsum(gam_pt.count(b) ? gam_pt[b] : 0.0);
            const double Gm = fw_allsum(gam_mr.count(b) ? gam_mr[b] : 0.0);
            if (!(G > 0)) continue;
            gsum += G;
            json e = {{"G_ohm", om * mu0 * uKu / G}, {"G_ohm_pointwise", om * mu0 * uKu / std::max(Gp, 1e-300)}};
            // a-posteriori resolution indicator: the consistent-flux and pointwise currents converge from opposite
            // sides near edges (rod tips, gaps); their disagreement bounds the error of G on this body
            const double dis = Gp > 0 ? std::fabs(G / Gp - 1) : 1.0;
            e["G_flux_pointwise_disagreement"] = dis;
            if (dis > kGDisagreementTol) {
              e["resolution_warning"] = true;
              g_disagree[b] = std::max(g_disagree[b], dis);
            }
            const spec::Material* mt = body_mat[b];
            double Rs0 = 0;
            bool miss = false;
            if (mt && !mt->rf.model.empty()) {
              try {  // the shared R_s models (axf::rs via spec::RFLoss), without the vortex term (added per point)
                Rs0 = mt->rf.Rs_base(om / (2 * pi));
                for (auto& [wk, wt] : mt->rf.warnings(om / (2 * pi))) rs_warn[mt->rf.owner + "/" + wk] = wt;
              } catch (const std::exception& ex) {
                e["note"] = std::string("R_s not available: ") + ex.what();
                rs_missing = miss = true;
              }
            }
            const bool ffw = mt && mt->rf.flux_flow && !miss;
            if (ffw && fw_allsum(ff_nodir_n.count(b) ? ff_nodir_n[b] : 0.0) > 0) ff_nodir.insert(b);
            const double Rff = ffw ? fw_allsum(gam_ff.count(b) ? gam_ff[b] : 0.0) / G : 0.0;
            const double Rs = Rs0 * (mt && !mt->rf.mr_model.empty() && b0src ? Gm / G : 1.0) + Rff;
            if (Rs > 0) {
              const double Qb = om * mu0 * uKu / (Rs * G);
              e["Rs_ohm"] = Rs;
              e["Rs_model"] = mt->rf.model;
              if (ffw) {  // order-of-magnitude model, reported separately
                e["Rs_flux_flow_ohm"] = Rff;
                e["Rs_flux_flow_perpendicular_ohm"] = fw_allsum(gam_ff_perp.count(b) ? gam_ff_perp[b] : 0.0) / G;
                e["flux_flow_orientation_factor"] = fw_allsum(gam_ff_s.count(b) ? gam_ff_s[b] : 0.0) / G;
                flux_flow_walls.insert(b);
              }
              e["Q"] = Qb;
              e["T_K"] = mt->rf.T_K;
              inv_Qw += 1.0 / Qb;
              TQ += mt->rf.T_K / Qb;
              any_rf = true;
            } else if (!miss) {
              e["note"] = cav.allow_lossless_walls ? "no rf data: lossless (cavity.allow_lossless_walls)"
                                                   : "no rf data: lossless";
              lossless.push_back(b);
            }
            bj[b] = e;
          }
          if (any_rf)
            for (auto it = bj.begin(); it != bj.end(); ++it)
              if (it->contains("Q")) (*it)["P_frac"] = (1.0 / (*it)["Q"].get<double>()) / inv_Qw;
          lj["walls"] = bj;
          if (gsum > 0) lj["G_ohm"] = om * mu0 * uKu / gsum;
          if (!body_names.empty() && !(gsum > 0 && std::isfinite(gsum) && ms_rel <= 1e-6))
            wall_current_failures.push_back(
                strcat(k, " (relative residual ", ms_rel, ", ", ms_its, " PCG iterations)"));
          if (!body_names.empty() && !(ms_rel <= 1e-6)) wall_solve_failed = true;   // ms_rel is global
          {
            // B = curl E / (i omega); B_rms^2 = int |curl E|^2 / (omega^2 V) (mu_r = 1 in the cavity assumed)
            const double gE = fw_allmax_at(pk_E, pk_E_x), gB = fw_allmax_at(pk_B, pk_B_x) / std::max(om, 1e-300),
                         gBk = fw_allmax_at(pk_Bk, pk_Bk_x) / std::max(om, 1e-300), gEf = fw_allmax(pk_Ef),
                         gBf = fw_allmax(pk_Bf) / std::max(om, 1e-300), gEn = fw_allmax_at(pk_En, pk_En_x),
                         gEnf = fw_allmax(pk_Enf), gBkf = fw_allmax(pk_Bkf) / std::max(om, 1e-300);
            const double Brms = V_cav > 0 && om > 0 ? std::sqrt(std::max(uKu, 0.0) / V_cav) / om : 0.0;
            json pw = {{"E_max", gE}, {"B_t_max", gB}, {"B_t_max_flux", gBk}, {"B_rms", Brms},
                       {"E_max_face_rms", gEf}, {"B_t_max_face_rms", gBf},
                       {"E_max_at", {pk_E_x[0], pk_E_x[1], pk_E_x[2]}}, {"B_t_max_at", {pk_B_x[0], pk_B_x[1], pk_B_x[2]}},
                       {"B_t_max_flux_face_rms", gBkf}, {"B_t_max_flux_at", {pk_Bk_x[0], pk_Bk_x[1], pk_Bk_x[2]}},
                       {"note_flux", "E_n_max_flux: largest |E_n| over the same wall quadrature points from the consistent-flux "
                                     "(charge) recovery eps_r E_n = M_bdr^-1 G^T M u (an H1 field on the whole cavity "
                                     "boundary; a diagnostic: on the SRF cell it converges no better than the pointwise "
                                     "field, and it mixes E_n across boundary corners); B_t_max_flux from the "
                                     "consistent-flux surface current"},
                       {"note", "largest |E| and |B_t| over the wall quadrature points (E normalized to int "
                                "eps_r |E|^2 dV = 1; B = curl E/(i omega) pointwise, B_t_max_flux from the "
                                "consistent-flux current; *_face_rms: largest face rms, robust to single-point outliers; *_at: "
                                "location of the pointwise maximum); B_rms over the (meshed) cavity volume, mu_r = 1. "
                                "cEpk_over_Brms = E_max/(c B_rms): E_pk in units of B_rms with c = 1"}};
            if (Brms > 0) {
              pw["Bpk_over_Brms"] = gB / Brms;
              pw["Bpk_flux_over_Brms"] = gBk / Brms;
              pw["cEpk_over_Brms"] = gE / (units::c * Brms);
              pw["Bpk_face_rms_over_Brms"] = gBf / Brms;
              pw["cEpk_face_rms_over_Brms"] = gEf / (units::c * Brms);
              if (en_flux) {
                pw["E_n_flux_solve_converged"] = en_converged;
                pw["E_n_max_flux"] = gEn;
                pw["E_n_max_flux_face_rms"] = gEnf;
                pw["E_n_max_flux_at"] = {pk_En_x[0], pk_En_x[1], pk_En_x[2]};
                pw["cEpk_flux_over_Brms"] = gEn / (units::c * Brms);
                pw["cEpk_flux_face_rms_over_Brms"] = gEnf / (units::c * Brms);
              }
              pw["Bpk_flux_face_rms_over_Brms"] = gBkf / Brms;
            }
            if (cav.wall_samples) {  // gather to rank 0 and write raw doubles (B columns are curl E: divide by omega)
              int cnt = static_cast<int>(wsamp.size());
              std::vector<int> cnts(P.nranks), offs(P.nranks, 0);
              MPI_Gather(&cnt, 1, MPI_INT, cnts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
              std::vector<double> all;
              if (P.rank == 0) {
                for (int q = 1; q < P.nranks; ++q) offs[q] = offs[q - 1] + cnts[q - 1];
                all.resize(static_cast<size_t>(offs.back()) + cnts.back());
              }
              MPI_Gatherv(wsamp.data(), cnt, MPI_DOUBLE, all.data(), cnts.data(), offs.data(), MPI_DOUBLE, 0,
                          MPI_COMM_WORLD);
              const auto path = std::filesystem::absolute(std::filesystem::path(o.workdir) / strcat("wall_samples_", k, ".f64"));
              if (P.rank == 0) {
                for (size_t q = 0; q + 7 < all.size(); q += 8) {  // B columns in field units
                  all[q + 6] /= std::max(om, 1e-300);
                  all[q + 7] /= std::max(om, 1e-300);
                }
                std::ofstream fo(path, std::ios::binary);
                fo.write(reinterpret_cast<const char*>(all.data()), static_cast<std::streamsize>(all.size() * sizeof(double)));
              }
              pw["samples_file"] = path.string();
              {
                long long tot = cnt / 8, all_tot = 0;
                MPI_Allreduce(&tot, &all_tot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
                pw["samples_count"] = all_tot;
                pw["samples_f_Hz"] = om / (2 * pi);
              }
              pw["samples_columns"] = {"x", "y", "z", "dA", "E", "E_n_flux", "B_t", "B_t_flux"};
              pw["samples_note"] = "raw float64, 8 columns per wall quadrature point: x, y, z [m], dA [m^2], |E| (the "
                                   "normal E on a PEC wall: its tangential part vanishes there), |E_n| from the "
                                   "consistent-flux recovery, |B_t| = |curl E|_t/omega, |B_t| from the consistent-flux "
                                   "current (E normalized to int eps_r |E|^2 = 1); rewritten by every solve in this workdir";
            }
            row["peak_wall_fields"] = pw;
          }
          if (any_rf && !rs_missing) {
            lj["Q_wall"] = 1.0 / inv_Qw;
            lj["T_over_Q_K"] = TQ;
          }
          if (!dl.empty()) {
            lj["dielectric"] = dl;
            lj["Q_dielectric"] = 1.0 / inv_Qd;
          }
          if (!lossless.empty()) {
            lj["lossless_walls"] = lossless;
            if (!cav.allow_lossless_walls) for (auto& b : lossless) lossless_bodies.insert(b);
          }
          if (cav.allow_lossless_walls) lj["walls_lossless_by_request"] = !lossless.empty();
          // Q0 needs every wall's loss: with no wall rf data at all, Q0 would be Q_dielectric alone, reported only
          // when lossless walls are explicitly allowed
          if (rs_missing) rs_missing_any = true;
          else if (any_rf || (inv_Qd > 0 && cav.allow_lossless_walls)) lj["Q0"] = 1.0 / (inv_Qw + inv_Qd);
          else if (inv_Qd > 0) q0_omitted = true;
          lj["note"] = "Q0 = omega U / P; walls: perturbative R_s(f) with H_t of the lossless mode; G_ohm = Q R_s "
                       "(geometry factor; the total assumes one R_s on all walls). H_t from the consistent-flux "
                       "surface current on PEC boundary faces and retained-conductor faces (G_ohm_pointwise: the "
                       "pointwise curl, a check; G_flux_pointwise_disagreement > 0.05 flags an unresolved body); "
                       "symmetry planes carry no loss";
          row["losses"] = lj;
        }
      }
      list.push_back(row);
    }
    for (auto& [wk, wt] : rs_warn) warnings.push_back("cavity: " + wt);
    if (!flux_flow_walls.empty()) {
      std::string nm;
      for (auto& b : flux_flow_walls) nm += (nm.empty() ? "" : ", ") + b;
      warnings.push_back(strcat("cavity: the R_s of walls ", nm, " includes a flux_flow (vortex) term "
                                "(Rs_flux_flow_ohm): an order-of-magnitude model with large uncertainty, weighted "
                                "per wall point by |H_t|^2 and the orientation factor sin^2(J_rf, B0) (B0 = "
                                "fullwave.source; Rs_flux_flow_perpendicular_ohm: without it). Current parallel to "
                                "the vortices is assumed lossless; misalignment, roughness and vortex tilt make it "
                                "lossy in practice (docs/physics/surface_resistance.md Sec. 4)"));
    }
    if (!ff_nodir.empty()) {
      std::string nm;
      for (auto& b : ff_nodir) nm += (nm.empty() ? "" : ", ") + b;
      warnings.push_back("cavity: flux_flow walls " + nm + " have points where the static field vanishes; the "
                         "orientation factor is set to 1 there");
    }
    if (rs_missing_any)
      warnings.push_back("cavity: some modes have walls without R_s at their frequency (see losses.walls); Q_wall and "
                         "Q0 are not reported for them");
    if (!lossless_bodies.empty()) {
      std::string nm;
      for (auto& b : lossless_bodies) nm += (nm.empty() ? "" : ", ") + b;
      warnings.push_back("cavity: the walls " + nm + " have no rf data and are lossless: Q_wall, Q0 and P_frac leave "
                         "out their loss (see losses.lossless_walls; for the domain wall set cavity.wall_material). "
                         "Give them an rf model, or set cavity.allow_lossless_walls if this is intended");
    }
    if (!wall_current_failures.empty()) {
      std::string nm;
      for (auto& m : wall_current_failures) nm += (nm.empty() ? "" : ", ") + m;
      warnings.push_back("cavity: the wall-current solve failed for mode(s) " + nm + ": G, Q0, Q_wall and the walls' "
                         "losses are missing or wrong for them (losses.wall_current_solve)." +
                         (wall_solve_failed ? worst_elements_note(P) : std::string()));
    }
    if (q0_omitted)
      warnings.push_back("cavity: no wall has rf data, so Q0 is not reported (it would be Q_dielectric alone); "
                         "Q_dielectric is. Set cavity.wall_material (with rf data) or cavity.allow_lossless_walls");
    if (!g_disagree.empty()) {
      std::string nm;
      for (auto& [b, v] : g_disagree) nm += (nm.empty() ? "" : ", ") + b + strcat(" (", v, ")");
      warnings.push_back(strcat("cavity: the consistent-flux and pointwise wall currents disagree by more than ",
                                kGDisagreementTol, " on ", nm, ": G and Q of those bodies are not resolved (typically "
                                "rod tips, gaps and edges). Refine there (the region's surface_size, or a small "
                                "region with mesh_size around the gap) until losses.walls[].G_flux_pointwise_"
                                "disagreement is small"));
    }
    // ---- mode overlaps eta_ij = |int E_i . B_j| / sqrt(int |E_i|^2 int |B_j|^2), B_j = curl E_j / (i omega_j)
    json overlaps;
    if (cav.enabled && cav.mode_overlaps && nmodes > 0) {
      mfem::ParMixedBilinearForm cform(&fes, &fes);
      cform.AddDomainIntegrator(new mfem::MixedVectorCurlIntegrator());
      cform.Assemble();
      cform.Finalize();
      std::unique_ptr<mfem::HypreParMatrix> Cm(cform.ParallelAssemble());
      mfem::ParBilinearForm m1(&fes), k1(&fes);
      m1.AddDomainIntegrator(new mfem::VectorFEMassIntegrator());
      k1.AddDomainIntegrator(new mfem::CurlCurlIntegrator());
      m1.Assemble(); m1.Finalize(); k1.Assemble(); k1.Finalize();
      std::unique_ptr<mfem::HypreParMatrix> M1(m1.ParallelAssemble()), K1(k1.ParallelAssemble());
      std::vector<mfem::Vector> CX(nmodes);
      std::vector<double> ee(nmodes), bb(nmodes), ee_w(nmodes), bb_w(nmodes);
      mfem::Vector t(fes.GetTrueVSize());
      for (int j = 0; j < nmodes; ++j) {
        CX[j].SetSize(t.Size());
        Cm->Mult(X[j], CX[j]);
        M1->Mult(X[j], t);
        ee[j] = fw_gdot(X[j], t);
        K1->Mult(X[j], t);
        bb[j] = fw_gdot(X[j], t);
        ee_w[j] = fw_gdot(X[j], MX[j]);  // int eps_r |E|^2 (= 1)
        A->Mult(X[j], t);
        bb_w[j] = fw_gdot(X[j], t);      // int |curl E|^2 / mu_r
      }
      // E.B is a pseudoscalar: over the mirror images (both modes of one model share the signs s_g) the full
      // cavity integral is (1/n) sum_g det(R_g) times the meshed part's, which vanishes for any symmetry plane
      double img_factor = 0;
      for (const auto& g : images) img_factor += g[0] * g[1] * g[2] / images.size();
      json eta = json::array(), eta_part = json::array(), eta_w = json::array();
      for (int i = 0; i < nmodes; ++i) {
        json rowj = json::array(), rowp = json::array(), roww = json::array();
        for (int j = 0; j < nmodes; ++j) {
          const double xc = std::fabs(fw_gdot(X[i], CX[j]));
          const double v = xc / std::sqrt(std::max(ee[i] * bb[j], 1e-300));
          rowj.push_back(std::fabs(img_factor) * v);
          rowp.push_back(v);
          roww.push_back(std::fabs(img_factor) * xc / std::sqrt(std::max(ee_w[i] * bb_w[j], 1e-300)));
        }
        eta.push_back(rowj);
        eta_part.push_back(rowp);
        eta_w.push_back(roww);
      }
      if (images.size() > 1) {
        overlaps["eta_meshed_part"] = eta_part;
        overlaps["symmetry_note"] = "under mesh.symmetry all modes of the model share the mirror parities, so the "
                                    "full-cavity overlap vanishes identically (E.B is odd under each mirror); eta of "
                                    "opposite-parity pairs needs a model that holds both (fewer symmetry planes)";
      }
      overlaps["eta"] = eta;
      overlaps["eta_energy"] = eta_w;
      overlaps.update(json{
                  {"definition", "eta[i][j] = |int E_i . B_j dV| / sqrt(int |E_i|^2 dV int |B_j|^2 dV), B_j = curl E_j "
                                 "/ (i omega_j), unweighted L2 norms over the cavity (rows: E of mode i, columns: B of "
                                 "mode j; modes in the order of the list). eta_energy: the same overlap normalized by "
                                 "the stored energies, sqrt(int eps_r |E_i|^2 dV int |B_j|^2 / mu_r dV); equal to eta "
                                 "in vacuum, the natural choice with dielectrics"}});
    }
    // eigenvectors are kept only for cavity designs (modes x DOFs doubles; nothing reads them otherwise)
    if (cav.enabled) {
      P.fw_modes = std::move(X);
      P.fw_mode_k2 = lam;
    } else {
      P.fw_modes.clear();
      P.fw_mode_k2.clear();
    }
    int missing_in_window = 0, near_partner = 0;
    if (solver.contains("completeness_check")) {
      missing_in_window = solver["completeness_check"].value("missing_in_window", 0);
      near_partner = solver["completeness_check"].value("near_degenerate_outside_window", 0);
    }
    if (missing_in_window > 0)
      warnings.push_back(strcat("3D eigenmodes: the completeness check found ", missing_in_window, " Ritz value(s) "
                                "inside the returned window that the main Lanczos run did not return (see "
                                "eigenmodes.solver.completeness_check); rerun with more modes or a different "
                                "target_f_Hz"));
    if (near_partner > 0)
      warnings.push_back(strcat("3D eigenmodes: ", near_partner, " mode(s) within 1e-4 (in k^2) of a returned mode lie "
                                "just outside the returned window: a near-degenerate pair is cut by the window edge; "
                                "request more modes to get both members"));
    const bool complete = finite_modes && nmodes == fw.modes && max_residual < 1e-4 && max_overlap < 1e-4 &&
                          (!shift_invert || solver.value("converged", false)) && missing_in_window == 0;
    if (!complete) warnings.push_back("3D eigenmodes incomplete or nonfinite, or residual/orthogonality gate failed; inspect requested/returned counts and residual diagnostics. Resolve eigensolver convergence before using these modes.");
    solver["seconds"] = fw_allmax(eig_timer.seconds());
    out["eigenmodes"] = {{"modes", list}, {"requested", fw.modes}, {"returned", nmodes},
                         {"converged", complete}, {"max_relative_residual", max_residual},
                         {"max_mass_overlap", max_overlap}, {"residual_tolerance", 1e-4}, {"solver", solver},
                         {"note", cav.enabled ? "modes of the cavity component (closed off elsewhere); E normalized to "
                                                "int eps_r |E|^2 dV = 1 (k^2-scaled units)"
                                              : "closed perfectly conducting domain boundary; perfect conductors "
                                                "eliminated, readout port open (all azimuthal orders)"}};
    if (shift_invert) out["eigenmodes"]["target_f_Hz"] = fw.target_f_Hz;
    if (!overlaps.is_null()) out["eigenmodes"]["overlaps"] = overlaps;
    if (!cavity_probe_info.is_null()) out["eigenmodes"]["field_probes"] = cavity_probe_info;
    if (cav.enabled) out["cavity"] = cav_json;
  }
  // what the allocator did for this analysis (a sweep, shift-invert eigenmodes, or both); the per-frequency records and
  // after_sweep are in fullwave.memory too when there was a sweep
  const json allocator = malloc_scope.Record(trim_mode, thresholds_mode);   // a named json: items() of a temporary dangles
  for (const auto& [key, value] : allocator.items()) out["memory"][key] = value;
  out["seconds"] = fw_allmax(timer.seconds());
  R["fullwave"] = out;
} catch (const SolverFailure& failure) {
  std::string note;
  try {
    note = worst_elements_note(*p_, "");
  } catch (const std::exception& e) {  // the original failure must survive a failing check
    note = strcat(" (The mesh check failed: ", e.what(), ")");
  }
  if (failure.hint().empty()) note.erase(0, 1);  // the leading space
  throw Error(failure.message(), failure.where(), failure.hint() + note);
}

}  // namespace axf
