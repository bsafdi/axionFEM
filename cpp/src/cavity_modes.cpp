// Fields of the axisymmetric cavity eigenmodes (designs with a `cavity` section; beta, docs/validation/cavity_2d.md).
//
// The eigenvectors of the cavity analysis are kept in AxiModel::fw_modes_ (TE: u = A_phi, TM: u = H_phi; e^{-i omega t}).
// Complex fields of a mode:
//   TM: H = u phi^,               E = (i / (omega eps)) curl(u phi^)
//   TE: E = i omega u phi^,        H = curl(u phi^) / mu
// so both are (i E, H) with real E, H. They are scaled to int eps_r |E|^2 dV = 1 over the cavity (the normalization of
// the form factor), which makes int mu_r |H|^2 dV = eps0 / mu0 = 1/Z0^2 at resonance (equal electric and magnetic
// energies; a check, not imposed).
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>

#include "axionfem/core.hpp"
#include "axionfem/model.hpp"
#include "axionfem/units.hpp"

namespace axf {

using namespace axi;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct ModePt {
  double E[3] = {0, 0, 0}, H[3] = {0, 0, 0}, flux = 0, eps_r = 1, mu_r = 1;
};

// Unscaled fields (eigenvector units) of mode m at a point of a cavity element e.
void eval_mode(const AxiWave& wv, bool tm, const Eigenmode& m, int e, const PointEval& pe, ModePt& p) {
  const double eps0 = 1.0 / (units::mu0 * units::c * units::c);
  const WaveRole& ro = wv.roles()[e];
  const auto& dd = wv.edofs(e);
  double u = 0, dr = 0, dz = 0;
  for (size_t i = 0; i < dd.size(); ++i) {
    u += m.x[dd[i]] * pe.N[i];
    dr += m.x[dd[i]] * pe.dN[2 * i];
    dz += m.x[dd[i]] * pe.dN[2 * i + 1];
  }
  const double cbr = -dz, cbz = dr + (pe.r > 0 ? u / pe.r : dr);  // curl(u phi^) = (r, z) components
  const double al = ro.alpha_at ? ro.alpha_at(e, pe) : ro.alpha;
  const double be = ro.beta_at ? ro.beta_at(e, pe) : ro.beta;
  const double w = m.omega;
  p = ModePt();
  p.flux = pe.r * u;
  if (tm) {  // alpha = 1/eps, beta = mu
    const double eps = 1.0 / al;
    p.eps_r = eps / eps0;
    p.mu_r = be / units::mu0;
    p.H[1] = u;
    p.E[0] = cbr / (w * eps);
    p.E[2] = cbz / (w * eps);
  } else {  // alpha = 1/mu, beta = eps
    p.eps_r = be / eps0;
    p.mu_r = 1.0 / (al * units::mu0);
    p.E[1] = w * u;
    p.H[0] = cbr * al;
    p.H[2] = cbz * al;
  }
}

// Sub-triangle lattice of the reference triangle: points (i/k, j/k).
void lattice(int k, std::vector<mfem::IntegrationPoint>& pts, std::vector<std::array<int, 3>>& tris) {
  std::map<std::pair<int, int>, int> id;
  for (int j = 0; j <= k; ++j)
    for (int i = 0; i + j <= k; ++i) {
      mfem::IntegrationPoint ip;
      ip.Set2(double(i) / k, double(j) / k);
      ip.weight = 0;
      id[{i, j}] = static_cast<int>(pts.size());
      pts.push_back(ip);
    }
  for (int j = 0; j < k; ++j)
    for (int i = 0; i + j < k; ++i) {
      tris.push_back({id[{i, j}], id[{i + 1, j}], id[{i, j + 1}]});
      if (i + j + 1 < k) tris.push_back({id[{i + 1, j}], id[{i + 1, j + 1}], id[{i, j + 1}]});
    }
}

}  // namespace

int AxiModel::cavity_mode_count(const std::string& family) const {
  auto it = fw_modes_.find(family);
  return it == fw_modes_.end() ? 0 : static_cast<int>(it->second.size());
}

namespace {

struct ModeCtx {
  const AxiWave* wv = nullptr;
  const Eigenmode* m = nullptr;
  bool tm = false;
  double norm = 0;
};

}  // namespace

// Mode lookup and normalization (shared by mesh() and sample()).
static ModeCtx mode_ctx(const std::map<std::string, std::vector<Eigenmode>>& modes, const AxiWave* te,
                        const AxiWave* tm, const Space& sp, const std::string& family, int index) {
  AXF_REQUIRE(!modes.empty(), "this design has no cavity eigenmodes (fields are kept only for designs with a "
              "`cavity` section and fullwave.modes > 0)", "/cavity", "Add a cavity section and solve again.");
  auto it = modes.find(family);
  AXF_REQUIRE(it != modes.end(), strcat("no cavity eigenmodes of family '", family, "'"), "/fullwave/families",
              "Use \"TE\" or \"TM\" (one of the families that were solved).");
  AXF_REQUIRE(index >= 0 && index < static_cast<int>(it->second.size()),
              strcat("cavity mode index ", index, " out of range for family ", family, " (", it->second.size(),
                     " modes)"), "/fullwave/modes", "Use an index from Results.cavity_modes().");
  ModeCtx c;
  c.tm = family == "TM";
  c.wv = c.tm ? tm : te;
  AXF_REQUIRE(c.wv != nullptr, "internal: the full-wave problem of family " + family + " is missing");
  c.m = &it->second[static_cast<size_t>(index)];
  // int eps_r |E|^2 dV (2 pi r) and the sign: largest E component at the quadrature point of largest |E| positive
  const RefTable& tab = sp.table(sp.quad_rule());
  const int ne = sp.mesh.GetNE();
  double I = 0, best = -1, sgn = 1;
  PointEval pe;
  ModePt p;
  for (int e = 0; e < ne; ++e) {
    const WaveRole& ro = c.wv->roles()[e];
    if (ro.master || ro.excluded) continue;
    for (int q = 0; q < tab.nq; ++q) {
      sp.eval(e, tab, q, pe);
      eval_mode(*c.wv, c.tm, *c.m, e, pe, p);
      const double e2 = p.E[0] * p.E[0] + p.E[1] * p.E[1] + p.E[2] * p.E[2];
      I += 2 * units::pi * pe.r * pe.w * p.eps_r * e2;
      if (e2 > best * (1 + 1e-9)) {
        best = e2;
        int k = 0;
        for (int j = 1; j < 3; ++j) if (std::fabs(p.E[j]) > std::fabs(p.E[k])) k = j;
        sgn = p.E[k] >= 0 ? 1.0 : -1.0;
      }
    }
  }
  AXF_REQUIRE(I > 0, strcat("internal: cavity mode ", family, "[", index, "] has zero electric energy"));
  c.norm = sgn / std::sqrt(I);
  return c;
}

static void push_pt(AxiModel::CavityModeField& F, const ModePt& p, double s) {
  for (int k = 0; k < 3; ++k) {
    F.E[k].push_back(s * p.E[k]);
    F.H[k].push_back(s * p.H[k]);
  }
  F.flux.push_back(s * p.flux);
  F.eps_r.push_back(p.eps_r);
  F.mu_r.push_back(p.mu_r);
}

static void push_nan(AxiModel::CavityModeField& F) {
  for (int k = 0; k < 3; ++k) {
    F.E[k].push_back(kNaN);
    F.H[k].push_back(kNaN);
  }
  F.flux.push_back(kNaN);
  F.eps_r.push_back(kNaN);
  F.mu_r.push_back(kNaN);
}

AxiModel::CavityModeField AxiModel::cavity_mode_mesh(const std::string& family, int index, int subdiv) const {
  AXF_REQUIRE(solved_, "solve the model before asking for cavity mode fields");
  AXF_REQUIRE(subdiv >= 1 && subdiv <= 8, strcat("subdiv must lie in 1..8 (got ", subdiv, ")"));
  const Space& sp = *sp_;
  const ModeCtx c = mode_ctx(fw_modes_, fw_te_.get(), fw_tm_.get(), sp, family, index);
  CavityModeField F;
  F.family = family;
  F.index = index;
  F.f_Hz = c.m->f_Hz;
  F.norm = std::fabs(c.norm);
  std::vector<mfem::IntegrationPoint> pts;
  std::vector<std::array<int, 3>> tris;
  lattice(subdiv, pts, tris);
  PointEval pe;
  ModePt p;
  for (int e = 0; e < sp.mesh.GetNE(); ++e) {
    const WaveRole& ro = c.wv->roles()[e];
    if (ro.master || ro.excluded) continue;
    const uint32_t base = static_cast<uint32_t>(F.r.size());
    for (auto& ip : pts) {
      sp.eval_at(e, ip, pe);
      eval_mode(*c.wv, c.tm, *c.m, e, pe, p);
      F.r.push_back(pe.r);
      F.z.push_back(pe.z);
      F.elem.push_back(e);
      push_pt(F, p, c.norm);
    }
    for (auto& t : tris)
      for (int j = 0; j < 3; ++j) F.tris.push_back(base + static_cast<uint32_t>(t[j]));
  }
  return F;
}

AxiModel::CavityModeField AxiModel::cavity_mode_sample(const std::string& family, int index,
                                                       const std::vector<double>& r,
                                                       const std::vector<double>& z) const {
  AXF_REQUIRE(solved_, "solve the model before asking for cavity mode fields");
  AXF_REQUIRE(r.size() == z.size(), "r and z must have the same length");
  const Space& sp = *sp_;
  const ModeCtx c = mode_ctx(fw_modes_, fw_te_.get(), fw_tm_.get(), sp, family, index);
  CavityModeField F;
  F.family = family;
  F.index = index;
  F.f_Hz = c.m->f_Hz;
  F.norm = std::fabs(c.norm);
  const int n = static_cast<int>(r.size());
  mfem::DenseMatrix P(2, n);
  for (int i = 0; i < n; ++i) {
    P(0, i) = r[i];
    P(1, i) = z[i];
  }
  mfem::Array<int> eid;
  mfem::Array<mfem::IntegrationPoint> ips;
  if (n > 0) sp.mesh.FindPoints(P, eid, ips, false);
  PointEval pe;
  ModePt p;
  for (int i = 0; i < n; ++i) {
    F.r.push_back(r[i]);
    F.z.push_back(z[i]);
    const int e = eid[i];
    if (e < 0 || c.wv->roles()[e].master || c.wv->roles()[e].excluded) {
      F.elem.push_back(-1);
      push_nan(F);
      continue;
    }
    sp.eval_at(e, ips[i], pe);
    eval_mode(*c.wv, c.tm, *c.m, e, pe, p);
    F.elem.push_back(e);
    push_pt(F, p, c.norm);
  }
  return F;
}

}  // namespace axf
