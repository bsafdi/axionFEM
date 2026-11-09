// Mechanics and heat: linear elasticity from the Lorentz loads, and the steady thermal model.
//
// This file turns a solved axisymmetric design into the two auxiliary analyses of
// cpp/src/axi_mech.cpp and writes results["mechanics"] and results["thermal"].  The physics lives
// here; axi::Elasticity and axi::Conduction only know about coefficients, loads and constraints.
//
// The structure is the set of elements whose material carries a "mechanical" block (the built-in
// winding / nb / superconductor / copper / iron have documented defaults); vacuum, air, gaps and
// ports have none and are excluded.  The thermal domain is defined the same way by the "thermal"
// material block.  Touching regions share mesh nodes, so the assembly is bonded: no contact, no
// sliding, no gaps.
//
// Loads on the structure:
//  * Lorentz body force in windings.  Azimuthal winding (current density J_phi):
//      f = J_phi phi^ x B = J_phi (B_z r^ - B_r z^).
//    Poloidal winding (poloidal current density (J_r, J_z) from the toroidal problem's unit
//    profile, azimuthal field B_phi):
//      f = (J_r r^ + J_z z^) x B_phi phi^ = (-J_z B_phi) r^ + (J_r B_phi) z^.
//    The field used is the total static field at the quadrature point, including the winding's own
//    field: that is the physical local force density (unlike a net force on a body, which has a
//    self-force floor).
//  * Maxwell pressure on superconductor surfaces, t = -(B^2/2 mu0) n with n the outward normal of
//    the superconductor and B taken on the vacuum side (the field vanishes inside).  This is the
//    force per area on the screening current sheet; it is applied to the shared surface nodes, so a
//    superconductor bonded to another part transmits it correctly.  Mirrors the surface-force
//    post-processing of cpp/src/model.cpp.
//  * Optional prescribed pressures (mechanics.pressures) on the faces between two named regions, or
//    between a region and the non-structural surroundings.
//  * Optional uniform cooldown: the stress-free strain -alpha of each material (skipped for
//    materials declared `warm`, which stay at room temperature).
#include "axionfem/model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "axionfem/core.hpp"
#include "axionfem/mech.hpp"
#include "axionfem/units.hpp"

namespace axf {

using namespace axi;
using units::mu0;
using units::pi;

namespace {

// Lattice of reference points on a triangle (vertices, edges and interior), where the extrema of a
// polynomial field are looked for.
const mfem::IntegrationRule& mech_lattice(int k) {
  static std::map<int, mfem::IntegrationRule> cache;
  auto it = cache.find(k);
  if (it != cache.end()) return it->second;
  mfem::IntegrationRule ir;
  for (int i = 0; i <= k; ++i)
    for (int j = 0; i + j <= k; ++j) {
      mfem::IntegrationPoint ip;
      ip.Set2(double(i) / k, double(j) / k);
      ip.weight = 0;
      ir.Append(ip);
    }
  return cache[k] = ir;
}

// Running extremes of one scalar over a region.
struct Extreme {
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  void add(double v) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  json to_json() const {
    if (lo > hi) return {{"min", 0.0}, {"max", 0.0}};
    return {{"min", lo}, {"max", hi}};
  }
};

// `PressureFace::load` tags below zero: the Maxwell traction computed from the solved field.
constexpr int kMaxwellSurface = -1;  // face of a superconductor REGION (the field vanishes inside)
constexpr int kMaxwellSheet = -2;    // zero-thickness superconducting sheet (a field on both sides)

struct PeakAt {
  double value = -1, r = 0, z = 0;
  void add(double v, double rr, double zz) {
    if (v > value) { value = v; r = rr; z = zz; }
  }
  json loc() const { return {{"r_m", r}, {"z_m", z}}; }
};

}  // namespace

void AxiModel::mechanics_thermal(const SolveOptions& o) {
  if (!d_.mechanics.enabled && !d_.thermal.enabled) return;
  Timer timer;
  const Space& sp = *sp_;
  mfem::Mesh& m = sp.mesh;
  const int ne = m.GetNE();
  json& R = results_;
  json& warnings = R["warnings"];
  for (auto& c : d_.curves)
    if (c.thin_layer() && d_.thermal.enabled)
      warnings.push_back("thermal: thin magnetic layer '" + c.name + "' has no thermal role (a zero-thickness "
                         "curve carries neither conduction nor heat capacity); mesh it as a region if it matters");

  // ------------------------------------------------------------ element -> region / material
  std::vector<std::string> reg(ne);
  std::vector<const spec::Material*> mat(ne, nullptr);
  std::vector<const spec::Winding*> wind(ne, nullptr);
  std::map<std::string, const spec::Winding*> wind_of;
  for (auto& w : d_.windings) wind_of[w.region] = &w;
  static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
  std::map<std::string, double> region_vol;  // 2 pi int r dr dz
  const RefTable& tab = sp.table(sp.quad_rule());
  {
    PointEval pe;
    for (int e = 0; e < ne; ++e) {
      reg[e] = region_name(m.GetAttribute(e));
      if (reg[e] == "air") {
        mat[e] = &vacuum;
      } else {
        const spec::Region* g = d_.find_region(reg[e]);
        mat[e] = g ? &d_.materials.at(g->material) : &vacuum;
        auto it = wind_of.find(reg[e]);
        if (it != wind_of.end()) wind[e] = it->second;
      }
      for (int q = 0; q < tab.nq; ++q) {
        sp.eval(e, tab, q, pe);
        region_vol[reg[e]] += 2 * pi * pe.r * pe.w;
      }
    }
  }
  auto is_sc = [&](int e) { return mat[e]->kind == spec::MaterialKind::Superconductor; };
  // regions that are matter but carry no data for an analysis: reported once each
  auto missing_data = [&](const char* what, const std::set<std::string>& names) {
    if (names.empty()) return;
    std::string list;
    for (auto& s : names) list += (list.empty() ? "" : ", ") + s;
    warnings.push_back(strcat("the ", what, " analysis skips the region(s) ", list,
                              ": their material has no \"", what, "\" block. Add one to the material "
                              "(see docs/design_format.md) to include them."));
  };

  // ============================================================ elasticity
  if (d_.mechanics.enabled) {
    const auto& ms = d_.mechanics;
    std::vector<ElasticRole> roles(ne);
    std::set<std::string> skipped, solid_regions;
    for (int e = 0; e < ne; ++e) {
      const spec::Material& mt = *mat[e];
      if (!mt.mech.given) {
        if (reg[e] != "air" && mt.kind != spec::MaterialKind::Vacuum) skipped.insert(reg[e]);
        continue;
      }
      ElasticRole& ro = roles[e];
      ro.solid = true;
      solid_regions.insert(reg[e]);
      const double E = mt.mech.E_Pa, nu = mt.mech.nu;
      ro.lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      ro.mu = E / (2 * (1 + nu));
      // cooldown: the stress-free strain is -alpha; `warm` parts stay at room temperature
      ro.eps_th = (ms.cooldown && !mt.warm) ? -mt.mech.alpha : 0.0;
    }
    missing_data("mechanical", skipped);
    {
      // Pure-displacement elasticity is not locking-free: as nu -> 1/2 the volumetric term lambda
      // div(u) dominates and the stresses come out over-stiff on a fixed mesh.  Measured on the Lame
      // benchmark (P2, h = 0.02): sigma_tt error 2.3e-3 at nu = 0.3, 7.2e-2 at 0.49, 0.72 at 0.499.
      std::set<std::string> soft;
      for (int e = 0; e < ne; ++e)
        if (roles[e].solid && mat[e]->mech.nu > 0.45) soft.insert(mat[e]->name);
      if (!soft.empty()) {
        std::string list;
        for (auto& s2 : soft) list += (list.empty() ? "" : ", ") + s2;
        warnings.push_back("mechanics: material(s) " + list +
                           " have nu > 0.45. The pure-displacement formulation is not locking-free: on "
                           "a fixed mesh the stresses are over-stiff by roughly 1/(1 - 2 nu) (7 % at "
                           "nu = 0.49, 70 % at nu = 0.499 on the Lame benchmark). Refine strongly, or "
                           "reduce nu if the material is not really incompressible.");
      }
      // A cooldown with no alpha is a silent no-op for that material.
      if (ms.cooldown) {
        std::set<std::string> noalpha;
        for (int e = 0; e < ne; ++e)
          if (roles[e].solid && !mat[e]->warm && mat[e]->mech.alpha == 0.0) noalpha.insert(mat[e]->name);
        if (!noalpha.empty()) {
          std::string list;
          for (auto& s2 : noalpha) list += (list.empty() ? "" : ", ") + s2;
          warnings.push_back("mechanics: mechanics.cooldown is on but material(s) " + list +
                             " have no (or zero) \"alpha\" in their \"mechanical\" block, so they "
                             "contract by nothing while the parts around them do. Give them an "
                             "integrated contraction alpha = (L_300K - L_cold)/L_300K, or mark them "
                             "\"warm\": true if they really stay at room temperature.");
        }
      }
    }
    AXF_REQUIRE(!solid_regions.empty(), "the mechanics analysis has no structure", "/mechanics",
                "Every region is vacuum-like or lacks a \"mechanical\" material block.");

    // ---- supports
    std::vector<SupportGroup> groups;
    for (size_t i = 0; i < ms.supports.size(); ++i) {
      const auto& s = ms.supports[i];
      SupportGroup g;
      g.name = s.region;
      g.fix_r = s.fix == "all" || s.fix == "r";
      g.fix_z = s.fix == "all" || s.fix == "z";
      if (!solid_regions.count(s.region)) {
        throw Error("support region '" + s.region + "' is not part of the structure",
                    strcat("/mechanics/supports/", i, "/region"),
                    "Only regions whose material has a \"mechanical\" block carry displacements.");
      }
      // ElasticRole::support holds ONE group index, so a second entry on the same region used to
      // silently replace the first ([{bar, r}, {bar, z}] left u_r unconstrained).
      for (int e = 0; e < ne; ++e) {
        if (!roles[e].solid || reg[e] != s.region || roles[e].support < 0) continue;
        throw Error("region '" + s.region + "' already has a support (entry " +
                        std::to_string(roles[e].support) + ")",
                    strcat("/mechanics/supports/", i, "/region"),
                    "A region carries one support: use \"fix\": \"all\" to fix both components at once.");
      }
      int n = 0;
      for (int e = 0; e < ne; ++e)
        if (roles[e].solid && reg[e] == s.region) { roles[e].support = static_cast<int>(groups.size()); ++n; }
      AXF_REQUIRE(n > 0, "support region '" + s.region + "' has no elements", strcat("/mechanics/supports/", i));
      groups.push_back(g);
    }

    // ---- Lorentz body force
    std::set<std::string> loaded_windings, dropped_windings;
    for (int e = 0; e < ne; ++e) {
      if (!wind[e]) continue;
      if (!roles[e].solid) { dropped_windings.insert(reg[e]); continue; }
      roles[e].body_force = true;
      loaded_windings.insert(wind[e]->name);
    }
    if (!dropped_windings.empty()) {
      std::string list;
      for (auto& s : dropped_windings) list += (list.empty() ? "" : ", ") + s;
      warnings.push_back("the Lorentz body force of winding region(s) " + list +
                         " is not applied: their material has no \"mechanical\" block");
    }
    // `wind` is copied: the callback outlives this function (the problem is kept for sample_mech).
    auto body = [this, wind](int e, const PointEval& pe, double& fr, double& fz) {
      fr = fz = 0;
      const spec::Winding* w = wind[e];
      if (!w) return;
      Sample s = eval_pe(e, pe);
      if (w->type == spec::WindingType::Azimuthal && a_lab_) {
        const double J = a_lab_->roles()[e].J + (coil_J_.empty() ? 0.0 : coil_J_[e]);
        fr = J * s.Bz;
        fz = -J * s.Br;
      } else if (w->type == spec::WindingType::Poloidal && h_lab_) {
        // G = La + (Lb - La) r u_s  ->  J_r = -(1/r) dG/dz, J_z = (1/r) dG/dr  (as in model.cpp)
        const int p = h_lab_->roles()[e].profile;
        if (p < 0 || pe.r <= 0) return;
        const double dL = h_lab_sol_.Lb[p] - h_lab_sol_.La[p];
        const auto& d = sp_->edofs(e);
        double us = 0, dr = 0, dz = 0;
        for (size_t i = 0; i < d.size(); ++i) {
          us += h_lab_sol_.unit(d[i]) * pe.N[i];
          dr += h_lab_sol_.unit(d[i]) * pe.dN[2 * i];
          dz += h_lab_sol_.unit(d[i]) * pe.dN[2 * i + 1];
        }
        const double Jr = -dL * dz, Jz = dL * (us + pe.r * dr) / pe.r;
        fr = -Jz * s.Bphi;
        fz = Jr * s.Bphi;
      }
    };

    // ---- pressure loads: index -1 = Maxwell pressure on a superconductor, >= 0 = prescribed
    std::vector<PressureFace> pfaces;
    std::set<std::string> sc_loaded, sc_dropped;
    for (int f = 0; f < m.GetNumFaces(); ++f) {
      int e1 = -1, e2 = -1;
      m.GetFaceElements(f, &e1, &e2);
      if (e1 < 0 || e2 < 0 || is_sc(e1) == is_sc(e2)) continue;
      const int sc = is_sc(e1) ? e1 : e2, other = is_sc(e1) ? e2 : e1;
      if (!roles[sc].solid) { sc_dropped.insert(reg[sc]); continue; }
      pfaces.push_back({f, sc, other, kMaxwellSurface});
      sc_loaded.insert(reg[sc]);
    }
    if (!sc_dropped.empty()) {
      std::string list;
      for (auto& s : sc_dropped) list += (list.empty() ? "" : ", ") + s;
      warnings.push_back("the magnetic surface pressure on superconductor region(s) " + list +
                         " is not applied: their material has no \"mechanical\" block");
    }
    // A zero-thickness superconducting sheet (`curves`) carries the screening current, so the Maxwell
    // traction (B_solid^2 - B_other^2) n /(2 mu0) acts on it; the sheet shares the surface nodes of
    // whatever solid it lies on, so that traction must be transmitted to the structure.  It used not to
    // be: a steel tube with a superconducting skin screening 0.1045 T reported hoop stress 0 instead of
    // -p b/t = -2.2e5 Pa.
    std::map<int, std::string> sc_sheet;  // curve boundary attribute -> curve name
    for (auto& c : d_.curves) {
      if (d_.materials.at(c.material).kind != spec::MaterialKind::Superconductor) continue;
      auto it = mb_.attr_of_curve.find(c.name);
      if (it != mb_.attr_of_curve.end()) sc_sheet[it->second] = c.name;
    }
    std::set<std::string> sheet_loaded, sheet_unloaded;
    for (auto& [attr, nm] : sc_sheet) sheet_unloaded.insert(nm);
    for (int f = 0; f < m.GetNumFaces() && !sc_sheet.empty(); ++f) {
      auto it = sc_sheet.find(sp.face_bdr_attr[f]);
      if (it == sc_sheet.end()) continue;
      int e1 = -1, e2 = -1;
      m.GetFaceElements(f, &e1, &e2);
      if (e1 < 0 || e2 < 0) continue;  // a sheet on the domain boundary has no structure to load
      const bool s1 = roles[e1].solid, s2 = roles[e2].solid;
      if (!s1 && !s2) continue;        // free-standing sheet: nothing to push on
      // Both sides share the face DOFs, so the nodal load is the same whichever side carries it; apply
      // it ONCE (applying it to both solids would double it).
      const int sol = s1 ? e1 : e2, oth = s1 ? e2 : e1;
      pfaces.push_back({f, sol, oth, kMaxwellSheet});
      sheet_loaded.insert(it->second);
      sheet_unloaded.erase(it->second);
    }
    for (auto& nm : sheet_loaded) {
      warnings.push_back("superconducting sheet '" + nm + "' has zero thickness, so no stress is "
                         "reported for the sheet itself; its magnetic surface traction "
                         "(B_in^2 - B_out^2)/(2 mu0) IS applied to the structure it lies on. Model it as "
                         "a thin region to get the sheet's own stress.");
    }
    for (auto& nm : sheet_unloaded) {
      warnings.push_back("superconducting sheet '" + nm + "' touches no region with a \"mechanical\" "
                         "material block on either side, so its magnetic surface traction is applied "
                         "nowhere and the sheet itself carries no stress (model it as a thin region, or "
                         "give the region it lies on a \"mechanical\" block).");
    }
    std::vector<double> pvalue;
    for (size_t i = 0; i < ms.pressures.size(); ++i) {
      const auto& pl = ms.pressures[i];
      const int tag = static_cast<int>(pvalue.size());
      pvalue.push_back(pl.p_Pa);
      auto side_is_from = [&](int e) {
        return pl.from == "vacuum" ? !roles[e].solid : reg[e] == pl.from;
      };
      int n = 0;
      for (int f = 0; f < m.GetNumFaces(); ++f) {
        int e1 = -1, e2 = -1;
        m.GetFaceElements(f, &e1, &e2);
        if (e1 < 0 || e2 < 0) continue;
        if (reg[e1] == pl.on && side_is_from(e2)) { pfaces.push_back({f, e1, e2, tag}); ++n; }
        else if (reg[e2] == pl.on && side_is_from(e1)) { pfaces.push_back({f, e2, e1, tag}); ++n; }
      }
      if (n == 0) {
        throw Error("pressure load on '" + pl.on + "' from '" + pl.from + "' touches no mesh face",
                    strcat("/mechanics/pressures/", i),
                    "The two sides must share a boundary in the (r, z) half-plane. Check the region names.");
      }
      AXF_REQUIRE(solid_regions.count(pl.on) > 0, "pressure load region '" + pl.on + "' is not part of the structure",
                  strcat("/mechanics/pressures/", i, "/on"));
    }

    elast_ = std::make_unique<Elasticity>(sp, roles, groups);
    elast_->set_body_force(body);
    elast_->set_pressure(pfaces, [this, pvalue](const PressureFace& pf, mfem::FaceElementTransformations& FT) {
      if (pf.load >= 0) return pvalue[pf.load];
      int e1 = -1, e2 = -1;
      sp_->mesh.GetFaceElements(pf.face, &e1, &e2);
      const mfem::IntegrationPoint& eip =
          pf.other_elem == e1 ? FT.GetElement1IntPoint() : FT.GetElement2IntPoint();
      Sample s = eval(pf.other_elem, eip);
      const double B2o = s.Br * s.Br + s.Bz * s.Bz + s.Bphi * s.Bphi;
      // A superconductor REGION is field free inside, so the pressure is the one-sided B^2/(2 mu0).
      if (pf.load == kMaxwellSurface) return B2o / (2 * mu0);
      // A zero-thickness SHEET has a field on both sides and the screening current feels the
      // difference; t = -p n with n out of solid_elem, so p = (B_other^2 - B_solid^2)/(2 mu0).
      const mfem::IntegrationPoint& eips =
          pf.solid_elem == e1 ? FT.GetElement1IntPoint() : FT.GetElement2IntPoint();
      Sample t = eval(pf.solid_elem, eips);
      const double B2s = t.Br * t.Br + t.Bz * t.Bz + t.Bphi * t.Bphi;
      return (B2o - B2s) / (2 * mu0);
    });
    log(1, "solving linear elasticity (Lorentz and surface loads)");
    elast_sol_ = elast_->solve(o.verbose);

    // The implementation keeps only the pressure part of the Maxwell traction,
    // t = [(B.n) B - B^2 n/2]/mu0 -> -B^2 n/(2 mu0), which is exact when B.n = 0.  On an ideal
    // superconductor B.n vanishes identically (psi = r A_phi and G = r H_phi are constant along its
    // surface), so this reports how far the discrete solution is from that, i.e. the size of the term
    // that was dropped relative to the one that was kept.
    double bn_rel = 0;
    {
      const mfem::IntegrationRule& irf = mfem::IntRules.Get(mfem::Geometry::SEGMENT, 2 * sp.order + 6);
      mfem::Vector nor(2);
      for (const auto& pf : pfaces) {
        if (pf.load >= 0) continue;
        int e1 = -1, e2 = -1;
        m.GetFaceElements(pf.face, &e1, &e2);
        mfem::FaceElementTransformations* FT = m.GetFaceElementTransformations(pf.face);
        for (int q = 0; q < irf.GetNPoints(); ++q) {
          FT->SetAllIntPoints(&irf.IntPoint(q));
          const mfem::IntegrationPoint& eip =
              pf.other_elem == e1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
          Sample sv = eval(pf.other_elem, eip);
          mfem::CalcOrtho(FT->Jacobian(), nor);
          const double nl = std::hypot(nor(0), nor(1));
          const double Bp = std::hypot(sv.Br, sv.Bz);  // B_phi is azimuthal, hence tangential
          if (!(nl > 0) || !(Bp > 0)) continue;
          bn_rel = std::max(bn_rel, std::fabs(sv.Br * nor(0) + sv.Bz * nor(1)) / (nl * Bp));
        }
      }
    }

    // ---- per-region extremes
    const int K = 2 * sp.order + 2;
    const RefTable& ltab = sp.table(mech_lattice(K));
    std::map<std::string, PeakAt> umax, vmmax;
    std::map<std::string, Extreme> hoop, srr, szz, srz;
    {
      PointEval pe;
      for (int e = 0; e < ne; ++e) {
        if (!roles[e].solid) continue;
        for (int q = 0; q < ltab.nq; ++q) {
          sp.eval(e, ltab, q, pe);
          double ur, uz, a, b, c, dd;
          elast_->displacement(elast_sol_, e, pe, ur, uz);
          elast_->stress(elast_sol_, e, pe, a, b, c, dd);
          umax[reg[e]].add(std::hypot(ur, uz), pe.r, pe.z);
          vmmax[reg[e]].add(Elasticity::von_mises(a, b, c, dd), pe.r, pe.z);
          srr[reg[e]].add(a);
          szz[reg[e]].add(b);
          hoop[reg[e]].add(c);
          srz[reg[e]].add(dd);
        }
      }
    }
    json rj = json::object();
    for (auto& [name, p] : umax) {
      const spec::Region* g = d_.find_region(name);
      if (!g) continue;  // the implicit "air" never carries mechanical data
      const spec::Material& mt = d_.materials.at(g->material);
      double eth = 0;
      for (int e = 0; e < ne; ++e)
        if (roles[e].solid && reg[e] == name) { eth = roles[e].eps_th; break; }
      rj[name] = {{"u_max_m", p.value},
                  {"u_max_location", p.loc()},
                  {"von_mises_max_Pa", vmmax[name].value},
                  {"von_mises_max_location", vmmax[name].loc()},
                  {"sigma_hoop_Pa", hoop[name].to_json()},
                  {"sigma_rr_Pa", srr[name].to_json()},
                  {"sigma_zz_Pa", szz[name].to_json()},
                  {"sigma_rz_Pa", srz[name].to_json()},
                  {"E_Pa", mt.mech.E_Pa},
                  {"nu", mt.mech.nu},
                  {"thermal_strain", eth}};
    }
    json sj = json::array();
    for (size_t i = 0; i < groups.size(); ++i)
      sj.push_back({{"region", groups[i].name},
                    {"fix", ms.supports[i].fix},
                    {"reaction_Fz_N", elast_sol_.reaction_Fz_N[i]},
                    {"reaction_radial_N_per_rad", elast_sol_.reaction_Fr_per_rad[i]}});
    json mech_checks = {{"equilibrium_Fz_round_off", elast_sol_.closure_Fz_round_off},
                        {"equilibrium_Fz_relative", elast_sol_.closure_Fz},
                        {"equilibrium_Fz_residual_N", elast_sol_.closure_Fz_N},
                        {"applied_Fz_N", elast_sol_.applied_Fz_N},
                        {"applied_radial_N_per_rad", elast_sol_.applied_Fr_per_rad},
                        {"load_scale_N", elast_sol_.load_scale_N},
                        {"axial_load_scale_N", elast_sol_.axial_load_scale_N},
                        {"assembly_scale_N", elast_sol_.assembly_scale_N},
                        {"rigid_axial_pin_Fz_N", elast_sol_.pin_Fz_N},
                        {"maxwell_traction_Bn_over_B_max", bn_rel},
                        {"maxwell_traction_note",
                         "the magnetic surface load is the pressure part of the Maxwell traction, "
                         "t = -B^2 n/(2 mu0); the full traction has an extra (B.n) B/mu0, which vanishes "
                         "on an ideal superconductor. maxwell_traction_Bn_over_B_max is the largest "
                         "|B.n|/|B_poloidal| the discrete field shows on those faces: the relative size "
                         "of the term that is dropped (0 if there are no magnetic pressure faces)"},
                        {"note", "axial equilibrium: sum(applied F_z) + sum(reaction F_z) = 0 is an exact identity "
                                 "of the discrete system, so the residual only measures round-off in the assembly. "
                                 "Read equilibrium_Fz_round_off (residual / assembly_scale_N, the L1 norm of the "
                                 "nodal load vector): it is always meaningful. equilibrium_Fz_relative divides by "
                                 "the applied axial load instead, and is meaningless when there is none - a cooldown "
                                 "prestress, for instance, is self-balanced and enters assembly_scale_N but not "
                                 "applied_Fz_N. The radial resultant does not balance at all (hoop stress carries it "
                                 "in axisymmetry)."}};
    R["mechanics"] = {{"cooldown", ms.cooldown},
                      {"regions", rj},
                      {"supports", sj},
                      {"lorentz_windings", std::vector<std::string>(loaded_windings.begin(), loaded_windings.end())},
                      {"magnetic_pressure_bodies", std::vector<std::string>(sc_loaded.begin(), sc_loaded.end())},
                      {"magnetic_pressure_sheets", std::vector<std::string>(sheet_loaded.begin(), sheet_loaded.end())},
                      {"prescribed_pressures", static_cast<int>(ms.pressures.size())},
                      {"checks", mech_checks},
                      {"contact", "touching regions share mesh nodes and are bonded (no contact or sliding)"},
                      {"solver", elast_sol_.diagnostics}};
    // ---- warnings
    // The free-assembly test compares an AXIAL resultant, so it must be scaled by the axial load
    // magnitude: load_scale_N mixes |f_r| and |f_z| and hid a real 28 kN net axial load behind a large
    // radial pressure.  The absolute floor suppresses the round-off of a dummy (1 A-turn) winding.
    const double zscale = std::max(elast_sol_.axial_load_scale_N, 1e-30);
    const double floor_N = 1e-9 * std::max({elast_sol_.assembly_scale_N, elast_sol_.load_scale_N, 1e-30});
    if (elast_sol_.closure_Fz > 1e-6 && elast_sol_.closure_Fz_round_off > 1e-10) {
      warnings.push_back(strcat("mechanics: axial equilibrium does not close (residual ",
                                elast_sol_.closure_Fz_N, " N, ", elast_sol_.closure_Fz,
                                " of the applied axial load); the stress results are suspect"));
    }
    for (size_t c = 0; c < elast_sol_.free_component_Fz_N.size(); ++c) {
      const double fz = elast_sol_.free_component_Fz_N[c];
      if (std::fabs(fz) <= 1e-3 * zscale || std::fabs(fz) <= floor_N) continue;
      std::set<std::string> names;
      for (int e = 0; e < ne; ++e)
        if (elast_sol_.component_of_elem[e] == static_cast<int>(c)) names.insert(reg[e]);
      std::string list;
      for (auto& s : names) list += (list.empty() ? "" : ", ") + s;
      warnings.push_back(strcat("mechanics: the unsupported assembly (", list, ") carries a net axial load of ", fz,
                                " N and cannot be in equilibrium; its axial displacement is pinned at one node and "
                                "is not physical. Add a support in mechanics.supports."));
    }
    if (groups.empty()) {
      warnings.push_back("mechanics: no supports were given; each unsupported assembly is free in z (its axial "
                         "displacement is reported relative to its own volume-weighted mean) and reports no "
                         "reaction forces");
    }
  }

  // ============================================================ conduction
  if (d_.thermal.enabled) {
    const auto& ts = d_.thermal;
    ktables_.clear();
    std::map<std::string, const KTable*> table_of;  // material name -> k(T)
    for (auto& [name, mt] : d_.materials) {
      if (!mt.therm.given || mt.therm.kT_T.empty()) continue;
      auto t = std::make_unique<KTable>();
      t->T = mt.therm.kT_T;
      t->k = mt.therm.kT_k;
      table_of[name] = t.get();
      ktables_.push_back(std::move(t));
    }
    std::vector<ThermalRole> roles(ne);
    std::set<std::string> skipped, domain;
    for (int e = 0; e < ne; ++e) {
      const spec::Material& mt = *mat[e];
      if (!mt.therm.given) {
        if (reg[e] != "air" && mt.kind != spec::MaterialKind::Vacuum) skipped.insert(reg[e]);
        continue;
      }
      ThermalRole& ro = roles[e];
      ro.solid = true;
      domain.insert(reg[e]);
      ro.k = mt.therm.k_const;
      auto it = table_of.find(mt.name);
      if (it != table_of.end()) ro.kt = it->second;
    }
    missing_data("thermal", skipped);
    AXF_REQUIRE(!domain.empty(), "the thermal analysis has no conducting region", "/thermal",
                "Every region is vacuum-like or lacks a \"thermal\" material block.");

    // ---- fixed temperatures
    std::map<int, double> fixed_curve;
    struct FixedSrc { std::string name; double T; bool is_region; };
    std::vector<FixedSrc> fixed_src;
    json fj = json::array();
    for (size_t i = 0; i < ts.fixed.size(); ++i) {
      const auto& f = ts.fixed[i];
      if (!f.region.empty()) {
        if (!domain.count(f.region)) {
          throw Error("fixed-temperature region '" + f.region + "' is not part of the thermal domain",
                      strcat("/thermal/fixed/", i, "/region"),
                      "Only regions whose material has a \"thermal\" block conduct heat.");
        }
        for (int e = 0; e < ne; ++e)
          if (roles[e].solid && reg[e] == f.region) { roles[e].fixed = true; roles[e].T_fix = f.T_K; }
        fixed_src.push_back({f.region, f.T_K, true});
        fj.push_back({{"region", f.region}, {"T_K", f.T_K}});
      } else {
        auto it = mb_.attr_of_curve.find(f.curve);
        AXF_REQUIRE(it != mb_.attr_of_curve.end(), "curve '" + f.curve + "' is not in the mesh",
                    strcat("/thermal/fixed/", i, "/curve"));
        fixed_curve[it->second] = f.T_K;
        fixed_src.push_back({f.curve, f.T_K, false});
        fj.push_back({{"curve", f.curve}, {"T_K", f.T_K}});
      }
    }
    // Two prescribed temperatures that meet on a mesh node would resolve as last-writer-wins (the
    // element loop above overwrites roles[e].T_fix, and fixed curves are applied on top of that), so a
    // design with two touching sinks would silently get one of the two temperatures.  Reject it instead.
    if (fixed_src.size() > 1) {
      std::vector<int> owner(sp.ndof(), -1);
      mfem::Array<int> bd;
      for (size_t k = 0; k < fixed_src.size(); ++k) {
        std::vector<int> dofs;
        if (fixed_src[k].is_region) {
          for (int e = 0; e < ne; ++e)
            if (roles[e].solid && reg[e] == fixed_src[k].name)
              for (int d : sp.edofs(e)) dofs.push_back(d);
        } else {
          const int attr = mb_.attr_of_curve.at(fixed_src[k].name);
          for (int be = 0; be < m.GetNBE(); ++be) {
            if (m.GetBdrAttribute(be) != attr) continue;
            sp.fes->GetBdrElementDofs(be, bd);
            for (int d : bd) dofs.push_back(d);
          }
        }
        for (int d : dofs) {
          const int o = owner[d];
          if (o >= 0 && o != static_cast<int>(k)) {
            const double Ta = fixed_src[o].T, Tb = fixed_src[k].T;
            if (std::fabs(Ta - Tb) > 1e-12 * std::max(1.0, std::max(std::fabs(Ta), std::fabs(Tb)))) {
              throw Error(strcat("conflicting prescribed temperatures: '", fixed_src[o].name, "' asks for ",
                                 Ta, " K and '", fixed_src[k].name, "' for ", Tb,
                                 " K at the same mesh node (r, z) = (", sp.r[d], ", ", sp.z[d], ")"),
                          "/thermal/fixed",
                          "Two prescribed-temperature regions or curves touch, so the temperature there "
                          "is whichever entry came last. Separate them with a conducting region, or give "
                          "them the same temperature.");
            }
          }
          owner[d] = static_cast<int>(k);
        }
      }
    }
    // ---- heat loads
    json lj = json::array();
    std::map<std::string, double> q_of_region;
    for (size_t i = 0; i < ts.loads.size(); ++i) {
      const auto& l = ts.loads[i];
      if (!domain.count(l.region)) {
        throw Error("heat load region '" + l.region + "' is not part of the thermal domain",
                    strcat("/thermal/loads/", i, "/region"),
                    "Only regions whose material has a \"thermal\" block conduct heat.");
      }
      const double V = region_vol[l.region];
      AXF_REQUIRE(V > 0, "heat load region '" + l.region + "' has zero volume", strcat("/thermal/loads/", i));
      const double q = l.total ? l.W / V : l.W_m3;
      q_of_region[l.region] += q;
      lj.push_back({{"region", l.region}, {"W_m3", q}, {"W", q * V}, {"volume_m3", V}});
      bool all_fixed = true;
      for (int e = 0; e < ne; ++e)
        if (roles[e].solid && reg[e] == l.region && !roles[e].fixed) all_fixed = false;
      if (all_fixed) {
        warnings.push_back("thermal: the heat load on '" + l.region +
                           "' sits in a region whose temperature is prescribed everywhere, so it is absorbed by the "
                           "boundary condition and changes nothing");
      }
    }
    for (int e = 0; e < ne; ++e) {
      if (!roles[e].solid) continue;
      auto it = q_of_region.find(reg[e]);
      if (it != q_of_region.end()) roles[e].q = it->second;
    }

    // Every connected part of the thermal domain needs a prescribed temperature somewhere, otherwise
    // its temperature level is undetermined (a singular system).  Say which regions are the problem.
    {
      int nc = 0;
      std::vector<int> comp = solid_components(sp, [&](int e) { return roles[e].solid; }, nc);
      std::vector<char> has_fix(nc, 0);
      std::vector<char> dof_fixed(sp.ndof(), 0);
      for (int e = 0; e < ne; ++e)
        if (roles[e].solid && roles[e].fixed)
          for (int d : sp.edofs(e)) dof_fixed[d] = 1;
      if (!fixed_curve.empty()) {
        mfem::Array<int> bd;
        for (int be = 0; be < m.GetNBE(); ++be) {
          if (!fixed_curve.count(m.GetBdrAttribute(be))) continue;
          sp.fes->GetBdrElementDofs(be, bd);
          for (int d : bd) dof_fixed[d] = 1;
        }
      }
      for (int e = 0; e < ne; ++e) {
        if (comp[e] < 0) continue;
        for (int d : sp.edofs(e))
          if (dof_fixed[d]) has_fix[comp[e]] = 1;
      }
      for (int c = 0; c < nc; ++c) {
        if (has_fix[c]) continue;
        std::set<std::string> names;
        for (int e = 0; e < ne; ++e)
          if (comp[e] == c) names.insert(reg[e]);
        std::string list;
        for (auto& s : names) list += (list.empty() ? "" : ", ") + s;
        throw Error("the conducting part made of region(s) " + list + " has no prescribed temperature",
                    "/thermal/fixed",
                    "It is thermally isolated from every heat sink, so its temperature level is undetermined. Add a "
                    "{\"region\"|\"curve\", \"T_K\"} entry that touches it, connect it to one, or give its material "
                    "no \"thermal\" block to keep it out of the analysis.");
      }
    }

    cond_ = std::make_unique<Conduction>(sp, roles, fixed_curve);
    log(1, "solving steady heat conduction");
    cond_sol_ = cond_->solve(o.verbose);

    // ---- per-region temperature extremes
    const int K = 2 * sp.order + 2;
    const RefTable& ltab = sp.table(mech_lattice(K));
    std::map<std::string, Extreme> trange;
    {
      PointEval pe;
      for (int e = 0; e < ne; ++e) {
        if (!roles[e].solid) continue;
        for (int q = 0; q < ltab.nq; ++q) {
          sp.eval(e, ltab, q, pe);
          trange[reg[e]].add(cond_->value(cond_sol_, e, pe));
        }
      }
    }
    json rj = json::object();
    for (auto& [name, t] : trange) {
      const spec::Region* g = d_.find_region(name);
      if (!g) continue;  // the implicit "air" never carries thermal data
      const spec::Material& mt = d_.materials.at(g->material);
      json e = {{"T_min_K", t.lo}, {"T_max_K", t.hi}, {"volume_m3", region_vol[name]}};
      if (mt.therm.kT_T.empty()) {
        e["k_W_mK"] = mt.therm.k_const;
      } else {
        KTable kt;
        kt.T = mt.therm.kT_T;
        kt.k = mt.therm.kT_k;
        e["k_W_mK"] = {{"at_T_min", kt.at(t.lo)}, {"at_T_max", kt.at(t.hi)}};
        e["k_table_T_range_K"] = {mt.therm.kT_T.front(), mt.therm.kT_T.back()};
        if (t.lo < mt.therm.kT_T.front() - 1e-12 || t.hi > mt.therm.kT_T.back() + 1e-12) {
          warnings.push_back(strcat("thermal: the temperature in '", name, "' reaches [", t.lo, ", ", t.hi,
                                    "] K, outside the k(T) table [", mt.therm.kT_T.front(), ", ",
                                    mt.therm.kT_T.back(), "] K; k is clamped to the end values there"));
        }
      }
      if (q_of_region.count(name)) e["q_W_m3"] = q_of_region[name];
      rj[name] = e;
    }
    R["thermal"] = {{"regions", rj},
                    {"fixed", fj},
                    {"loads", lj},
                    {"T_min_K", cond_sol_.T_min},
                    {"T_max_K", cond_sol_.T_max},
                    {"checks", {{"generated_W", cond_sol_.generated_W},
                                {"outflow_W", cond_sol_.outflow_W},
                                {"heat_removed_W", cond_sol_.heat_removed_W},
                                {"heat_added_W", cond_sol_.heat_added_W},
                                {"balance_relative", cond_sol_.balance_rel},
                                {"note", "outflow = -2 pi sum over the prescribed-temperature DOFs of (K T - F); it "
                                         "equals the generated power exactly for the discrete system, so "
                                         "balance_relative only measures assembly round-off. outflow is the NET "
                                         "flow and cancels when heat enters at one prescribed temperature and "
                                         "leaves at another: heat_removed_W is the load on the sinks that absorb "
                                         "heat (the cold-stage heat load) and heat_added_W what the warm "
                                         "boundaries supply; balance_relative is scaled by the largest of the "
                                         "three"}}},
                    {"solver", cond_sol_.diagnostics}};
    if (!cond_sol_.converged) {
      warnings.push_back(strcat("thermal: the k(T) Picard iteration did not converge in ",
                                cond_sol_.picard_iterations, " iterations (relative change ",
                                cond_sol_.picard_change, "); results are NOT reliable"));
    }
    if (cond_sol_.generated_W != 0 && cond_sol_.balance_rel > 1e-8) {
      warnings.push_back(strcat("thermal: the heat balance does not close (relative residual ",
                                cond_sol_.balance_rel, ")"));
    }
  }
  R["timing"]["mechanics_thermal_s"] = timer.seconds();
}

// ---------------------------------------------------------------- sampling
std::vector<AxiModel::MechSample> AxiModel::sample_mech(const std::vector<double>& r,
                                                       const std::vector<double>& z) const {
  AXF_REQUIRE(solved_, "solve the model before sampling");
  AXF_REQUIRE(r.size() == z.size(), "r and z must have the same length");
  AXF_REQUIRE(elast_ || cond_, "the design has no mechanics or thermal analysis", "",
              "Add a \"mechanics\" or \"thermal\" section to the design.");
  const int n = static_cast<int>(r.size());
  mfem::DenseMatrix pts(2, n);
  for (int i = 0; i < n; ++i) {
    pts(0, i) = r[i];
    pts(1, i) = z[i];
  }
  mfem::Array<int> eid;
  mfem::Array<mfem::IntegrationPoint> ips;
  sp_->mesh.FindPoints(pts, eid, ips, false);
  std::vector<MechSample> out(n);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  PointEval pe;
  for (int i = 0; i < n; ++i) {
    MechSample& s = out[i];
    // Outside the structural / thermal domain the quantity does not exist, so it is NaN, not zero: a
    // zero displacement or a 0 K temperature is a legitimate value elsewhere and would be
    // indistinguishable from "not computed here". `solid` and `thermal` say which fields are defined.
    s.ur = s.uz = s.s_rr = s.s_zz = s.s_tt = s.s_rz = s.von_mises = s.T_K = nan;
    if (eid[i] < 0) continue;
    const int e = eid[i];
    s.region = sp_->mesh.GetAttribute(e);
    sp_->eval_at(e, ips[i], pe);
    if (elast_ && elast_->roles()[e].solid) {
      s.solid = true;
      elast_->displacement(elast_sol_, e, pe, s.ur, s.uz);
      elast_->stress(elast_sol_, e, pe, s.s_rr, s.s_zz, s.s_tt, s.s_rz);
      s.von_mises = axi::Elasticity::von_mises(s.s_rr, s.s_zz, s.s_tt, s.s_rz);
    }
    if (cond_ && cond_->roles()[e].solid) {
      s.thermal = true;
      s.T_K = cond_->value(cond_sol_, e, pe);
    }
  }
  return out;
}

}  // namespace axf
