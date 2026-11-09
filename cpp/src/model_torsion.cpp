// Vibrations: the design-level driver of the axisymmetric m = 0 torsional analysis.
// Physics and formulation: axionfem/torsion.hpp.
//
// The structure is the same set of elements as for static elasticity (a material with a "mechanical"
// block), with the additional requirement of a positive density; regions that have a mechanical
// block but no density are skipped with a warning naming them, because without a density there is
// no inertia and the dynamic problem is undefined.  Per-region overrides of the shear modulus and
// density are accepted in the request, so a study can sweep wall thicknesses and materials without
// editing the design.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>

#include <Eigen/Dense>
#include <set>
#include <string>
#include <vector>

#include "axionfem/core.hpp"
#include "axionfem/expr.hpp"
#include "axionfem/model.hpp"
#include "axionfem/omp_shim.hpp"
#include "axionfem/torsion.hpp"
#include "axionfem/units.hpp"

namespace axf {

using namespace axi;
using units::pi;

namespace {

// Mesh-resolution guard of the harmonic path: elements per shear wavelength N = c_T/(f_max h), h the longest edge
// of an element, minimum over the structure.  From the convergence study against the exact finite-tube modal sum
// (benchmarks/torsion_highfreq.py B1, docs/validation/torsion_highfreq.md; band 0.8-1 MHz,
// eta = 0.01 and 0.003, inner-traction and volume loads; the worst band-mean error over the cases):
//   P1: 0.85 % at N = 24.1, 9.1 % at 7.35;   P2: 1.00 % at 2.98, 0.21 % at 4.46, 4.2 % at 1.53 (coarsest studied);
//   P3: 0.08 % at 2.15, 1.1 % at 1.53, 0.68 % at 1.28 (coarsest studied).
// warn: below the resolution that keeps the band-mean error under about 1 %.
// refuse: P1 below about 10 %; P2 and P3 below the coarsest resolution studied (outside the validated range).
constexpr double kWarnPerWavelength[4] = {0.0, 24.0, 3.5, 2.0};
constexpr double kRefusePerWavelength[4] = {0.0, 7.0, 1.5, 1.25};

double table_at(const double* t, int order) { return t[std::max(1, std::min(order, 3))]; }

}  // namespace

json AxiModel::torsion(const json& req) const {
  AXF_REQUIRE(solved_, "torsion: solve the design first");
  const Space& sp = *sp_;
  mfem::Mesh& m = sp.mesh;
  const int ne = m.GetNE();

  // ---------------------------------------------------- strict request keys
  static const std::set<std::string> keys = {"nmodes", "shapes", "lanczos_extra", "load", "loads", "freqs_Hz",
                                             "band", "eta", "regions", "dofs", "verbose", "solver",
                                             "mesh_check"};
  AXF_REQUIRE(req.is_object(), "torsion: the request must be a JSON object");
  for (auto it = req.begin(); it != req.end(); ++it)
    AXF_REQUIRE(keys.count(it.key()), strcat("torsion: unknown request key '", it.key(), "'"), "",
                "Known keys: nmodes, shapes, lanczos_extra, load, loads, freqs_Hz, band, eta, regions, dofs, "
                "verbose, solver, mesh_check.");
  const bool want_harmonic = req.contains("freqs_Hz") || req.contains("band");
  AXF_REQUIRE(!want_harmonic || req.contains("load") || req.contains("loads"),
              "torsion: freqs_Hz or band needs a load: `load` (one azimuthal body-force value per DOF) or "
              "`loads` (body forces on regions, tractions on surfaces or curves)", "",
              "Example: loads=[{\"surface\": {\"on\": \"tube\", \"from\": \"vacuum\"}, \"traction_Pa\": 1.0}].");
  AXF_REQUIRE(!(req.contains("freqs_Hz") && req.contains("band")),
              "torsion: give freqs_Hz or band, not both");
  const json mc = req.value("mesh_check", json("error"));
  AXF_REQUIRE(mc.is_string() && (mc == "error" || mc == "warn" || mc == "off"),
              "torsion: mesh_check must be \"error\", \"warn\" or \"off\"", "/torsion/mesh_check");
  // a negative loss factor is an active structure (Re Y < 0, S_q < 0); a negative mode count is an error
  AXF_REQUIRE(req.value("eta", 0.0) >= 0.0, "torsion: eta must be >= 0 (a structural loss factor)");
  AXF_REQUIRE(req.value("nmodes", 0) >= 0, "torsion: nmodes must be >= 0");
  AXF_REQUIRE(req.value("lanczos_extra", 0) >= 0, "torsion: lanczos_extra must be >= 0");
  const json over = req.value("regions", json::object());
  AXF_REQUIRE(over.is_object(), "torsion: regions must be an object {<region name>: {...}}");
  static const std::set<std::string> region_keys = {"mu_Pa", "E_Pa", "nu", "density", "exclude"};
  for (auto it = over.begin(); it != over.end(); ++it) {
    AXF_REQUIRE(it.key() == "air" || d_.find_region(it.key()) != nullptr,
                strcat("torsion: regions names '", it.key(), "', which is not a region of the design"));
    AXF_REQUIRE(it.value().is_object(), strcat("torsion: regions.", it.key(), " must be an object"));
    for (auto jt = it.value().begin(); jt != it.value().end(); ++jt)
      AXF_REQUIRE(region_keys.count(jt.key()),
                  strcat("torsion: unknown key '", jt.key(), "' in regions.", it.key()), "",
                  "Known keys: mu_Pa, E_Pa (with optional nu), density, exclude.");
    AXF_REQUIRE(!it.value().contains("nu") || it.value().contains("E_Pa"),
                strcat("torsion: regions.", it.key(), ".nu is used only with E_Pa"));
    AXF_REQUIRE(!(it.value().contains("mu_Pa") && it.value().contains("E_Pa")),
                strcat("torsion: regions.", it.key(), " gives both mu_Pa and E_Pa; give one"));
  }

  // ---------------------------------------------------- element -> region / material / role
  static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
  std::vector<TorsionRole> roles(ne);
  std::vector<std::string> reg(ne);
  std::set<std::string> no_mech, no_rho, excluded, solid_regions;
  for (int e = 0; e < ne; ++e) {
    reg[e] = region_name(m.GetAttribute(e));
    const spec::Material* mt = &vacuum;
    if (reg[e] != "air") {
      const spec::Region* g = d_.find_region(reg[e]);
      if (g) mt = &d_.materials.at(g->material);
    }
    double mu = 0.0, rho = mt->density;
    bool have = mt->mech.given;
    if (have) mu = mt->mech.E_Pa / (2.0 * (1.0 + mt->mech.nu));
    if (over.contains(reg[e])) {
      const json& o = over.at(reg[e]);
      if (o.contains("mu_Pa")) { mu = o.at("mu_Pa").get<double>(); have = true; }
      if (o.contains("E_Pa")) {
        const double nu = o.value("nu", have ? mt->mech.nu : 0.3);
        mu = o.at("E_Pa").get<double>() / (2.0 * (1.0 + nu));
        have = true;
      }
      if (o.contains("density")) rho = o.at("density").get<double>();
      if (o.value("exclude", false)) {
        excluded.insert(reg[e]);
        continue;
      }
    }
    if (!have) {
      if (reg[e] != "air" && mt->kind != spec::MaterialKind::Vacuum) no_mech.insert(reg[e]);
      continue;
    }
    if (!(rho > 0)) { no_rho.insert(reg[e]); continue; }
    if (!(mu > 0)) { no_mech.insert(reg[e]); continue; }
    roles[e].solid = true;
    roles[e].mu = mu;
    roles[e].rho = rho;
    solid_regions.insert(reg[e]);
  }

  json out;
  out["warnings"] = json::array();
  auto warn_list = [&](const char* what, const std::set<std::string>& s) {
    if (s.empty()) return;
    std::string list;
    for (auto& x : s) list += (list.empty() ? "" : ", ") + x;
    out["warnings"].push_back(strcat("the torsional analysis skips the region(s) ", list, ": ", what,
                                     ". Give the material a \"mechanical\" block and a positive "
                                     "\"density\", or override it with "
                                     "regions: {<name>: {mu_Pa, density}}."));
  };
  warn_list("no mechanical data", no_mech);
  warn_list("no density", no_rho);
  if (!excluded.empty()) {
    std::string list;
    for (auto& x : excluded) list += (list.empty() ? "" : ", ") + x;
    out["excluded_regions"] = list;
  }
  {
    std::string list;
    for (auto& x : solid_regions) list += (list.empty() ? "" : ", ") + x;
    out["structure_regions"] = list;
  }

  Torsion T(sp, roles);
  out["diagnostics"] = T.diagnostics();

  if (req.value("dofs", false)) {
    std::vector<double> rr, zz;
    for (int d = 0; d < sp.ndof(); ++d) {
      rr.push_back(sp.r[d]);
      zz.push_back(sp.z[d]);
    }
    out["dof_r"] = rr;
    out["dof_z"] = zz;
    out["rtol"] = sp.rtol;
  }

  // ---------------------------------------------------- loads: nodal `load` and/or named `loads`
  std::vector<double> load;
  const bool has_load = req.contains("load") || req.contains("loads");
  std::vector<double> F(T.n_free(), 0.0);
  std::set<std::string> loaded_regions;   // for the resolution guard
  if (req.contains("load")) {
    load = req.at("load").get<std::vector<double>>();
    AXF_REQUIRE(static_cast<int>(load.size()) == sp.ndof(),
                strcat("torsion: load has ", load.size(), " values but the space has ", sp.ndof(), " DOFs"),
                "/torsion/load", "Get the DOF coordinates with dofs=True and give one value per DOF.");
    out["load"] = T.load_moments(load);   // what the FEM integrates
    F = T.load_vector(load);
    for (int e = 0; e < ne; ++e) {
      if (!roles[e].solid) continue;
      for (int d : sp.edofs(e))
        if (load[d] != 0.0) {
          loaded_regions.insert(reg[e]);
          break;
        }
    }
  }
  if (req.contains("loads")) {
    const json& L = req.at("loads");
    AXF_REQUIRE(L.is_array() && !L.empty(), "torsion: loads must be a non-empty list", "/torsion/loads");
    std::map<std::string, std::vector<int>> elems_of;
    for (int e = 0; e < ne; ++e) elems_of[reg[e]].push_back(e);
    // the value of an entry: a number, an expression in r and z [m], or one value per DOF (P_p interpolant)
    auto make_fn = [&](const json& v, const std::string& where) -> TorsionLoadFn {
      if (v.is_number()) {
        const double c = v.get<double>();
        return [c](int, const PointEval&) { return c; };
      }
      if (v.is_string()) {
        const std::string txt = v.get<std::string>();
        ParamMap probe{{"r", 1.0}, {"z", 0.0}};
        eval_expr(txt, probe, where);   // fails here, with the location, on a syntax error or unknown name
        return [txt, where](int, const PointEval& pe) {
          ParamMap pm{{"r", pe.r}, {"z", pe.z}};
          return eval_expr(txt, pm, where);
        };
      }
      AXF_REQUIRE(v.is_array(), "torsion: a load value must be a number, an expression in r and z, or a list "
                                "with one value per DOF", where);
      auto vals = std::make_shared<std::vector<double>>(v.get<std::vector<double>>());
      AXF_REQUIRE(static_cast<int>(vals->size()) == sp.ndof(),
                  strcat("torsion: the load list has ", vals->size(), " values, the space ", sp.ndof(), " DOFs"),
                  where, "Get the DOF coordinates with dofs=True.");
      return [vals, &sp](int e, const PointEval& pe) {
        const auto& d = sp.edofs(e);
        double g = 0.0;
        for (size_t i = 0; i < d.size(); ++i) g += pe.N[i] * (*vals)[d[i]];
        return g;
      };
    };
    json applied = json::array();
    TorsionLoadMoments total;
    for (size_t li = 0; li < L.size(); ++li) {
      const json& en = L[li];
      const std::string where = strcat("/torsion/loads/", li);
      AXF_REQUIRE(en.is_object(), "torsion: each loads entry must be an object", where);
      const int kinds = static_cast<int>(en.contains("region")) + static_cast<int>(en.contains("surface")) +
                        static_cast<int>(en.contains("curve"));
      AXF_REQUIRE(kinds == 1, "torsion: a loads entry needs exactly one of region, surface, curve", where,
                  "{\"region\": R, \"g_N_m3\": v} (body force), {\"surface\": {\"on\": R, \"from\": S or "
                  "\"vacuum\"}, \"traction_Pa\": v}, {\"curve\": C, \"traction_Pa\": v}.");
      for (auto it = en.begin(); it != en.end(); ++it) {
        static const std::set<std::string> ek = {"region", "surface", "curve", "g_N_m3", "traction_Pa", "name"};
        AXF_REQUIRE(ek.count(it.key()), strcat("torsion: unknown key '", it.key(), "' in a loads entry"), where,
                    "Known keys: region, surface, curve, g_N_m3, traction_Pa, name.");
      }
      TorsionLoadMoments mom;
      json rec{{"index", li}};
      if (en.contains("name")) rec["name"] = en.at("name");
      if (en.contains("region")) {
        const std::string R = en.at("region").get<std::string>();
        AXF_REQUIRE(en.contains("g_N_m3") && !en.contains("traction_Pa"),
                    "torsion: a region load takes g_N_m3 (a body force), not traction_Pa", where);
        AXF_REQUIRE(solid_regions.count(R), strcat("torsion: load region '", R, "' is not part of the structure"),
                    where + "/region", strcat("Structure regions: ", out["structure_regions"].get<std::string>(),
                                              ". A region needs a mechanical block and a density."));
        mom = T.add_volume_load(F, elems_of[R], make_fn(en.at("g_N_m3"), where + "/g_N_m3"));
        rec["region"] = R;
        loaded_regions.insert(R);
      } else {
        AXF_REQUIRE(en.contains("traction_Pa") && !en.contains("g_N_m3"),
                    "torsion: a surface or curve load takes traction_Pa (azimuthal, along +phi), not g_N_m3", where);
        std::vector<std::pair<int, int>> faces;
        if (en.contains("surface")) {
          const json& sf = en.at("surface");
          AXF_REQUIRE(sf.is_object() && sf.contains("on"),
                      "torsion: surface must be {\"on\": R, \"from\": S or \"vacuum\"}", where + "/surface");
          const std::string on = sf.at("on").get<std::string>();
          const std::string from = sf.value("from", std::string("vacuum"));
          AXF_REQUIRE(solid_regions.count(on), strcat("torsion: surface.on names '", on,
                                                      "', which is not part of the structure"), where + "/surface/on");
          AXF_REQUIRE(from == "vacuum" || from == "air" || d_.find_region(from) != nullptr,
                      strcat("torsion: surface.from names '", from, "', which is not a region"), where + "/surface/from",
                      "Use a region name, or \"vacuum\" for every element outside the structure.");
          for (int f = 0; f < m.GetNumFaces(); ++f) {
            int e1 = -1, e2 = -1;
            m.GetFaceElements(f, &e1, &e2);
            auto is_from = [&](int e) {
              if (e < 0) return from == "vacuum";             // the domain boundary
              return from == "vacuum" ? !roles[e].solid : reg[e] == from;
            };
            if (e1 >= 0 && reg[e1] == on && roles[e1].solid && is_from(e2)) faces.push_back({f, e1});
            else if (e2 >= 0 && reg[e2] == on && roles[e2].solid && is_from(e1)) faces.push_back({f, e2});
          }
          AXF_REQUIRE(!faces.empty(), strcat("torsion: no mesh face between '", on, "' and '", from, "'"),
                      where + "/surface", "The two sides must share a boundary in the (r, z) half-plane.");
          rec["surface"] = json{{"on", on}, {"from", from}};
        } else {
          const std::string C = en.at("curve").get<std::string>();
          auto it = mb_.attr_of_curve.find(C);
          AXF_REQUIRE(it != mb_.attr_of_curve.end(), strcat("torsion: '", C, "' is not a curve of the design"),
                      where + "/curve");
          for (int f = 0; f < m.GetNumFaces(); ++f) {
            if (sp.face_bdr_attr[f] != it->second) continue;
            int e1 = -1, e2 = -1;
            m.GetFaceElements(f, &e1, &e2);
            // both sides share the face DOFs: apply the traction once, through a structure side
            if (e1 >= 0 && roles[e1].solid) faces.push_back({f, e1});
            else if (e2 >= 0 && roles[e2].solid) faces.push_back({f, e2});
          }
          AXF_REQUIRE(!faces.empty(), strcat("torsion: curve '", C, "' touches no structure element"),
                      where + "/curve", "A traction on a curve loads the structure the curve lies on or bounds.");
          rec["curve"] = C;
        }
        mom = T.add_face_load(F, faces, make_fn(en.at("traction_Pa"), where + "/traction_Pa"));
        for (const auto& fe : faces) loaded_regions.insert(reg[fe.second]);
        rec["faces"] = faces.size();
      }
      rec.update(mom.to_json());
      applied.push_back(rec);
      total.add(mom);
    }
    out["loads"] = json{{"entries", applied}, {"total", total.to_json()},
                        {"note", "integrals of the loads as applied (exact at quadrature points; full ring)"}};
  }

  const int nmodes = req.value("nmodes", 0);
  if (nmodes > 0) {
    const bool shapes = req.value("shapes", false) || has_load;
    TorsionModes md = T.modes(nmodes, shapes, req.value("lanczos_extra", 24),
                              req.value("verbose", 0));
    out["modes"] = json{{"f_Hz", md.f_Hz},
                        {"residual", md.residual},
                        {"modal_mass_kg", md.modal_mass},
                        {"diagnostics", md.diagnostics}};
    const auto& unconv = md.diagnostics["unconverged_modes"];
    if (!unconv.empty())
      out["warnings"].push_back(strcat("torsion: ", unconv.size(), " mode(s) have a relative residual above "
                                       "1e-6 (indices ", unconv.dump(), "); raise lanczos_extra"));
    const json& inertia = md.diagnostics["inertia"];
    if (inertia.contains("complete") && inertia["complete"].is_null())
      out["warnings"].push_back("torsion: the inertia check could not factorize K - lam_bar M; the completeness of "
                                "the returned modes is not checked");
    if (inertia.contains("complete") && inertia["complete"].is_boolean() && !inertia["complete"].get<bool>())
      out["warnings"].push_back(strcat("torsion: the inertia count finds ", inertia["eigenvalues_below"].dump(),
                                       " eigenvalues up to the highest returned one, but ",
                                       inertia["returned"].dump(), " modes were returned: a mode (or a copy "
                                       "of a repeated one) is missing; raise lanczos_extra or nmodes"));
    if (has_load) out["modes"]["participation"] = T.participation_F(md, F);
    if (req.value("shapes", false)) out["modes"]["shape"] = md.shape;
  }

  if (has_load && want_harmonic) {
    std::vector<double> freqs;
    json band_spec;
    double band_length = 0.0;
    if (req.contains("band")) {
      band_spec = req.at("band");
      AXF_REQUIRE(band_spec.is_object() && band_spec.contains("f_Hz") && band_spec.at("f_Hz").size() == 2,
                  "torsion: band must be {\"f_Hz\": [f_min, f_max], \"n\": N}", "/torsion/band");
      for (auto it = band_spec.begin(); it != band_spec.end(); ++it)
        AXF_REQUIRE(it.key() == "f_Hz" || it.key() == "n" || it.key() == "length_m",
                    strcat("torsion: unknown key '", it.key(), "' in band"), "/torsion/band",
                    "Known keys: f_Hz, n, length_m.");
      const double f0 = band_spec.at("f_Hz")[0].get<double>();
      const double f1 = band_spec.at("f_Hz")[1].get<double>();
      const int n = band_spec.value("n", 201);
      AXF_REQUIRE(f0 > 0 && f1 > f0, "torsion: band.f_Hz must be [f_min, f_max] with 0 < f_min < f_max",
                  "/torsion/band/f_Hz");
      AXF_REQUIRE(n >= 5 && n % 2 == 1, "torsion: band.n must be odd and >= 5 (the uncertainty of the band "
                  "mean compares the full grid with every second point)", "/torsion/band/n");
      if (band_spec.contains("length_m")) {
        band_length = band_spec.at("length_m").get<double>();
        AXF_REQUIRE(band_length > 0, "torsion: band.length_m must be positive (the structure's axial length)",
                    "/torsion/band/length_m");
      }
      for (int i = 0; i < n; ++i) freqs.push_back(f0 + (f1 - f0) * i / (n - 1));
    } else {
      freqs = req.at("freqs_Hz").get<std::vector<double>>();
    }
    AXF_REQUIRE(!freqs.empty(), "torsion: no frequencies", "/torsion/freqs_Hz");
    for (double f : freqs) AXF_REQUIRE(f > 0, "torsion: frequencies must be positive", "/torsion/freqs_Hz");
    const double eta = req.value("eta", 1e-4);
    AXF_REQUIRE(eta >= 0, "torsion: eta must be >= 0", "/torsion/eta");

    // ---- mesh-resolution guard (elements per shear wavelength at the highest frequency).  It refuses when the
    // 1st percentile of the elements of any loaded region (or of the whole structure) is below the refuse
    // threshold, so that one sliver in an unloaded part does not refuse a request but an under-resolved loaded
    // region does; it warns when any region's minimum is below the warn threshold.
    const double fmax = *std::max_element(freqs.begin(), freqs.end());
    json res = T.resolution(fmax, [&](int e) { return reg[e]; });
    const double warn_n = table_at(kWarnPerWavelength, sp.order);
    const double refuse_n = table_at(kRefusePerWavelength, sp.order);
    res["warn_below"] = warn_n;
    res["refuse_below"] = refuse_n;
    res["loaded_regions"] = loaded_regions;
    res["rule"] = "refuse when the 1st percentile of a loaded region, or of the structure, is below refuse_below; "
                  "warn when any region's minimum is below warn_below";
    const double have = res["elements_per_wavelength_min"].get<double>();
    double gate = res["elements_per_wavelength_p01"].get<double>();
    std::string gate_where = "the structure";
    for (const auto& r : loaded_regions) {
      if (!res["per_region"].contains(r)) continue;
      const double p01 = res["per_region"][r]["p01"].get<double>();
      if (p01 < gate) {
        gate = p01;
        gate_where = "loaded region '" + r + "'";
      }
    }
    res["elements_per_wavelength_gate"] = gate;
    if (mc != "off" && (have < warn_n || gate < refuse_n)) {
      const std::string wr = res["worst_region"].get<std::string>();
      const bool refuse = mc == "error" && gate < refuse_n;
      std::string tail;
      if (refuse) tail = "mesh_check=\"warn\" would proceed anyway.";
      else if (mc == "warn" && gate < refuse_n) tail = "Proceeding because mesh_check=\"warn\".";
      else tail = "The 1st percentile of every loaded region is above the refuse threshold, so it proceeds.";
      const std::string msg = strcat(
          "torsion: the mesh has ", have, " elements per shear wavelength (minimum, region '", wr, "'; 1st "
          "percentile of ", gate_where, ": ", gate, ") at ", fmax, " Hz (order ", sp.order, "); the band-mean "
          "mobility needs >= ", warn_n, " for about 1 % error, and ", refuse_n, " is the coarsest validated "
          "(docs/physics/torsion_highfreq.md). Set the region's mesh_size to <= c_T/(f_max ", warn_n,
          ") or raise mesh.order. ", tail);
      AXF_REQUIRE(!refuse, msg, "/torsion/mesh_check");
      out["warnings"].push_back(msg);
    }

    // ---- solve
    TorsionHarmonicOptions opt;
    opt.eta = eta;
    opt.verbose = req.value("verbose", 0);
    const json sv = req.value("solver", json::object());
    AXF_REQUIRE(sv.is_object(), "torsion: solver must be an object", "/torsion/solver");
    for (auto it = sv.begin(); it != sv.end(); ++it)
      AXF_REQUIRE(it.key() == "backend" || it.key() == "parallel_frequencies",
                  strcat("torsion: unknown key '", it.key(), "' in solver"), "/torsion/solver",
                  "Known keys: backend (auto, eigen, strumpack), parallel_frequencies.");
    const std::string be = sv.value("backend", std::string("auto"));
    AXF_REQUIRE(be == "auto" || be == "eigen" || be == "strumpack",
                "torsion: solver.backend must be auto, eigen or strumpack", "/torsion/solver/backend");
    opt.backend = be == "auto" ? -1 : (be == "eigen" ? 0 : 1);
    const json pf = sv.value("parallel_frequencies", json(1));
    if (pf.is_string()) {
      AXF_REQUIRE(pf == "auto", "torsion: solver.parallel_frequencies must be a positive integer or \"auto\"",
                  "/torsion/solver/parallel_frequencies");
      opt.parallel_frequencies = 0;   // measured choice in Torsion::harmonic_F
    } else {
      AXF_REQUIRE(pf.is_number_integer() && pf.get<int>() >= 1,
                  "torsion: solver.parallel_frequencies must be a positive integer or \"auto\"",
                  "/torsion/solver/parallel_frequencies");
      opt.parallel_frequencies = pf.get<int>();
    }
    json timing;
    auto rs = T.harmonic_F(F, freqs, opt, &timing);
    json h = json::array();
    std::vector<double> failed;
    for (const auto& r : rs) {
      h.push_back(json{{"f_Hz", r.f_Hz}, {"q_re", r.q_re}, {"q_im", r.q_im},
                       {"Y_re", r.Y_re}, {"Y_im", r.Y_im}, {"ok", r.lu_ok}});
      if (!r.lu_ok) failed.push_back(r.f_Hz);
    }
    out["harmonic"] = json{{"eta", eta}, {"response", h}, {"resolution", res}, {"timing", timing},
                           {"convention", "e^{-i omega t}: (K (1 - i eta) - omega^2 M) u = F, "
                                          "Y = -i omega q; Re Y > 0 for a passive structure"}};
    if (!failed.empty())
      out["warnings"].push_back(strcat("torsion: the factorization failed at ", failed.size(), " frequency(ies) (",
                                       json(failed).dump(), " Hz); their rows have ok = 0 and Y = 0"));
    if (!req.contains("band"))
      out["harmonic"]["note"] = "single-frequency values are not certified by the mesh guard: it is calibrated on "
                                "band means, and at its thresholds the median single-frequency error is 4-22 % "
                                "(validation B1); use band, or a finer mesh, for individual frequencies";
    if (req.contains("band")) {
      // a failed factorization would enter the mean as Y = 0
      AXF_REQUIRE(failed.empty(), strcat("torsion: the factorization failed at ", failed.size(),
                                         " frequency(ies) of the band (", json(failed).dump(),
                                         " Hz), so the band mean is not computed"),
                  "/torsion/band", "Check the materials (mu, density > 0) and eta; try solver.backend = \"eigen\".");
      // Band mean of Re Y (trapezoid) and two diagnostics:
      //  quadrature_uncertainty: |mean on the full grid - mean on every second point| (resolution of the peaks);
      //  ripple_sem: standard error of the 8 sub-band means of Re Y minus its quadratic trend in f.  A
      //    diagnostic of the modal ripple, NOT a bound or an estimate of the finite-length scatter: in the
      //    cross-validation X it is below |F/A - 1| in 8 of 108 rows (all among the 14 above 1 %) and about 350x
      //    above it at the median.
      //  edge_estimate (with band.length_m = L): rms(ripple) sqrt(2) (c_T/L)/(pi B), the band mean of a ripple
      //    with the T(0,1) even-mode spacing at the worst edge phase, c_T the largest in
      //    the structure; assumes B >> c_T/L.  A conservative scale, not an estimate: in validation X it is 58x
      //    |F/A - 1| at the median and below it in 6 of 108 rows, by up to 17 %.
      const int n = static_cast<int>(freqs.size());
      auto trap = [&](int stride) {
        double s2 = 0.0, wsum = 0.0;
        for (int i = 0; i + stride < n; i += stride) {
          const double hseg = freqs[i + stride] - freqs[i];
          s2 += 0.5 * hseg * (rs[i].Y_re + rs[i + stride].Y_re);
          wsum += hseg;
        }
        return s2 / wsum;
      };
      const double mean = trap(1);
      const double coarse = trap(2);
      const double fc = 0.5 * (freqs.front() + freqs.back());
      const double fh = 0.5 * (freqs.back() - freqs.front());
      Eigen::Matrix3d AtA = Eigen::Matrix3d::Zero();
      Eigen::Vector3d Atb = Eigen::Vector3d::Zero();
      for (int i = 0; i < n; ++i) {
        const double x = (freqs[i] - fc) / fh;
        const Eigen::Vector3d phi(1.0, x, x * x);
        AtA += phi * phi.transpose();
        Atb += phi * rs[i].Y_re;
      }
      const Eigen::Vector3d cfit = AtA.ldlt().solve(Atb);
      std::vector<double> resid(n);
      double rms = 0.0;
      for (int i = 0; i < n; ++i) {
        const double x = (freqs[i] - fc) / fh;
        resid[i] = rs[i].Y_re - (cfit[0] + cfit[1] * x + cfit[2] * x * x);
        rms += resid[i] * resid[i];
      }
      rms = std::sqrt(rms / n);
      const double df = freqs[1] - freqs[0];
      json bj{{"f_Hz", band_spec.at("f_Hz")}, {"n", n}, {"mean_Re_Y", mean},
              {"quadrature_uncertainty", std::fabs(mean - coarse)}, {"ripple_rms", rms}, {"df_Hz", df},
              {"half_power_width_at_f_min_Hz", eta * freqs.front()}};
      // ripple_sem: 8 disjoint sub-bands of at least 2 points each (n >= 17)
      const int nsub = 8;
      if (n >= 17) {
        std::vector<double> sub, subres;
        for (int b = 0; b < nsub; ++b) {
          const int lo = b * n / nsub, hi = (b + 1) * n / nsub;   // [lo, hi): disjoint
          double s2 = 0.0, sr = 0.0;
          for (int i = lo; i < hi; ++i) {
            s2 += rs[i].Y_re;
            sr += resid[i];
          }
          sub.push_back(s2 / (hi - lo));
          subres.push_back(sr / (hi - lo));
        }
        double sm = 0.0, ss = 0.0;
        for (double x : subres) sm += x;
        sm /= nsub;
        for (double x : subres) ss += (x - sm) * (x - sm);
        bj["subband_means"] = sub;
        bj["ripple_sem"] = std::sqrt(ss / (nsub - 1) / nsub);
      } else {
        bj["ripple_sem"] = nullptr;
      }
      if (band_length > 0) {
        double cmax = 0.0;
        for (const auto& ro : roles)
          if (ro.solid) cmax = std::max(cmax, std::sqrt(ro.mu / ro.rho));
        bj["length_m"] = band_length;
        bj["edge_estimate"] = rms * std::sqrt(2.0) * (cmax / band_length) / (pi * (freqs.back() - freqs.front()));
      }
      bj["note"] = "mean_Re_Y: trapezoid over the band. quadrature_uncertainty: |full grid - every second point| "
                   "(the coarse grid's error, an upper estimate of the grid's). ripple_sem: standard error of 8 "
                   "disjoint sub-band means of Re Y minus its quadratic trend, a diagnostic of the modal ripple, "
                   "not a bound or an estimate of the finite-length scatter (n >= 17). edge_estimate (needs "
                   "band.length_m): rms ripple x sqrt(2) (c_T/L)/(pi B), a conservative scale of the band "
                   "mean's finite-length deviation from a smooth value (on uniform tubes 58x the deviation at "
                   "the median, below it in 6 of 108 cases by up to 17 %; assumes B >> c_T/L; "
                   "validation/torsion_highfreq.md, X)";
      out["harmonic"]["band"] = bj;
      if (df > 0.5 * eta * freqs.front())
        out["warnings"].push_back(strcat("torsion: the band grid step ", df, " Hz exceeds half the resonance "
                                         "half-power width eta f_min = ", eta * freqs.front(),
                                         " Hz: the peaks are undersampled; raise band.n (see "
                                         "band.quadrature_uncertainty)"));
    }
  }
  if (nmodes <= 0 && !want_harmonic) {
    std::string what = "nothing was computed";
    if (has_load && req.value("dofs", false)) what = "only the DOF coordinates and the applied load were returned";
    else if (has_load) what = "only the applied load's integrals (load, loads) were returned";
    else if (req.value("dofs", false)) what = "only the DOF coordinates were returned";
    out["note"] = strcat("torsion: no nmodes and no freqs_Hz in the request: ", what);
  }
  return out;
}

}  // namespace axf
