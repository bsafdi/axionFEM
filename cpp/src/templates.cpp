#include "axionfem/templates.hpp"

#include <map>

#include "axionfem/core.hpp"

namespace axf::templates {

namespace {

struct Param {
  json value;
  std::string units, meaning;
};

struct Template {
  std::string name, description;
  std::vector<std::pair<std::string, Param>> params;
  json body;  // design without "parameters"
};

json rect(const std::string& r0, const std::string& r1, const std::string& z0, const std::string& z1,
          const std::string& fillet = "") {
  json s = {{"r", {r0, r1}}, {"z", {z0, z1}}};
  if (!fillet.empty()) s["fillet"] = fillet;
  return {{"rect", s}};
}

json frame(const std::string& r0, const std::string& r1, const std::string& z0, const std::string& z1,
           const std::string& t, const std::string& fillet = "") {
  json s = {{"r", {r0, r1}}, {"z", {z0, z1}}, {"thickness", t}};
  if (!fillet.empty()) s["fillet"] = fillet;
  return {{"frame", s}};
}

std::vector<Template> all() {
  std::vector<Template> T;

  // ---------------------------------------------------------------- toroid
  {
    Template t;
    t.name = "toroid";
    t.description =
        "Toroidal magnet (ABRACADABRA / DMRadio-GUT topology). The static field B_phi fills the rectangular box "
        "y R < r < R, |z| < H/2, wound by a poloidal winding of thickness tw. A superconducting sheath of thickness "
        "ts encloses the winding with clearance g and carries the readout current around the central hole through a "
        "meridional slit (azimuthal pickup). For a long toroid the geometry factor is g = y^2 ln(1/y), largest at y = e^(-1/2).";
    t.params = {{"R", {2.0, "m", "outer radius of the field box"}},
                {"y", {"exp(-0.5)", "", "inner radius of the field box as a fraction of R"}},
                {"H", {2.0, "m", "height of the field box"}},
                {"tw", {"0.01*R", "m", "winding thickness"}},
                {"g", {"0.005*R", "m", "clearance between winding and sheath"}},
                {"ts", {"0.005*R", "m", "sheath wall thickness"}},
                {"f", {"0.02*R", "m", "outer corner radius of the sheath"}},
                {"NI", {1.0e6, "A", "total ampere-turns of the toroidal winding"}}};
    t.body = {
        {"regions",
         {{{"name", "winding"},
           {"shape", frame("y*R-tw", "R+tw", "-H/2-tw", "H/2+tw", "tw")},
           {"material", "winding"}},
          {{"name", "sheath"},
           {"shape", frame("y*R-tw-g-ts", "R+tw+g+ts", "-H/2-tw-g-ts", "H/2+tw+g+ts", "ts", "f")},
           {"material", "nb"}}}},
        {"windings", {{{"name", "toroid"}, {"region", "winding"}, {"type", "poloidal"}, {"ampere_turns", "NI"}}}},
        {"superconductors",
         {{{"body", "sheath"}, {"azimuthal", {{"net_current", 0}}}, {"poloidal", {{"net_current", 0}}},
           {"azimuthal_slit", true}}}},
        {"pickup", {{"type", "azimuthal"}, {"conductor", "sheath"}}},
        {"mesh", {{"order", 2}, {"size", "max(R, H)/30"}, {"surface_size", "min(ts, g, tw)"}}}};
    T.push_back(t);
  }

  // ---------------------------------------------------------------- solenoid + coax
  {
    Template t;
    t.name = "solenoid";
    t.description =
        "Solenoid magnet with an immersed shorted coaxial pickup (DMRadio-m^3 topology). The winding (inner radius R, "
        "thickness tw, height H) produces an axial field that penetrates the coax; the coax walls (copper: transparent "
        "to the static field, perfect conductors at RF) have inner radius x R and outer radius R - g. The readout port "
        "spans the bottom gap between the conductors; the top is shorted. For a long solenoid "
        "the geometry factor is g = (1 - x^2)^2 / (4 ln 1/x), largest at x = 0.534.";
    t.params = {{"R", {1.0, "m", "winding inner radius"}},
                {"H", {2.0, "m", "winding height"}},
                {"x", {0.534, "", "coax inner radius as a fraction of R"}},
                {"tw", {"0.01*R", "m", "winding thickness"}},
                {"g", {"0.005*R", "m", "radial clearance between coax outer wall and winding"}},
                {"tc", {"0.005*R", "m", "coax wall thickness"}},
                {"tp", {"0.01*R", "m", "height of the readout port gap"}},
                {"NI", {1.0e6, "A", "total ampere-turns of the solenoid"}}};
    t.body = {
        {"regions",
         {{{"name", "winding"}, {"shape", rect("R", "R+tw", "-H/2", "H/2")}, {"material", "winding"}},
          {{"name", "coax_inner"}, {"shape", rect("x*R-tc", "x*R", "-H/2", "H/2")}, {"material", "copper"}},
          {{"name", "coax_outer"}, {"shape", rect("R-g-tc", "R-g", "-H/2", "H/2")}, {"material", "copper"}},
          {{"name", "coax_short"}, {"shape", rect("x*R-tc", "R-g", "H/2-tc", "H/2")}, {"material", "copper"}},
          {{"name", "port"}, {"shape", rect("x*R", "R-g-tc", "-H/2", "-H/2+tp")}, {"material", "vacuum"}}}},
        {"windings", {{{"name", "solenoid"}, {"region", "winding"}, {"type", "azimuthal"}, {"ampere_turns", "NI"}}}},
        {"pickup", {{"type", "poloidal"}, {"port", "port"}}},
        {"benchmarks", {"optimistic", "conservative_solenoid"}},
        {"mesh", {{"order", 2}, {"size", "max(R, H)/30"}, {"surface_size", "min(tc, g)"}}}};
    T.push_back(t);
  }

  // ---------------------------------------------------------------- CORE
  {
    Template t;
    t.name = "core";
    t.description =
        "CORE topology (DMRadio-Core): a split solenoid of inner radius Rc, thickness tw and height H drives an "
        "axial field through the bore; a superconducting toroidal cavity (inner radius a, outer radius b, height h, "
        "wall w, corner radius f) surrounds it and carries the readout current poloidally. The cavity's inner wall "
        "has a circumferential neck gap of height gn at z = 0, which is the readout port (tuning capacitor to an "
        "on-axis readout); the coil is split by gc to let the neck through. The cavity must stay superconducting: "
        "check the superconductor surface field. The neck lips are rounded with radius fl (fl = 0 gives sharp lips, "
        "where the surface field is singular and grows under mesh refinement). The cavity's inner corners are "
        "rounded with radius max(f - w, 0) and its outer corners with f. Optional iron caps above and below the "
        "cavity (iron_t > 0, built-in nonlinear 'iron') route return flux. "
        "For a long CORE the geometry factor is g = y^2 ln(1/y) with y = a/b.";
    t.params = {{"Rc", {1.0, "m", "coil inner radius"}},
                {"tw", {"0.02*Rc", "m", "coil thickness"}},
                {"H", {2.0, "m", "total coil height (both halves)"}},
                {"gc", {"0.02*Rc", "m", "gap between the two coil halves"}},
                {"cl", {"0.05*Rc", "m", "radial clearance between coil and cavity inner wall"}},
                {"a", {"Rc+tw+cl+w", "m", "cavity inner radius (cavity-side surface of the inner wall)"}},
                {"b", {"2.5*Rc", "m", "cavity outer radius"}},
                {"h", {"H", "m", "cavity height"}},
                {"w", {"0.01*Rc", "m", "cavity wall thickness"}},
                {"f", {"0.05*Rc", "m", "cavity outer corner radius"}},
                {"gn", {"0.01*Rc", "m", "neck gap height"}},
                {"fl", {"0.3*w", "m", "neck lip corner radius (0 = sharp lips; must be < w/2)"}},
                {"NI", {1.0e6, "A", "total ampere-turns (both halves)"}},
                {"iron_t", {0.0, "m", "iron cap thickness (0 = no iron)"}},
                {"iron_r0", {"0", "m", "iron cap inner radius"}},
                {"iron_r1", {"b+w", "m", "iron cap outer radius"}},
                {"iron_dz", {"0.02*Rc", "m", "clearance between cavity outer surface and each iron cap"}}};
    t.body = {
        {"regions",
         {{{"name", "coil_upper"}, {"shape", rect("Rc", "Rc+tw", "gc/2", "H/2")}, {"material", "winding"}},
          {{"name", "coil_lower"}, {"shape", rect("Rc", "Rc+tw", "-H/2", "-gc/2")}, {"material", "winding"}},
          // neck lips: short pieces with rounded corners (radius fl) replace the cut ends of the inner wall
          {{"name", "cavity"},
           {"shape",
            {{"union",
              {{{"difference",
                 {frame("a-w", "b+w", "-h/2-w", "h/2+w", "w", "f"), rect("a-w", "a", "-gn/2-w", "gn/2+w")}}},
               rect("a-w", "a", "gn/2", "gn/2+2*w", "fl"),
               rect("a-w", "a", "-gn/2-2*w", "-gn/2", "fl")}}}},
           {"material", "nb"}},
          // the port spans the flat part of the lips (no cusps against their rounded corners)
          {{"name", "neck"}, {"shape", rect("a-w+fl", "a-fl", "-gn/2", "gn/2")}, {"material", "vacuum"}},
          {{"name", "iron_cap_upper"},
           {"shape", rect("iron_r0", "iron_r1", "h/2+w+iron_dz", "h/2+w+iron_dz+iron_t")},
           {"material", "iron"},
           {"enabled", "iron_t"}},
          {{"name", "iron_cap_lower"},
           {"shape", rect("iron_r0", "iron_r1", "-h/2-w-iron_dz-iron_t", "-h/2-w-iron_dz")},
           {"material", "iron"},
           {"enabled", "iron_t"}}}},
        {"windings",
         {{{"name", "upper"}, {"region", "coil_upper"}, {"type", "azimuthal"}, {"ampere_turns", "NI/2"}},
          {{"name", "lower"}, {"region", "coil_lower"}, {"type", "azimuthal"}, {"ampere_turns", "NI/2"}}}},
        {"superconductors", {{{"body", "cavity"}, {"azimuthal", {{"net_current", 0}}}}}},
        {"pickup", {{"type", "poloidal"}, {"port", "neck"}}},
        {"cost", {{"sc_wall_cap", 0.086}}},
        {"mesh", {{"order", 2}, {"size", "max(b, H)/30"}, {"surface_size", "min(w, gn)/2"}}}};
    T.push_back(t);
  }
  return T;
}

}  // namespace

json list() {
  json out = json::array();
  for (auto& t : all()) {
    json p = json::object();
    for (auto& [k, v] : t.params) p[k] = {{"default", v.value}, {"units", v.units}, {"meaning", v.meaning}};
    out.push_back({{"name", t.name}, {"description", t.description}, {"parameters", p}});
  }
  return out;
}

json make(const std::string& name, const json& overrides) {
  for (auto& t : all()) {
    if (t.name != name) continue;
    json d = {{"schema", "axionfem.design/1"}, {"name", name}, {"description", t.description},
              {"template", name}};
    json params = json::object();
    for (auto& [k, v] : t.params) params[k] = v.value;
    json body = t.body;
    if (overrides.is_object()) {
      for (auto it = overrides.begin(); it != overrides.end(); ++it) {
        if (params.contains(it.key())) {
          params[it.key()] = it.value();
        } else if (it.key() == "parameters" && it.value().is_object()) {
          for (auto p = it.value().begin(); p != it.value().end(); ++p) {
            if (!params.contains(p.key())) {
              throw Error("template '" + name + "' has no parameter '" + p.key() + "'", "/parameters/" + p.key(),
                          "Call templates.list() to see the parameters of each template.");
            }
            params[p.key()] = p.value();
          }
        } else if (it.key() == "name") {
          d["name"] = it.value();
        } else if (it.key() == "description" || it.key() == "regions" || it.key() == "curves" ||
                   it.key() == "windings" || it.key() == "superconductors" || it.key() == "pickup" ||
                   it.key() == "benchmarks" || it.key() == "scan" || it.key() == "cost" || it.key() == "mesh" ||
                   it.key() == "materials" || it.key() == "conductors" || it.key() == "hts" ||
                   it.key() == "background" || it.key() == "fullwave" || it.key() == "notes") {
          body[it.key()] = it.value();
        } else {
          std::string known;
          for (auto p = params.begin(); p != params.end(); ++p) known += (known.empty() ? "" : ", ") + p.key();
          throw Error("template '" + name + "' has no parameter or section '" + it.key() + "'",
                      "/parameters/" + it.key(), "Parameters of this template: " + known + ".");
        }
      }
    }
    d["parameters"] = params;
    for (auto it = body.begin(); it != body.end(); ++it) d[it.key()] = it.value();
    if (d.contains("template")) {
      d["notes"] = "generated from template '" + name + "'";
      d.erase("template");
    }
    return d;
  }
  std::string names;
  for (auto& t : all()) names += (names.empty() ? "" : ", ") + t.name;
  throw Error("unknown template '" + name + "'", "", "Available templates: " + names + ".");
}

}  // namespace axf::templates
