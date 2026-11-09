// Parametric design templates.  Each template returns a complete design
// document whose geometry is written in terms of named parameters, so that a
// person or an agent changes a design by editing "parameters".
#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace axf::templates {

using json = nlohmann::json;

// [{"name", "description", "parameters": {name: {"default", "units", "meaning"}}}]
json list();

// Instantiate a template; `overrides` replaces parameter defaults (numbers or
// expressions) and may also override top-level sections (e.g. "mesh").
json make(const std::string& name, const json& overrides = json::object());

}  // namespace axf::templates
