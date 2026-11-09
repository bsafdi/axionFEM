// Headless PNG snapshots of a solved design (axionfem.scene/1, see
// docs/scene_format.md).  Pure C++: a software rasterizer with supersampling,
// an embedded bitmap font and exact colormap tables; PNG encoding through the
// vendored stb_image_write.  Deterministic: the same scene and options give
// the same bytes.  See docs/render.md.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace axf::render {

struct RenderOptions {
  std::string view = "section";  // "section" (meridional cut, mirrored) or "3d" (cutaway)
  std::string quantity = "Bmag";  // a key of scene.field.quantities, or "none" (materials only)
  bool log_scale = false;         // log10 colour scale (signed quantities: log10 |q|)
  int width = 1600, height = 1000;
  bool field_lines = true;        // draw scene.field_lines.static / .pickup
  bool annotate = true;           // results panel (summary numbers, region legend)
  double cutaway_deg = 270, elev_deg = 25, azim_deg = -60;  // 3d view
  std::string theme = "light";    // "light" or "dark"
  // Extensions (defaults keep the automatic behaviour).
  double vmin = NAN, vmax = NAN;  // fixed colour range in the quantity's units (NaN: automatic)
  int supersample = 2;            // render at this factor and box-filter down (1..4)
  std::string title;              // replaces the automatic title when non-empty
};

// An 8-bit RGB raster plus where the plot landed, so that callers (and tests)
// can map pixels back to metres.
struct Image {
  int width = 0, height = 0;
  std::vector<unsigned char> rgb;  // row-major, top row first, 3 bytes per pixel
  // Plot frame in output pixels [x0, x1) x [y0, y1) and, for the section view,
  // the data window it shows: x = +-r in metres (horizontal) and z (vertical,
  // up).  For the 3d view the window fields are NaN.
  int plot_x0 = 0, plot_y0 = 0, plot_x1 = 0, plot_y1 = 0;
  double x_min = NAN, x_max = NAN, z_min = NAN, z_max = NAN;
};

// Render a scene to an RGB raster.  Throws axf::Error on malformed input.
Image render_image(const nlohmann::json& scene, const RenderOptions& o = {});

// PNG bytes of a raster, and of a scene rendered with the given options.
std::vector<unsigned char> encode_png(const Image& img);
std::vector<unsigned char> render_png(const nlohmann::json& scene, const RenderOptions& o = {});

// Render and write a PNG file; write raw PNG bytes to a file.
void write_png(const std::string& path, const nlohmann::json& scene, const RenderOptions& o = {});
void write_file(const std::string& path, const std::vector<unsigned char>& bytes);

// Options from a JSON object with the field names above, e.g.
// {"view": "3d", "quantity": "Br", "width": 1200}.  Unknown keys and wrong
// types raise axf::Error naming the key.
RenderOptions options_from_json(const nlohmann::json& j);
nlohmann::json options_to_json(const RenderOptions& o);

}  // namespace axf::render
