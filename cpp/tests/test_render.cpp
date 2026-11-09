// Tests for the PNG snapshot renderer (axf::render).
//
// 1. A synthetic scene with known content checks orientation (z up, r to the
//    right, mirror image at r < 0), equal aspect, colormap values, NaN
//    handling (material shows through), determinism and error messages.
// 2. If the sample scenes runs/<name>/scene.json exist, both views of each are
//    rendered to PNG files in the current directory and checked for a valid
//    PNG header, the requested size and non-blank content.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "axionfem/core.hpp"
#include "axionfem/render.hpp"

using json = nlohmann::json;
namespace R = axf::render;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (cond) {                                                              \
      ++g_pass;                                                              \
    } else {                                                                 \
      ++g_fail;                                                              \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                        \
  } while (0)

static std::string b64(const void* data, size_t n) {
  static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto* p = static_cast<const unsigned char*>(data);
  std::string o;
  size_t i = 0;
  for (; i + 2 < n; i += 3) {
    const uint32_t v = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
    o += t[(v >> 18) & 63], o += t[(v >> 12) & 63], o += t[(v >> 6) & 63], o += t[v & 63];
  }
  if (i < n) {
    uint32_t v = p[i] << 16;
    if (i + 1 < n) v |= p[i + 1] << 8;
    o += t[(v >> 18) & 63], o += t[(v >> 12) & 63];
    o += (i + 1 < n) ? t[(v >> 6) & 63] : '=';
    o += '=';
  }
  return o;
}
static json f32(const std::vector<float>& v, int cols) {
  return {{"dtype", "float32"}, {"shape", {static_cast<long>(v.size()) / cols, cols}}, {"data", b64(v.data(), v.size() * 4)}};
}
static json u32(const std::vector<uint32_t>& v, int cols) {
  return {{"dtype", "uint32"}, {"shape", {static_cast<long>(v.size()) / cols, cols}}, {"data", b64(v.data(), v.size() * 4)}};
}

// Field on a structured grid over [0, 1.05] x [-1.05, 1.05]: Bmag = z + 1.05
// (increases upwards), Br = r z (signed).  A superconducting block occupies
// r in [0.5, 0.7], z in [0.2, 0.6]; the field is NaN there (as excluded).
static json synthetic_scene() {
  const int nr = 42, nz = 84;
  std::vector<float> V, bm, br;
  std::vector<uint32_t> I;
  auto in_block = [](double r, double z) { return r > 0.47 && r < 0.73 && z > 0.17 && z < 0.63; };
  for (int j = 0; j <= nz; ++j)
    for (int i = 0; i <= nr; ++i) {
      const double r = 1.05 * i / nr, z = -1.05 + 2.1 * j / nz;
      V.push_back(static_cast<float>(r));
      V.push_back(static_cast<float>(z));
      bm.push_back(in_block(r, z) ? NAN : static_cast<float>(z + 1.05));
      br.push_back(in_block(r, z) ? NAN : static_cast<float>(r * z));
    }
  for (int j = 0; j < nz; ++j)
    for (int i = 0; i < nr; ++i) {
      const uint32_t a = j * (nr + 1) + i, b = a + 1, c = a + nr + 1, d = c + 1;
      I.insert(I.end(), {a, b, d, a, d, c});
    }
  std::vector<float> bv = {0.5f, 0.2f, 0.7f, 0.2f, 0.7f, 0.6f, 0.5f, 0.6f};
  std::vector<uint32_t> bi = {0, 1, 2, 0, 2, 3};
  std::vector<float> outline = {0.5f, 0.2f, 0.7f, 0.2f, 0.7f, 0.6f, 0.5f, 0.6f, 0.5f, 0.2f};
  // a few static field lines (vertical segments at r = 0.3)
  std::vector<float> fl;
  for (int k = 0; k < 20; ++k) {
    const float z0 = -1.0f + 0.1f * k;
    fl.insert(fl.end(), {0.3f, z0, 0.3f, z0 + 0.1f});
  }
  json scene = {
      {"schema", "axionfem.scene/1"},
      {"name", "synthetic"},
      {"symmetry", "axisymmetric"},
      {"units", "m"},
      {"extent", {{"r_max", 0.7}, {"z_min", 0.2}, {"z_max", 0.6}}},
      {"view", {{"r_max", 1.0}, {"z_min", -1.0}, {"z_max", 1.0}}},
      {"regions",
       json::array({{{"name", "block"},
                     {"role", "superconductor"},
                     {"material", "nb"},
                     {"color", "#7c8ea6"},
                     {"triangles", {{"vertices", f32(bv, 2)}, {"indices", u32(bi, 3)}}},
                     {"outline", json::array({f32(outline, 2)})}}})},
      {"curves", json::array({{{"name", "sheet"}, {"role", "superconductor"}, {"color", "#7c8ea6"},
                               {"segments", f32({0.85f, -0.6f, 0.85f, -0.3f, 0.85f, -0.3f, 0.95f, -0.2f}, 2)}}})},
      {"field",
       {{"vertices", f32(V, 2)},
        {"indices", u32(I, 3)},
        {"quantities", {{"Bmag", f32(bm, 1)}, {"Br", f32(br, 1)}}},
        {"units", {{"Bmag", "T"}, {"Br", "T"}}}}},
      {"field_lines", {{"static", f32(fl, 2)}, {"pickup", f32({}, 2)}}},
      {"summary", {{"coupling", {{"Psi_natural_T_m3", 0.25}, {"L_p_H", 3.3e-7}, {"W_mag_J", 5.9e5}}}}}};
  return scene;
}

struct RGB {
  double r, g, b;
};
static RGB pixel(const R::Image& im, int x, int y) {
  const unsigned char* p = &im.rgb[3 * (static_cast<size_t>(y) * im.width + x)];
  return {p[0] / 255.0, p[1] / 255.0, p[2] / 255.0};
}
// Mean colour of a (2k+1)^2 window around the pixel showing (x, z) in metres.
static RGB at_xz(const R::Image& im, double x, double z, int k = 2) {
  const double fx = (x - im.x_min) / (im.x_max - im.x_min), fz = (im.z_max - z) / (im.z_max - im.z_min);
  const int px = static_cast<int>(im.plot_x0 + fx * (im.plot_x1 - im.plot_x0));
  const int py = static_cast<int>(im.plot_y0 + fz * (im.plot_y1 - im.plot_y0));
  RGB s{0, 0, 0};
  int n = 0;
  for (int dy = -k; dy <= k; ++dy)
    for (int dx = -k; dx <= k; ++dx) {
      const RGB c = pixel(im, px + dx, py + dy);
      s.r += c.r, s.g += c.g, s.b += c.b, ++n;
    }
  return {s.r / n, s.g / n, s.b / n};
}
static double cdist(RGB a, RGB b) { return std::fabs(a.r - b.r) + std::fabs(a.g - b.g) + std::fabs(a.b - b.b); }

static double variance(const R::Image& im) {
  double s = 0, s2 = 0;
  for (unsigned char c : im.rgb) s += c, s2 += double(c) * c;
  const double n = static_cast<double>(im.rgb.size());
  return s2 / n - (s / n) * (s / n);
}

static bool png_ok(const std::vector<unsigned char>& b, int w, int h) {
  static const unsigned char sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  if (b.size() < 33 || std::memcmp(b.data(), sig, 8) != 0) return false;
  if (std::memcmp(b.data() + 12, "IHDR", 4) != 0) return false;
  auto be = [&](size_t o) { return (uint32_t(b[o]) << 24) | (uint32_t(b[o + 1]) << 16) | (uint32_t(b[o + 2]) << 8) | b[o + 3]; };
  const bool tail = std::memcmp(b.data() + b.size() - 8, "IEND", 4) == 0;
  return be(16) == static_cast<uint32_t>(w) && be(20) == static_cast<uint32_t>(h) && b[24] == 8 && b[25] == 2 && tail;
}

static void test_synthetic() {
  const json scene = synthetic_scene();
  R::RenderOptions o;
  o.width = 800;
  o.height = 500;
  o.annotate = false;
  const R::Image im = R::render_image(scene, o);
  CHECK(im.width == 800 && im.height == 500 && im.rgb.size() == 800u * 500u * 3u);
  CHECK(im.plot_x1 > im.plot_x0 + 100 && im.plot_y1 > im.plot_y0 + 100);
  // equal aspect: metres per pixel equal in x and z (to a pixel)
  const double kx = (im.plot_x1 - im.plot_x0) / (im.x_max - im.x_min);
  const double kz = (im.plot_y1 - im.plot_y0) / (im.z_max - im.z_min);
  CHECK(std::fabs(kx - kz) * (im.x_max - im.x_min) < 1.5);
  // the axis is at the centre of the plot
  CHECK(std::fabs(0.5 * (im.plot_x0 + im.plot_x1) - (im.plot_x0 - im.x_min * kx)) < 1.0);
  // colours: Bmag = z + 1.05 on [0, 2.1] -> viridis: top yellow, bottom purple
  const RGB top = at_xz(im, 0.85, 0.9), bot = at_xz(im, 0.85, -0.9);
  std::printf("  top (%.2f %.2f %.2f) bottom (%.2f %.2f %.2f)\n", top.r, top.g, top.b, bot.r, bot.g, bot.b);
  CHECK(top.r > 0.75 && top.g > 0.75 && top.b < 0.35);  // viridis(0.93) ~ (0.87, 0.89, 0.10)
  CHECK(bot.b > bot.g && bot.r < 0.4);                   // viridis(0.07) ~ (0.28, 0.13, 0.45)
  // mirror image about the axis
  CHECK(cdist(at_xz(im, -0.85, 0.9), top) < 0.03);
  CHECK(cdist(at_xz(im, -0.85, -0.9), bot) < 0.03);
  CHECK(cdist(at_xz(im, -0.2, -0.5), at_xz(im, 0.2, -0.5)) < 0.03);
  // the block (NaN field) shows its material colour, in the upper half only
  const RGB blk = at_xz(im, 0.6, 0.4, 4), blkm = at_xz(im, -0.6, 0.4, 4), below = at_xz(im, 0.6, -0.4, 4);
  std::printf("  block (%.2f %.2f %.2f) mirrored (%.2f %.2f %.2f) below (%.2f %.2f %.2f)\n", blk.r, blk.g, blk.b,
              blkm.r, blkm.g, blkm.b, below.r, below.g, below.b);
  CHECK(blk.b > blk.r && std::fabs(blk.r - 0.45) < 0.12 && std::fabs(blk.b - 0.60) < 0.12);
  CHECK(cdist(blk, blkm) < 0.06);
  CHECK(cdist(below, blk) > 0.2);
  // signed quantity: diverging map, B_r = r z > 0 upper right (warm), < 0 lower right (cool)
  o.quantity = "Br";
  const R::Image ib = R::render_image(scene, o);
  const RGB warm = at_xz(ib, 0.85, 0.85), cool = at_xz(ib, 0.85, -0.85), mid = at_xz(ib, 0.85, 0.0);
  CHECK(warm.r > warm.b + 0.3 && cool.b > cool.r + 0.3);
  CHECK(std::fabs(mid.r - mid.b) < 0.1);  // ~0 maps to the neutral centre
  // log scale and fixed range run
  o.quantity = "Bmag";
  o.log_scale = true;
  CHECK(variance(R::render_image(scene, o)) > 50);
  o.log_scale = false;
  o.vmin = 0;
  o.vmax = 1;
  const R::Image iv = R::render_image(scene, o);
  CHECK(cdist(at_xz(iv, 0.85, 0.5), at_xz(iv, 0.85, 0.9)) < 0.03);  // both above vmax: clamped
  // determinism
  o = R::RenderOptions();
  o.width = 640;
  o.height = 400;
  const auto p1 = R::render_png(scene, o), p2 = R::render_png(scene, o);
  CHECK(p1 == p2);
  CHECK(png_ok(p1, 640, 400));
  // 3d view of the synthetic scene
  o.view = "3d";
  const R::Image i3 = R::render_image(scene, o);
  CHECK(variance(i3) > 50);
  o.quantity = "none";
  CHECK(variance(R::render_image(scene, o)) > 50);
  R::write_file("render_synthetic_section.png", p1);
  // camera / option sweep: must not throw and must draw something
  struct Cam {
    double elev, azim, cut;
    int ssamp;
    const char* theme;
  };
  for (const Cam& c : {Cam{-30, 90, 180, 1, "light"}, Cam{90, 0, 45, 3, "dark"}, Cam{-90, -10, 359, 2, "light"},
                       Cam{10, 200, 360, 2, "dark"}, Cam{0, -60, 270, 4, "light"}}) {
    R::RenderOptions a;
    a.view = "3d";
    a.width = 480;
    a.height = 320;
    a.elev_deg = c.elev;
    a.azim_deg = c.azim;
    a.cutaway_deg = c.cut;
    a.supersample = c.ssamp;
    a.theme = c.theme;
    const R::Image ic = R::render_image(scene, a);
    CHECK(variance(ic) > 20);
  }
}

static void test_errors() {
  const json scene = synthetic_scene();
  bool threw = false;
  try {
    R::RenderOptions o;
    o.quantity = "Bnope";
    R::render_image(scene, o);
  } catch (const axf::Error& e) {
    threw = std::string(e.what()).find("use one of") != std::string::npos;
  }
  CHECK(threw);
  threw = false;
  try {
    R::options_from_json({{"veiw", "3d"}});
  } catch (const axf::Error& e) {
    threw = std::string(e.what()).find("veiw") != std::string::npos;
  }
  CHECK(threw);
  threw = false;
  try {
    R::options_from_json({{"view", "side"}});
  } catch (const axf::Error&) {
    threw = true;
  }
  CHECK(threw);
  const R::RenderOptions o = R::options_from_json({{"view", "3d"}, {"width", 1200}, {"log_scale", true}, {"vmax", nullptr}});
  CHECK(o.view == "3d" && o.width == 1200 && o.log_scale && std::isnan(o.vmax));
  CHECK(R::options_from_json(R::options_to_json(o)).width == 1200);
  threw = false;
  try {
    json bad = scene;
    bad["field"]["indices"]["data"] = "AAAA";
    R::render_image(bad, R::RenderOptions());
  } catch (const axf::Error& e) {
    threw = std::string(e.what()).find("/field/indices") != std::string::npos;
  }
  CHECK(threw);
}

static void test_samples() {
  const std::string root = AXF_SOURCE_DIR;
  struct Variant {
    std::string tag;
    R::RenderOptions o;
  };
  for (const std::string name : {"toroid", "solenoid", "core"}) {
    const std::string path = root + "/runs/" + name + "/scene.json";
    std::ifstream f(path);
    if (!f.good()) {
      std::printf("  skip %s (no %s)\n", name.c_str(), path.c_str());
      continue;
    }
    const json scene = json::parse(f);
    std::vector<Variant> vs;
    vs.push_back({"section", R::RenderOptions()});
    R::RenderOptions o3;
    o3.view = "3d";
    vs.push_back({"3d", o3});
    if (name == "solenoid") {
      R::RenderOptions a;
      a.quantity = "Br";
      vs.push_back({"section_Br", a});
      a = R::RenderOptions();
      a.log_scale = true;
      a.theme = "dark";
      vs.push_back({"section_log_dark", a});
      a = R::RenderOptions();
      a.view = "3d";
      a.quantity = "none";
      vs.push_back({"3d_none", a});
      a = R::RenderOptions();
      a.view = "3d";
      a.cutaway_deg = 360;
      a.elev_deg = 15;
      a.width = 1200;
      a.height = 800;
      vs.push_back({"3d_closed", a});
    }
    if (name == "toroid") {
      R::RenderOptions a;
      a.quantity = "pickup_Bmag";
      vs.push_back({"section_pickup", a});
      a = R::RenderOptions();
      a.annotate = false;
      a.width = 1000;
      a.height = 900;
      vs.push_back({"section_plain", a});
    }
    if (name == "core") {
      R::RenderOptions a;
      a.quantity = "none";
      vs.push_back({"section_none", a});
      a = R::RenderOptions();
      a.view = "3d";
      a.quantity = "Bz";
      a.theme = "dark";
      a.elev_deg = 35;
      a.azim_deg = -120;
      vs.push_back({"3d_Bz_dark", a});
    }
    for (const Variant& v : vs) {
      const auto t0 = std::chrono::steady_clock::now();
      const R::Image im = R::render_image(scene, v.o);
      const auto t1 = std::chrono::steady_clock::now();
      const auto png = R::encode_png(im);
      const auto t2 = std::chrono::steady_clock::now();
      const std::string out = "render_" + name + "_" + v.tag + ".png";
      R::write_file(out, png);
      const double var = variance(im);
      std::printf("  %-36s render %.3f s, encode %.3f s, %zu bytes, variance %.0f\n", out.c_str(),
                  std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(),
                  png.size(), var);
      CHECK(png_ok(png, v.o.width, v.o.height));
      CHECK(var > 100);
      CHECK(im.plot_x1 > im.plot_x0 && im.plot_y1 > im.plot_y0);
    }
  }
}

int main() {
  try {
    std::printf("synthetic scene\n");
    test_synthetic();
    std::printf("errors\n");
    test_errors();
    std::printf("sample scenes\n");
    test_samples();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "unexpected exception: %s\n", e.what());
    return 2;
  }
  std::printf("test_render: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
