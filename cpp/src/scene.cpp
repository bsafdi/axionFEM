// Scene export for the viewer (axionfem.scene/1) and VTK output.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>

#include "axionfem/core.hpp"
#include "axionfem/model.hpp"
#include "axionfem/units.hpp"

namespace axf {

using namespace axi;

namespace {

std::string base64(const void* data, size_t n) {
  static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const auto* p = static_cast<const unsigned char*>(data);
  std::string out;
  out.reserve(4 * ((n + 2) / 3));
  size_t i = 0;
  for (; i + 2 < n; i += 3) {
    uint32_t v = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
    out += tbl[(v >> 18) & 63];
    out += tbl[(v >> 12) & 63];
    out += tbl[(v >> 6) & 63];
    out += tbl[v & 63];
  }
  if (i < n) {
    uint32_t v = p[i] << 16;
    if (i + 1 < n) v |= p[i + 1] << 8;
    out += tbl[(v >> 18) & 63];
    out += tbl[(v >> 12) & 63];
    out += (i + 1 < n) ? tbl[(v >> 6) & 63] : '=';
    out += '=';
  }
  return out;
}

json f32(const std::vector<double>& v, int cols) {
  std::vector<float> f(v.begin(), v.end());
  return {{"dtype", "float32"}, {"shape", {static_cast<long>(v.size()) / cols, cols}},
          {"data", base64(f.data(), f.size() * sizeof(float))}};
}

json u32(const std::vector<uint32_t>& v, int cols) {
  return {{"dtype", "uint32"}, {"shape", {static_cast<long>(v.size()) / cols, cols}},
          {"data", base64(v.data(), v.size() * sizeof(uint32_t))}};
}

// Sub-triangle lattice on the reference triangle: points (i/k, j/k).
struct SubTri {
  mfem::IntegrationRule pts;
  std::vector<std::array<int, 3>> tris;
};
const SubTri& subtri(int k) {
  static std::map<int, SubTri> cache;
  auto it = cache.find(k);
  if (it != cache.end()) return it->second;
  SubTri s;
  std::map<std::pair<int, int>, int> id;
  for (int j = 0; j <= k; ++j)
    for (int i = 0; i + j <= k; ++i) {
      mfem::IntegrationPoint ip;
      ip.Set2(double(i) / k, double(j) / k);
      ip.weight = 0;
      id[{i, j}] = s.pts.GetNPoints();
      s.pts.Append(ip);
    }
  for (int j = 0; j < k; ++j)
    for (int i = 0; i + j < k; ++i) {
      s.tris.push_back({id[{i, j}], id[{i + 1, j}], id[{i, j + 1}]});
      if (i + j + 1 < k) s.tris.push_back({id[{i + 1, j}], id[{i + 1, j + 1}], id[{i, j + 1}]});
    }
  return cache[k] = s;
}

std::string role_color(const std::string& role) {
  if (role == "superconductor") return "#7c8ea6";
  if (role == "winding") return "#c46a2c";
  if (role == "iron") return "#5b5f66";
  if (role == "conductor") return "#b8862b";
  if (role == "port") return "#2f9e6f";
  return "#9fb7d6";
}


// Shared-point display mesh over a set of elements: mesh vertices, plus edge
// midpoints when subdiv = 2 (each triangle -> 4).  Each point records an owner
// element and the reference coordinates in that element.
struct DisplayMesh {
  std::vector<int> owner;
  std::vector<mfem::IntegrationPoint> ref;
  std::vector<uint32_t> tris;
};

DisplayMesh display_mesh(const mfem::Mesh& m, const std::vector<int>& elems, int subdiv) {
  DisplayMesh D;
  std::map<int, uint32_t> vid, eid;
  static const double vref[3][2] = {{0, 0}, {1, 0}, {0, 1}};
  static const double eref[3][2] = {{0.5, 0}, {0.5, 0.5}, {0, 0.5}};
  mfem::Array<int> v, ed, eo;
  for (int e : elems) {
    m.GetElementVertices(e, v);
    uint32_t c[3], mid[3];
    for (int i = 0; i < 3; ++i) {
      auto it = vid.find(v[i]);
      if (it == vid.end()) {
        mfem::IntegrationPoint ip;
        ip.Set2(vref[i][0], vref[i][1]);
        ip.weight = 0;
        it = vid.emplace(v[i], static_cast<uint32_t>(D.owner.size())).first;
        D.owner.push_back(e);
        D.ref.push_back(ip);
      }
      c[i] = it->second;
    }
    if (subdiv >= 2) {
      m.GetElementEdges(e, ed, eo);
      for (int j = 0; j < 3; ++j) {
        auto it = eid.find(ed[j]);
        if (it == eid.end()) {
          mfem::IntegrationPoint ip;
          ip.Set2(eref[j][0], eref[j][1]);
          ip.weight = 0;
          it = eid.emplace(ed[j], static_cast<uint32_t>(D.owner.size())).first;
          D.owner.push_back(e);
          D.ref.push_back(ip);
        }
        mid[j] = it->second;
      }
      const uint32_t t[4][3] = {{c[0], mid[0], mid[2]}, {mid[0], c[1], mid[1]}, {mid[2], mid[1], c[2]},
                                {mid[0], mid[1], mid[2]}};
      for (auto& tt : t) D.tris.insert(D.tris.end(), tt, tt + 3);
    } else {
      D.tris.insert(D.tris.end(), c, c + 3);
    }
  }
  return D;
}

}  // namespace

// shared with the 3D scene (model3d.cpp)
json scene_f32(const std::vector<double>& v, int cols) { return f32(v, cols); }
json scene_u32(const std::vector<uint32_t>& v, int cols) { return u32(v, cols); }

json AxiModel::scene(const SceneOptions& o) const {
  AXF_REQUIRE(solved_, "solve the model before exporting a scene");
  const Space& sp = *sp_;
  const mfem::Mesh& m = sp.mesh;
  const int ne = m.GetNE();
  const int k = std::max(1, o.subdiv);

  // roles
  auto role_of = [&](const std::string& reg) -> std::string {
    if (reg == "air") return "vacuum";
    const spec::Region* g = d_.find_region(reg);
    if (!g) return "vacuum";
    for (auto& w : d_.windings) if (w.region == reg) return "winding";
    if (d_.pickup && d_.pickup->port == reg) return "port";
    switch (d_.materials.at(g->material).kind) {
      case spec::MaterialKind::Superconductor: return "superconductor";
      case spec::MaterialKind::Iron: return "iron";
      case spec::MaterialKind::Conductor: return "conductor";
      case spec::MaterialKind::Linear: return "magnetic";
      default: return "vacuum";
    }
  };

  // device extent (non-air regions)
  double rmax = 0, zmin = 1e300, zmax = -1e300;
  for (auto& g : d_.regions) {
    double a, b, c, e;
    g.shape.bbox(a, b, c, e);
    rmax = std::max(rmax, b);
    zmin = std::min(zmin, c);
    zmax = std::max(zmax, e);
  }
  for (auto& c : d_.curves) {
    double a, b, cc, e;
    c.path.bbox(a, b, cc, e);
    rmax = std::max(rmax, b);
    zmin = std::min(zmin, cc);
    zmax = std::max(zmax, e);
  }
  const double zc = 0.5 * (zmin + zmax), hz = 0.5 * (zmax - zmin);
  const double half = std::max(rmax, hz) * o.view_scale;
  const double vr = half, vz0 = zc - half, vz1 = zc + half;

  // ---------------------------------------------------------------- regions
  json regions = json::array();
  std::map<int, std::vector<int>> elems_of_attr;
  for (int e = 0; e < ne; ++e) elems_of_attr[m.GetAttribute(e)].push_back(e);
  PointEval pe;
  for (auto& [attr, elems] : elems_of_attr) {
    std::string reg = region_name(attr);
    if (reg == "air") continue;
    DisplayMesh D = display_mesh(m, elems, std::min(2, k));
    std::vector<double> V(2 * D.owner.size());
    for (size_t i = 0; i < D.owner.size(); ++i) {
      sp.eval_at(D.owner[i], D.ref[i], pe);
      V[2 * i] = pe.r;
      V[2 * i + 1] = pe.z;
    }
    const std::vector<uint32_t>& I = D.tris;
    // outline: boundary loops sampled along curved edges
    std::vector<char> in(ne, 0);
    for (int e : elems) in[e] = 1;
    auto loops = boundary_loops(sp, in);
    json outlines = json::array();
    for (auto& L : loops) {
      std::vector<double> P;
      for (int f : L) {
        mfem::FaceElementTransformations* FT = const_cast<mfem::Mesh&>(m).GetFaceElementTransformations(f);
        const int ko = std::max(1, o.outline_samples);
        for (int i = 0; i <= ko; ++i) {
          mfem::IntegrationPoint ip;
          ip.x = double(i) / ko;
          FT->SetAllIntPoints(&ip);
          mfem::Vector x(2);
          FT->Transform(ip, x);
          P.push_back(x(0));
          P.push_back(x(1));
        }
      }
      outlines.push_back(f32(P, 2));
    }
    const spec::Region* g = d_.find_region(reg);
    std::string role = role_of(reg);
    regions.push_back({{"name", reg},
                       {"role", role},
                       {"material", g ? g->material : "vacuum"},
                       {"color", role_color(role)},
                       {"triangles", {{"vertices", f32(V, 2)}, {"indices", u32(I, 3)}}},
                       {"outline", outlines}});
  }
  // curves (sheets)
  json curves = json::array();
  for (auto& c : d_.curves) {
    int attr = mb_.attr_of_curve.at(c.name);
    std::vector<double> S;
    for (int be = 0; be < m.GetNBE(); ++be) {
      if (m.GetBdrAttribute(be) != attr) continue;
      mfem::ElementTransformation* T = const_cast<mfem::Mesh&>(m).GetBdrElementTransformation(be);
      mfem::Vector x(2);
      for (int i = 0; i < k; ++i) {
        for (int s = 0; s < 2; ++s) {
          mfem::IntegrationPoint ip;
          ip.x = double(i + s) / k;
          T->SetIntPoint(&ip);
          T->Transform(ip, x);
          S.push_back(x(0));
          S.push_back(x(1));
        }
      }
    }
    const std::string crole = c.thin_layer() ? "iron" : "superconductor";  // a thin magnetic layer, or a sheet
    curves.push_back({{"name", c.name}, {"role", crole}, {"color", role_color(crole)},
                      {"segments", f32(S, 2)}});
  }

  json scene = {{"schema", "axionfem.scene/1"},
                {"name", d_.name},
                {"symmetry", "axisymmetric"},
                {"units", "m"},
                {"extent", {{"r_max", rmax}, {"z_min", zmin}, {"z_max", zmax}}},
                {"view", {{"r_max", vr}, {"z_min", vz0}, {"z_max", vz1}}},
                {"regions", regions},
                {"curves", curves}};

  // ---------------------------------------------------------------- fields
  if (o.fields) {
    std::vector<int> sel;
    for (int e = 0; e < ne; ++e) {
      if (role_of(region_name(m.GetAttribute(e))) == "superconductor") continue;
      double r, z;
      element_center(sp, e, r, z);
      if (r <= 1.05 * vr && z >= vz0 - 0.05 * half && z <= vz1 + 0.05 * half) sel.push_back(e);
    }
    DisplayMesh D = display_mesh(m, sel, std::min(2, k));
    const size_t nv = D.owner.size();
    std::vector<double> V(2 * nv), Bm(nv), Br(nv), Bz(nv), Bp(nv), Ps(nv), PB(nv), PPs(nv);
    std::vector<PointEval> pes(nv);
    for (size_t i = 0; i < nv; ++i) sp.eval_at(D.owner[i], D.ref[i], pes[i]);  // serial (MFEM shape evaluation)
#pragma omp parallel for schedule(dynamic, 256)
    for (long ii = 0; ii < static_cast<long>(nv); ++ii) {
      size_t i = static_cast<size_t>(ii);
      Sample smp = eval_pe(D.owner[i], pes[i]);
      V[2 * i] = pes[i].r;
      V[2 * i + 1] = pes[i].z;
      Br[i] = smp.Br;
      Bz[i] = smp.Bz;
      Bp[i] = smp.Bphi;
      Bm[i] = std::sqrt(smp.Br * smp.Br + smp.Bz * smp.Bz + smp.Bphi * smp.Bphi);
      Ps[i] = smp.psi;
      PB[i] = std::sqrt(smp.pBr * smp.pBr + smp.pBz * smp.pBz + smp.pBphi * smp.pBphi);
      PPs[i] = smp.ppsi;
    }
    const std::vector<uint32_t>& I = D.tris;
    auto range = [](const std::vector<double>& v) {
      double a = 1e300, b = -1e300;
      for (double x : v) if (std::isfinite(x)) { a = std::min(a, x); b = std::max(b, x); }
      return json::array({a, b});
    };
    json q = {{"Bmag", f32(Bm, 1)}, {"Br", f32(Br, 1)}, {"Bz", f32(Bz, 1)}, {"Bphi", f32(Bp, 1)},
              {"psi", f32(Ps, 1)}, {"pickup_Bmag", f32(PB, 1)}};
    json ranges = {{"Bmag", range(Bm)}, {"Br", range(Br)}, {"Bz", range(Bz)}, {"Bphi", range(Bp)},
                   {"psi", range(Ps)}, {"pickup_Bmag", range(PB)}};
    // flux-function contours (poloidal field lines), marching triangles
    auto contours = [&](const std::vector<double>& Pv) {
      std::vector<double> seg;
      double pmin = 1e300, pmax = -1e300;
      for (size_t i = 0; i < nv; ++i) {
        if (std::isfinite(Pv[i])) {
          pmin = std::min(pmin, Pv[i]);
          pmax = std::max(pmax, Pv[i]);
        }
      }
      if (!(pmax > pmin)) return seg;
      const int nl = std::max(4, o.contours);
      for (size_t t = 0; t + 2 < I.size(); t += 3) {
        size_t a = I[t], b = I[t + 1], c = I[t + 2];
        double va = Pv[a], vb = Pv[b], vc = Pv[c];
        for (int l = 1; l < nl; ++l) {
          double lev = pmin + (pmax - pmin) * l / nl;
          double pts[4];
          int n = 0;
          auto edge = [&](size_t i, size_t j, double vi, double vj) {
            if ((vi - lev) * (vj - lev) < 0 && n < 4) {
              double tt = (lev - vi) / (vj - vi);
              pts[n++] = V[2 * i] + tt * (V[2 * j] - V[2 * i]);
              pts[n++] = V[2 * i + 1] + tt * (V[2 * j + 1] - V[2 * i + 1]);
            }
          };
          edge(a, b, va, vb);
          edge(b, c, vb, vc);
          edge(c, a, vc, va);
          if (n == 4) seg.insert(seg.end(), pts, pts + 4);
        }
      }
      return seg;
    };
    std::vector<double> seg = a_lab_ ? contours(Ps) : std::vector<double>{};
    std::vector<double> pseg = a_pick_ ? contours(PPs) : std::vector<double>{};
    q["pickup_psi"] = f32(PPs, 1);
    ranges["pickup_psi"] = range(PPs);
    json units = {{"Bmag", "T"}, {"Br", "T"}, {"Bz", "T"}, {"Bphi", "T"}, {"psi", "Wb/rad"},
                  {"pickup_Bmag", "T/A"}, {"pickup_psi", "Wb/(rad A)"}};
    if (fw_scene_frequency() > 0) {  // full-wave RF fields per readout ampere (amplitudes)
      std::vector<double> RB(nv), RE(nv);
#pragma omp parallel for schedule(dynamic, 256)
      for (long ii = 0; ii < static_cast<long>(nv); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        axi::cplx Bc[3], Ec[3];
        fw_fields(D.owner[i], pes[i], Bc, Ec);
        RB[i] = std::sqrt(std::norm(Bc[0]) + std::norm(Bc[1]) + std::norm(Bc[2]));
        RE[i] = std::sqrt(std::norm(Ec[0]) + std::norm(Ec[1]) + std::norm(Ec[2]));
      }
      q["rf_Bmag"] = f32(RB, 1);
      q["rf_Emag"] = f32(RE, 1);
      ranges["rf_Bmag"] = range(RB);
      ranges["rf_Emag"] = range(RE);
      units["rf_Bmag"] = "T/A";
      units["rf_Emag"] = "V/(m A)";
    }
    scene["field"] = {{"vertices", f32(V, 2)}, {"indices", u32(I, 3)}, {"quantities", q}, {"ranges", ranges},
                      {"units", units}};
    if (fw_scene_frequency() > 0) scene["field"]["rf_frequency_Hz"] = fw_scene_frequency();
    scene["field_lines"] = {{"static", f32(seg, 2)}, {"pickup", f32(pseg, 2)}};
  }
  // summary of results
  json sum = json::object();
  if (results_.contains("coupling")) sum["coupling"] = results_["coupling"];
  if (results_.contains("fields")) sum["fields"] = results_["fields"];
  if (results_.contains("benchmarks")) sum["benchmarks"] = results_["benchmarks"];
  if (results_.contains("geometry")) sum["geometry"] = results_["geometry"];
  if (results_.contains("forces")) {
    json f = results_["forces"];
    if (f.is_object()) f.erase("distributions");  // per-sample arrays: megabytes; the viewer shows net forces
    sum["forces"] = f;
  }
  if (results_.contains("pickup") && results_["pickup"].is_object()) {
    json p = results_["pickup"];
    p.erase("solver");
    sum["pickup"] = p;
  }
  scene["summary"] = sum;
  scene["design"] = d_.source;
  return scene;
}

void AxiModel::write_vtu(const std::string& path, const SceneOptions& o) const {
  AXF_REQUIRE(solved_, "solve the model before writing VTK output");
  const Space& sp = *sp_;
  const mfem::Mesh& m = sp.mesh;
  const int ne = m.GetNE();
  const int k = std::max(1, o.subdiv);
  const SubTri& st = subtri(k);
  const RefTable& tab = sp.table(st.pts);
  const int npts = st.pts.GetNPoints();
  std::ofstream f(path);
  AXF_REQUIRE(f.good(), "cannot write '" + path + "'");
  const size_t nv = static_cast<size_t>(ne) * npts, nt = static_cast<size_t>(ne) * st.tris.size();
  std::vector<Sample> S(nv);
  std::vector<double> R(nv), Z(nv);
#pragma omp parallel for schedule(dynamic, 128)
  for (int e = 0; e < ne; ++e) {
    PointEval p;
    for (int q = 0; q < npts; ++q) {
      sp.eval(e, tab, q, p);
      size_t i = static_cast<size_t>(e) * npts + q;
      S[i] = eval_pe(e, p);
      R[i] = p.r;
      Z[i] = p.z;
    }
  }
  f << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "<UnstructuredGrid><Piece NumberOfPoints=\"" << nv << "\" NumberOfCells=\"" << nt << "\">\n"
    << "<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  f.precision(10);
  for (size_t i = 0; i < nv; ++i) f << R[i] << " " << Z[i] << " 0\n";
  f << "</DataArray></Points>\n<Cells><DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
  for (int e = 0; e < ne; ++e)
    for (auto& t : st.tris)
      f << e * npts + t[0] << " " << e * npts + t[1] << " " << e * npts + t[2] << "\n";
  f << "</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
  for (size_t c = 1; c <= nt; ++c) f << 3 * c << "\n";
  f << "</DataArray>\n<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
  for (size_t c = 0; c < nt; ++c) f << "5\n";
  f << "</DataArray></Cells>\n<PointData Scalars=\"Bmag\">\n";
  auto arr = [&](const char* name, auto get) {
    f << "<DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">\n";
    for (size_t i = 0; i < nv; ++i) f << get(S[i]) << "\n";
    f << "</DataArray>\n";
  };
  arr("Bmag", [](const Sample& s) { return std::sqrt(s.Br * s.Br + s.Bz * s.Bz + s.Bphi * s.Bphi); });
  arr("Br", [](const Sample& s) { return s.Br; });
  arr("Bz", [](const Sample& s) { return s.Bz; });
  arr("Bphi", [](const Sample& s) { return s.Bphi; });
  arr("psi", [](const Sample& s) { return s.psi; });
  arr("pickup_Bmag", [](const Sample& s) { return std::sqrt(s.pBr * s.pBr + s.pBz * s.pBz + s.pBphi * s.pBphi); });
  arr("region", [](const Sample& s) { return static_cast<double>(s.region); });
  f << "</PointData>\n</Piece></UnstructuredGrid></VTKFile>\n";
}

}  // namespace axf
