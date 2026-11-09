// Full-wave surface losses of the 3D readout ports (surface_losses3d.hpp; docs/physics/fullwave_surface_losses.md).
// The faces are found on the serial mesh, identically on every rank; each rank keeps the quadrature points whose
// field-side element it holds and evaluates the port fields there; one reduction makes the sums the same everywhere.
#include "surface_losses3d.hpp"

#include <cmath>
#include <limits>
#include <set>

#include "axionfem/axi.hpp"
#include "axionfem/core.hpp"

namespace axf {

namespace {

using cplx = std::complex<double>;

// the largest |R_loss| / |X| above which the first-order (perturbative) loss is flagged: the neglected order is about
// its square (docs/physics/fullwave_surface_losses.md, Validity)
constexpr double kPerturbationFlag = 0.1;

// Raise a rank-local error on every rank: the lowest failing rank's message is broadcast, and every rank throws it.
// Returns on every rank when no rank failed. Collective.
void surface_loss_agree(bool failed, const std::string& local_error, MPI_Comm comm) {
  int rank = 0;
  int nranks = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  int who = failed ? rank : nranks;
  MPI_Allreduce(MPI_IN_PLACE, &who, 1, MPI_INT, MPI_MIN, comm);
  if (who == nranks) return;
  int length = rank == who ? static_cast<int>(local_error.size()) : 0;
  MPI_Bcast(&length, 1, MPI_INT, who, comm);
  std::string message = rank == who ? local_error : std::string(length, ' ');
  MPI_Bcast(message.data(), length, MPI_CHAR, who, comm);
  throw Error("3D full wave surface losses: " + message, "/materials",
              "The error arose on MPI rank " + std::to_string(who) + " while the surface-loss faces were set up.");
}

json matrix_json(const Eigen::MatrixXcd& m) {
  json re = json::array();
  json im = json::array();
  for (int j = 0; j < m.rows(); ++j) {
    json rr = json::array();
    json ii = json::array();
    for (int k = 0; k < m.cols(); ++k) {
      rr.push_back(m(j, k).real());
      ii.push_back(m(j, k).imag());
    }
    re.push_back(rr);
    im.push_back(ii);
  }
  return {{"re", re}, {"im", im}};
}

}  // namespace

FwSurfaceLosses::FwSurfaceLosses(const Model3D::Impl& P, const spec::Design& d, const std::vector<int>& epml,
                                 bool static_field)
    : P_(P), pm_(*P.pmesh), fes_(*P.fes), order_(P.order), static_field_(static_field) {
  Timer timer;
  static const spec::Material vacuum = spec::builtin_materials().at("vacuum");
  mfem::Mesh& sm = *P.serial;
  const int nser = sm.GetNE();
  std::vector<const spec::Material*> ser_mat(nser);
  std::vector<std::string> ser_region(nser);
  for (int s2 = 0; s2 < nser; ++s2) {
    auto it = P.region_of_attr.find(P.ser_attr[s2]);
    ser_region[s2] = it != P.region_of_attr.end() ? it->second : "air";
    const spec::Region* g = d.find_region(ser_region[s2]);
    ser_mat[s2] = g ? &d.materials.at(g->material) : &vacuum;
  }
  auto conductor = [&](int s2) {
    const spec::MaterialKind kind = ser_mat[s2]->kind;
    return kind == spec::MaterialKind::Superconductor || kind == spec::MaterialKind::Conductor;
  };
  auto in_shell = [&](int s2) { return P.sh_attr > 0 && P.ser_attr[s2] == P.sh_attr; };
  // the faces of pickup.losses, on the serial mesh: the surface and region lists are the same on every rank
  const std::vector<SurfaceLossFace> faces = surface_loss_faces(sm, conductor, in_shell);
  faces_ = static_cast<long>(faces.size());
  std::set<std::string> surface_set, region_set;
  for (const SurfaceLossFace& lf : faces) {
    surface_set.insert(ser_region[lf.conductor]);
    region_set.insert(ser_region[lf.field]);
  }
  surfaces_.assign(surface_set.begin(), surface_set.end());
  regions_.assign(region_set.begin(), region_set.end());
  std::map<std::string, int> surface_index, region_index;
  for (size_t s = 0; s < surfaces_.size(); ++s) surface_index[surfaces_[s]] = static_cast<int>(s);
  for (size_t r = 0; r < regions_.size(); ++r) region_index[regions_[r]] = static_cast<int>(r);
  material_.assign(surfaces_.size(), nullptr);
  pair_.assign(surfaces_.size(), std::vector<char>(regions_.size(), 0));
  for (const SurfaceLossFace& lf : faces) {
    const int s = surface_index.at(ser_region[lf.conductor]);
    material_[s] = ser_mat[lf.conductor];
    pair_[s][region_index.at(ser_region[lf.field])] = 1;
  }
  for (const spec::Material* m : material_) {
    any_rf_ = any_rf_ || !m->rf.model.empty();
    any_mr_ = any_mr_ || !m->rf.mr_model.empty();
  }
  // PEC boundary groups other than symmetry planes and the outer sphere: conductor surfaces without volume elements
  // (mesh.remove_pec holes, imported PEC groups), which carry no loss here
  for (const auto& g : P.mesh_info.value("boundary_groups", json::array())) {
    if (g.value("type", std::string("pec")) != "pec" || g.contains("axis")) continue;
    if (g.value("attr", 0) == axi::kOuterAttr) continue;
    pec_groups_.push_back(g.value("name", strcat("attr_", g.value("attr", 0))));
    const spec::Region* r = d.find_region(g.value("conductor", std::string()));
    if (r && !d.materials.at(r->material).rf.model.empty()) holes_with_rf_.push_back(r->name);
  }
  // this rank's quadrature points: the faces whose field side it holds (a PML layer is no physical field side)
  std::vector<int> ser2loc(nser, -1);
  for (size_t e = 0; e < P.loc2ser.size(); ++e) ser2loc[P.loc2ser[e]] = static_cast<int>(e);
  const mfem::IntegrationRule& rule = surface_loss_rule(order_);
  const bool static_B = static_field_ && P.A_lab;
  long pml_local = 0;
  mfem::Vector nor(3), Bs(3), x(3);
  // only the ranks holding a face's field side visit it, so an error here would be rank-local: it is caught and
  // raised on every rank after one agreement, before the collectives below;
  // a flag, not the message, says whether this rank failed (an exception may carry an empty message)
  bool failed = false;
  std::string local_error;
  try {
    for (const SurfaceLossFace& lf : faces) {
      const int element = ser2loc[lf.field];
      if (element < 0) continue;
      if (!epml.empty() && epml[element] >= 0) {
        ++pml_local;
        continue;
      }
      mfem::FaceElementTransformations* FT = sm.GetFaceElementTransformations(lf.face);
      const bool side1 = FT->Elem1No == lf.field;
      const spec::Material& cm = *ser_mat[lf.conductor];
      const double mu = surface_loss_mu(*ser_mat[lf.field]);
      const int slot = surface_index.at(ser_region[lf.conductor]) * static_cast<int>(regions_.size()) +
                       region_index.at(ser_region[lf.field]);
      for (int q = 0; q < rule.GetNPoints(); ++q) {
        const mfem::IntegrationPoint& ip = rule.IntPoint(q);
        FT->SetAllIntPoints(&ip);
        mfem::CalcOrtho(FT->Jacobian(), nor);
        Point p;
        p.element = element;
        p.ip = side1 ? FT->GetElement1IntPoint() : FT->GetElement2IntPoint();
        p.dA = ip.weight * nor.Norml2();
        p.inv_mu2 = 1.0 / (mu * mu);
        for (int k = 0; k < 3; ++k) p.n[k] = nor(k) / std::max(nor.Norml2(), 1e-300);
        p.slot = slot;
        if (!cm.rf.mr_model.empty() && static_field_) {
          // the static field on the local element (the same field as the MQS losses' gathered copy): B_lab and an
          // imposed background
          mfem::ElementTransformation* T = pm_.GetElementTransformation(element);
          T->SetIntPoint(&p.ip);
          Bs = 0.0;
          if (static_B) P.A_lab->GetCurl(*T, Bs);
          T->Transform(p.ip, x);
          double b[3];
          P.bg_at(x, b);
          for (int k = 0; k < 3; ++k) Bs(k) += b[k];
          p.mr = surface_loss_mr_weighted(cm, Bs.Norml2(), 1.0);
        }
        points_.push_back(p);
      }
    }
  } catch (const std::exception& error) {
    failed = true;
    local_error = error.what();
    if (local_error.empty()) local_error = "an error without a message";
  } catch (...) {
    failed = true;
    local_error = "an error that is not a std::exception";
  }
  const MPI_Comm comm = pm_.GetComm();
  surface_loss_agree(failed, local_error, comm);
  long counts[2] = {pml_local, static_cast<long>(points_.size())};
  MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_LONG, MPI_SUM, comm);
  pml_faces_ = counts[0];
  points_total_ = counts[1];
  double seconds = timer.seconds();
  MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
  setup_seconds_ = seconds;
}

void FwSurfaceLosses::Evaluate(double f, const std::vector<const mfem::Vector*>& re,
                               const std::vector<const mfem::Vector*>& im, const Eigen::MatrixXcd& Z,
                               json& primary_row) {
  Timer timer;
  const MPI_Comm comm = pm_.GetComm();
  const int np = static_cast<int>(re.size());
  // the exact dZ/domega of the row (the same on every rank), when the solve computed it; as the adaptive sweep, only a
  // derivative whose scope says exact is used (the losses are post-processing and add nothing to A(omega) or A')
  Eigen::MatrixXcd dZ_matrix;
  const Eigen::MatrixXcd* dZ = nullptr;
  const bool exact = primary_row.value("dZ_domega_scope", std::string()).rfind("exact", 0) == 0;
  if (primary_row.contains("dZ_domega_H") && exact) {
    const json& d = primary_row["dZ_domega_H"];
    dZ_matrix.resize(np, np);
    for (int j = 0; j < np; ++j)
      for (int k = 0; k < np; ++k) {
        const json& a = d["re"][j][k];
        const json& b = d["im"][j][k];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        dZ_matrix(j, k) = cplx(a.is_number() ? a.get<double>() : nan, b.is_number() ? b.get<double>() : nan);
      }
    dZ = &dZ_matrix;
  }
  const int ns = static_cast<int>(surfaces_.size());
  const int nr = static_cast<int>(regions_.size());
  const double w = 2 * units::pi * f;
  std::vector<std::unique_ptr<mfem::ParGridFunction>> gre(np), gim(np);
  for (int k = 0; k < np; ++k) {
    gre[k] = std::make_unique<mfem::ParGridFunction>(&fes_);
    gim[k] = std::make_unique<mfem::ParGridFunction>(&fes_);
    gre[k]->SetFromTrueDofs(*re[k]);
    gim[k]->SetFromTrueDofs(*im[k]);
  }
  // sums: per (surface, region) slot and port the Gamma part; per surface the Hermitian matrix oint H_j^* . H_k dA,
  // re and im (diagonal: Gamma); the same weighted by the magnetoresistance; per surface and port oint |H . n|^2 dA
  const size_t o_matrix = static_cast<size_t>(ns) * nr * np;
  const size_t o_mr = o_matrix + static_cast<size_t>(ns) * np * np * 2;
  const size_t o_normal = o_mr + static_cast<size_t>(ns) * np * np * 2;
  const size_t total = o_normal + static_cast<size_t>(ns) * np;
  std::vector<double> sums(total, 0.0);
  std::vector<mfem::Vector> cre(np, mfem::Vector(3)), cim(np, mfem::Vector(3));
  for (const Point& p : points_) {
    mfem::ElementTransformation* T = pm_.GetElementTransformation(p.element);
    T->SetIntPoint(&p.ip);
    for (int k = 0; k < np; ++k) {
      gre[k]->GetCurl(*T, cre[k]);
      gim[k]->GetCurl(*T, cim[k]);
    }
    // H = curl E / (i omega mu): H_j^* . H_k = (curl E_j)^* . curl E_k / (omega mu)^2
    const double scale = p.dA * p.inv_mu2 / (w * w);
    const int s = p.slot / nr;
    for (int j = 0; j < np; ++j) {
      sums[static_cast<size_t>(p.slot) * np + j] += scale * ((cre[j] * cre[j]) + (cim[j] * cim[j]));
      const double nre = cre[j](0) * p.n[0] + cre[j](1) * p.n[1] + cre[j](2) * p.n[2];
      const double nim = cim[j](0) * p.n[0] + cim[j](1) * p.n[1] + cim[j](2) * p.n[2];
      sums[o_normal + static_cast<size_t>(s) * np + j] += scale * (nre * nre + nim * nim);
      for (int k = 0; k < np; ++k) {
        const double hre = (cre[j] * cre[k]) + (cim[j] * cim[k]);
        const double him = (cre[j] * cim[k]) - (cim[j] * cre[k]);
        const size_t at = ((static_cast<size_t>(s) * np + j) * np + k) * 2;
        sums[o_matrix + at] += scale * hre;
        sums[o_matrix + at + 1] += scale * him;
        sums[o_mr + at] += scale * p.mr * hre;
        sums[o_mr + at + 1] += scale * p.mr * him;
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, sums.data(), static_cast<int>(total), MPI_DOUBLE, MPI_SUM, comm);
  auto matrix = [&](size_t offset, int s) {
    Eigen::MatrixXcd m(np, np);
    for (int j = 0; j < np; ++j)
      for (int k = 0; k < np; ++k) {
        const size_t at = offset + ((static_cast<size_t>(s) * np + j) * np + k) * 2;
        m(j, k) = cplx(sums[at], sums[at + 1]);
      }
    return m;
  };
  // R_s(f) per surface (NaN: not available at f, with the reason)
  std::vector<double> Rs(ns, std::numeric_limits<double>::quiet_NaN());
  std::vector<std::string> Rs_missing(ns);
  for (int s = 0; s < ns; ++s) {
    if (material_[s]->rf.model.empty()) continue;
    try {
      Rs[s] = material_[s]->rf.Rs(f);
    } catch (const std::exception& error) {
      Rs_missing[s] = error.what();
    }
  }
  // the Gamma that R_s multiplies: magnetoresistance-weighted where the surface has a model and a static field
  auto weighted = [&](int s) { return !material_[s]->rf.mr_model.empty() && static_field_; };
  Eigen::MatrixXcd R_matrix = Eigen::MatrixXcd::Zero(np, np);
  json gamma_matrices = json::object();
  bool R_complete = any_rf_;
  for (int s = 0; s < ns; ++s) {
    if (np > 1) gamma_matrices[surfaces_[s]] = matrix_json(matrix(o_matrix, s));
    if (material_[s]->rf.model.empty()) continue;
    if (!std::isfinite(Rs[s])) {
      R_complete = false;
      continue;
    }
    R_matrix += Rs[s] * matrix(weighted(s) ? o_mr : o_matrix, s);
  }
  json& ports = primary_row["port_solutions"];
  for (int k = 0; k < np; ++k) {
    const cplx z = Z(k, k);
    const double X = -z.imag();   // omega L (e^{-i omega t}: Z = R - i omega L)
    const double L = X / w;
    // the tuned pole's inductance: L_eff = L + (omega / 2) dL/domega = (X / omega + dX/domega) / 2
    double L_eff = std::numeric_limits<double>::quiet_NaN();
    if (dZ && std::isfinite((*dZ)(k, k).imag())) L_eff = 0.5 * (L - (*dZ)(k, k).imag());
    json row;
    json gamma = json::object();
    json surfaces = json::object();
    double R_loss = 0;
    double TQ = 0;
    double TQ_tuned = 0;
    double normal = 0;
    double all = 0;
    json missing = json::array();
    for (int s = 0; s < ns; ++s) {
      const spec::Material& mt = *material_[s];
      const size_t diag = ((static_cast<size_t>(s) * np + k) * np + k) * 2;
      const double G0 = sums[o_matrix + diag];
      normal += sums[o_normal + static_cast<size_t>(s) * np + k];
      all += G0;
      json by_region = json::object();
      for (int r = 0; r < nr; ++r)
        if (pair_[s][r]) by_region[regions_[r]] = sums[(static_cast<size_t>(s) * nr + r) * np + k];
      json e = {{"Gamma", G0}, {"Gamma_by_field_region", by_region}};
      double G = G0;
      if (weighted(s)) {
        G = sums[o_mr + diag];
        e["Gamma_magnetoresistance"] = G;
      }
      if (!mt.rf.model.empty()) {
        if (!std::isfinite(Rs[s])) {
          e["note"] = "R_s not available at this frequency: " + Rs_missing[s];
          missing.push_back(surfaces_[s]);
        } else {
          const double R = Rs[s] * G;
          e["R_s_ohm"] = Rs[s];
          e["R_eff_ohm"] = R;
          e["T_K"] = mt.rf.T_K;
          if (L > 0 && R > 0) {
            e["Q"] = w * L / R;
            e["T_over_Q_K"] = mt.rf.T_K * R / (w * L);
            TQ += mt.rf.T_K * R / (w * L);
          }
          if (L > 0 && L_eff > 0 && R > 0) {
            e["Q_tuned"] = w * L_eff / R;
            TQ_tuned += mt.rf.T_K * R / (w * L_eff);
          }
          R_loss += R;
        }
      }
      gamma[surfaces_[s]] = G0;
      surfaces[surfaces_[s]] = e;
    }
    row["surface_Gamma"] = gamma;
    row["surface_losses"] = {{"surfaces", surfaces}, {"H_normal_share", all > 0 ? normal / all : 0.0}};
    if (std::isfinite(L_eff)) row["L_eff_H"] = L_eff;
    if (any_rf_ && missing.empty()) {
      row["R_loss_ohm"] = R_loss;
      if (std::fabs(X) > 0) row["R_loss_over_abs_X"] = R_loss / std::fabs(X);
      if (L > 0 && R_loss > 0) {
        // Q_total adds Re Z (radiation, bulk, boundary losses). A true relative residual rho can move Re Z by up to
        // about rho |X|, so Q_total = omega L / (R_loss + max(Re Z, 0)) is resolved where rho |X| is small against
        // R_loss + max(Re Z, 0): it is withheld where rho |X| > kQTotalResolution (R_loss + max(Re Z, 0)). Where the
        // conductor loss dominates it is kept even when Re Z itself is not resolved (Re_Z_unresolved in the row)
        const json& port = k < static_cast<int>(ports.size()) ? ports[k] : primary_row;
        const json& solver = port.contains("solver") ? port["solver"] : json::object();
        const json& rho_json = solver.contains("relative_residual") ? solver["relative_residual"] : json();
        const double rho = rho_json.is_number() ? rho_json.get<double>() : std::numeric_limits<double>::quiet_NaN();
        const double R_total = R_loss + std::max(z.real(), 0.0);
        const double uncertainty = std::isfinite(rho) ? rho * std::fabs(X) : 0.0;
        const bool resolved = uncertainty <= kQTotalResolution * R_total;
        row["Q_loss"] = w * L / R_loss;
        if (resolved) row["Q_total"] = w * L / R_total;
        row["T_over_Q_K"] = TQ;
        if (L_eff > 0) {
          row["Q_loss_tuned"] = w * L_eff / R_loss;
          if (resolved) row["Q_total_tuned"] = w * L_eff / R_total;
          row["T_over_Q_tuned_K"] = TQ_tuned;
        }
        if (!resolved)
          row["surface_losses"]["Q_total_withheld"] = {
              {"residual_times_abs_X_over_R_total", uncertainty / R_total}, {"true_relative_residual", rho},
              {"reason", strcat("the solve's true relative residual can move Re Z by up to about residual x |X|, which "
                                "is more than ", kQTotalResolution, " of R_loss + max(Re Z, 0) here: Q_total is not "
                                "resolved (Q_loss and Q_loss_tuned are)")}};
      }
    } else if (any_rf_) {
      row["surface_losses"]["R_loss_omitted"] = "R_s is not available at this frequency for the surfaces listed";
      row["surface_losses"]["surfaces_without_R_s"] = missing;
    }
    if (k < static_cast<int>(ports.size())) ports[k].update(row);
    if (k == 0) primary_row.update(row);
  }
  if (np > 1) {
    primary_row["surface_Gamma_matrix"] = gamma_matrices;
    if (R_complete) primary_row["R_loss_matrix_ohm"] = matrix_json(R_matrix);
  }
  double seconds = timer.seconds();
  MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
  primary_row["surface_loss_seconds"] = seconds;
  evaluation_seconds_ += seconds;
  ++evaluations_;
}

json FwSurfaceLosses::Summary(const json& sweep, bool adaptive, json& warnings) const {
  json out;
  out["engine"] = "3D full wave: perturbative surface loss of the perfectly conducting structure's port fields, "
                  "H = curl E / (i omega mu) on the field side, over the faces of pickup.losses";
  out["definitions"] = {
      {"Gamma", "oint |H|^2 dA / |I|^2 per surface, unit incoming current at the port (surface_Gamma; "
                "dimensionless: H per ampere is in 1/m)"},
      {"R_eff_ohm", "R_s(f) Gamma (Gamma_magnetoresistance with a magnetoresistance model)"},
      {"R_loss_ohm", "sum of R_eff over the surfaces with rf data: 2 P_loss / |I|^2, the power the conductor surfaces "
                     "dissipate (to first order the series resistance they add to Re Z)"},
      {"Q_loss_tuned", "the headline Q: omega L_eff / R_loss, L_eff = L + (omega / 2) dL/domega from the exact "
                       "dZ/domega: the Q of the readout resonance when an ideal capacitor tunes the port to f "
                       "(= 2 omega W_m / P_loss for a closed lossless structure; L > 0 only)"},
      {"Q_loss", "omega L / R_loss, L = -Im Z / omega: the reactance's Q, the key shared with 2D and MQS (2 pi f L_p / "
                 "R_eff there); beyond MQS it is not a resonator Q and tends to 0 at a self-resonance"},
      {"Q_total", "omega L / (R_loss + max(Re Z, 0)): with radiation, bulk and boundary losses (Q_total_tuned with "
                  "L_eff); withheld where the solve's true relative residual times |X| exceeds 0.01 of R_loss + "
                  "max(Re Z, 0)"},
      {"T_over_Q_K", "sum over the rf surfaces of T_b R_eff,b / (omega L) (T_over_Q_tuned_K: with L_eff); radiation "
                     "and bulk loss are not in it, as their temperatures are not known"},
      {"R_loss_over_abs_X", "R_loss / |Im Z|: the perturbation parameter; the neglected order is about its square"},
      {"R_loss_matrix_ohm", "multiport: sum_b R_s,b oint H_j^* . H_k dA (Hermitian): the dissipated power 1/2 I^dagger "
                            "R I of any combination of port currents. It equals the (symmetric) impedance perturbation "
                            "only for in-phase port fields (closed, lossless); with radiation or bulk loss do not add "
                            "it to Z"}};
  json surfaces = json::object();
  json lossless = json::array();
  for (size_t s = 0; s < surfaces_.size(); ++s) {
    const spec::Material& mt = *material_[s];
    json regions = json::array();
    for (size_t r = 0; r < regions_.size(); ++r)
      if (pair_[s][r]) regions.push_back(regions_[r]);
    json e = {{"material", mt.name}, {"field_side_regions", regions}};
    if (!mt.rf.model.empty()) {
      e["rf_model"] = mt.rf.model;
      e["T_K"] = mt.rf.T_K;
      if (!mt.rf.mr_model.empty())
        e["magnetoresistance"] = static_field_ ? "weighted by the static field B_lab"
                                               : "not applied (no static solve: fullwave.rf_only)";
    } else {
      e["rf_model"] = nullptr;
      lossless.push_back(surfaces_[s]);
    }
    surfaces[surfaces_[s]] = e;
  }
  out["surfaces"] = surfaces;
  out["faces"] = faces_;
  out["quadrature_points"] = points_total_;
  out["quadrature_degree"] = 2 * order_ + 2;
  if (pml_faces_ > 0) out["pml_faces_excluded"] = pml_faces_;
  if (!lossless.empty()) out["surfaces_without_rf_data"] = lossless;
  if (!any_rf_) out["note"] = "no surface has rf loss data: Gamma only (give the conductor materials an rf block)";
  out["scope"] = "conductor and superconductor regions meshed as volumes; PEC boundary groups (mesh.remove_pec holes, "
                 "imported PEC groups) and a closed outer sphere carry no loss here. Re Z keeps the bulk, Robin and "
                 "PML losses of the solve itself; R_loss adds the conductor surfaces, which those do not contain";
  if (adaptive)
    out["adaptive_sweep"] = "the losses are evaluated at the solved frequencies (the sweep rows) only: the adaptive "
                            "model and its dense curve carry Z and Psi, not R_loss (docs/physics/"
                            "fullwave_surface_losses.md)";
  // the perturbation parameter and the normal-field check over the rows whose solve passed its residual gate (an
  // unconverged field's losses are reported in its row, but left out here)
  double worst = 0;
  double worst_f = 0;
  int worst_port = 0;
  int flagged = 0;
  int unaccepted = 0;
  int withheld = 0;
  double normal_share = 0;
  for (const json& row : sweep)
    for (const json& port : row.value("port_solutions", json::array())) {
      if (!port.contains("surface_losses")) continue;
      if (!port.value("solver", json::object()).value("accepted", false)) {
        ++unaccepted;
        continue;
      }
      normal_share = std::max(normal_share, port["surface_losses"].value("H_normal_share", 0.0));
      if (port["surface_losses"].contains("Q_total_withheld")) ++withheld;
      if (!port.contains("R_loss_over_abs_X")) continue;
      const double v = port["R_loss_over_abs_X"].get<double>();
      if (v > kPerturbationFlag) ++flagged;
      if (v > worst) {
        worst = v;
        worst_f = row["f_Hz"].get<double>();
        worst_port = port.value("port_index", 0);
      }
    }
  out["H_normal_share_max"] = normal_share;
  if (unaccepted > 0)
    out["port_rows_with_unaccepted_solve"] = {
        {"count", unaccepted},
        {"note", "the field did not pass the residual gate (solver.accepted false, with a warning): its losses are in "
                 "the row but not reliable, and left out of perturbation and H_normal_share_max"}};
  if (any_rf_) {
    out["perturbation"] = {{"max_R_loss_over_abs_X", worst}, {"at_Hz", worst_f}, {"port_index", worst_port},
                           {"flag_above", kPerturbationFlag}, {"rows_above_flag", flagged}};
    if (flagged > 0)
      warnings.push_back(strcat("3D full wave surface losses: R_loss/|X| exceeds ", kPerturbationFlag, " in ", flagged,
                                " port rows (largest ", worst, " at ", worst_f, " Hz, port ", worst_port, "): the "
                                "perturbative loss is unreliable there (near a self-resonance, within about R_loss/|X| "
                                "of its width, or Q of order 1); the neglected order is about (R_loss/|X|)^2"));
  }
  if (withheld > 0) {
    out["Q_total_withheld_port_rows"] = withheld;
    warnings.push_back(strcat("3D full wave surface losses: Q_total is withheld in ", withheld, " port rows, where "
                              "the solve's true relative residual times |X| exceeds ", kQTotalResolution, " of R_loss "
                              "+ max(Re Z, 0), so the residual can move Q_total by more than that; Q_loss and "
                              "Q_loss_tuned are unaffected (a tighter GMRES tolerance, AXF_FW_RTOL, lowers the residual "
                              "where double precision allows)"));
  }
  if (!pec_groups_.empty()) out["pec_boundary_groups_without_loss"] = pec_groups_;
  if (faces_ == 0 && !pec_groups_.empty()) {
    std::string names;
    for (const std::string& n : pec_groups_) names += (names.empty() ? "" : ", ") + n;
    out["note"] = "no conductor or superconductor is meshed as a volume: the conductors are PEC boundary groups, "
                  "which carry no loss here";
    warnings.push_back("3D full wave surface losses: no surface loss is computed: the conductors of this model are PEC "
                       "boundary groups (" + names + "), which carry no loss here; mesh them as conductor regions "
                       "with an rf block for their loss");
  }
  if (!holes_with_rf_.empty()) {
    std::string names;
    for (const std::string& n : holes_with_rf_) names += (names.empty() ? "" : ", ") + n;
    warnings.push_back("3D full wave surface losses: the conductors " + names + " are PEC boundaries (mesh.remove_pec) "
                       "and carry no surface loss here, although their materials have rf data; mesh them as volumes "
                       "for their loss");
  }
  if (pml_faces_ > 0)
    warnings.push_back(strcat("3D full wave surface losses: ", pml_faces_, " conductor faces border PML layers, where "
                              "the field is not physical; they are left out of Gamma"));
  if (any_mr_ && !static_field_)
    warnings.push_back("3D full wave surface losses: a magnetoresistance model needs the static field, which "
                       "fullwave.rf_only does not compute; R_s is taken at zero field");
  out["setup_seconds"] = setup_seconds_;
  out["evaluation_seconds"] = evaluation_seconds_;
  out["evaluations"] = evaluations_;
  return out;
}

}  // namespace axf
