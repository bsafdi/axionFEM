#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

#include "axionfem/axi.hpp"
#include "axionfem/core.hpp"
#include "axionfem/sym_solver.hpp"
#include "axionfem/units.hpp"

namespace axf::axi {

using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;
using units::pi;

namespace {
struct UF {
  std::vector<int> p;
  int add() { p.push_back(static_cast<int>(p.size())); return p.back(); }
  int find(int a) { while (p[a] != a) a = p[a] = p[p[a]]; return a; }
  void unite(int a, int b) { a = find(a); b = find(b); if (a != b) p[std::max(a, b)] = std::min(a, b); }
};
}  // namespace

ToroidalProblem::ToroidalProblem(const Space& sp, const MeshBundle& mb, std::vector<HRole> roles,
                                 std::vector<Profile> profiles, BodyConstraintFn constraints)
    : sp_(sp), mb_(mb), roles_(std::move(roles)), profiles_(std::move(profiles)), constraints_(std::move(constraints)) {
  AXF_REQUIRE(static_cast<int>(roles_.size()) == sp_.mesh.GetNE(), "internal: role table size mismatch");
}

void ToroidalProblem::field(const HSolution& s, int e, const PointEval& pe, double& H, double& mu) const {
  const HRole& ro = roles_[e];
  if (ro.kind == HKind::Excluded) {
    H = 0;
    mu = 0;
    return;
  }
  mu = ro.mu_at ? ro.mu_at(e, pe) : ro.mu;
  if (ro.kind == HKind::Level) {
    H = pe.r > 0 ? s.G_elem[e] / pe.r : 0.0;
    return;
  }
  const int p = ro.profile;
  double us = 0;
  const auto& d = sp_.edofs(e);
  for (size_t i = 0; i < d.size(); ++i) us += s.unit(d[i]) * pe.N[i];
  H = (pe.r > 0 ? s.La[p] / pe.r : 0.0) + (s.Lb[p] - s.La[p]) * us;
}

HSolution ToroidalProblem::solve(int verbose) {
  Timer timer;
  const mfem::Mesh& m = sp_.mesh;
  const int ne = m.GetNE(), nf = m.GetNumFaces();
  const int np = static_cast<int>(profiles_.size());
  auto is_kind = [&](int e, HKind k) { return roles_[e].kind == k; };

  // ------------------------------------------------------------ level components
  int ncomp = 0;
  std::vector<int> comp = components(
      sp_, [&](int e) { return is_kind(e, HKind::Level); }, [&](int) { return true; }, ncomp);
  // excluded bodies
  int nbody = 0;
  std::vector<int> body = components(
      sp_, [&](int e) { return is_kind(e, HKind::Excluded); }, [&](int) { return true; }, nbody);
  for (int e = 0; e < ne; ++e) roles_[e].body = body[e];

  // Union-find nodes: [0] = ZERO (axis / infinity), then components, then winding loops.
  UF uf;
  const int ZERO = uf.add();
  std::vector<int> comp_node(ncomp);
  for (int c = 0; c < ncomp; ++c) comp_node[c] = uf.add();
  // components touching axis or outer boundary are fixed to zero
  for (int be = 0; be < m.GetNBE(); ++be) {
    int a = m.GetBdrAttribute(be);
    if (a != kAxisAttr && a != kOuterAttr) continue;
    int f = m.GetBdrElementFaceIndex(be), e1, e2;
    m.GetFaceElements(f, &e1, &e2);
    if (e1 >= 0 && comp[e1] >= 0) uf.unite(comp_node[comp[e1]], ZERO);
  }

  // Region membership per profile
  std::vector<std::vector<char>> in_prof(np, std::vector<char>(ne, 0));
  for (int e = 0; e < ne; ++e) if (roles_[e].kind == HKind::Profile) in_prof[roles_[e].profile][e] = 1;

  // For each profile: terminal nodes and the faces carrying each terminal.
  std::vector<int> nodeA(np, -1), nodeB(np, -1);   // A: outer/outside, B: inner/inside
  std::vector<std::vector<int>> facesA(np), facesB(np);
  auto other_side_node = [&](int f, int self_elem) -> int {
    int e1, e2;
    m.GetFaceElements(f, &e1, &e2);
    int o = (e1 == self_elem) ? e2 : e1;
    if (o < 0) {
      int ba = sp_.face_bdr_attr[f];
      return (ba == kAxisAttr || ba == kOuterAttr) ? ZERO : -2;
    }
    if (roles_[o].kind == HKind::Level) return comp_node[comp[o]];
    if (roles_[o].kind == HKind::Excluded) return -1;  // Neumann
    return -3;  // another profile
  };
  auto face_elem_in = [&](int f, const std::vector<char>& set) {
    int e1, e2;
    m.GetFaceElements(f, &e1, &e2);
    return (e1 >= 0 && set[e1]) ? e1 : e2;
  };
  for (int p = 0; p < np; ++p) {
    auto loops = boundary_loops(sp_, in_prof[p]);
    AXF_REQUIRE(!loops.empty(), "profile region '" + profiles_[p].name + "' has no elements");
    if (profiles_[p].kind == Profile::Kind::Winding) {
      if (loops.size() != 2) {
        throw Error(strcat("poloidal winding '", profiles_[p].name, "' must be a ring-shaped region with exactly one hole "
                           "(found ", loops.size(), " boundary loops)"),
                    "/windings", "A toroidal-field winding wraps the cross-section: use a 'frame' shape (or a "
                    "difference of two shapes) so its current can circulate around the enclosed box.");
      }
      nodeA[p] = uf.add();  // outer loop
      nodeB[p] = uf.add();  // inner loop
      for (int L = 0; L < 2; ++L) {
        int node = L == 0 ? nodeA[p] : nodeB[p];
        for (int f : loops[L]) {
          (L == 0 ? facesA[p] : facesB[p]).push_back(f);
          int o = other_side_node(f, face_elem_in(f, in_prof[p]));
          if (o == -3) throw Error("winding '" + profiles_[p].name + "' touches another current-carrying region", "/windings");
          if (o >= 0) uf.unite(node, o);
        }
      }
    } else {
      // port: terminals are the distinct level nodes around it
      std::map<int, std::vector<int>> by_node;
      for (auto& L : loops) {
        for (int f : L) {
          int o = other_side_node(f, face_elem_in(f, in_prof[p]));
          if (o == -3) throw Error("port '" + profiles_[p].name + "' touches another current-carrying region", "/pickup");
          if (o >= 0) by_node[o].push_back(f);
        }
      }
      if (by_node.size() != 2) {
        throw Error(strcat("readout port '", profiles_[p].name, "' must separate exactly two current-free regions "
                           "(found ", by_node.size(), ")"),
                    "/pickup/port", "The port region should span the gap between two conductors, with the pickup "
                    "cavity on one side and the outside on the other.");
      }
      auto it = by_node.begin();
      int n0 = it->first;
      auto f0 = it->second;
      ++it;
      int n1 = it->first;
      auto f1 = it->second;
      // inside = not connected to ZERO, or the side containing inside_point
      bool z0 = uf.find(n0) == uf.find(ZERO), z1 = uf.find(n1) == uf.find(ZERO);
      bool swap = false;
      if (profiles_[p].has_inside_point) {
        mfem::DenseMatrix pts(2, 1);
        pts(0, 0) = profiles_[p].inside_r;
        pts(1, 0) = profiles_[p].inside_z;
        mfem::Array<int> eid;
        mfem::Array<mfem::IntegrationPoint> ips;
        const_cast<mfem::Mesh&>(m).FindPoints(pts, eid, ips, false);
        AXF_REQUIRE(eid[0] >= 0 && roles_[eid[0]].kind == HKind::Level,
                    "pickup inside_point is not inside a current-free region", "/pickup/inside_point");
        int nin = comp_node[comp[eid[0]]];
        if (uf.find(nin) == uf.find(n0)) swap = true;
        else AXF_REQUIRE(uf.find(nin) == uf.find(n1), "pickup inside_point is not adjacent to the port", "/pickup/inside_point");
      } else if (z1 && !z0) {
        swap = true;
      } else if (z0 == z1) {
        throw Error("cannot tell which side of port '" + profiles_[p].name + "' is the pickup cavity", "/pickup",
                    "Give pickup.inside_point = [r, z] inside the cavity.");
      }
      if (swap) {
        std::swap(n0, n1);
        std::swap(f0, f1);
      }
      nodeA[p] = n0;  // outside
      nodeB[p] = n1;  // inside
      facesA[p] = f0;
      facesB[p] = f1;
    }
  }

  // ------------------------------------------------------------ unit profiles (FE)
  const int nd = sp_.ndof();
  HSolution s;
  s.unit.SetSize(nd);
  s.unit = 0.0;
  json prof_diag = json::array();
  for (int p = 0; p < np; ++p) {
    // DOF sets
    std::vector<int> kind(nd, 0);  // 0 unused, 1 free, 2 dirichlet
    std::vector<double> val(nd, 0.0);
    for (int e = 0; e < ne; ++e) if (in_prof[p][e]) for (int d : sp_.edofs(e)) kind[d] = 1;
    mfem::Array<int> fd;
    auto set_dir = [&](const std::vector<int>& faces, double G) {
      for (int f : faces) {
        sp_.fes->GetFaceDofs(f, fd);
        for (int d : fd) {
          kind[d] = 2;
          val[d] = sp_.r[d] > 0 ? G / sp_.r[d] : 0.0;
        }
      }
    };
    set_dir(facesA[p], 0.0);
    set_dir(facesB[p], 1.0);
    for (int d = 0; d < nd; ++d) if (kind[d] == 1 && sp_.r[d] == 0.0) { kind[d] = 2; val[d] = 0.0; }
    std::vector<int> idx(nd, -1);
    int nfree = 0;
    for (int d = 0; d < nd; ++d) if (kind[d] == 1) idx[d] = nfree++;
    std::vector<Trip> trip;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nfree);
    PointEval pe;
    const RefTable& tab = sp_.table(sp_.quad_rule());
    for (int e = 0; e < ne; ++e) {
      if (!in_prof[p][e]) continue;
      const auto& d = sp_.edofs(e);
      const int n = static_cast<int>(d.size());
      Eigen::MatrixXd Ke = Eigen::MatrixXd::Zero(n, n);
      for (int q = 0; q < tab.nq; ++q) {
        sp_.eval(e, tab, q, pe);
        double wr = pe.w * pe.r;
        for (int i = 0; i < n; ++i) {
          double ai = -pe.dshape(i, 1), bi = pe.dshape(i, 0) + pe.shape(i) / pe.r;
          for (int j = 0; j < n; ++j) {
            double aj = -pe.dshape(j, 1), bj = pe.dshape(j, 0) + pe.shape(j) / pe.r;
            Ke(i, j) += wr * (ai * aj + bi * bj);
          }
        }
      }
      for (int i = 0; i < n; ++i) {
        if (kind[d[i]] != 1) continue;
        for (int j = 0; j < n; ++j) {
          if (kind[d[j]] == 1) trip.emplace_back(idx[d[i]], idx[d[j]], Ke(i, j));
          else rhs(idx[d[i]]) -= Ke(i, j) * val[d[j]];
        }
      }
    }
    Eigen::VectorXd x = Eigen::VectorXd::Zero(nfree);
    if (nfree > 0) {
      SpMat K(nfree, nfree);
      K.setFromTriplets(trip.begin(), trip.end());
      SymSolver ch(K);
      if (ch.info() != Eigen::Success) throw SolverFailure("profile system for '" + profiles_[p].name + "' is singular");
      x = ch.solve(rhs);
    }
    for (int d = 0; d < nd; ++d) {
      if (kind[d] == 1) s.unit(d) = x(idx[d]);
      else if (kind[d] == 2) s.unit(d) = val[d];
    }
    prof_diag.push_back({{"name", profiles_[p].name}, {"free_dofs", nfree}});
  }

  // ------------------------------------------------------------ level equations
  // canonical node ids
  std::map<int, int> canon;  // root -> unknown index (or -1 for ZERO)
  const int zroot = uf.find(ZERO);
  int nunk = 0;
  auto unk = [&](int node) -> int {
    int rt = uf.find(node);
    if (rt == zroot) return -1;
    auto it = canon.find(rt);
    if (it != canon.end()) return it->second;
    canon[rt] = nunk;
    return nunk++;
  };
  for (int c = 0; c < ncomp; ++c) unk(comp_node[c]);
  for (int p = 0; p < np; ++p) { unk(nodeA[p]); unk(nodeB[p]); }

  struct Eq { std::map<int, double> coef; double rhs; std::string what; };
  std::vector<Eq> eqs;
  auto add_diff = [&](int nb, int na, double rhs, const std::string& what) {
    Eq q;
    int ib = unk(nb), ia = unk(na);
    if (ib >= 0) q.coef[ib] += 1.0;
    if (ia >= 0) q.coef[ia] -= 1.0;
    q.rhs = rhs;
    q.what = what;
    eqs.push_back(q);
  };
  for (int p = 0; p < np; ++p) {
    add_diff(nodeB[p], nodeA[p], profiles_[p].current / (2 * pi),
             (profiles_[p].kind == Profile::Kind::Winding ? "winding " : "port ") + profiles_[p].name);
  }

  // excluded bodies enclosing cavities
  json body_diag = json::array();
  for (int b = 0; b < nbody; ++b) {
    std::vector<char> inb(ne, 0);
    std::set<std::string> names;
    for (int e = 0; e < ne; ++e) if (body[e] == b) { inb[e] = 1; names.insert(mb_.region_of_attr.at(m.GetAttribute(e))); }
    auto loops = boundary_loops(sp_, inb);
    std::vector<std::string> nm(names.begin(), names.end());
    auto loop_nodes = [&](const std::vector<int>& L) {
      std::set<int> nodes;
      for (int f : L) {
        int e1, e2;
        m.GetFaceElements(f, &e1, &e2);
        int o = (e1 >= 0 && inb[e1]) ? e2 : e1;
        if (o < 0) {
          int ba = sp_.face_bdr_attr[f];
          if (ba == kAxisAttr || ba == kOuterAttr) nodes.insert(zroot);
          continue;
        }
        if (roles_[o].kind == HKind::Level) nodes.insert(uf.find(comp_node[comp[o]]));
        else if (roles_[o].kind == HKind::Profile) {
          int p = roles_[o].profile;
          // which terminal of the profile touches this face?
          bool onA = std::find(facesA[p].begin(), facesA[p].end(), f) != facesA[p].end();
          bool onB = std::find(facesB[p].begin(), facesB[p].end(), f) != facesB[p].end();
          if (onA) nodes.insert(uf.find(nodeA[p]));
          else if (onB) nodes.insert(uf.find(nodeB[p]));
          else nodes.insert(-100 - p);  // current enters the body through a port side: no simple constraint
        }
      }
      return nodes;
    };
    std::set<int> outer = loop_nodes(loops[0]);
    for (size_t h = 1; h < loops.size(); ++h) {
      std::set<int> inner = loop_nodes(loops[h]);
      spec::Constraint c = constraints_(nm);
      // A net-current constraint relates two levels; an enclosed-flux constraint integrates whatever fills
      // the hole (several levels and port profiles, e.g. a sealed cavity with the readout port inside).
      if (c.kind == spec::Constraint::Kind::NetCurrent &&
          (inner.size() != 1 || outer.size() != 1 || *inner.begin() < 0 || *outer.begin() < -1))
        continue;
      if (inner.empty()) continue;
      int nin = *inner.begin(), nout = outer.empty() ? zroot : *outer.begin();
      std::string what = "superconductor " + nm.front() + " hole " + std::to_string(h);
      if (c.kind == spec::Constraint::Kind::NetCurrent) {
        // net poloidal current of the body around this hole = 2 pi (G_in - G_out)
        Eq q;
        int ii = (nin == zroot) ? -1 : canon.at(nin), io = (nout == zroot) ? -1 : canon.at(nout);
        if (ii >= 0) q.coef[ii] += 1.0;
        if (io >= 0) q.coef[io] -= 1.0;
        q.rhs = c.value / (2 * pi);
        q.what = what + " (net poloidal current)";
        eqs.push_back(q);
      } else {
        // toroidal flux enclosed: int mu G / r dr dz over the hole interior = Phi
        std::vector<char> inside(ne, 0);
        std::vector<int> stack;
        for (int f : loops[h]) {
          int e1, e2;
          m.GetFaceElements(f, &e1, &e2);
          int o = inb[e1] ? e2 : e1;
          if (o >= 0 && !inside[o]) { inside[o] = 1; stack.push_back(o); }
        }
        while (!stack.empty()) {
          int e = stack.back();
          stack.pop_back();
          for (auto [n, f] : sp_.nbr[e]) {
            (void)f;
            if (!inside[n] && !inb[n]) { inside[n] = 1; stack.push_back(n); }
          }
        }
        Eq q;
        q.rhs = c.value;
        PointEval pe;
        const RefTable& tab = sp_.table(sp_.quad_rule());
        for (int e = 0; e < ne; ++e) {
          if (!inside[e] || roles_[e].kind == HKind::Excluded) continue;
          for (int qq = 0; qq < tab.nq; ++qq) {
            sp_.eval(e, tab, qq, pe);
            double mu = roles_[e].mu_at ? roles_[e].mu_at(e, pe) : roles_[e].mu;
            if (roles_[e].kind == HKind::Level) {
              int k = unk(comp_node[comp[e]]);
              if (k >= 0) q.coef[k] += pe.w * mu / pe.r;
            } else {
              int p = roles_[e].profile;
              double us = 0;
              const auto& d = sp_.edofs(e);
              for (size_t i = 0; i < d.size(); ++i) us += s.unit(d[i]) * pe.N[i];
              // H = La/r + (Lb - La) us  -> flux density mu H
              int ka = unk(nodeA[p]), kb = unk(nodeB[p]);
              if (ka >= 0) q.coef[ka] += pe.w * mu * (1.0 / pe.r - us);
              if (kb >= 0) q.coef[kb] += pe.w * mu * us;
            }
          }
        }
        q.what = what + " (enclosed toroidal flux)";
        eqs.push_back(q);
      }
      body_diag.push_back({{"body", nm}, {"hole", h}, {"constraint", eqs.back().what}});
    }
  }

  // ------------------------------------------------------------ solve levels
  std::vector<double> L(nunk, 0.0);
  json lev_diag;
  if (nunk > 0) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(eqs.size(), nunk);
    Eigen::VectorXd b(eqs.size());
    for (size_t i = 0; i < eqs.size(); ++i) {
      for (auto& [k, v] : eqs[i].coef) A(i, k) += v;
      b(i) = eqs[i].rhs;
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    qr.setThreshold(1e-10);
    int rank = static_cast<int>(qr.rank());
    if (rank < nunk) {
      // find an undetermined node to report
      std::string where;
      for (int c = 0; c < ncomp; ++c) {
        if (unk(comp_node[c]) < 0) continue;
        for (int e = 0; e < ne; ++e) {
          if (comp[e] == c) {
            double rr, zz;
            element_center(sp_, e, rr, zz);
            where = strcat("current-free region near (r, z) = (", rr, ", ", zz, ") in '",
                           mb_.region_of_attr.at(m.GetAttribute(e)), "'");
            break;
          }
        }
        if (!where.empty()) break;
      }
      throw Error(strcat("the toroidal-field problem is underdetermined (", nunk, " cavity levels, rank ", rank, "); ",
                         where),
                  "/superconductors", "A cavity enclosed by conductors needs a winding, a readout port, or a "
                  "superconductor poloidal constraint (net_current or flux).");
    }
    Eigen::VectorXd x = qr.solve(b);
    double resid = (A * x - b).norm() / std::max(1.0, b.norm());
    if (resid > 1e-8) {
      std::string list;
      for (auto& q : eqs) list += q.what + "; ";
      throw Error(strcat("the toroidal-field constraints are inconsistent (relative residual ", resid, "): ", list),
                  "/superconductors", "Two constraints fix the same cavity level differently; remove one.");
    }
    for (int k = 0; k < nunk; ++k) L[k] = x(k);
    lev_diag = {{"unknowns", nunk}, {"equations", eqs.size()}, {"residual", resid}};
  }
  auto level = [&](int node) { int k = unk(node); return k < 0 ? 0.0 : L[k]; };
  s.G_elem.assign(ne, 0.0);
  for (int e = 0; e < ne; ++e) if (roles_[e].kind == HKind::Level) s.G_elem[e] = level(comp_node[comp[e]]);
  s.termA = facesA;
  s.termB = facesB;
  s.La.resize(np);
  s.Lb.resize(np);
  for (int p = 0; p < np; ++p) {
    s.La[p] = level(nodeA[p]);
    s.Lb[p] = level(nodeB[p]);
  }

  // energy
  double W = 0;
  {
    const RefTable& tab = sp_.table(sp_.quad_rule());
#pragma omp parallel reduction(+ : W)
    {
      PointEval pe;
#pragma omp for schedule(dynamic, 256)
      for (int e = 0; e < ne; ++e) {
        if (roles_[e].kind == HKind::Excluded) continue;
        for (int q = 0; q < tab.nq; ++q) {
          sp_.eval(e, tab, q, pe);
          double H, mu;
          field(s, e, pe, H, mu);
          W += pe.w * pe.r * mu * H * H;
        }
      }
    }
  }
  s.energy = pi * W;
  s.diagnostics = {{"level_regions", ncomp}, {"excluded_bodies", nbody}, {"profiles", prof_diag},
                   {"levels", lev_diag}, {"constraints", body_diag}, {"seconds", timer.seconds()}};
  (void)nf;
  (void)verbose;
  return s;
}

}  // namespace axf::axi
