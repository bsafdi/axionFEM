// The order-2 full-wave error indicator's local problem (fw_local_residual, fw_local_residual.hpp) is posed on the
// MM-orthogonal complement range(Q) of the local gradients, and does not depend on the tree that picks its basis.
// On an affine, skewed ND_3 tetrahedron, for three
// conductor patterns (none: every interior element; a perfect-conductor face; a perfect-conductor edge), lossless and
// lossy, at k h of about 0.3:
//   - a manufactured local error e* = Q v (v random) with the load b = A e* is returned to 1e-12, and the span(C)
//     formulation (span(C) with the unprojected mass) misses it by more than 1e-3: no ties are involved;
//   - a second tree, different by construction (one row of the first tree zeroed in a copy of G_r), gives the same
//     magnetic and electric parts and form to 1e-12, where the span(C) formulation differs by more than 1e-6;
//   - a relative perturbation of 1e-15 of G_r (the level at which OpenBLAS's kernel choice moves MFEM's bases) moves
//     them by less than 1e-12.
// MPI-free: it builds with a serial MFEM and with AXF_MPI=OFF (fw_local_residual.cpp is compiled in every
// configuration), and it gives the same verdict under every OpenBLAS kernel family.
#include "../src/fw_local_residual.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

using rmat = Eigen::MatrixXd;
using cvec = Eigen::VectorXcd;
using cmat = Eigen::MatrixXcd;
using cplx = std::complex<double>;

namespace {

// a small deterministic generator (std::uniform_real_distribution is implementation-defined)
struct Lcg {
  unsigned long long state = 20261031ULL;
  double operator()() {   // in [-1, 1)
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / 4503599627370496.0 - 1.0;
  }
};

struct Element {
  rmat MM, CC, G;                                // ND_3 mass and curl-curl, H1_3 gradients in ND_3 (reference)
  std::vector<char> face0_nd, face0_h1;          // dofs with a tangential trace (ND) or a value (H1) on face 0
  std::vector<char> edge0_nd, edge0_h1;          // the same on edge 0
};

// MM, CC and G on the affine image x = J xi of the reference tetrahedron, and the dofs on face 0 and edge 0
Element element(const Eigen::Matrix3d& J) {
  mfem::ND_FECollection nd_fec(3, 3);
  mfem::H1_FECollection h1_fec(3, 3);
  const mfem::FiniteElement* nd = nd_fec.FiniteElementForGeometry(mfem::Geometry::TETRAHEDRON);
  const mfem::FiniteElement* h1 = h1_fec.FiniteElementForGeometry(mfem::Geometry::TETRAHEDRON);
  const int n = nd->GetDof();
  const int m = h1->GetDof();
  mfem::IsoparametricTransformation reference;
  reference.SetIdentityTransformation(mfem::Geometry::TETRAHEDRON);
  mfem::DenseMatrix g;
  nd->ProjectGrad(*h1, reference, g);
  Element el;
  el.G = Eigen::Map<const rmat>(g.Data(), g.Height(), g.Width());
  const double det = J.determinant();
  const Eigen::Matrix3d invJ = J.inverse();
  const Eigen::Matrix3d gm = invJ * invJ.transpose();   // covariant: phi = J^-T phi_ref
  const Eigen::Matrix3d gc = J.transpose() * J;         // curls: curl phi = J curl_ref / det J
  el.MM = rmat::Zero(n, n);
  el.CC = rmat::Zero(n, n);
  const mfem::IntegrationRule& rule = mfem::IntRules.Get(mfem::Geometry::TETRAHEDRON, 6);
  mfem::DenseMatrix vs(n, 3), cs(n, 3);
  for (int k = 0; k < rule.GetNPoints(); ++k) {
    nd->CalcVShape(rule.IntPoint(k), vs);
    nd->CalcCurlShape(rule.IntPoint(k), cs);
    const Eigen::Map<const rmat> v(vs.Data(), n, 3);
    const Eigen::Map<const rmat> c(cs.Data(), n, 3);
    const double w = rule.IntPoint(k).weight;
    el.MM += w * std::abs(det) * v * gm * v.transpose();
    el.CC += w / std::abs(det) * c * gc * c.transpose();
  }
  // face 0: vertices 1, 2, 3 (x + y + z = 1, normal (1, 1, 1)); edge 0: vertices 0 and 1 (the x axis), as MFEM's
  // reference tetrahedron numbers them (the same sampling as model3d_fw_indicator.cpp's Residual setup)
  el.face0_nd.assign(n, 0);
  el.face0_h1.assign(m, 0);
  el.edge0_nd.assign(n, 0);
  el.edge0_h1.assign(m, 0);
  mfem::Vector hs(m);
  const double face_points[4][3] = {{0.5, 0.2, 0.3}, {0.3, 0.55, 0.15}, {0.2, 0.1, 0.7}, {0.34, 0.33, 0.33}};
  for (const auto& p : face_points) {
    mfem::IntegrationPoint ip;
    ip.Set3(p[0], p[1], p[2]);
    nd->CalcVShape(ip, vs);
    h1->CalcShape(ip, hs);
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d x(vs(i, 0), vs(i, 1), vs(i, 2));
      if (x.cross(Eigen::Vector3d(1, 1, 1)).squaredNorm() > 1e-20) el.face0_nd[i] = 1;
    }
    for (int l = 0; l < m; ++l)
      if (std::fabs(hs(l)) > 1e-10) el.face0_h1[l] = 1;
  }
  for (double s : {0.15, 0.4, 0.65, 0.9}) {
    mfem::IntegrationPoint ip;
    ip.Set3(s, 0.0, 0.0);
    nd->CalcVShape(ip, vs);
    h1->CalcShape(ip, hs);
    for (int i = 0; i < n; ++i)
      if (std::fabs(vs(i, 0)) > 1e-10) el.edge0_nd[i] = 1;
    for (int l = 0; l < m; ++l)
      if (std::fabs(hs(l)) > 1e-10) el.edge0_h1[l] = 1;
  }
  return el;
}

// one conductor pattern: the free ND dofs (not on the conductor) and G_r (the gradients of the H1 nodes off the
// conductor, which is the reference; without a conductor vertex 0 fixes the constant)
struct Pattern {
  std::string name;
  std::vector<int> free;
  rmat MM, CC, Gr;
};

Pattern pattern(const Element& el, const std::string& name, const std::vector<char>* nd_on,
                const std::vector<char>* h1_on) {
  Pattern p;
  p.name = name;
  const int n = static_cast<int>(el.MM.rows());
  const int m = static_cast<int>(el.G.cols());
  for (int i = 0; i < n; ++i)
    if (!nd_on || !(*nd_on)[i]) p.free.push_back(i);
  std::vector<int> columns;
  for (int l = 0; l < m; ++l) {
    if (h1_on ? (*h1_on)[l] : l == 0) continue;
    columns.push_back(l);
  }
  const int nf = static_cast<int>(p.free.size());
  p.MM.resize(nf, nf);
  p.CC.resize(nf, nf);
  p.Gr.resize(nf, static_cast<int>(columns.size()));
  for (int r = 0; r < nf; ++r) {
    for (int c = 0; c < nf; ++c) {
      p.MM(r, c) = el.MM(p.free[r], p.free[c]);
      p.CC(r, c) = el.CC(p.free[r], p.free[c]);
    }
    for (size_t c = 0; c < columns.size(); ++c) p.Gr(r, static_cast<int>(c)) = el.G(p.free[r], columns[c]);
  }
  return p;
}

axf::FwLocalResidual local(const Pattern& p, const rmat& Gr, const std::vector<int>& complement, const cvec& b,
                           cplx eps, double w2) {
  const rmat MG = p.MM * Gr;
  Eigen::LLT<rmat> llt(Gr.transpose() * MG);
  return axf::fw_local_residual(p.MM, p.CC, Gr, MG, llt, complement, b, cplx(1.0, 0.0), eps, w2);
}

// the span(C) formulation, for comparison: the local problem on span(C) with the unprojected mass block (then x -= P x)
cvec previous(const Pattern& p, const std::vector<int>& complement, const cvec& b, cplx eps, double w2) {
  const int nc = static_cast<int>(complement.size());
  const rmat& Gr = p.Gr;
  Eigen::LLT<rmat> llt(Gr.transpose() * p.MM * Gr);
  const rmat P = Gr * llt.solve(rmat(Gr.transpose() * p.MM));
  const cvec rhs = b - (P.transpose()).cast<cplx>() * b;
  cmat A(nc, nc);
  cvec rc(nc);
  for (int r = 0; r < nc; ++r) {
    for (int c = 0; c < nc; ++c)
      A(r, c) = p.CC(complement[r], complement[c]) - w2 * eps * p.MM(complement[r], complement[c]);
    rc(r) = rhs(complement[r]);
  }
  const cvec y = A.partialPivLu().solve(rc);
  cvec x = cvec::Zero(p.MM.rows());
  for (int r = 0; r < nc; ++r) x(complement[r]) = y(r);
  return x - P.cast<cplx>() * x;
}

double rel(double a, double b) {
  return std::abs(a - b) / std::max(std::abs(a), std::abs(b));
}

double magnetic(const Pattern& p, const cvec& x) {
  return x.dot(p.CC.cast<cplx>() * x).real();
}

}  // namespace

int main() {
  Eigen::Matrix3d J;
  J << 1.0, 0.31, -0.12, 0.05, 0.83, 0.27, 0.18, -0.09, 1.12;
  const Element el = element(J);
  std::vector<Pattern> patterns;
  patterns.push_back(pattern(el, "no conductor", nullptr, nullptr));
  patterns.push_back(pattern(el, "a conductor face", &el.face0_nd, &el.face0_h1));
  patterns.push_back(pattern(el, "a conductor edge", &el.edge0_nd, &el.edge0_h1));
  Lcg u;
  const double w2 = 0.09;   // k^2 with element edges of order 1: k h about 0.3
  int failures = 0;
  auto fail = [&](const std::string& what) {
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
  };
  for (const Pattern& p : patterns) {
    const int nf = static_cast<int>(p.free.size());
    const std::vector<int> first = axf::fw_gradient_complement(p.Gr);   // the complement of the first tree
    if (static_cast<int>(first.size()) != nf - p.Gr.cols()) fail(p.name + ": the complement has the wrong size");
    // a second tree, different by construction: zero the first tree row (the first free dof outside the complement)
    // in a copy of G_r; it cannot be in that copy's tree. Without gradients (m' = 0) there is no tree
    const bool gradients = p.Gr.cols() > 0;
    std::vector<int> other = first;
    if (gradients) {
      int tree_row = -1;
      for (int i = 0; i < nf && tree_row < 0; ++i)
        if (!std::binary_search(first.begin(), first.end(), i)) tree_row = i;
      rmat Gz = p.Gr;
      Gz.row(tree_row).setZero();
      other = axf::fw_gradient_complement(Gz);
      if (other == first) fail(p.name + ": the second tree coincides with the first");
    }
    rmat Gp = p.Gr;
    for (int i = 0; i < Gp.rows(); ++i)
      for (int j = 0; j < Gp.cols(); ++j) Gp(i, j) *= 1.0 + 1e-15 * u();
    const std::vector<int> perturbed = axf::fw_gradient_complement(Gp);
    cvec b(nf), v(nf);
    for (int i = 0; i < nf; ++i) {
      b(i) = cplx{u(), u()};   // braced: the two draws in order on every compiler
      v(i) = cplx{u(), u()};
    }
    for (cplx eps : {cplx(1.0, 0.0), cplx(1.0, -0.4)}) {
      const cmat A = p.CC.cast<cplx>() - (w2 * eps) * p.MM.cast<cplx>();
      // the manufactured local error e* = Q v and its load b* = A e*
      Eigen::LLT<rmat> llt(p.Gr.transpose() * p.MM * p.Gr);
      const rmat P = p.Gr * llt.solve(rmat(p.Gr.transpose() * p.MM));
      const cvec e_star = v - P.cast<cplx>() * v;
      const cvec b_star = A * e_star;
      const auto manufactured = local(p, p.Gr, first, b_star, eps, w2);
      const double d_star = (manufactured.x - e_star).norm() / e_star.norm();
      const double d_star_prev = (previous(p, first, b_star, eps, w2) - e_star).norm() / e_star.norm();
      // its energies and form, from e* itself: |nu| e*^H CC e*, w^2 |eps| e*^H MM e*, e*^T A e* (nu = 1)
      const cvec CCe = p.CC.cast<cplx>() * e_star;
      const cvec MMe = p.MM.cast<cplx>() * e_star;
      const double magnetic_star = e_star.dot(CCe).real();
      const double electric_star = w2 * std::abs(eps) * e_star.dot(MMe).real();
      const cplx form_star = (e_star.transpose() * CCe)(0) - w2 * eps * (e_star.transpose() * MMe)(0);
      const double d_parts = std::max({rel(manufactured.magnetic, magnetic_star),
                                       rel(manufactured.electric, electric_star),
                                       std::abs(manufactured.form - form_star) / std::abs(form_star)});
      // two trees, and G_r perturbed
      const auto a = local(p, p.Gr, first, b, eps, w2);
      const auto c = local(p, p.Gr, other, b, eps, w2);
      const auto q = local(p, Gp, perturbed, b, eps, w2);
      const double d_tree = std::max({rel(a.magnetic, c.magnetic), rel(a.electric, c.electric),
                                      std::abs(a.form - c.form) / std::abs(a.form)});
      const double d_pert = std::max({rel(a.magnetic, q.magnetic), rel(a.electric, q.electric),
                                      std::abs(a.form - q.form) / std::abs(a.form)});
      const double prev_first = magnetic(p, previous(p, first, b, eps, w2));
      const double d_prev = rel(prev_first, magnetic(p, previous(p, other, b, eps, w2)));
      std::printf("%s, eps = %g%+gi (%d free dofs, %zu in the complement): manufactured error returned to %.1e "
                  "(span(C) formulation %.1e), its parts and form to %.1e; another tree %.1e (span(C) %.1e); G_r "
                  "perturbed %.1e\n", p.name.c_str(), eps.real(), eps.imag(), nf, first.size(), d_star, d_star_prev,
                  d_parts, d_tree, d_prev, d_pert);
      if (!(d_star < 1e-12)) fail(p.name + ": the manufactured local error is not returned (limit 1e-12)");
      if (!(d_parts < 1e-12)) fail(p.name + ": the magnetic or electric part or the form differs from e*'s");
      if (!(d_star_prev > 1e-3)) fail(p.name + ": the span(C) formulation returns the manufactured error too");
      if (!(d_pert < 1e-12)) fail(p.name + ": the local estimate depends on a 1e-15 perturbation of G_r");
      if (gradients && !(d_tree < 1e-12)) fail(p.name + ": the local estimate depends on the tree");
      if (gradients && !(d_prev > 1e-6)) fail(p.name + ": the span(C) formulation does not depend on the tree here");
      if (!(a.magnetic > 0.0) || !(a.electric > 0.0)) fail(p.name + ": non-positive parts");
    }
  }
  // the heuristic k h warning (fw_local_kh_warning) at known shapes: the reference tetrahedron (h_max sqrt 2, smallest
  // height 1/sqrt 3), a flat sliver (thickness 0.03; smallest height 0.042 of its shortest edge) and a
  // 20:1 Kuhn needle (thin, but not a sliver: smallest height 0.71 of its shortest edge)
  auto shape = [](const Eigen::Matrix3d& Jt) {
    double vtx[4][3] = {{0, 0, 0}};
    for (int i = 1; i < 4; ++i)
      for (int c = 0; c < 3; ++c) vtx[i][c] = Jt(c, i - 1);
    return axf::fw_local_shape(vtx);
  };
  Eigen::Matrix3d sliver, needle;
  sliver << 1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 0.03;
  needle << 20, 20, 20, 0, 1, 1, 0, 0, 1;
  needle /= 20.0;
  const axf::FwLocalShape ref_shape = shape(Eigen::Matrix3d::Identity());
  const axf::FwLocalShape sliver_shape = shape(sliver);
  const axf::FwLocalShape needle_shape = shape(needle);
  if (std::abs(ref_shape.h_max - std::sqrt(2.0)) > 1e-14 || std::abs(ref_shape.min_height - 1 / std::sqrt(3.0)) > 1e-14)
    fail("fw_local_shape: the reference tetrahedron's h_max or smallest height");
  struct Case {
    const char* what;
    axf::FwLocalShape s;
    double kh;
    int flags;
  };
  const Case cases[] = {{"reference at k h_max 0.5", ref_shape, 0.5, 0},
                        {"reference at k h_max 1.2", ref_shape, 1.2, 1},
                        {"sliver at k h_max 0.15", sliver_shape, 0.15, 0},
                        {"sliver at k h_max 0.25", sliver_shape, 0.25, 2},
                        {"sliver at k h_max 1.2", sliver_shape, 1.2, 3},
                        {"needle at k h_max 0.5", needle_shape, 0.5, 0}};
  for (const Case& c : cases) {
    const int flags = axf::fw_local_kh_warning(c.s, c.kh / c.s.h_max);
    std::printf("k h warning, %s: flags %d\n", c.what, flags);
    if (flags != c.flags) fail(std::string("fw_local_kh_warning, ") + c.what);
  }
  std::printf(failures ? "test_fw_indicator_local FAILED\n" : "test_fw_indicator_local OK\n");
  return failures ? 1 : 0;
}
