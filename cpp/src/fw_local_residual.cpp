// The order-2 full-wave error indicator's local problem on one element (fw_local_residual.hpp).
#include "fw_local_residual.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace axf {

namespace {

using cplx = std::complex<double>;
using cvec = Eigen::VectorXcd;
using cmat = Eigen::MatrixXcd;
using rmat = Eigen::MatrixXd;

// a real matrix times a complex vector
cvec times(const rmat& A, const cvec& x) {
  cvec y(A.rows());
  y.real() = A * x.real();
  y.imag() = A * x.imag();
  return y;
}

// the solution of a real SPD system for a complex right-hand side
cvec solve(const Eigen::LLT<rmat>& llt, const cvec& b) {
  cvec x(b.size());
  x.real() = llt.solve(b.real());
  x.imag() = llt.solve(b.imag());
  return x;
}

}  // namespace

std::vector<int> fw_gradient_complement(const rmat& Gr) {
  const int nf = static_cast<int>(Gr.rows());
  std::vector<char> tree(nf, 0);
  if (Gr.cols() > 0) {
    const Eigen::ColPivHouseholderQR<rmat> qr(Gr.transpose());
    for (int k = 0; k < qr.rank(); ++k) tree[qr.colsPermutation().indices()(k)] = 1;
  }
  std::vector<int> complement;
  for (int i = 0; i < nf; ++i)
    if (!tree[i]) complement.push_back(i);
  return complement;
}

FwLocalResidual fw_local_residual(const rmat& MMf, const rmat& CCf, const rmat& Gr, const rmat& MG,
                                  const Eigen::LLT<rmat>& llt, const std::vector<int>& complement, const cvec& bf,
                                  cplx nu, cplx eps, double w2) {
  const int nf = static_cast<int>(MMf.rows());
  const int nc = static_cast<int>(complement.size());
  const int mp = static_cast<int>(Gr.cols());
  const bool gradients = mp > 0;
  // Q^T b = b - MM G_r S^-1 G_r^T b. The test functions are Q v: their gradient parts would carry the traction's
  // inconsistent normal flux
  cvec rhs = bf;
  if (gradients) rhs -= times(MG, solve(llt, times(Gr.transpose(), bf)));
  // the complement block of the mass (MM - MG S^-1 MG^T)_cc and of the curl-curl: only the nc x nc block is formed,
  // nc^2 m' flops for the correction
  rmat Mc(nc, nc), Cc(nc, nc);
  cvec rc(nc);
  rmat MGc(nc, mp);
  for (int r = 0; r < nc; ++r) {
    for (int c = 0; c < nc; ++c) {
      Mc(r, c) = MMf(complement[r], complement[c]);
      Cc(r, c) = CCf(complement[r], complement[c]);
    }
    if (gradients) MGc.row(r) = MG.row(complement[r]);
    rc(r) = rhs(complement[r]);
  }
  if (gradients) Mc.noalias() -= MGc * llt.solve(rmat(MGc.transpose()));
  FwLocalResidual out;
  cvec yc(nc);
  if (nu.imag() == 0.0 && eps.imag() == 0.0) {   // lossless: one real factorization for both parts
    const rmat Ac = nu.real() * Cc - (w2 * eps.real()) * Mc;
    const Eigen::PartialPivLU<rmat> lu(Ac);
    yc.real() = lu.solve(rc.real());
    yc.imag() = lu.solve(rc.imag());
  } else {
    const cmat A = nu * Cc.cast<cplx>() - (w2 * eps) * Mc.cast<cplx>();
    const Eigen::PartialPivLU<cmat> lu(A);
    yc = lu.solve(rc);
  }
  // x = Q C y = C y - G_r S^-1 MG^T C y
  out.x = cvec::Zero(nf);
  for (int r = 0; r < nc; ++r) out.x(complement[r]) = yc(r);
  if (gradients) out.x -= times(Gr, solve(llt, times(MGc.transpose(), yc)));
  const cvec CCx = times(CCf, out.x);
  const cvec MMx = times(MMf, out.x);
  out.magnetic = std::abs(nu) * out.x.dot(CCx).real();
  out.electric = w2 * std::abs(eps) * out.x.dot(MMx).real();
  out.form = nu * (out.x.transpose() * CCx)(0) - w2 * eps * (out.x.transpose() * MMx)(0);
  return out;
}

FwLocalShape fw_local_shape(const double (&v)[4][3]) {
  Eigen::Vector3d p[4];
  for (int i = 0; i < 4; ++i) p[i] = Eigen::Vector3d(v[i][0], v[i][1], v[i][2]);
  FwLocalShape s;
  s.min_edge = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j) {
      const double l = (p[i] - p[j]).norm();
      s.h_max = std::max(s.h_max, l);
      s.min_edge = std::min(s.min_edge, l);
    }
  const double volume6 = std::abs((p[1] - p[0]).dot((p[2] - p[0]).cross(p[3] - p[0])));
  double max_area2 = 0.0;   // twice the largest face's area
  const int faces[4][3] = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};
  for (const auto& f : faces) max_area2 = std::max(max_area2, (p[f[1]] - p[f[0]]).cross(p[f[2]] - p[f[0]]).norm());
  s.min_height = max_area2 > 0.0 ? volume6 / max_area2 : 0.0;   // 3 V / A = (6 V) / (2 A)
  return s;
}

int fw_local_kh_warning(const FwLocalShape& shape, double k) {
  const double kh = k * shape.h_max;
  int flags = 0;
  if (kh > kFwLocalKhBound) flags |= 1;
  if (shape.min_height < kFwLocalSliverRatio * shape.min_edge && kh > kFwLocalSliverKhBound) flags |= 2;
  return flags;
}

}  // namespace axf
