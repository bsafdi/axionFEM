// Internal (not installed): the state of NavierStokesSolver (its Impl), shared with BoussinesqSolver (which builds on
// the same assembled Stokes system, convection kernel, dual norm and monolithic direct matrix).
#pragma once

#include <mfem.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "axionfem/fluids/navier_stokes.hpp"
#include "mhd_internal.hpp"

namespace axf::fluids {

inline DistCsr to_dist(const mfem::HypreParMatrix& M0, MPI_Comm comm) {
  mfem::HypreParMatrix& M = const_cast<mfem::HypreParMatrix&>(M0);
  mfem::SparseMatrix m;
  M.MergeDiagAndOffd(m);
  DistCsr D;
  D.comm = comm;
  D.n_global = M.GetGlobalNumRows();
  D.row_begin = M.RowPart()[0];
  D.local_rows = m.Height();
  D.rowptr.assign(m.GetI(), m.GetI() + D.local_rows + 1);
  D.col.assign(m.GetJ(), m.GetJ() + m.NumNonZeroElems());
  D.val.assign(m.GetData(), m.GetData() + m.NumNonZeroElems());
  D.sum_duplicates();
  return D;
}

inline DistCsr to_dist_sparse(const mfem::SparseMatrix& m, MPI_Comm comm) {
  DistCsr D;
  D.comm = comm;
  D.n_global = m.Height();
  D.row_begin = 0;
  D.local_rows = m.Height();
  D.rowptr.assign(m.GetI(), m.GetI() + D.local_rows + 1);
  D.col.assign(m.GetJ(), m.GetJ() + m.NumNonZeroElems());
  D.val.assign(m.GetData(), m.GetData() + m.NumNonZeroElems());
  D.sum_duplicates();
  return D;
}

inline double gdot(MPI_Comm comm, const mfem::Vector& a, const mfem::Vector& b) { return mfem::InnerProduct(comm, a, b); }

inline int comm_size(MPI_Comm c) {
  int s = 1;
  MPI_Comm_size(c, &s);
  return s;
}


struct NavierStokesSolver::Impl {
  mfem::ParMesh* mesh = nullptr;
  MPI_Comm comm = MPI_COMM_WORLD;
  std::unique_ptr<StokesSolver> S;
  bool axi = false;
  int vdim = 2, order = 2;
  int nu_loc = 0, np_loc = 0;
  // the assembled Stokes system (owned by S; valid after S->assemble())
  const mfem::HypreParMatrix *A_el = nullptr, *A0 = nullptr, *B = nullptr, *Bt = nullptr, *B0 = nullptr;
  mfem::Vector F, rhs_p, lift, mean_weights;
  mfem::Array<int> ess;
  std::vector<char> ess_mask;  // velocity true DOFs (local)
  bool nullspace = false;
  long pin_local = -1;         // local pressure index of the pinned DOF on its owner (PinPressure), else -1
  // the monolithic direct-path matrix with the Stokes solver's nullspace treatment and a given velocity block
  std::function<DistCsr(const mfem::HypreParMatrix&)> mono;
  // convection
  std::unique_ptr<mfem::ParNonlinearForm> N;
  ConvectionIntegrator* conv = nullptr;  // owned by N
  // mass matrices: M_u, M_p (full) and the eliminated versions for the dual norm (DIAG_ONE) / PTC (DIAG_ZERO)
  std::unique_ptr<mfem::HypreParMatrix> Mu, Mp, Mu_one, Mu_zero, Mp_one;
  DistCsr Mu_csr, Mp_csr;
  std::unique_ptr<SparseLU> Mu_lu, Mp_lu;
  std::unique_ptr<mfem::Coefficient> one, rweight;
  int qm = 6;
  // the current state (true DOFs; internal pressure, not normalized)
  mfem::Vector u, p;
  double scale_final = 1.0;
  bool solved = false;
  std::unique_ptr<mfem::ParGridFunction> ug, pg;
  // MHD: the Lorentz coupling (null without MHD): extra unknowns a (potential) appended to the pressure vector
  std::unique_ptr<MhdCoupling> M;
  int na = 0;
  std::unique_ptr<mfem::HypreParMatrix> Kuu_h;
  HYPRE_BigInt kuu_rows[2] = {0, 0};
  DistCsr Ma_csr;
  std::unique_ptr<SparseLU> Ma_lu;
  mfem::SparseMatrix* Kuu_sp = nullptr;
  // transient steps (transient_ns.hpp): direct solves repeated on another LU configuration (cumulative) and the last one
  long lu_fallbacks_solve = 0;
  std::string lu_fallback_name;
  // time term: residual += zero_ess(tr_c Mt u - tr_hist); Jacobian += tr_c Mtz (Mt = Mu, Mtz = Mu_zero)
  double tr_c = 0.0;
  mfem::Vector tr_hist;
  const mfem::HypreParMatrix *Mt = nullptr, *Mtz = nullptr;
  std::unique_ptr<mfem::HypreParMatrix> oseen_full;  // O(w) with no elimination (semi-implicit residual)
  mfem::Vector advect;

  // ---------------------------------------------------------------------------------------------- operators
  void zero_ess(mfem::Vector& v) const {
    for (int i : ess) v(i) = 0.0;
  }
  // R(x) of the system actually solved: free velocity rows, pressure rows (pinned row 0)
  void residual(const mfem::Vector& uu, const mfem::Vector& pp, double s, bool with_conv, mfem::Vector& Ru,
                mfem::Vector& Rp) const {
    Ru.SetSize(nu_loc);
    Rp.SetSize(np_loc + na);
    mfem::Vector t(nu_loc);
    mfem::Vector pph(const_cast<double*>(pp.GetData()), np_loc), Rph(Rp.GetData(), np_loc);
    A0->Mult(uu, Ru);
    Ru *= s;
    Bt->Mult(pph, t);
    Ru += t;
    Ru -= F;
    if (with_conv) {
      if (oseen_full) oseen_full->Mult(uu, t);
      else N->Mult(uu, t);
      Ru += t;
    }
    if (tr_c != 0.0) {  // time step: (rho alpha_0 / dt) M u - history
      mfem::Vector q(nu_loc);
      Mt->Mult(uu, q);
      q *= tr_c;
      q -= tr_hist;
      Ru += q;
    }
    if (M) {  // the Lorentz force -(J x B, v) and the potential rows (linear in (u, phi))
      mfem::Vector a(const_cast<double*>(pp.GetData()) + np_loc, na), Ra;
      M->add_force(uu, a, Ru);
      M->aux_residual(uu, a, Ra);
      for (int i = 0; i < na; ++i) Rp(np_loc + i) = Ra(i);
    }
    zero_ess(Ru);
    B->Mult(uu, Rph);  // eliminated columns: the free part B_free u
    Rph -= rhs_p;      // rhs_p = P(-B_D u_D): the projected lift part
    if (pin_local >= 0) Rp(static_cast<int>(pin_local)) = 0.0;
  }
  double dual(const mfem::Vector& Ru, const mfem::Vector& Rp) const {
    std::vector<double> bu(Ru.GetData(), Ru.GetData() + nu_loc), bp(Rp.GetData(), Rp.GetData() + np_loc);
    std::vector<double> yu = Mu_lu->solve(bu), yp = Mp_lu->solve(bp);
    double loc = 0, glob = 0;
    for (int i = 0; i < nu_loc; ++i) loc += bu[i] * yu[i];
    for (int i = 0; i < np_loc; ++i) loc += bp[i] * yp[i];
    if (M) {
      std::vector<double> ba(Rp.GetData() + np_loc, Rp.GetData() + np_loc + na);
      std::vector<double> ya = Ma_lu->solve(ba);
      for (int i = 0; i < na; ++i) loc += ba[i] * ya[i];
    }
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(std::max(glob, 0.0));
  }
  // the monolithic matrix with the aux (potential) rows and columns appended after the [u; p; borders] layout
  DistCsr extend(const DistCsr& D) const {
    const int nD = D.local_rows;
    const mfem::SparseMatrix &Kua = M->Kua_e(), &Kau = M->Kau_e(), &Kaa = M->Kaa_e();
    DistCsr E;
    E.comm = D.comm;
    E.n_global = nD + na;
    E.row_begin = 0;
    E.local_rows = nD + na;
    E.rowptr.assign(1, 0);
    for (int r = 0; r < nD; ++r) {
      for (int k = D.rowptr[r]; k < D.rowptr[r + 1]; ++k) {
        E.col.push_back(D.col[k]);
        E.val.push_back(D.val[k]);
      }
      if (r < nu_loc)
        for (int k = Kua.GetI()[r]; k < Kua.GetI()[r + 1]; ++k) {
          E.col.push_back(nD + Kua.GetJ()[k]);
          E.val.push_back(Kua.GetData()[k]);
        }
      E.rowptr.push_back(static_cast<int>(E.col.size()));
    }
    for (int i = 0; i < na; ++i) {
      for (int k = Kau.GetI()[i]; k < Kau.GetI()[i + 1]; ++k) {
        E.col.push_back(Kau.GetJ()[k]);
        E.val.push_back(Kau.GetData()[k]);
      }
      for (int k = Kaa.GetI()[i]; k < Kaa.GetI()[i + 1]; ++k) {
        E.col.push_back(nD + Kaa.GetJ()[k]);
        E.val.push_back(Kaa.GetData()[k]);
      }
      E.rowptr.push_back(static_cast<int>(E.col.size()));
    }
    E.sum_duplicates();
    return E;
  }
  // convective gradient at u (mode), essential rows and columns removed (zero, no diagonal)
  std::unique_ptr<mfem::HypreParMatrix> conv_grad(const mfem::Vector& uu, ConvectionIntegrator::GradMode m) const {
    conv->set_grad_mode(m);
    auto& G = dynamic_cast<mfem::HypreParMatrix&>(N->GetGradient(uu));
    std::unique_ptr<mfem::HypreParMatrix> C(new mfem::HypreParMatrix(G));
    conv->set_grad_mode(ConvectionIntegrator::GradMode::Newton);
    C->EliminateBC(ess, mfem::Operator::DIAG_ZERO);
    return C;
  }
  // velocity block s A (+ convective gradient) (+ c M_u; the time mass Mtz)
  std::unique_ptr<mfem::HypreParMatrix> velocity_block(double s, const mfem::HypreParMatrix* C, double cmass) const {
    std::unique_ptr<mfem::HypreParMatrix> J(new mfem::HypreParMatrix(*A_el));
    *J *= s;
    if (C) J.reset(mfem::Add(1.0, *J, 1.0, *C));
    if (Kuu_h) J.reset(mfem::Add(1.0, *J, 1.0, *Kuu_h));
    if (cmass != 0.0) J.reset(mfem::Add(1.0, *J, cmass, *Mtz));
    return J;
  }
};

}  // namespace axf::fluids
