// Internal (not installed): the assembled Stokes system, read by NavierStokesSolver through the friend declaration in
// StokesSolver.
#pragma once

#include <mfem.hpp>

#include <map>
#include <memory>
#include <vector>

#include "axionfem/fluids/stokes.hpp"

namespace axf::fluids {

struct StokesSolver::Impl {
  int vdim = 2;
  bool direct = true;
  mfem::Array<int> ess_tdofs;   // all eliminated velocity true DOFs (Dirichlet, axis, slip normal/corner)
  mfem::Array<int> dir_tdofs;   // Dirichlet + axis only
  std::vector<char> ess_mask;   // local true DOFs of the monolithic layout (velocity rows)
  mfem::Array<int> offsets;     // block offsets of the true-DOF vectors (u, p)
  std::unique_ptr<mfem::ParBilinearForm> a, mp;
  std::unique_ptr<mfem::ParMixedBilinearForm> b;
  std::unique_ptr<mfem::ParLinearForm> f, pmean;
  std::unique_ptr<mfem::HypreParMatrix> A, B, Bt, Mp, B0, A0;
  std::unique_ptr<mfem::HypreParMatrix> Ae, Be;  // eliminated rows/cols (kept for refresh_data)
  std::unique_ptr<mfem::BlockVector> rhs, x;
  mfem::Vector Fvec;          // assembled (f, v) + <g, v>_traction (+ prescribed periodic drive), true DOFs, Cartesian
  mfem::Vector mean_weights;  // int phi_k (r-weighted in axisymmetry), pressure true DOFs
  // slip: u = R w on the velocity true DOFs (identity except 2x2 rotations at slip nodes)
  bool rotated = false;
  mfem::SparseMatrix Rdiag;
  std::unique_ptr<mfem::HypreParMatrix> R;
  mfem::Array<int> slip_ess;  // rotated-frame DOFs eliminated with value 0
  // periodic drives: per periodic condition (in order), the load vector c = int phi . d_hat (Cartesian)
  struct Drive {
    int attribute = 0;
    PeriodicDrive kind = PeriodicDrive::None;
    double value = 0, length = 1;
    mfem::Vector c;  // Cartesian
  };
  std::vector<Drive> drives;
  // extra (bordered) unknowns of the direct path: flow-rate multipliers (velocity part) and the mean multiplier
  struct Border {
    mfem::Vector u, p;  // local true-DOF blocks (rotated frame; essential entries zero)
    double rhs = 0;
  };
  std::vector<Border> borders;  // flow-rate constraints only (the mean multiplier is added in monolithic())
  std::vector<double> lambda;   // values of all bordered unknowns of the last direct solve (flow rates first)
  // markers and coefficients kept alive for the forms
  std::vector<mfem::Array<int>> markers;
  std::vector<std::unique_ptr<mfem::VectorCoefficient>> vcoefs;
  std::vector<std::unique_ptr<mfem::Coefficient>> scoefs;
  mfem::ConstantCoefficient zero{0.0}, minus_one{-1.0}, one{1.0};
  std::unique_ptr<mfem::Coefficient> inv_nu, mp_coef, r_coef, rmean;
  std::unique_ptr<mfem::VectorCoefficient> rforce;
  std::map<int, BoundaryCondition> bc;  // attribute -> condition (periodic partners included, as Periodic)
  std::vector<int> periodic_order;      // attributes carrying a Periodic condition, in the user's order
  bool assembled = false;
  long pp_rank = -1, pp_local = -1;  // pressure-point node (owner rank, local true DOF)
  mfem::Coefficient* S(mfem::Coefficient* c) {
    scoefs.emplace_back(c);
    return c;
  }
  mfem::VectorCoefficient* V(mfem::VectorCoefficient* c) {
    vcoefs.emplace_back(c);
    return c;
  }
};

}  // namespace axf::fluids
