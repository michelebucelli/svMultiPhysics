// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

//-------------------------------------------------------------------------
// Graduate [sic] minimum residual algorithm is implemented here for vector
// and scaler problems.
//-------------------------------------------------------------------------

#include "gmres.h"

#include "fsils_api.hpp"

#include "add_bc_mul.h"
#include "bcast.h"
#include "dot.h"
#include "norm.h"
#include "omp_la.h"
#include "spar_mul.h"

#include "Array3.h"
#include "DebugMsg.h"

#include <limits>
#include <math.h>

namespace gmres {

namespace {

void bc_pre(fsi_linear_solver::FSILS_lhsType& lhs, fsi_linear_solver::FSILS_subLsType& ls, const int dof,
    const int mynNo, const int nNo)
{
  int nsd = dof - 1;
  Array<double> v(nsd,nNo);

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto &face = lhs.face[faIn];
    if (face.coupledFlag) {
      if (face.sharedFlag) {
        v = 0.0;

        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < nsd; i++) {
            v(i,Ac) = face.valM(i,a);
          }
        }

        face.nS = pow(norm::fsi_ls_normv(nsd, mynNo, lhs.commu, v), 2.0);

      } else { 
        face.nS = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < nsd; i++) {
            face.nS = face.nS + pow(face.valM(i,a), 2.0);
          }
        }
      }
    }
  }
}

/// @brief Two local inner products sharing one Krylov basis vector, for one
/// unknown per node.
///
/// Both products stream q once, so the pair costs a single pass over it.
///
/// @param[in] nNo Number of nodes to include, i.e. the nodes owned by this
///   process.
/// @param[in] q Krylov basis vector, the operand common to both products.
/// @param[in] a First vector to multiply q with.
/// @param[in] b Second vector to multiply q with.
/// @param[out] qa Local part of the inner product of q and a.
/// @param[out] qb Local part of the inner product of q and b.
void nc_dot2_s(const int nNo, const Vector<double> &q, const Vector<double> &a,
               const Vector<double> &b, double &qa, double &qb) {
  double sum_a = 0.0;
  double sum_b = 0.0;

  for (int i = 0; i < nNo; i++) {
    sum_a = sum_a + q(i) * a(i);
    sum_b = sum_b + q(i) * b(i);
  }

  qa = sum_a;
  qb = sum_b;
}

/// @brief Two local inner products sharing one Krylov basis vector, for 'dof'
/// unknowns per node.
///
/// Both products stream q once, so the pair costs a single pass over it. The
/// small unknown counts are written out separately, as in dot.cpp, to keep the
/// node loop free of a nested trip count.
///
/// @param[in] dof Number of unknowns per node.
/// @param[in] nNo Number of nodes to include, i.e. the nodes owned by this
///   process.
/// @param[in] q Krylov basis vector, the operand common to both products.
/// @param[in] a First vector to multiply q with.
/// @param[in] b Second vector to multiply q with.
/// @param[out] qa Local part of the inner product of q and a.
/// @param[out] qb Local part of the inner product of q and b.
void nc_dot2_v(const int dof, const int nNo, const Array<double> &q,
               const Array<double> &a, const Array<double> &b, double &qa,
               double &qb) {
  double sum_a = 0.0;
  double sum_b = 0.0;

  switch (dof) {
    case 1: {
      for (int i = 0; i < nNo; i++) {
        sum_a = sum_a + q(0,i)*a(0,i);
        sum_b = sum_b + q(0,i)*b(0,i);
      }
    } break;

    case 2: {
      for (int i = 0; i < nNo; i++) {
        sum_a = sum_a + q(0,i)*a(0,i) + q(1,i)*a(1,i);
        sum_b = sum_b + q(0,i)*b(0,i) + q(1,i)*b(1,i);
      }
    } break;

    case 3: {
      for (int i = 0; i < nNo; i++) {
        sum_a = sum_a + q(0,i)*a(0,i) + q(1,i)*a(1,i) + q(2,i)*a(2,i);
        sum_b = sum_b + q(0,i)*b(0,i) + q(1,i)*b(1,i) + q(2,i)*b(2,i);
      }
    } break;

    case 4: {
      for (int i = 0; i < nNo; i++) {
        sum_a = sum_a + q(0,i)*a(0,i) + q(1,i)*a(1,i) + q(2,i)*a(2,i) + q(3,i)*a(3,i);
        sum_b = sum_b + q(0,i)*b(0,i) + q(1,i)*b(1,i) + q(2,i)*b(2,i) + q(3,i)*b(3,i);
      }
    } break;

    default: {
      for (int i = 0; i < nNo; i++) {
        for (int k = 0; k < dof; k++) {
          sum_a = sum_a + q(k,i)*a(k,i);
          sum_b = sum_b + q(k,i)*b(k,i);
        }
      }
    } break;
  }

  qa = sum_a;
  qb = sum_b;
}

/// @brief Norm of the newest Krylov vector before it was orthogonalized.
///
/// Recovered from column 'i' of the Hessenberg matrix: h(j,i) is the component
/// of the vector along the j-th orthonormal basis vector, and h(i+1,i) is the
/// part left orthogonal to all of them. Only squares are summed, so the result
/// is free of cancellation.
///
/// @param[in] h Hessenberg matrix, with column 'i' already filled.
/// @param[in] i Index of the current Arnoldi step.
///
/// @return The norm the vector had before orthogonalization.
double pre_orth_norm(const Array<double> &h, const int i) {
  double w_norm = h(i + 1, i) * h(i + 1, i);

  for (int j = 0; j <= i; j++) {
    w_norm = w_norm + h(j, i) * h(j, i);
  }

  return sqrt(w_norm);
}

/// @brief Orthogonalize the newest Krylov vector against the preceding ones,
/// for one unknown per node.
///
/// Fills column 'i' of the Hessenberg matrix and normalizes u(:,i+1) in place.
///
/// Modified Gram-Schmidt sweeps the basis one vector at a time, each inner
/// product taken against the partially updated vector, so its inner products
/// cannot be batched. The same arithmetic is produced here in one shot by
/// projecting on the whole basis at once and correcting the coefficients with
/// the measured non-orthogonality of the basis:
///
///     w <- w - Q y,   (I + L) y = Q^T w
///
/// where L holds the strictly lower triangular part of Q^T Q. A basis that is
/// orthonormal to working precision leaves L at zero and the coefficients are
/// those of classical Gram-Schmidt; L carries whatever orthogonality the basis
/// has actually lost, which is what reproduces the modified sweep. Q^T w is a
/// batch of independent inner products, so a step costs two reductions -- one
/// for those coefficients together with the newest row of L, one for the norm
/// of the projected vector -- instead of one per basis vector.
///
/// A subdiagonal entry that is negligible against the norm the vector had
/// before orthogonalization is a lucky breakdown: the Krylov space is invariant
/// and already contains the solution. In that case h(i+1,i) is set to zero and
/// the vector is left unnormalized, which drives err(i+1) to zero through the
/// Givens rotations the caller applies next.
///
/// @param[in] lhs FSILS left-hand side structure, used for its communicator.
/// @param[in] nNo Number of nodes stored on this process, ghost nodes included.
/// @param[in] mynNo Number of nodes owned by this process.
/// @param[in] i Index of the current Arnoldi step.
/// @param[in,out] u Krylov basis. Column i+1 holds the vector to orthogonalize
///   and is overwritten with the new orthonormal basis vector.
/// @param[in,out] h Hessenberg matrix. Column 'i' is overwritten.
/// @param[in,out] gram Strictly lower triangular part of Q^T Q for the basis
///   built in this restart cycle. Row 'i' is written, rows above it are read.
void orthogonalize_s(fsi_linear_solver::FSILS_lhsType &lhs, const int nNo,
                     const int mynNo, const int i, Array<double> &u,
                     Array<double> &h, Array<double> &gram) {
  auto w = u.rcol(i + 1);
  auto q_i = u.rcol(i);

  // Q^T w in the leading i+1 entries, the newest row of L in the rest, so that
  // the two travel in a single reduction.
  Vector<double> buffer(2 * i + 1);

  for (int j = 0; j < i; j++) {
    nc_dot2_s(mynNo, u.rcol(j), w, q_i, buffer(j), buffer(i + 1 + j));
  }
  buffer(i) = dot::fsils_nc_dot_s(mynNo, q_i, w);

  bcast::fsils_bcast_v(2 * i + 1, buffer, lhs.commu);

  for (int j = 0; j < i; j++) {
    gram(i, j) = buffer(i + 1 + j);
  }

  // Forward substitution for (I + L) y = Q^T w. The solution is written
  // straight into the Hessenberg column, so the entries already solved for act
  // as the running right-hand side.
  for (int j = 0; j <= i; j++) {
    h(j, i) = buffer(j);
    for (int m = 0; m < j; m++) {
      h(j, i) = h(j, i) - gram(j, m) * h(m, i);
    }
  }

  for (int j = 0; j <= i; j++) {
    omp_la::omp_sum_s(nNo, -h(j, i), w, u.rcol(j));
  }

  h(i + 1, i) = norm::fsi_ls_norms(mynNo, lhs.commu, w);

  if (h(i + 1, i) >
      std::numeric_limits<double>::epsilon() * pre_orth_norm(h, i)) {
    omp_la::omp_mul_s(nNo, 1.0 / h(i + 1, i), w);
  } else {
    h(i + 1, i) = 0.0;
  }
}

/// @brief Orthogonalize the newest Krylov vector against the preceding ones,
/// for 'dof' unknowns per node.
///
/// Fills column 'i' of the Hessenberg matrix and normalizes u(:,:,i+1) in
/// place. The scheme is the one described on orthogonalize_s(): the modified
/// Gram-Schmidt coefficients are obtained from a batched projection on the
/// whole basis, corrected by the measured non-orthogonality of that basis, at
/// two reductions per step.
///
/// A subdiagonal entry that is negligible against the norm the vector had
/// before orthogonalization is a lucky breakdown: the Krylov space is invariant
/// and already contains the solution. In that case h(i+1,i) is set to zero and
/// the vector is left unnormalized, which drives err(i+1) to zero through the
/// Givens rotations the caller applies next.
///
/// @param[in] lhs FSILS left-hand side structure, used for its communicator.
/// @param[in] dof Number of unknowns per node.
/// @param[in] nNo Number of nodes stored on this process, ghost nodes included.
/// @param[in] mynNo Number of nodes owned by this process.
/// @param[in] i Index of the current Arnoldi step.
/// @param[in,out] u Krylov basis. Slice i+1 holds the vector to orthogonalize
///   and is overwritten with the new orthonormal basis vector.
/// @param[in,out] h Hessenberg matrix. Column 'i' is overwritten.
/// @param[in,out] gram Strictly lower triangular part of Q^T Q for the basis
///   built in this restart cycle. Row 'i' is written, rows above it are read.
void orthogonalize_v(fsi_linear_solver::FSILS_lhsType &lhs, const int dof,
                     const int nNo, const int mynNo, const int i,
                     Array3<double> &u, Array<double> &h,
                     Array<double> &gram) {
  auto w = u.rslice(i + 1);
  auto q_i = u.rslice(i);

  // Q^T w in the leading i+1 entries, the newest row of L in the rest, so that
  // the two travel in a single reduction.
  Vector<double> buffer(2 * i + 1);

  for (int j = 0; j < i; j++) {
    nc_dot2_v(dof, mynNo, u.rslice(j), w, q_i, buffer(j), buffer(i + 1 + j));
  }
  buffer(i) = dot::fsils_nc_dot_v(dof, mynNo, q_i, w);

  bcast::fsils_bcast_v(2 * i + 1, buffer, lhs.commu);

  for (int j = 0; j < i; j++) {
    gram(i, j) = buffer(i + 1 + j);
  }

  // Forward substitution for (I + L) y = Q^T w. The solution is written
  // straight into the Hessenberg column, so the entries already solved for act
  // as the running right-hand side.
  for (int j = 0; j <= i; j++) {
    h(j, i) = buffer(j);
    for (int m = 0; m < j; m++) {
      h(j, i) = h(j, i) - gram(j, m) * h(m, i);
    }
  }

  for (int j = 0; j <= i; j++) {
    omp_la::omp_sum_v(dof, nNo, -h(j, i), w, u.rslice(j));
  }

  h(i + 1, i) = norm::fsi_ls_normv(dof, mynNo, lhs.commu, w);

  if (h(i + 1, i) >
      std::numeric_limits<double>::epsilon() * pre_orth_norm(h, i)) {
    omp_la::omp_mul_v(dof, nNo, 1.0 / h(i + 1, i), w);
  } else {
    h(i + 1, i) = 0.0;
  }
}

} // namespace

/// @brief Solver the system Val * X = R.
///
/// Reproduces the Fortran 'GMRES' subroutine.
//
void gmres(fsi_linear_solver::FSILS_lhsType &lhs,
           fsi_linear_solver::FSILS_subLsType &ls, const int dof,
           const Array<double> &Val, const Array<double> &R, Array<double> &X) {
#define n_debug_gmres
#ifdef debug_gmres
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  using namespace fsi_linear_solver;

  int nNo = lhs.nNo;
  int mynNo = lhs.mynNo;
  #ifdef debug_gmres
  dmsg << "dof: " << dof;
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  dmsg << "ls.sD: " << ls.sD;
  dmsg << "ls.mItr: " << ls.mItr;
  dmsg << "ls.absTol: " << ls.absTol;
  dmsg << "ls.relTol: " << ls.relTol;
  #endif

  Array<double> h(ls.sD+1,ls.sD), gram(ls.sD,ls.sD); 
  Array3<double> u(dof,nNo,ls.sD+1); 
  Array<double> unCondU(dof,nNo);
  Vector<double> y(ls.sD), c(ls.sD), s(ls.sD), err(ls.sD+1);

  double time = fsi_linear_solver::fsils_cpu_t();
  ls.success = false;
  double eps = 0.0;
  int last_i = 0;
  X = 0.0;

  for (int l = 0; l < ls.mItr; l++) {
    #ifdef debug_gmres
    dmsg;
    dmsg << "---------- l " << l+1 << " ----------";
    #endif

    if (l == 0) {
      u.set_slice(0, R);
    } else {
      auto u_slice = u.rslice(0);
      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof,  Val, X, u_slice);

      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, X, u_slice);

      ls.itr = ls.itr + 1;
      u.set_slice(0, R - u_slice);
    }

    for (auto& face : lhs.face) {
      if (face.coupledFlag) {
        auto u_slice = u.rslice(0);
        auto unCondU = u.rslice(0);
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU, u_slice);
        break; 
      }
    }

    err[0] = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u.rslice(0));
    #ifdef debug_gmres
    dmsg << "err(1): " << err[0];
    #endif
    if (err[0] == 0.0) { 
      throw std::runtime_error("FSILS: A zero matrix norm has been computed. This is probably caused by ill-posed boundary conditions.");
    }

    if (l == 0) {
      eps = err[0];
      ls.iNorm = eps;
      ls.fNorm = eps;
      eps = std::max(ls.absTol, ls.relTol*eps);
    }
    #ifdef debug_gmres
    dmsg << "eps: " << eps;
    #endif

    ls.dB = ls.fNorm;
    auto u_slice = u.rslice(0);
    u_slice = u_slice / err(0);

    for (int i = 0; i < ls.sD; i++) {
      #ifdef debug_gmres
      dmsg;
      dmsg << "----- i " << i+1 << " -----";
      #endif
      last_i = i;
      auto u_slice = u.rslice(i);
      auto u_slice_1 = u.rslice(i+1);
      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof,  Val, u_slice, u_slice_1);

      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, u_slice, u_slice_1);

      ls.itr = ls.itr + 1;

      for (auto& face : lhs.face) {
        if (face.coupledFlag) {
          auto u_slice_1 = u.rslice(i+1);
          auto unCondU = u.rslice(i+1);
          add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU, u_slice_1);
          break;
        }
      }

      orthogonalize_v(lhs, dof, nNo, mynNo, i, u, h, gram);

      for (int j = 0; j <= i-1; j++) {
        double tmp = c(j)*h(j,i) + s(j)*h(j+1,i);
        h(j+1,i) = -s(j)*h(j,i) + c(j)*h(j+1,i);
        h(j,i) = tmp;
      }

      double tmp = sqrt(h(i,i)*h(i,i) + h(i+1,i)*h(i+1,i));
      c(i) = h(i,i) / tmp;
      s(i) = h(i+1,i) / tmp;
      h(i,i) = tmp;
      h(i+1,i) = 0.0;
      err(i+1) = -s(i)*err(i);
      err(i) = c(i)*err(i);
      #ifdef debug_gmres
      dmsg;
      dmsg << "tmp: " << tmp;
      dmsg << "err(i): " << err(i);
      dmsg << "err(i+1): " << err(i+1);
      dmsg << "eps: " << eps;
      #endif

      if (fabs(err(i+1)) < eps) {
        ls.success = true;
        break;
      }
    } // for int i = 0; i < ls.sD

    if (last_i >= ls.sD) {
      last_i = ls.sD - 1;
    }

    for (int i = 0; i <= last_i; i++) {
      y(i) = err(i);
    }

    for (int j = last_i; j >= 0; j--) { 
      for (int k = j+1; k <= last_i; k++) {
        y(j) = y(j) - h(j,k)*y(k);
      }
      y(j) = y(j) / h(j,j);
    }

    for (int j = 0; j <= last_i; j++) {
      omp_la::omp_sum_v(dof, nNo, y(j), X, u.rslice(j));
    }

    ls.fNorm = fabs(err(last_i+1));
    if (ls.success) {
      break;
    }

  } // for l = 0; l < ls.mItr

  ls.callD = fsi_linear_solver::fsils_cpu_t() - time + ls.callD;
  ls.dB  = 10.0 * log(ls.fNorm / ls.dB);

  #ifdef debug_gmres
  dmsg << "Done";
  #endif
}

//---------
// gmres_s
//---------
// Reproduces the Fortran 'GMRESS' subroutine.
//
void gmres_s(fsi_linear_solver::FSILS_lhsType& lhs, fsi_linear_solver::FSILS_subLsType& ls, const int dof,
    const Vector<double>& Val, Vector<double>& R)
{
  #define n_debug_gmres_s
  #ifdef debug_gmres_s
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  #endif

  using namespace fsi_linear_solver;

  bool flag = false;
  int nNo = lhs.nNo;
  int mynNo = lhs.mynNo;
  #ifdef debug_gmres_s
  dmsg << "dof: " << dof;
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  dmsg << "ls.sD: " << ls.sD;
  dmsg << "ls.mItr: " << ls.mItr;
  dmsg << "ls.absTol: " << ls.absTol;
  dmsg << "ls.relTol: " << ls.relTol;
  #endif

  Array<double> h(ls.sD+1,ls.sD), gram(ls.sD,ls.sD);
  Array<double> u(nNo,ls.sD+1);
  Vector<double> X(nNo), y(ls.sD), c(ls.sD), s(ls.sD), err(ls.sD+1);

  ls.callD = fsi_linear_solver::fsils_cpu_t();
  ls.success = false;
  double eps = norm::fsi_ls_norms(mynNo, lhs.commu, R);
  ls.iNorm = eps;
  ls.fNorm = eps;
  eps = std::max(ls.absTol, ls.relTol*eps);
  ls.itr = 0;
  int last_i = 0;
  #ifdef debug_gmres_s
  dmsg << "ls.iNorm: " << ls.iNorm;
  dmsg << "eps: " << eps;
  #endif

  if (ls.iNorm <= ls.absTol) {
    ls.callD = std::numeric_limits<double>::epsilon();
    ls.dB = 0.0;
    ls.success = true;
    return; 
  }

  for (int l = 0; l < ls.mItr; l++) {
    #ifdef debug_gmres_s
    dmsg;
    dmsg << "======== l " << l+1 << " ======== ";
    #endif
    ls.dB = ls.fNorm;
    ls.itr = ls.itr + 1;
    auto u_col = u.col(0);
    spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, Val, X, u_col);
    u.set_col(0, R - u_col);

    err[0] = norm::fsi_ls_norms(mynNo, lhs.commu, u.col(0));
    if (err[0] == 0.0) { 
      throw std::runtime_error("FSILS: A zero matrix norm has been computed. This is probably caused by ill-posed boundary conditions.");
    }

    u_col = u.col(0) / err[0];
    u.set_col(0, u_col);
    #ifdef debug_gmres_s
    dmsg << "err(1): " << err[0];
    #endif


    for (int i = 0; i < ls.sD; i++) {
      #ifdef debug_gmres_s
      dmsg;
      dmsg << "----- i " << i+1 << " ----- ";
      #endif
      ls.itr = ls.itr + 1;
      last_i = i;
      auto u_col = u.col(i);
      auto u_col_1 = u.col(i+1);
      spar_mul::fsils_spar_mul_ss(lhs, lhs.rowPtr, lhs.colPtr, Val, u_col, u_col_1);
      u.set_col(i+1, u_col_1);

      orthogonalize_s(lhs, nNo, mynNo, i, u, h, gram);

      for (int j = 0; j <= i-1; j++) {
        double tmp = c(j)*h(j,i) + s(j)*h(j+1,i);
        h(j+1,i) = -s(j)*h(j,i) + c(j)*h(j+1,i);
        h(j,i) = tmp;
      }

      double tmp = sqrt(h(i,i)*h(i,i) + h(i+1,i)*h(i+1,i));
      c(i) = h(i,i) / tmp;
      s(i) = h(i+1,i) / tmp;
      h(i,i) = tmp;
      h(i+1,i) = 0.0;
      err(i+1) = -s(i)*err(i);
      err(i) = c(i)*err(i);
      #ifdef debug_gmres_s
      dmsg << "err(i+1): " << err(i+1);
      dmsg << "tmp: " << tmp;
      #endif

      if (fabs(err(i+1)) < eps) {
        ls.success = true;
        break;
      }
    } // for int i = 0; i < ls.sD

    if (last_i >= ls.sD) {
      last_i = ls.sD - 1;
    }

    for (int i = 0; i <= last_i; i++) {
      y(i) = err(i);
    }

    for (int j = last_i; j >= 0; j--) { 
      for (int k = j+1; k <= last_i; k++) {
        y(j) = y(j) - h(j,k)*y(k);
      }
      y(j) = y(j) / h(j,j);
    }

    for (int j = 0; j <= last_i; j++) {
      omp_la::omp_sum_s(nNo, y(j), X, u.col(j));
    }

    ls.fNorm = fabs(err(last_i+1));
    if (ls.success) {
      break;
    }
  }

  R = X;
  ls.callD = fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB  = 10.0 * log(ls.fNorm / ls.dB);
}

//---------
// gmres_v
//---------
// Generalized minimum residual algorithm implemented vector problems.
//
// The Array3::rslice() method is used to create an Array object with 
// data directly referenced to the Array3 data. This eliminates the overhead 
// of copying data to and from an Array3 object.
//
// Reproduces the Fortran 'GMRESV' subroutine.
//
void gmres_v(fsi_linear_solver::FSILS_lhsType& lhs, fsi_linear_solver::FSILS_subLsType& ls, const int dof,
    const Array<double>& Val, Array<double>& R)
{
  using namespace fsi_linear_solver;

  #define n_debug_gmres_v
  #ifdef debug_gmres_v
  DebugMsg dmsg(__func__,  lhs.commu.task);
  dmsg.banner();
  double time = fsi_linear_solver::fsils_cpu_t();
  #endif

  bool flag = false;
  int nNo = lhs.nNo;
  int mynNo = lhs.mynNo;
  #ifdef debug_gmres_v
  dmsg << "dof: " << dof;
  dmsg << "nNo: " << nNo;
  dmsg << "mynNo: " << mynNo;
  dmsg << "ls.sD: " << ls.sD;
  dmsg << "ls.mItr: " << ls.mItr;
  dmsg << "ls.absTol: " << ls.absTol;
  dmsg << "ls.relTol: " << ls.relTol;
  #endif

  Array<double> h(ls.sD+1,ls.sD), gram(ls.sD,ls.sD), X(dof,nNo);
  Array3<double> u(dof,nNo,ls.sD+1);
  Array<double> unCondU(dof,nNo);
  Vector<double> y(ls.sD), c(ls.sD), s(ls.sD), err(ls.sD+1);

  ls.callD = fsi_linear_solver::fsils_cpu_t();
  ls.success = false;
  double eps = norm::fsi_ls_normv(dof, mynNo, lhs.commu, R);
  ls.iNorm = eps;
  ls.fNorm = eps;
  eps = std::max(ls.absTol, ls.relTol*eps);
  ls.itr = 0;
  int last_i = 0;
  #ifdef debug_gmres_v
  dmsg << "ls.iNorm: " << ls.iNorm;
  dmsg << "eps: " << eps;
  #endif

  bc_pre(lhs, ls, dof, mynNo, nNo);

  if (ls.iNorm <= ls.absTol) {
    ls.callD = std::numeric_limits<double>::epsilon();
    ls.dB = 0.0;
    ls.success = true;
    return; 
  }

  for (int l = 0; l < ls.mItr; l++) {
    #ifdef debug_gmres_v
    dmsg << "===== l " << l+1 << " ===== ";
    #endif
    ls.dB = ls.fNorm;
    ls.itr = ls.itr + 1;
    auto u_slice = u.rslice(0);
    spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof,  Val, X, u_slice);

    add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, X, u_slice);

    u_slice = R - u_slice;

    for (auto& face : lhs.face) {
      if (face.coupledFlag && flag) {
        auto u_slice = u.rslice(0);
        auto unCondU = u.rslice(0);
        add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU, u_slice);
        break;
      }
    }

    err[0] = norm::fsi_ls_normv(dof, mynNo, lhs.commu, u.rslice(0));
    if (err[0] == 0.0) { 
      throw std::runtime_error("FSILS: A zero matrix norm has been computed. This is probably caused by ill-posed boundary conditions.");
    }

    u_slice = u.rslice(0) / err[0];
    #ifdef debug_gmres_v
    dmsg << "err(1): " << err[0];
    #endif

    for (int i = 0; i < ls.sD; i++) {
      #ifdef debug_gmres_v
      dmsg << "----- i " << i+1 << " ----- ";
      #endif
      ls.itr = ls.itr + 1;
      last_i = i;
      auto u_slice = u.rslice(i);
      auto u_slice_1 = u.rslice(i+1);
      spar_mul::fsils_spar_mul_vv(lhs, lhs.rowPtr, lhs.colPtr, dof,  Val, u_slice, u_slice_1);

      add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_ADD, dof, u_slice, u_slice_1);

      for (auto& face : lhs.face) {
        if (face.coupledFlag && flag) {
          auto u_slice_1 = u.rslice(i+1);
          auto unCondU = u.rslice(i+1);
          add_bc_mul::add_bc_mul(lhs, BcopType::BCOP_TYPE_PRE, dof, unCondU, u_slice_1);
          break;
        }
      }

      orthogonalize_v(lhs, dof, nNo, mynNo, i, u, h, gram);

      for (int j = 0; j <= i-1; j++) {
        double tmp = c(j)*h(j,i) + s(j)*h(j+1,i);
        h(j+1,i) = -s(j)*h(j,i) + c(j)*h(j+1,i);
        h(j,i) = tmp;
      }

      double tmp = sqrt(h(i,i)*h(i,i) + h(i+1,i)*h(i+1,i));
      c(i) = h(i,i) / tmp;
      s(i) = h(i+1,i) / tmp;
      h(i,i) = tmp;
      h(i+1,i) = 0.0;
      err(i+1) = -s(i)*err(i);
      err(i) = c(i)*err(i);
      #ifdef debug_gmres_v
      dmsg << "err(i+1): " << err(i+1);
      dmsg << "tmp: " << tmp;
      #endif

      if (fabs(err(i+1)) < eps) {
        ls.success = true;
        break;
      }
    } // for int i = 0; i < ls.sD

    if (last_i >= ls.sD) {
      last_i = ls.sD - 1;
    }

    for (int i = 0; i <= last_i; i++) {
      y(i) = err(i);
    }

    for (int j = last_i; j >= 0; j--) { 
      for (int k = j+1; k <= last_i; k++) {
        y(j) = y(j) - h(j,k)*y(k);
      }
      y(j) = y(j) / h(j,j);
    }

    for (int j = 0; j <= last_i; j++) {
      omp_la::omp_sum_v(dof, nNo, y(j), X, u.rslice(j));
    }

    ls.fNorm = fabs(err(last_i+1));
    if (ls.success) {
      break;
    }
  }

  R = X;
  ls.callD = fsi_linear_solver::fsils_cpu_t() - ls.callD;
  ls.dB  = 10.0 * log(ls.fNorm / ls.dB);

  #ifdef debug_gmres_v
  double exec_time = fsi_linear_solver::fsils_cpu_t() - time;
  dmsg << "Execution time: " << exec_time;
  dmsg << "Done";
  #endif
}

};


