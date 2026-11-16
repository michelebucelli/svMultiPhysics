// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "add_bc_mul.h"

#include "dot.h"

namespace add_bc_mul {

/// @brief The contribution of coupled BCs is added to the matrix-vector
/// product operation. Depending on the type of operation (adding the
/// contribution or computing the PC contribution) different
/// coefficients are used.
///
/// For reference, see 
/// Moghadam et al. 2013 eq. 27 (https://doi.org/10.1016/j.jcp.2012.07.035) and
/// Moghadam et al. 2013b (https://doi.org/10.1007/s00466-013-0868-1).
///
/// Reproduces code in ADDBCMUL.f.
///
/// The operator of a coupled face is K^BC = res * v * u^T. The left factor
/// v = face.valM is the integral of the shape functions times the normal over
/// the face. The right factor u is the derivative of the flux with respect to
/// the unknown, divided by res: it is v, plus face.cap_valM if the face has a
/// cap, plus face.shape_valM if the flux varies with the face geometry.
///
/// @param[in] lhs The left-hand side of the linear system. 0D resistance is
///   stored in the face(i).res field.
/// @param[in] op_Type The type of operation (addition or PC contribution).
/// @param[in] dof The number of degrees of freedom.
/// @param[in] X The input vector.
/// @param[in,out] Y The current matrix-vector product (Y = K*X), to which
///   K^BC * X is added. The coefficient that replaces res is different if
///   preconditioning.
void add_bc_mul(FSILS_lhsType& lhs, const BcopType op_Type, const int dof, const Array<double>& X, Array<double>& Y)
{
  Vector<double> coef(lhs.nFaces); 
  Array<double> v(dof,lhs.nNo);

  if (op_Type == BcopType::BCOP_TYPE_ADD) {
    for (int i = 0; i < lhs.nFaces; i++) {
      coef(i) = lhs.face[i].res;
    }
  } else if (op_Type == BcopType::BCOP_TYPE_PRE) {
    for (int i = 0; i < lhs.nFaces; i++) {
      coef(i) = -lhs.face[i].res / (1.0 + (lhs.face[i].res*lhs.face[i].nS));
    }
  } else { 
    //PRINT *, "FSILS: op_Type is not defined"
    //STOP "FSILS: FATAL ERROR"
  }

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];
    int nsd = std::min(face.dof, dof);

    if (face.coupledFlag) {
      // If face is shared across procs
      if (face.sharedFlag) {
        v = 0.0;
        // Setting vector v = int{N_A n_i} dGamma, plus the variation of the
        // flux with the face geometry: the right factor of the operator
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < nsd; i++) {
            v(i,Ac) = face.valM(i,a);
            if (face.has_shape_derivative) {
              v(i,Ac) += face.shape_valM(i,a);
            }
          }
        }
        // Computing S = coef * v^T * X
        double S = coef(faIn) * dot::fsils_dot_v(dof, lhs.mynNo, lhs.commu, v, X);
        
        // Add cap surface contribution to S
        // Cap surfaces contribute to flow rate but not pressure
        if (face.has_cap) {
          // Compute sparse cap contribution directly over cap nodes.
          // This avoids forming a full dof x nNo temporary array.
          double Scap = 0.0;
          for (int a = 0; a < face.cap_glob.size(); a++) {
            int Ac = face.cap_glob(a);
            if (Ac < 0) continue;  // cap node not on this rank
            for (int i = 0; i < nsd; i++) {
              Scap += face.cap_valM(i,a) * X(i,Ac);
            }
          }
          if (lhs.commu.nTasks > 1) {
            double tmp = 0.0;
            MPI_Allreduce(&Scap, &tmp, 1, cm_mod::mpreal, MPI_SUM, lhs.commu.comm);
            Scap = tmp;
          }
          S = S + coef(faIn) * Scap;
        }
        // Computing Y = Y + int{N_A n_i} dGamma * S
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < nsd; i++) {
            Y(i,Ac) = Y(i,Ac) + face.valM(i,a)*S;
          }
        }

      } 
      // If face is not shared across procs
      else  {
        // Computing S = coef * v^T * X
        double S = 0.0;
        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < nsd; i++) {
            S = S + face.valM(i,a)*X(i,Ac);
            if (face.has_shape_derivative) {
              S = S + face.shape_valM(i,a)*X(i,Ac);
            }
          }
        }
        
        // If capping surface is present add its contribution to S
        if (face.has_cap) {
          for (int a = 0; a < face.cap_glob.size(); a++) {
            int Ac = face.cap_glob(a);
            if (Ac < 0) continue;  // cap node not on this rank
            for (int i = 0; i < nsd; i++) {
              S = S + face.cap_valM(i,a)*X(i,Ac);
            }
          }
        }
        S = coef(faIn) * S;

        for (int a = 0; a < face.nNo; a++) {
          int Ac = face.glob(a);
          for (int i = 0; i < nsd; i++) {
            Y(i,Ac) = Y(i,Ac) + face.valM(i,a)*S;
          }
        }
      }
    }
  }

}

};
