// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "fils_struct.hpp"

namespace bcast {

using namespace fsi_linear_solver;

void fsils_bcast(double& u, FSILS_commuType& commu);

/**
 * @brief Sum the leading entries of a vector over all processors.
 *
 * The reduction is in place, so the size of 'u' is left as it is and a vector
 * longer than 'n' may be passed; the entries past 'n' are untouched.
 *
 * @param[in] n Number of leading entries to reduce.
 * @param[in,out] u Vector holding this processor's contribution, overwritten
 *   with the sum over all processors.
 * @param[in] commu FSILS communicator structure.
 */
void fsils_bcast_v(const int n, Vector<double>& u, FSILS_commuType& commu);

};
