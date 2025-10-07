//
// Created by behrooz on 28/09/22.
//

#include <algorithm>
#include <functional>
#include <iostream>
#include "csv_utils.h"
#include "parth_solver.h"

namespace PARTH {

bool ParthSolverAPI::factorize(void) {
    cholmod_factorize(chol_A, chol_L, &A_cm);
    return A_cm.status != CHOLMOD_NOT_POSDEF;
}

} // namespace PARTH
