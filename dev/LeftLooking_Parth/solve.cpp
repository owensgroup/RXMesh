//
// Created by behrooz on 25/10/22.
//

#include <iostream>
#include "parth_solver.h"

namespace PARTH {

void ParthSolverAPI::solve(std::vector<double>& rhs, std::vector<double>& result){
  if (!chol_b) {
    chol_b = cholmod_allocate_dense(A_n, 1, A_n, CHOLMOD_REAL, &A_cm);
    chol_b_tmp = chol_b->x;
  }
  chol_b->x = rhs.data();

  if (chol_sol) {
    cholmod_free_dense(&chol_sol, &A_cm);
  }

  chol_sol = cholmod_solve(CHOLMOD_A, chol_L, chol_b, &A_cm);

  result.resize(rhs.size());
  memcpy(result.data(), chol_sol->x, result.size() * sizeof(result[0]));
}

} // namespace PARTH