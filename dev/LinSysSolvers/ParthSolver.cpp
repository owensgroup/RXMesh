//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef USE_SUITESPARSE

#include "ParthSolver.hpp"
#include <cassert>
#include <iostream>
#include "omp.h"
#include <spdlog/spdlog.h>

namespace RXMESH_SOLVER {
ParthSolver::~ParthSolver(){}

ParthSolver::ParthSolver(){}

void ParthSolver::setMatrix(int*              p,
                              int*              i,
                              double*           x,
                              int               A_N,
                              int               NNZ)
{
    solver.setMatrix(p, i, x, A_N, NNZ);
    this->N   = A_N;
    this->NNZ = NNZ;
}

void ParthSolver::innerAnalyze_pattern(std::vector<int>& user_defined_perm)
{
  solver.analyze(user_defined_perm);
}

void ParthSolver::innerFactorize(void)
{
    solver.factorize();
}

void ParthSolver::innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
{
    solver.solve(rhs, result);
}


void ParthSolver::resetSolver()
{
    cholmod_clean_memory();

    A       = NULL;
    L       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
    bx           = NULL;
}

void ParthSolver::save_factor(
    const std::string &filePath) {
    cholmod_sparse *spm = cholmod_factor_to_sparse(solver.chol_L, &solver.A_cm);

    FILE *out = fopen(filePath.c_str(), "w");
    assert(out);

    cholmod_write_sparse(out, spm, NULL, "", &cm);

    fclose(out);
}

LinSysSolverType ParthSolver::type() const
{
    return LinSysSolverType::PARTH_SOLVER;
};
}  // namespace RXMESH_SOLVER

#endif
