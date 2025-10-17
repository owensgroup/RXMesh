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
    L_NNZ = solver.A_cm.lnz * 2 - N;
}

void ParthSolver::innerFactorize(void)
{
    solver.factorize();
}

void ParthSolver::innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
{
    std::vector<double> rhs_std(rhs.data(), rhs.data() + rhs.size());
    std::vector<double> result_std;
    solver.solve(rhs_std, result_std);
    result = Eigen::Map<Eigen::VectorXd>(result_std.data(), result_std.size());
}


void ParthSolver::resetSolver()
{
}

void ParthSolver::save_factor(
    const std::string &filePath) {
    cholmod_sparse *spm = cholmod_factor_to_sparse(solver.chol_L, &solver.A_cm);

    FILE *out = fopen(filePath.c_str(), "w");
    assert(out);

    cholmod_write_sparse(out, spm, NULL, "", &solver.A_cm);

    fclose(out);
}

LinSysSolverType ParthSolver::type() const
{
    return LinSysSolverType::PARTH_SOLVER;
};
}  // namespace RXMESH_SOLVER

#endif
