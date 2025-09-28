//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef RXMESH_WITH_SUITESPARSE

#include "CUDSSSolver.hpp"
#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>
#include "omp.h"

namespace RXMESH_SOLVER {
CUDSSSolver::~CUDSSSolver()
{
    cudssCreate(&handle);
    cudssConfigCreate(&config);
    cudssDataCreate(handle, &data);
}

CUDSSSolver::CUDSSSolver()
{
}


void CUDSSSolver::setMatrix(int*              p,
                              int*              i,
                              double*           x,
                              int               A_N,
                              int               NNZ)
{
    // cudssMatrixCreateCsr(&A, ... rowOffsets, colIndices, values, ...);
    // cudssMatrixCreateDn(&b, ... bvalues, ...);
    // cudssMatrixCreateDn(&x, ... xvalues, ...);
}

void CUDSSSolver::innerAnalyze_pattern(std::vector<int>& user_defined_perm)
{

}

void CUDSSSolver::innerFactorize(void)
{

}

void CUDSSSolver::innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
{
}


void CUDSSSolver::resetSolver()
{

}

LinSysSolverType CUDSSSolver::type() const
{
    return LinSysSolverType::GPU_CUDSS;
};
}  // namespace RXMESH_SOLVER

#endif
