//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef USE_STRUMPACK

#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>
#include "STRUMPACKSolver.hpp"

#include <spdlog/spdlog.h>
#include "omp.h"



namespace RXMESH_SOLVER {
STRUMPACKSolver::~STRUMPACKSolver()
{

}

STRUMPACKSolver::STRUMPACKSolver()
{
    spss.options().set_rel_tol(1e-14);

    /** options for the inner solver */

}


void STRUMPACKSolver::setMatrix(int* p, int* i, double* x, int A_N, int NNZ)
{
    _A = strumpack::CSRMatrix<double,int>(A_N, p, i, x);
    spss.options().enable_symmetric();
    spss.options().enable_positive_definite();

    spss.solver().options().enable_symmetric();
    spss.solver().options().enable_positive_definite();

    spss.solver().options().set_Krylov_solver(strumpack::KrylovSolver::DIRECT);
    spss.options().set_matching(strumpack::MatchingJob::NONE);
    spss.solver().options().set_matching(strumpack::MatchingJob::NONE);

    spss.options().set_reordering_method(strumpack::ReorderingStrategy::METIS);
    spss.options().enable_gpu();
    spss.set_matrix(_A);
}

void STRUMPACKSolver::innerAnalyze_pattern(std::vector<int>& user_defined_perm)
{
    strumpack::ReturnCode code = spss.reorder();
    if (code != strumpack::ReturnCode::SUCCESS) {
        spdlog::error("Reordering faces a problem");
    }
}

void STRUMPACKSolver::innerFactorize(void)
{
    strumpack::ReturnCode return_code = spss.factor();
    if (return_code != strumpack::ReturnCode::SUCCESS) {
        if (return_code == strumpack::ReturnCode::MATRIX_NOT_SET) {
            spdlog::error("The matrix is not set correctly.");
        }
        spdlog::error("Factorization faces a problem");
    }

}

void STRUMPACKSolver::innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
{
    _rhs = strumpack::DenseMatrix<double>(rhs.rows(), 1);
    for (int i = 0; i < rhs.rows(); i++) {
        _rhs(i,0) = rhs[i];
    }
    _x = strumpack::DenseMatrix<double>(rhs.rows(), 1);

    spss.solve(_rhs, _x);
    result.resize(rhs.rows());
    memcpy(result.data(), _x.data(), result.rows() * sizeof(double));
}


void STRUMPACKSolver::resetSolver()
{
}

LinSysSolverType STRUMPACKSolver::type() const
{
    return LinSysSolverType::GPU_STRUMPACK;
};

int STRUMPACKSolver::getFactorNNZ()
{
    return spss.solver().factor_nonzeros();
}

}  // namespace RXMESH_SOLVER

#endif