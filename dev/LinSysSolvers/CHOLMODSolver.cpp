//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef USE_SUITESPARSE

#include "CHOLMODSolver.hpp"
#include <cassert>
#include <iostream>
#include "omp.h"
#include <spdlog/spdlog.h>

namespace RXMESH_SOLVER {
CHOLMODSolver::~CHOLMODSolver()
{
    if (A) {
        A->i = Ai;
        A->p = Ap;
        A->x = Ax;
        cholmod_free_sparse(&A, &cm);
    }

    cholmod_free_factor(&L, &cm);

    if (b) {
        b->x = bx;
        cholmod_free_dense(&b, &cm);
    }

    if (x_solve) {
        cholmod_free_dense(&x_solve, &cm);
    }

    cholmod_finish(&cm);
}

CHOLMODSolver::CHOLMODSolver()
{
    cholmod_start(&cm);
    bx      = NULL;
    A       = NULL;
    L       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
}

void CHOLMODSolver::cholmod_clean_memory()
{
    if (A) {
        A->i = Ai;
        A->p = Ap;
        A->x = Ax;
        cholmod_free_sparse(&A, &cm);
    }

    if (b) {
        b->x = bx;
        cholmod_free_dense(&b, &cm);
    }

    if (x_solve) {
        cholmod_free_dense(&x_solve, &cm);
    }

    A       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
}

void CHOLMODSolver::setMatrix(int*              p,
                              int*              i,
                              double*           x,
                              int               A_N,
                              int               NNZ)
{
    assert(p[A_N] == NNZ);
    this->N   = A_N;
    this->NNZ = NNZ;

    this->cholmod_clean_memory();

    if (!A) {
        A = cholmod_allocate_sparse(
            N, N, NNZ, true, true, -1, CHOLMOD_REAL, &cm);
        this->Ap = A->p;
        this->Ax = A->x;
        this->Ai = A->i;
        // -1: upper right part will be ignored during computation
    }

    A->p = p;
    A->i = i;
    A->x = x;
}

void CHOLMODSolver::innerAnalyze_pattern(std::vector<int>& user_defined_perm)
{
    cholmod_free_factor(&L, &cm);

    cm.supernodal = CHOLMOD_SUPERNODAL;
    if (user_defined_perm.size() == N) {
        spdlog::info("Using user provided permutation.");
        cm.nmethods           = 1;
        cm.method[0].ordering = CHOLMOD_GIVEN;
        L                     = cholmod_analyze_p(A, user_defined_perm.data(), NULL, 0, &cm);
    } else {
        spdlog::info("Using METIS permutation.");
        cm.nmethods           = 1;
        cm.method[0].ordering = CHOLMOD_METIS;
        L                     = cholmod_analyze(A, &cm);
    }
    assert(L != nullptr);
    if (L == nullptr) {
        std::cerr << "ERROR during symbolic factorization:" << std::endl;
    }
    L_NNZ = cm.lnz * 2 - N;
}

void CHOLMODSolver::innerFactorize(void)
{
    cholmod_factorize(A, L, &cm);
    if (cm.status == CHOLMOD_NOT_POSDEF) {
        std::cerr << "ERROR during numerical factorization - code: "
                  << std::endl;
    }
}

void CHOLMODSolver::innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
{
    if (!b) {
        b  = cholmod_allocate_dense(N, 1, N, CHOLMOD_REAL, &cm);
        bx = b->x;
    }
    b->x = rhs.data();

    if (x_solve) {
        cholmod_free_dense(&x_solve, &cm);
    }

    x_solve = cholmod_solve(CHOLMOD_A, L, b, &cm);

    result.conservativeResize(rhs.size());
    memcpy(result.data(), x_solve->x, result.rows() * sizeof(result[0]));
}


void CHOLMODSolver::resetSolver()
{
    cholmod_clean_memory();

    A       = NULL;
    L       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
    bx           = NULL;
}

LinSysSolverType CHOLMODSolver::type() const
{
    return LinSysSolverType::CPU_CHOLMOD;
};
}  // namespace RXMESH_SOLVER

#endif
