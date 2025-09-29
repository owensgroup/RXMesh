//
// Created by Behrooz on 2025-09-10.
//

#include "get_factor_nnz.h"
#include <cassert>
#include <iostream>



namespace RXMESH_SOLVER {
//Return the factor's nnz using CHOLMOD analysis. The input matrix should be CSC with only lower part represented.
int get_factor_nnz(int* Ap, int* Ai, double* Ax, int N, int NNZ, std::vector<int>& perm)
{
    int factor_NNZ = -1;
    cholmod_common cm;
    cholmod_start(&cm);
    cholmod_sparse *A = nullptr;
    cholmod_factor *L = nullptr;
    void *Ai_tmp, *Ap_tmp, *Ax_tmp;
    
    try {
        A = cholmod_allocate_sparse(N, N, NNZ, true, true, -1, CHOLMOD_REAL, &cm);
        if (A == nullptr) {
            std::cerr << "ERROR: Failed to allocate CHOLMOD sparse matrix" << std::endl;
            cholmod_finish(&cm);
            return -1;
        }
        
        // Save original pointers for cleanup
        Ap_tmp = A->p;
        Ax_tmp = A->x;
        Ai_tmp = A->i;

        // Set matrix data
        A->p = Ap;
        A->i = Ai;
        A->x = Ax;

        // Configure CHOLMOD for analysis
        cm.nmethods = 1;
        cm.supernodal = CHOLMOD_SUPERNODAL;
        cm.method[0].ordering = CHOLMOD_GIVEN;
        
        assert(perm.size() == N);
        L = cholmod_analyze_p(A, perm.data(), NULL, 0, &cm);
        if (L == nullptr) {
            std::cerr << "ERROR: CHOLMOD symbolic factorization failed" << std::endl;
            factor_NNZ = -1;
        } else {
            factor_NNZ = cm.lnz * 2 - N;
        }

        // Cleanup - restore original pointers before freeing
        A->i = Ai_tmp;
        A->p = Ap_tmp;
        A->x = Ax_tmp;
        cholmod_free_sparse(&A, &cm);

        if (L != nullptr) {
            cholmod_free_factor(&L, &cm);
        }
    } catch (...) {
        std::cerr << "ERROR: Exception caught in get_factor_nnz" << std::endl;
        factor_NNZ = -1;
    }
    
    cholmod_finish(&cm);
    return factor_NNZ;
}
}
