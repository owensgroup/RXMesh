//
// Created by Behrooz on 2025-09-10.
//


#pragma once
#include <cholmod.h>
#include <vector>
//Return the factor's nnz using CHOLMOD analysis
namespace RXMESH_SOLVER {
    int get_factor_nnz(int* Ap, int* Ai, double* Ax, int N, int NNZ, std::vector<int>& perm);
}


