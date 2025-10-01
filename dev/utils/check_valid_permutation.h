//
// Created by Behrooz on 2025-09-10.
//


#pragma once

#include <cholmod.h>
#include <vector>
//Return the factor's nnz using CHOLMOD analysis
namespace RXMESH_SOLVER {
    bool check_valid_permutation(int* perm, int n);
}

