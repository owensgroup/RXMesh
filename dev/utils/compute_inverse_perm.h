//
// Created by Behrooz on 2025-09-10.
//


#pragma once

#include <vector>
//Return the factor's nnz using CHOLMOD analysis
namespace RXMESH_SOLVER {
    void compute_inverse_perm(std::vector<int>& perm, std::vector<int>& inv_perm);
}


