//
// Created by Behrooz on 2025-09-10.
//

#include "compute_inverse_perm.h"
#include <cassert>
#include <iostream>



namespace RXMESH_SOLVER {
//Return the factor's nnz using CHOLMOD analysis. The input matrix should be CSC with only lower part represented.
void compute_inverse_perm(std::vector<int>& perm, std::vector<int>& inv_perm)
{
    assert(!perm.empty());
    inv_perm.clear();
    inv_perm.resize(perm.size(), -1);
    for (int j = 0; j < perm.size(); j++)
    {
        inv_perm[perm[j]] = j;
    }
}
}
