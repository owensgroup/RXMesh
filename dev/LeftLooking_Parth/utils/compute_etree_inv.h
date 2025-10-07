//
// Created by Behrooz on 2025-09-10.
//


#pragma once

#include <cholmod.h>
#include <vector>
//Return the factor's nnz using CHOLMOD analysis
namespace PARTH_SOLVER {
    void compute_etree_inv(
        int n, ///<[in] number of nodes inside the elimination tree
        const int *parent_vector,   ///<[in] parent vector of the tree
        std::vector<int> &tree_ptr, ///<[out] pointer array in CSC format
        std::vector<int> &tree_set  ///<[out] index array in CSC format
    );
}

