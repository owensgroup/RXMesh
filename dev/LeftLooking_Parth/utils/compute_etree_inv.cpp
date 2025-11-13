//
// Created by Behrooz on 2025-09-10.
//

#include "compute_etree_inv.h"
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

namespace PARTH_SOLVER {
//Return the factor's nnz using CHOLMOD analysis. The input matrix should be CSC with only lower part represented.
void compute_etree_inv(
    int n, ///<[in] number of nodes inside the elimination tree
    const int *parent_vector,   ///<[in] parent vector of the tree
    std::vector<int> &tree_ptr, ///<[out] pointer array in CSC format
    std::vector<int> &tree_set  ///<[out] index array in CSC format
){
    // Creating the inverse elemination tree
    std::vector<int> number_of_child(n, 0);
    for (int i = 0; i < n; i++) {
        if (parent_vector[i] != -1) {
            number_of_child[parent_vector[i]]++;
        }
    }

    tree_ptr.clear();
    tree_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; i++) {
        tree_ptr[i + 1] = number_of_child[i] + tree_ptr[i];
    }

    std::vector<int> child_cnt(n, 0);
    tree_set.resize(tree_ptr[n]);
    for (int s = 0; s < n; s++) {
        int parent = parent_vector[s];
        if (parent != -1) {
            int start_idx = tree_ptr[parent];
            int start_child_idx = child_cnt[parent];
            tree_set[start_idx + start_child_idx] = s;
            child_cnt[parent]++;
        }
    }

#ifndef NDEBUG
    for (int s = 0; s < n; s++) {
        assert(child_cnt[s] == number_of_child[s]);
        int parent = parent_vector[s];
        if (parent != -1) {
            auto start = tree_set.begin() + tree_ptr[parent];
            auto end = tree_set.begin() + tree_ptr[parent + 1];
            assert(std::find(start, end, s) != end);
        }
    }

#endif
}

}
