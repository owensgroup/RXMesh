//
// Created by Behrooz on 2025-09-10.
//

#include "check_valid_permutation.h"
#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

namespace RXMESH_SOLVER {
//Return the factor's nnz using CHOLMOD analysis. The input matrix should be CSC with only lower part represented.
bool check_valid_permutation(int* perm, int n){
    std::vector<bool> marker(n, false);
    for(int i = 0; i < n; i++) {
        if(perm[i] < 0 || perm[i] >= n) {
            std::cerr << "ERROR: Invalid permutation. Element " << i << " has value " << perm[i] << " which is out of range [0, " << n-1 << "]" << std::endl;
            return false;
        }
        if(marker[perm[i]]) {
            std::cerr << "ERROR: Invalid permutation. Element " << perm[i] << " appears more than once." << std::endl;
            return false;
        }
        marker[perm[i]] = true;
    }

    //Check to see there is no marker left
    for(int i = 0; i < n; i++) {
        if (marker[i] == false) {
            std::cerr << "ERROR: Invalid permutation. Element " << i << " is missing." << std::endl;
            return false;
        }
    }
    return true;
}

}
