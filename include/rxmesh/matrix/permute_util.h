#pragma once

#include <stdint.h>
#include <algorithm>
#include <vector>

/**
 * @brief given a permutation array, verify that it is a unique permutation,
 * i.e., every index is assigned to one permutation. This is done by sorting the
 * array and then checking if every entry in the array matches its index.
 */
bool is_unique_permutation(uint32_t size, int* h_permute)
{
    std::vector<int> permute(size);
    std::memcpy(permute.data(), h_permute, size * sizeof(int));

    std::sort(permute.begin(), permute.end());

    for (int i = 0; i < int(permute.size()); ++i) {
        if (i != permute[i]) {
            return false;
        }
    }

    return true;
}