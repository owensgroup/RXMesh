#pragma once

#include <algorithm>
#include <string>

namespace rxmesh {
/**
 * @brief The enum class for choosing different reorder types
 * NONE for No Reordering Applied, SYMRCM for Symmetric Reverse Cuthill-McKee
 * permutation, SYMAMD for Symmetric Approximate Minimum Degree Algorithm based
 * on Quotient Graph, NSTDIS for Nested Dissection, GPUMGND is a GPU modified
 * generalized nested dissection permutation, and GPUND is GPU nested dissection
 */
enum class PermuteMethod
{
    NONE    = 0,
    SYMRCM  = 1,
    SYMAMD  = 2,
    NSTDIS  = 3,
    GPUMGND = 4,
    GPUND   = 5
};

inline PermuteMethod string_to_permute_method(std::string prem)
{
    std::transform(prem.begin(), prem.end(), prem.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    if (prem == "symrcm") {
        return PermuteMethod::SYMRCM;
    } else if (prem == "symamd") {
        return PermuteMethod::SYMAMD;
    } else if (prem == "nstdis") {
        return PermuteMethod::NSTDIS;
    } else if (prem == "gpumgnd") {
        return PermuteMethod::GPUMGND;
    } else if (prem == "gpund") {
        return PermuteMethod::GPUND;
    } else {
        return PermuteMethod::NONE;
    }
}


inline std::string permute_method_to_string(PermuteMethod prem)
{
    if (prem == PermuteMethod::SYMRCM) {
        return "symrcm";
    } else if (prem == PermuteMethod::SYMAMD) {
        return "symamd";
    } else if (prem == PermuteMethod::NSTDIS) {
        return "nstdis";
    } else if (prem == PermuteMethod::GPUMGND) {
        return "gpumgnd";
    } else if (prem == PermuteMethod::GPUND) {
        return "gpund";
    } else {
        return "none";
    }
}

}  // namespace rxmesh