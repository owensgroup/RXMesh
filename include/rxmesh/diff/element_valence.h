#pragma once

#include "rxmesh/types.h"

namespace rxmesh {

/**
 * @brief Return the number of incident element based on a query operation,
 * e.g., 3 for FV query operation
 */
template <Op op>
constexpr int element_valence()
{
    // dynamic
    if constexpr (op == Op::VV || op == Op::VE || op == Op::VF) {
        return 0;
    }

    if constexpr (op == Op::V || op == Op::E || op == Op::F) {
        return 1;
    }

    if constexpr (op == Op::EV || op == Op::EF) {
        return 2;
    }

    if constexpr (op == Op::FV || op == Op::FE || op == Op::FF) {
        return 3;
    }


    //??
    if constexpr (op == Op::EE) {
        return -1;
    }
}

}  // namespace rxmesh