#pragma once

#include "rxmesh/rxmesh.h"

namespace RXMESH {

/**
 * io_elements()
 */
void __device__ __host__ __inline__ io_elements(const Op& op,
                                                ELEMENT&  source_ele,
                                                ELEMENT&  output_ele)
{
    if (op == Op::VV || op == Op::VE || op == Op::VF) {
        source_ele = ELEMENT::VERTEX;
    } else if (op == Op::EV || op == Op::EE || op == Op::EF) {
        source_ele = ELEMENT::EDGE;
    } else if (op == Op::FV || op == Op::FE || op == Op::FF) {
        source_ele = ELEMENT::FACE;
    }
    if (op == Op::VV || op == Op::EV || op == Op::FV) {
        output_ele = ELEMENT::VERTEX;
    } else if (op == Op::VE || op == Op::EE || op == Op::FE) {
        output_ele = ELEMENT::EDGE;
    } else if (op == Op::VF || op == Op::EF || op == Op::FF) {
        output_ele = ELEMENT::FACE;
    }
}
}  // namespace RXMESH