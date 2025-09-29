//
// Created by behrooz on 2025-09-29.
//

#include "ordering.h"
#include "metis_ordering.h"
#include <cassert>
#include <iostream>

namespace RXMESH_SOLVER {


Ordering *Ordering::create(const RXMESH_Ordering_Type type) {
    switch (type) {
        case RXMESH_Ordering_Type::METIS:
            return new MetisOrdering();
        default:
            std::cerr << "Uknown linear system solver type" << std::endl;
            return nullptr;
    }
}

}