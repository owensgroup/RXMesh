//
// Created by behrooz on 2025-09-29.
//

#include "ordering.h"
#include <cassert>
#include <iostream>
#include "metis_ordering.h"
#include "neutral_ordering.h"
#include "rxmesh_ordering.h"
#include "poc_ordering.h"

namespace RXMESH_SOLVER {


Ordering *Ordering::create(const RXMESH_Ordering_Type type) {
    switch (type) {
        case RXMESH_Ordering_Type::METIS:
            return new MetisOrdering();
        case RXMESH_Ordering_Type::RXMESH_ND:
            return new RXMeshOrdering();
        case RXMESH_Ordering_Type::POC_ND:
            return new POCOrdering();
        case RXMESH_Ordering_Type::NEUTRAL:
            return new NeutralOrdering();
        default:
            std::cerr << "Unknown Ordering type" << std::endl;
            return nullptr;
    }
}

}