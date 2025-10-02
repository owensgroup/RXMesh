//
// Created by behrooz on 2025-09-29.
//

#include "metis_ordering.h"


#include <cassert>
#include <iostream>
#include <metis.h>
#include "ordering.h"

namespace RXMESH_SOLVER {

MetisOrdering::~MetisOrdering()
{
}

void MetisOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp = Gp;
    this->Gi = Gi;
    this->G_N = G_N;
    this->G_NNZ = NNZ;
}

void MetisOrdering::compute_permutation(std::vector<int>& perm)
{
    idx_t N = G_N;
    idx_t NNZ = Gp[G_N];
    perm.resize(G_N);
    if (NNZ == 0) {
        assert(G_N != 0);
        for (int i = 0; i < G_N; i++) {
#ifndef NDEBUG
            //      std::cout << "WARNING: This decomposition does not have edges"
            //                << std::endl;
#endif
            perm[i] = i;
        }
        return;
    }
    // TODO add memory allocation protection later like CHOLMOD

    std::vector<int> tmp(G_N);
    METIS_NodeND(&N, Gp, Gi, NULL, NULL, perm.data(), tmp.data());
}


RXMESH_Ordering_Type MetisOrdering::type() const
{
    return  RXMESH_Ordering_Type::METIS;
}
std::string MetisOrdering::typeStr() const
{
    return "METIS";
}

}