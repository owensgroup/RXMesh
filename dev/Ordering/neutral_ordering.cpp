//
// Created by behrooz on 2025-09-29.
//

#include "neutral_ordering.h"


#include <cassert>
#include <iostream>
#include <metis.h>
#include "ordering.h"

namespace RXMESH_SOLVER {

NeutralOrdering::~NeutralOrdering()
{
}

void NeutralOrdering::setGraph(int* Gp, int* Gi, int G_N, int NNZ)
{
    this->Gp = Gp;
    this->Gi = Gi;
    this->G_N = G_N;
    this->G_NNZ = NNZ;
}

void NeutralOrdering::compute_permutation(std::vector<int>& perm)
{
    perm.resize(G_N, -1);
    for (int i = 0; i < G_N; i++) {
        perm[i] = i;
    }
}


RXMESH_Ordering_Type NeutralOrdering::type() const
{
    return  RXMESH_Ordering_Type::NEUTRAL;
}


}