#pragma once

#include "rxmesh/context.h"
#include "rxmesh/kernels/update_dispatcher.cuh"

template <uint32_t blockThreads>
__global__ static void edge_flip(rxmesh::Context context)
{
    using namespace rxmesh;
}