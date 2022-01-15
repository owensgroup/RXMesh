#pragma once

#include "rxmesh/context.h"
#include "rxmesh/kernels/update_dispatcher.cuh"

template <uint32_t blockThreads>
__global__ static void edge_flip(rxmesh::Context context)
{
    using namespace rxmesh;

    // flip one edge (the edge assigned to thread 0) in each patch
    auto should_flip = [&](const EdgeHandle& edge) -> bool {
        if (threadIdx.x == 0) {
            return true;
        } else {
            return false;
        }
    };

    update_block_dispatcher<DynOp::EdgeFlip, blockThreads>(context,
                                                           should_flip);
}