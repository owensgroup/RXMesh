#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "nd_partition_manager.cuh"
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

// TODO: max levels = shared memory / patch size
template <uint32_t blockThreads>
__global__ static void nd_main(rxmesh::Context                   context,
                               rxmesh::VertexAttribute<uint16_t> v_ordering,
                               uint16_t                          req_levels)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    PartitionManager<blockThreads> coarsen_graph(
        block, context, shrd_alloc, req_levels);

    // iteration num known before kernel -> shared mem known before kernel
    int i = 0;
    while (i < req_levels) {
        coarsen_graph.matching(block, shared_alloc, i);
        coarsen_graph.coarsening(block, i);
        ++i;
    }

    // multi-level bipartition one block per patch
    coarsen_graph.partition(block, shared_alloc, i);

    i -= 1;
    while (i > 0) {
        coarsen_graph.uncoarsening(block,
                     i);
        // TODO: refinement 
        // refinement(block, shared_alloc, i);

        --i;
    }
}
