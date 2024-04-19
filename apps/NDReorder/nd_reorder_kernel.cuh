#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "nd_partition_manager.cuh"

// TODO: test function for shared mem allocated
__forceinline__ __device__ unsigned dynamic_shmem_size()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned total_shmem_size()
{
    unsigned ret; 
    asm volatile ("mov.u32 %0, %total_smem_size;" : "=r"(ret));
    return ret;
}

// TODO: max levels = shared memory / patch size
template <uint32_t blockThreads>
__global__ static void nd_main(rxmesh::Context                   context,
                               rxmesh::VertexAttribute<uint16_t> v_ordering,
                               rxmesh::VertexAttribute<uint16_t> attr_matched_v,
                               rxmesh::EdgeAttribute<uint16_t> attr_active_e,
                               uint16_t                          req_levels)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    // Init the struct and alloc the shared memory
    PartitionManager<blockThreads> coarsen_graph(
        block, context, shrd_alloc, req_levels);

    // TODO: partition quality and the matching quality analysis 
    
    // Start the matching process and the result are saved in bit masks
    coarsen_graph.matching(block, attr_matched_v, attr_active_e, 0);
    coarsen_graph.coarsening(block, 0);

    coarsen_graph.matching(block, attr_matched_v, attr_active_e, 1);
    coarsen_graph.coarsening(block, 1);

    coarsen_graph.matching(block, attr_matched_v, attr_active_e, 2);
    coarsen_graph.coarsening(block, 2);

    coarsen_graph.matching(block, attr_matched_v, attr_active_e, 3);
    coarsen_graph.coarsening(block, 3);

    coarsen_graph.matching(block, attr_matched_v, attr_active_e, 4);
    coarsen_graph.coarsening(block, 4);

    // coarsen_graph.partition(block, 5);

    // // iteration num known before kernel -> shared mem known before kernel
    // int i = 0;
    // while (i < req_levels) {
    //     coarsen_graph.matching(block, i);
    //     coarsen_graph.coarsening(block, i);
    //     ++i;
    // }

    // // multi-level bipartition one block per patch
    // coarsen_graph.partition(block, i);

    // i -= 1;
    // while (i > 0) {
    //     coarsen_graph.uncoarsening(block, i);
    //     // TODO: refinement 
    //     // refinement(block, shared_alloc, i);

    //     --i;
    // }

    // coarsen_graph.genrate_reordering(block, v_ordering);

    // TMP: Check that debug mode is working
    // if (idx == 0)
    //     assert(1 == 0);
}
