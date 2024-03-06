#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "nd_partition_manager.cuh"
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

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
                               uint16_t                          req_levels)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    //TMP: check the shared mem given by the driver vs the actual used
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
        printf("total shmem: %u dyn shmem: %u\n", total_shmem_size(), dynamic_shmem_size());

    // TODO: test constructor
    PartitionManager<blockThreads> coarsen_graph(
        block, context, shrd_alloc, req_levels);

    if (idx == 0)
        printf("total shmem: %u dyn shmem: %u\n", total_shmem_size(), dynamic_shmem_size());
    
    coarsen_graph.matching(block, 0);

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
