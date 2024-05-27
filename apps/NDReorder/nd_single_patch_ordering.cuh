#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "nd_partition_manager.cuh"

template <uint32_t blockThreads>
__global__ static void nd_single_patch_main(
    rxmesh::Context                   context,
    rxmesh::VertexAttribute<uint16_t> v_ordering,
    rxmesh::VertexAttribute<uint16_t> attr_matched_v,
    rxmesh::EdgeAttribute<uint16_t>   attr_active_e,
    uint16_t                          req_levels)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    // Init the struct and alloc the shared memory
    PartitionManager<blockThreads> partition_manager(
        block, context, shrd_alloc, req_levels);

    // TODO: partition quality and the matching quality analysis

    // Start the matching process and the result are saved in bit masks
    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 0);
    partition_manager.local_coarsening(block, 0);

    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 1);
    partition_manager.local_coarsening(block, 1);

    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 2);
    partition_manager.local_coarsening(block, 2);

    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 3);
    partition_manager.local_coarsening(block, 3);

    partition_manager.local_matching(block, attr_matched_v, attr_active_e, 4);
    partition_manager.local_coarsening(block, 4);

    partition_manager.local_multi_level_partition(block, 5, 2);

    // // iteration num known before kernel -> shared mem known before kernel
    // int i = 0;
    // while (i < req_levels) {
    //     partition_manager.matching(block, i);
    //     partition_manager.coarsening(block, i);
    //     ++i;
    // }

    // // multi-level bipartition one block per patch
    // partition_manager.partition(block, i);

    // i -= 1;
    // while (i > 0) {
    //     partition_manager.uncoarsening(block, i);
    //     // TODO: refinement
    //     // refinement(block, shared_alloc, i);

    //     --i;
    // }

    // partition_manager.genrate_reordering(block, v_ordering);

    // TMP: Check that debug mode is working
    // if (idx == 0)
    //     assert(1 == 0);
}


template <uint32_t blockThreads>
__global__ static void nd_single_patch_test_v_count(rxmesh::Context context,
                                                    uint16_t*       v_count)
{
    using namespace rxmesh;
    // auto           block = cooperative_groups::this_thread_block();
    // ShmemAllocator shrd_alloc;

    // // Init the struct and alloc the shared memory
    // PartitionManager<blockThreads> partition_manager(
    //     block, context, shrd_alloc, 1);

    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx == 0) {
        for (int i = 0; i < context.m_num_patches[0]; i++) {
            v_count[1] += context.m_patches_info[i].num_vertices[0];
            v_count[2] += context.m_patches_info[i].get_num_owned<VertexHandle>();
        }
    }
}