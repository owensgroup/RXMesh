#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"

#include "rxmesh/matrix/nd_patch.cuh"

namespace rxmesh {

template <uint32_t blockThreads>
__global__ static void nd_single_patch(Context                   context,
                                       VertexAttribute<int>      v_ordering,
                                       VertexAttribute<uint16_t> attr_matched_v,
                                       EdgeAttribute<uint16_t>   attr_active_e)
{

    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    PatchND<blockThreads> pnd(block, context, shrd_alloc);


    // matching and coarsening
    // int i = 0;
    // while (i < req_levels) {
    //    pm.local_matching(block, attr_matched_v, attr_active_e, i);
    //    pm.local_coarsening(block, i);
    //    ++i;
    //}
    //
    //// multi-level bipartition
    // pm.local_multi_level_partition(block, req_levels);
    //
    // i -= 1;
    // while (i > 0) {
    //     pm.local_uncoarsening(block, i);
    //     // TODO: refinement
    //     // refinement(block, shared_alloc, i);
    //     --i;
    // }
    //
    // pm.local_genrate_reordering(block, v_ordering);
}
}  // namespace rxmesh