#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"

#include "rxmesh/matrix/nd_patch.cuh"


namespace rxmesh {

template <uint32_t blockThreads, int maxCoarsenLevels>
__global__ static void nd_single_patch(Context              context,
                                       VertexAttribute<int> v_ordering,
                                       VertexAttribute<int> attr_v,
                                       EdgeAttribute<int>   attr_e,
                                       VertexAttribute<int> attr_v1)
{

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;


    PatchND<blockThreads, maxCoarsenLevels> pnd(block, context, shrd_alloc);


    // matching and coarsening
    int l = 0;


    while (l < maxCoarsenLevels) {

        pnd.edge_matching(block, l, attr_v, attr_e);
        pnd.coarsen(block, l);

        int num_active_vertices = pnd.num_active_vertices(block);

        if (num_active_vertices <= 32) {
            break;
        }
        ++l;
    }

    block.sync();
    pnd.bipartition_coarse_graph(block);

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