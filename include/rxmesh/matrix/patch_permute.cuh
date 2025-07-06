#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"

#include "rxmesh/matrix/min_deg_patch.cuh"
#include "rxmesh/matrix/nd_patch.cuh"

#include "rxmesh/matrix/kmeans_patch.cuh"

namespace rxmesh {

template <uint32_t blockThreads, int maxCoarsenLevels>
__global__ static void patch_permute_nd(Context              context,
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


template <uint32_t blockThreads>
__global__ static void patch_permute_kmeans(Context                   context,
                                            VertexAttribute<uint16_t> v_permute,
                                            int                       threshold)
{
    if (blockIdx.x >= context.get_num_patches()) {
        return;
    }

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;


    PatchKMeans<blockThreads> pkm(block, context, shrd_alloc);

    int num_v = pkm.num_active_vertices(block);

    // if (num_v < threshold) {
    //     return;
    // }

    pkm.partition(block);

    pkm.extract_separator(block);

    pkm.assign_permutation(block, v_permute);
}

template <uint32_t blockThreads>
__global__ static void patch_permute_min_deg(
    Context                   context,
    VertexAttribute<uint16_t> v_permute)
{
    if (blockIdx.x != 1) {
        return;
    }

    if (blockIdx.x >= context.get_num_patches()) {
        return;
    }

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;


    PatchMinDeg<blockThreads> pmd(block, context, shrd_alloc);

    pmd.permute(block, v_permute);
}


}  // namespace rxmesh