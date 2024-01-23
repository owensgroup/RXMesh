#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "nd_coarsen_manager.cuh"
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <typename T, uint32_t blockThreads>
__global__ static void nd_main(const rxmesh::Context context)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CoarsePatchinfo<blockThreads> coarsen_graph(block, context, shrd_alloc);

    // iteration num known before kernel -> shared mem known before kernel
    int i = 0;
    while (i < 1) {
        // device matching query specifically on CoarsePatchinfo
        // heavy_edge_matching(coarsen_graph, matching_array);

        // // coarsen graph
        // graph_coarsening(coarsen_graph, matching_array);

        // coarsen_graph = coarsen_graph->next_level_coarsen;

        i++;
    }

    // partition(coarsen_graph, ordering_array);

    // while(coarsen_graph->prev_level_coarsen != null) {
    //     // uncoarsen graph
    //     graph_uncoarsening(coarsen_graph, ordering_array);

    //     refinement(coarsen_graph, ordering_array);

    //     coarsen_graph = coarsen_graph->prev_level_coarsen;
    // }
}
