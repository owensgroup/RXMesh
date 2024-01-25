#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "nd_coarsen_manager.cuh"
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

// using void has type error?
template <typename T, uint32_t blockThreads>
__device__ __inline__ int random_matching(rxmesh::Context& context, uint16_t* matching_arr) {


    using namespace rxmesh;
    
    auto rm_lambda = [&](VertexHandle v_id, VertexIterator& vi){
        VertexHandle match_v = vi[0];

        uint16_t v_local_id = v_id.local_id();
        uint16_t match_v_local_id = match_v.local_id();

        atomicCAS(&matching_arr[match_v_local_id], INVALID16, v_local_id);
        // TODO: matching in progress

        // 1. two hop implementation
        // 2. admed implementation

    };
    
    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    query.dispatch<Op::VV>(block, shrd_alloc, rm_lambda);

    query.dispatch<Op::VE>(block, shrd_alloc, ve_lambda);

    return 0;
}


template <typename T, uint32_t blockThreads>
__global__ static void nd_main(rxmesh::Context context)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CoarsePatchinfo<blockThreads> coarsen_graph(block, context, shrd_alloc);

    // TODO: coarsen and do the multi-level partition and then refinement

    // iteration num known before kernel -> shared mem known before kernel
    int i = 0;
    while (i < 1) {
        // device matching query specifically on CoarsePatchinfo
        random_matching(coarsen_graph, matching_array, shrd_alloc);

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
