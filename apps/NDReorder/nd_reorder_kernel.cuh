#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "nd_partition_manager.cuh"
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

// TODO: create a branch in main repo and sync with this

// VE query example computed from EV (s_ev)
// 1. Copy s_ev into s_ve_offset
// 2. call v_e(num_vertices, num_edges, s_ve_offset, s_ve_output, nullptr);
// for(uint16_t v=threadIdx.x; v<num_vertices; v+=blockthreads){
//     uint16_t start = s_ve_offset[v];
//     uint16_t end = s_ve_offset[v+1];
//     for(uint16_t e=start; e<end;e++){
//         uint16_t edge = s_ve_output[e];
//
//     }
// }

// EV query example
// for(uint16_t e=threadIdx.x; e< num_edges; e+= blockThreads){
//     uint16_t v0_local_id = s_ev[2*e];
//     uint16_t v1_local_id = s_ev[2*e+1];
// }

// TODO: max levels = shared memory / patch size
//
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
    // int i = 0;
    // while (i < req_levels) {
    //     // device matching query specifically on PartitionManager

    //     // shared mem preprocessing here
    //     // calculating VE
    //     // used by both matching and coarsening

    //     matching<blockThreads>(block,
    //                            shrd_alloc,
    //                            coarsen_graph.get_ev(i),
    //                            coarsen_graph.get_matched_edges_bitmask(i),
    //                            coarsen_graph.get_matched_vertices_bitmask(i),
    //                            coarsen_graph.patch_info,
    //                            i);

    //     // // coarsen graph
    //     coarsening(block,
    //                shrd_alloc,
    //                coarsen_graph.get_ev(i),
    //                coarsen_graph.get_matched_edges_bitmask(i),
    //                coarsen_graph.get_matched_vertices_bitmask(i),
    //                coarsen_graph.patch_info,
    //                i);

    //     // shared mem deallocation
    //     // deallocate the VE

    //     ++i;
    // }

    // // multi-level bipartition one block per patch
    // partition(coarsen_graph);

    // while (i > 0) {
    //     uncoarsening(block,
    //                  context,
    //                  shrd_alloc,
    //                  coarsen_graph.m_s_matched_edges,
    //                  coarsen_graph.m_s_matched_vertices,
    //                  s_partition_a_v,
    //                  s_partition_b_v,
    //                  coarsen_graph.m_s_p0_vertices,
    //                  coarsen_graph.m_s_p1_vertices);

    //     // refinement(coarsen_graph, ordering_array);

    //     --i;
    // }
}
