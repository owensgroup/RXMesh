#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "nd_coarsen_manager.cuh"
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

template <uint32_t blockThreads>
__device__ __inline__ void matching(cooperative_groups::thread_block& block,
                                    rxmesh::ShmemAllocator&  shrd_alloc,
                                    const rxmesh::PatchInfo& patch_info,
                                    uint16_t*                s_ev,
                                    rxmesh::Bitmask&         matched_edges,
                                    rxmesh::Bitmask&         matched_vertices,
                                    uint16_t num_vertices,
                                    uint16_t num_edges,
                                    uint16_t curr_level)
{
    using namespace rxmesh;

    // TODO: num_edges/_vertices stored in the coarsen manager level by level
    __shared__ uint16_t s_num_active_vertices[1];
    s_num_active_vertices[0] = num_vertices;

    // TODO: use edge priority to replace the id for selecting edges
    uint16_t* s_e_chosen_by_v = shrd_alloc.alloc<uint16_t>(num_vertices);
    uint16_t* s_ve_offset     = shrd_alloc.alloc<uint16_t>(num_edges * 2);
    uint16_t* s_ve_output     = shrd_alloc.alloc<uint16_t>(num_edges * 2);

    rxmesh::Bitmask active_edges = Bitmask(num_edges, shrd_alloc);

    // Copy EV to offset array
    for (uint16_t i = threadIdx.x; i < num_edges * 2; i += blockThreads) {
        uint16_t s_ve_offset[i] = s_ev[i];
    }

    block.sync();

    // Get VE data here to avoid redundant computation
    v_e(num_vertices, num_edges, s_ve_offset, s_ve_output, nullptr);

    // TODO: add descriptions for every lambda
    while (float(s_num_active_vertices[0]) / float(s_num_vertices) > 0.25) {
        // reset the tmp array
        fill_n<blockThreads>(s_e_chosen_by_v, max_e_cap, uint16_t(0));
        block.sync();

        // VE operation
        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockthreads) {
            uint16_t start = s_ve_offset[v];
            uint16_t end   = s_ve_offset[v + 1];

            if (!coarsen_owned(LocalVertexT(v), curr_level, patch_info)) {
                continue;
            }

            uint16_t tgt_e_id = 0;

            // query for one ring edges
            for (uint16_t e = start; e < end; e++) {
                uint16_t edge = s_ve_output[e];

                if (!coarsen_owned(LocalEdgeT(e), curr_level, patch_info)) {
                    continue;
                }

                if (active_edges(e_local_id) && e_local_id > tgt_e_id) {
                    tgt_e_id = e_local_id;
                }
            }

            // TODO: assert memory access
            assert(v < num_vertices);
            s_e_chosen_by_v[v] = tgt_e_id;
        }

        block.sync();

        // EV operation
        for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
            uint16_t v0_local_id = s_ev[2 * e];
            uint16_t v1_local_id = s_ev[2 * e + 1];

            uint16_t v0_chosen_id = s_e_chosen_by_v[v0_local_id];
            uint16_t v1_chosen_id = s_e_chosen_by_v[v1_local_id];

            if (!coarsen_owned(LocalEdgeT(e), curr_level, patch_info)) {
                continue;
            }

            if (local_id == v1_chosen_id && local_id == v0_chosen_id) {
                matched_vertices.set(v0_local_id, true);
                matched_vertices.set(v1_local_id, true);
                matched_edges.set(e, true);
            }
        }

        block.sync();

        // VE operation
        for (uint16_t v = threadIdx.x; v < num_vertices; v += blockthreads) {
            uint16_t start = s_ve_offset[v];
            uint16_t end   = s_ve_offset[v + 1];

            if (!coarsen_owned(LocalEdgeT(v), curr_level, patch_info)) {
                continue;
            }

            if (matched_vertices(v)) {
                for (uint16_t e = start; e < end; e++) {
                    uint16_t e_local_id = s_ve_output[e];

                    if (coarsen_owned(
                            LocalEdgeT(e_local_id), curr_level, patch_info)) {
                        active_edges.set(e_local_id, false);
                    }
                }

                // count active vertices
                atomicAdd(&s_num_active_vertices[0], 1);
            }
        }

        block.sync();
    }

    shrd_alloc.dealloc<uint16_t>(num_vertices);
    shrd_alloc.dealloc<uint16_t>(2 * num_edges);
    shrd_alloc.dealloc<uint16_t>(2 * num_edges);
    shrd_alloc.dealloc(active_edges.num_bytes());

    // 1. two hop implementation
    //    -> traditional MIS/MM
    // 2. admed implementation
    //    -> priority function pi
    //    -> CAS to resolve conflict
    //    ->
    // 3. the kamesh parallel HEM implementation
    //    ->
    //    ->
}

template <uint32_t blockThreads>
__device__ __inline__ void coarsening(cooperative_groups::thread_block& block,
                                      rxmesh::ShmemAllocator&  shrd_alloc,
                                      const rxmesh::PatchInfo& patch_info,
                                      uint16_t*                s_ev,
                                      uint16_t*                s_mapping,
                                      rxmesh::Bitmask&         matched_edges,
                                      rxmesh::Bitmask&         matched_vertices,
                                      uint16_t                 num_vertices,
                                      uint16_t                 num_edges,
                                      uint16_t                 curr_level)
{
    // EV operation
    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        if (matched_edges(local_id)) {
            assert(matched_vertices(v0_local_id));
            assert(matched_vertices(v1_local_id));

            uint16_t coarse_id =
                v0_local_id < v1_local_id ? v0_local_id : v1_local_id;
            s_mapping[v0_local_id] = coarse_id;
            s_mapping[v1_local_id] = coarse_id;
        } else {
            if (!matched_vertices(v0_local_id)) {
                atomicCAS(&s_mapping[v0_local_id], INVALID16, v0_local_id);
            }

            if (!matched_vertices(v1_local_id)) {
                atomicCAS(&s_mapping[v1_local_id], INVALID16, v1_local_id);
            }
        }
    }

    block.sync();

    for (uint16_t e = threadIdx.x; e < num_edges; e += blockThreads) {
        uint16_t v0_local_id = s_ev[2 * e];
        uint16_t v1_local_id = s_ev[2 * e + 1];

        uint16_t v0_coarse_id =
            s_mapping[v0_local_id] < s_mapping[v1_local_id] ?
                s_mapping[v0_local_id] :
                s_mapping[v1_local_id];
        uint16_t v1_coarse_id =
            s_mapping[v0_local_id] < s_mapping[v1_local_id] ?
                s_mapping[v0_local_id] :
                s_mapping[v1_local_id];

        uint16_t tmp_coarse_edge_id =
            v0_coarse_id * num_vertices + v1_coarse_id;

        // TODO: needs extra mapping array for uncoarsening, may be solved just
        // using bitmask
        // TODO: sort and reduction for tmp_coarse_edge_id
        uint16_t coarse_edge_id = tmp_coarse_edge_id;

        atomicCAS(&s_ev[2 * coarse_edge_id + 0], 0, v0_coarse_id);
        atomicCAS(&s_ev[2 * coarse_edge_id + 1], 0, v1_coarse_id);
    }

    block.sync();

    shrd_alloc.dealloc<uint16_t>(num_vertices);
}

// direct function call
template <uint32_t blockThreads>
__device__ __inline__ void partition(cooperative_groups::thread_block& block,
                                     rxmesh::Context&                  context,
                                     rxmesh::ShmemAllocator& shrd_alloc)
{
    // TODO: use the active bitmask for non-continuous v_id
    // TODO: check the size indicating

    // bi_assignment_ggp(
    //     /*cooperative_groups::thread_block& */ block,
    //     /* const uint16_t                   */ num_vertices,
    //     /* const Bitmask&                   */ s_owned_v,
    //     /* const Bitmask&                   */ s_active_v,
    //     /* const uint16_t*                  */ m_s_vv_offset,
    //     /* const uint16_t*                  */ m_s_vv,
    //     /* Bitmask&                         */ s_assigned_v,
    //     /* Bitmask&                         */ s_current_frontier_v,
    //     /* Bitmask&                         */ s_next_frontier_v,
    //     /* Bitmask&                         */ s_partition_a_v,
    //     /* Bitmask&                         */ s_partition_b_v,
    //     /* int                              */ num_iter);
}

template <uint32_t blockThreads>
__device__ __inline__ void uncoarsening(cooperative_groups::thread_block& block,
                                        rxmesh::ShmemAllocator& shrd_alloc,
                                        uint16_t*               s_ev,
                                        uint16_t*               s_mapping,
                                        rxmesh::Bitmask&        matched_edges,
                                        rxmesh::Bitmask& matched_vertices,
                                        rxmesh::Bitmask& s_partition_a_v,
                                        rxmesh::Bitmask& s_partition_b_v,
                                        rxmesh::Bitmask& s_separator_v,
                                        rxmesh::Bitmask& s_coarse_partition_a_v,
                                        rxmesh::Bitmask& s_coarse_partition_b_v,
                                        rxmesh::Bitmask& s_coarse_separator_v,
                                        uint16_t         num_vertices,
                                        uint16_t         num_edges)
{
    //TODO: make this a coarsen manager
    //TODO: all the calculation for shared mem in one place
    uint16_t* s_ve_offset = shrd_alloc.alloc<uint16_t>(num_edges * 2);
    uint16_t* s_ve_output = shrd_alloc.alloc<uint16_t>(num_edges * 2);

    // Copy EV to offset array
    for (uint16_t i = threadIdx.x; i < num_edges * 2; i += blockThreads) {
        uint16_t s_ve_offset[i] = s_ev[i];
    }

    block.sync();

    // Get VE data here to avoid redundant computation
    v_e(num_vertices, num_edges, s_ve_offset, s_ve_output, nullptr);

    for (uint16_t v = threadIdx.x; v < num_vertices; v += blockthreads) {
        uint16_t start = s_ve_offset[v];
        uint16_t end   = s_ve_offset[v + 1];

        s_partition_a_v(v) = s_coarse_partition_a_v(s_mapping(v));
        s_partition_b_v(v) = s_coarse_partition_b_v(s_mapping(v));
        s_separator_v(v)   = s_coarse_separator_v(s_mapping(v));
    }

    block.sync();

    
}


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
    int i = 0;
    while (i < req_levels) {
        // device matching query specifically on PartitionManager

        // shared mem preprocessing here
        // calculating VE
        // used by both matching and coarsening

        matching<blockThreads>(block,
                               shrd_alloc,
                               coarsen_graph.get_ev(i),
                               coarsen_graph.get_matched_edges_bitmask(i),
                               coarsen_graph.get_matched_vertices_bitmask(i),
                               coarsen_graph.patch_info,
                               i);

        // // coarsen graph
        coarsening(block,
                   shrd_alloc,
                   coarsen_graph.get_ev(i),
                   coarsen_graph.get_matched_edges_bitmask(i),
                   coarsen_graph.get_matched_vertices_bitmask(i),
                   coarsen_graph.patch_info,
                   i);

        // shared mem deallocation
        // deallocate the VE

        ++i;
    }

    // multi-level bipartition one block per patch
    partition(coarsen_graph);

    while (i > 0) {
        uncoarsening(block,
                     context,
                     shrd_alloc,
                     coarsen_graph.m_s_matched_edges,
                     coarsen_graph.m_s_matched_vertices,
                     s_partition_a_v,
                     s_partition_b_v,
                     coarsen_graph.m_s_p0_vertices,
                     coarsen_graph.m_s_p1_vertices);

        // refinement(coarsen_graph, ordering_array);

        --i;
    }
}
