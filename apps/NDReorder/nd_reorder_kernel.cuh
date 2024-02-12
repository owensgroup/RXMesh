#pragma once
#include "rxmesh/attribute.h"
#include "rxmesh/context.h"
#include "rxmesh/query.cuh"
#include "rxmesh/util/vector.h"

#include "nd_coarsen_manager.cuh"
#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

template <uint32_t blockThreads>
__device__ __inline__ void matching(cooperative_groups::thread_block& block,
                                    rxmesh::Context&                  context,
                                    rxmesh::ShmemAllocator& shrd_alloc,
                                    rxmesh::Bitmask&        active_edges,
                                    rxmesh::Bitmask&        matched_edges,
                                    rxmesh::Bitmask&        matched_vertices,
                                    uint16_t*               s_num_vertices)
{
    using namespace rxmesh;

    Query<blockThreads> query(context);
    __shared__ uint16_t s_num_active_vertices[1];
    s_num_active_vertices[0] = *s_num_vertices;

    // edge degree as the priority function
    const uint16_t max_edge_cap =
        static_cast<uint16_t>(context.m_max_num_edges[0]);
    uint16_t* s_edge_pi = shrd_alloc.alloc<uint16_t>(max_edge_cap);
    fill_n<blockThreads>(s_edge_pi, max_edge_cap, uint16_t(0));

    // TODO: replace e_iter[i].local_id() with e_iter.local(i).
    // calculate priority based on edge degree
    auto edge_pi_lambda = [&](VertexHandle v_id, EdgeIterator& e_iter) {
        uint32_t patch_id    = v_id.patch_id();
        uint16_t edge_degree = (e_iter.size() - 1) << 8;  // intuition
        for (uint32_t i = 0; i < e_iter.size(); ++i) {
            uint32_t edge_patch_id = e_iter[i].patch_id();
            if (patch_id == edge_patch_id) {
                uint16_t local_edge_idx = e_iter[i].local_id();
                atomicAdd(&s_edge_pi[local_edge_idx],
                          edge_degree + local_edge_idx);
            }
        }
    };

    query.dispatch<Op::VE>(block, shrd_alloc, edge_pi_lambda);

    matched_edges.reset();
    matched_vertices.reset();

    const uint16_t max_vertex_cap =
        static_cast<uint16_t>(context.m_max_num_vertices[0]);
    uint16_t* s_edge_chosen_by_v = shrd_alloc.alloc<uint16_t>(max_vertex_cap);

    while (float(s_num_active_vertices[0]) / float(s_num_vertices) > 0.25) {
        // reset the tmp array
        fill_n<blockThreads>(s_edge_chosen_by_v, max_edge_cap, uint16_t(0));

        auto choose_edge_lambda = [&](VertexHandle v_id, EdgeIterator& e_iter) {
            uint32_t patch_id = v_id.patch_id();
            uint16_t local_id = v_id.local_id();

            uint16_t tgt_edge_id = 0;
            uint16_t edge_pi_val = 0;

            for (uint32_t i = 0; i < e_iter.size(); ++i) {
                uint32_t edge_patch_id = e_iter[i].patch_id();
                uint32_t edge_local_id = e_iter[i].local_id();
                // e_iter.local(i); // local index with ribbon counted

                // determine using the edge id or the edge pi here
                if (patch_id == edge_patch_id && active_edges(edge_local_id) &&
                    edge_local_id > tgt_edge_id) {
                    tgt_edge_id = edge_local_id;
                }
            }

            // assert check tgt_edge_id should not be 0
            // assert check local_id small than max_vertex_cap
            s_edge_chosen_by_v[local_id] = tgt_edge_id;
        };

        query.dispatch<Op::VE>(block, shrd_alloc, choose_edge_lambda);
        block.sync();

        auto filter_edge_lambda = [&](EdgeHandle e_id, VertexIterator& v_iter) {
            uint16_t local_id     = e_id.local_id();
            uint16_t v0_chosen_id = s_edge_chosen_by_v[v_iter[0].local_id()];
            uint16_t v1_chosen_id = s_edge_chosen_by_v[v_iter[1].local_id()];
            if (local_id == v0_chosen_id && local_id == v1_chosen_id) {
                matched_edges.set(local_id, true);
                matched_vertices.set(v_iter[0].local_id(), true);
                matched_vertices.set(v_iter[1].local_id(), true);
            }
        };

        query.dispatch<Op::EV>(block, shrd_alloc, filter_edge_lambda);
        block.sync();

        auto deactive_ring_edge_lambda = [&](VertexHandle  v_id,
                                             EdgeIterator& e_iter) {
            uint32_t patch_id = v_id.patch_id();
            uint16_t local_id = v_id.local_id();

            bool is_matched = false;

            for (uint32_t i = 0; i < e_iter.size(); ++i) {
                uint32_t edge_patch_id = e_iter[i].patch_id();
                uint32_t edge_local_id = e_iter[i].local_id();

                // determine using the edge id or the edge pi here
                if (patch_id == edge_patch_id && matched_edges(edge_local_id)) {
                    is_matched = true;
                }
            }

            if (is_matched) {
                for (uint32_t i = 0; i < e_iter.size(); ++i) {
                    uint32_t edge_patch_id = e_iter[i].patch_id();
                    uint32_t edge_local_id = e_iter[i].local_id();

                    // determine using the edge id or the edge pi here
                    if (patch_id == edge_patch_id) {
                        active_edges.set(edge_local_id, false);
                    }
                }
            }
        };

        query.dispatch<Op::VE>(block, shrd_alloc, filter_edge_lambda);
        block.sync();

        // count the active edges
        auto count_active_lambda = [&](VertexHandle  v_id,
                                       EdgeIterator& e_iter) {
            uint16_t local_v_id = v_id.local_id();
            if (matched_vertices(local_v_id)) {
                atomicAdd(&s_num_active_vertices[0], 1);
            }
        };

        query.dispatch<Op::VE>(block, shrd_alloc, count_active_lambda);
        block.sync();
    }

    // atomicCAS(&matching_arr[match_v_local_id], INVALID16, v_local_id);
    // TODO: matching in progress

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
                                      rxmesh::Context&                  context,
                                      rxmesh::ShmemAllocator& shrd_alloc,
                                      uint16_t*               s_ev,
                                      rxmesh::Bitmask&        matched_edges,
                                      rxmesh::Bitmask&        matched_vertices)
{
    const uint16_t max_edge_cap =
        static_cast<uint16_t>(context.m_max_num_edges[0]);

    const uint16_t max_vertex_cap =
        static_cast<uint16_t>(context.m_max_num_vertices[0]);
    uint16_t* s_coarsen_v = shrd_alloc.alloc<uint16_t>(max_vertex_cap);
    fill_n<blockThreads>(s_coarsen_v, max_vertex_cap, uint16_t(0));

    auto update_coarsen_v_lambda = [&](EdgeHandle      e_id,
                                       VertexIterator& v_iter) {
        uint16_t local_id    = e_id.local_id();
        uint16_t v0_local_id = v_iter[0].local_id();
        uint16_t v1_local_id = v_iter[1].local_id();
        if (matched_edges(local_id)) {
            assert(matched_vertices(v0_local_id));
            assert(matched_vertices(v1_local_id));

            s_coarsen_v[v0_local_id] =
                v0_local_id < v1_local_id ? v0_local_id : v1_local_id;
            s_coarsen_v[v1_local_id] =
                v0_local_id < v1_local_id ? v0_local_id : v1_local_id;
        } else {
            if (!matched_vertices(v0_local_id)) {
                atomicCAS(&s_coarsen_v, 0, v0_local_id);
            }

            if (!matched_vertices(v1_local_id)) {
                atomicCAS(&s_coarsen_v, 0, v1_local_id);
            }
        }

        // can I do this?
        block.sync();
        // fill ev with new edges

        atomicCAS(&s_ev[2 * max_edge_cap + 2 * local_id + 0],
                  0,
                  s_coarsen_v[v0_local_id]);
        atomicCAS(&s_ev[2 * max_edge_cap + 2 * local_id + 1],
                  0,
                  s_coarsen_v[v1_local_id]);
    };

    query.dispatch<Op::EV>(block, shrd_alloc, update_coarsen_v_lambda);
    block.sync();
}

// direct function call
template <uint32_t blockThreads>
__device__ __inline__ void partition(cooperative_groups::thread_block& block,
                                     rxmesh::Context&                  context,
                                     rxmesh::ShmemAllocator& shrd_alloc,
                                     ...)
{

    bi_assignment_ggp(
        /*cooperative_groups::thread_block& */ block,
        /* const uint16_t                   */ num_vertices,
        /* const Bitmask&                   */ s_owned_v,
        /* const Bitmask&                   */ s_active_v,
        /* const uint16_t*                  */ m_s_vv_offset,
        /* const uint16_t*                  */ m_s_vv,
        /* Bitmask&                         */ s_assigned_v,
        /* Bitmask&                         */ s_current_frontier_v,
        /* Bitmask&                         */ s_next_frontier_v,
        /* Bitmask&                         */ s_partition_a_v,
        /* Bitmask&                         */ s_partition_b_v,
        /* int                              */ num_iter);
}

template <uint32_t blockThreads>
__device__ __inline__ void uncoarsening(cooperative_groups::thread_block& block,
                                        rxmesh::Context&        context,
                                        rxmesh::ShmemAllocator& shrd_alloc,
                                        uint16_t*               s_ev,
                                        rxmesh::Bitmask&        matched_edges,
                                        rxmesh::Bitmask& matched_vertices,
                                        rxmesh::Bitmask& s_partition_a_v,
                                        rxmesh::Bitmask& s_partition_b_v,
                                        rxmesh::Bitmask& s_mapped_partition_a_v,
                                        rxmesh::Bitmask& s_mapped_partition_b_v)
{
    auto update_coarsen_v_lambda = [&](EdgeHandle      e_id,
                                       VertexIterator& v_iter) {
        uint16_t local_id    = e_id.local_id();
        uint16_t v0_local_id = v_iter[0].local_id();
        uint16_t v1_local_id = v_iter[1].local_id();
        if (matched_edges(local_id)) {
            assert(matched_vertices(v0_local_id));
            assert(matched_vertices(v1_local_id));

            uint16_t final_id =
                v0_local_id < v1_local_id ? v0_local_id : v1_local_id;
            s_mapped_partition_a_v(v0_local_id) = s_partition_a_v(final_id);
            s_mapped_partition_b_v(v0_local_id) = s_partition_b_v(final_id);
            s_mapped_partition_a_v(v1_local_id) = s_partition_a_v(final_id);
            s_mapped_partition_b_v(v1_local_id) = s_partition_b_v(final_id);
        } else {
            if (!matched_vertices(v0_local_id)) {
                s_mapped_partition_a_v(v0_local_id) = s_partition_a_v(final_id);
                s_mapped_partition_b_v(v0_local_id) = s_partition_b_v(final_id);
            }

            if (!matched_vertices(v1_local_id)) {
                s_mapped_partition_a_v(v1_local_id) = s_partition_a_v(final_id);
                s_mapped_partition_b_v(v1_local_id) = s_partition_b_v(final_id);
            }
        }
    };

    query.dispatch<Op::EV>(block, shrd_alloc, update_coarsen_v_lambda);
    block.sync();
}


// TODO: finish the full framework FIRST
template <uint32_t blockThreads>
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
        uint16_t* test;
        // device matching query specifically on CoarsePatchinfo
        matching<blockThreads>(block,
                               context,
                               shrd_alloc,
                               coarsen_graph.m_s_active_edges,
                               coarsen_graph.m_s_matched_edges,
                               coarsen_graph.m_s_matched_vertices,
                               coarsen_graph.m_s_num_vertices);

        // // coarsen graph
        coarsening(block,
                         context,
                         shrd_alloc,
                         coarsen_graph.m_s_ev,
                         coarsen_graph.m_s_matched_edges,
                         coarsen_graph.m_s_matched_vertices);

        // coarsen_graph = coarsen_graph->next_level_coarsen;

        ++i;
    }

    // multi-level bipartition one block per patch
    partition(coarsen_graph, ordering_array);

    // while(coarsen_graph->prev_level_coarsen != null) {
    //     // uncoarsen graph
    //     graph_uncoarsening(coarsen_graph, ordering_array);

    //     refinement(coarsen_graph, ordering_array);

    //     coarsen_graph = coarsen_graph->prev_level_coarsen;
    // }
}
