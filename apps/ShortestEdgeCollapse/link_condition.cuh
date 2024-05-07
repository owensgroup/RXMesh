#pragma once
#include "rxmesh/cavity_manager.cuh"

/**
 * @brief Check for link condition (used for edge collapse and edge flip) or
 * each edge marked in edge_mask, all threads in the block collaborate to check
 * the edge link condition of this edge. edge_mask stores 1 for edges that we
 * will check for. If the condition passes, the bit corresponding to the edge
 * will not change. If the edge fails the link condition, we flip the edge bit
 * from 1 to 0
 */
template <uint32_t blockThreads>
__inline__ __device__ void link_condition(
    cooperative_groups::thread_block& block,
    const rxmesh::PatchInfo&          patch_info,
    rxmesh::Query<blockThreads>&      ev_query,
    rxmesh::Bitmask&                  edge_mask,
    rxmesh::Bitmask&                  v0_mask,
    rxmesh::Bitmask&                  v1_mask)
{
    using namespace rxmesh;

    __shared__ int s_num_shared_one_ring;
    for (uint16_t e = 0; e < edge_mask.size(); ++e) {

        if (edge_mask(e)) {
            // the edge two end vertices
            const VertexIterator iter =
                ev_query.template get_iterator<VertexIterator>(e);

            const uint16_t v0 = iter.local(0);
            const uint16_t v1 = iter.local(1);

            if (threadIdx.x == 0) {
                s_num_shared_one_ring = 0;
            }

            v0_mask.reset(block);
            v1_mask.reset(block);
            block.sync();

            // each thread will be assigned to an edge (including not-owned one)
            // and mark in v0_mask/v1_mask if one of its two ends are v0/v1
            for_each_edge(
                patch_info,
                [&](EdgeHandle eh) {
                    if (eh.local_id() == e) {
                        return;
                    }
                    const VertexIterator v_iter =
                        ev_query.template get_iterator<VertexIterator>(
                            eh.local_id());

                    const uint16_t vv0 = v_iter.local(0);
                    const uint16_t vv1 = v_iter.local(1);


                    if (vv0 == v0) {
                        v0_mask.set(vv1, true);
                    }
                    if (vv0 == v1) {
                        v1_mask.set(vv1, true);
                    }

                    if (vv1 == v0) {
                        v0_mask.set(vv0, true);
                    }
                    if (vv1 == v1) {
                        v1_mask.set(vv0, true);
                    }
                },
                true);
            block.sync();

            for (int v = threadIdx.x; v < v0_mask.size(); v += blockThreads) {
                if (v0_mask(v) && v1_mask(v)) {
                    ::atomicAdd(&s_num_shared_one_ring, 1);
                }
            }

            block.sync();
            if (s_num_shared_one_ring > 2) {
                edge_mask.reset(e, true);
            }
        }
    }
}
