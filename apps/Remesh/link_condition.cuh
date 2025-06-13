#pragma once

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
    rxmesh::Bitmask&                  v1_mask,
    const int                         v0_index_in_iter,
    const int                         v1_index_in_iter)
{
    using namespace rxmesh;

    __shared__ int s_num_shared_one_ring;
    for (uint16_t e = 0; e < edge_mask.size(); ++e) {

        if (edge_mask(e)) {
            // the edge two end vertices
            const VertexIterator iter =
                ev_query.template get_iterator<VertexIterator>(e);

            const uint16_t v0 = iter.local(v0_index_in_iter);
            const uint16_t v1 = iter.local(v1_index_in_iter);

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
                    if (eh.local_id() == e &&
                        eh.patch_id() == patch_info.patch_id) {
                        return;
                    }
                    const VertexIterator v_iter =
                        ev_query.template get_iterator<VertexIterator>(
                            eh.local_id());

                    const uint16_t vv0 = v_iter.local(v0_index_in_iter);
                    const uint16_t vv1 = v_iter.local(v1_index_in_iter);


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

template <uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    edge_link_condition(const rxmesh::Context         context,
                        rxmesh::EdgeAttribute<int8_t> edge_link)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    Query<blockThreads> query(context);
    const PatchInfo&    patch_info = query.get_patch_info();

    Bitmask v0_mask(patch_info.num_vertices[0], shrd_alloc);
    Bitmask v1_mask(patch_info.num_vertices[0], shrd_alloc);

    query.prologue<Op::EV>(block, shrd_alloc);
    block.sync();

    __shared__ int s_num_shared_one_ring;


    const uint16_t num_edges = patch_info.num_edges[0];

    for (uint16_t e = 0; e < num_edges; ++e) {

        if (patch_info.is_owned(LocalEdgeT(e))) {
            // the edge two end vertices
            const VertexIterator iter =
                query.template get_iterator<VertexIterator>(e);

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
                    if (eh.local_id() == e &&
                        eh.patch_id() == patch_info.patch_id) {
                        return;
                    }
                    const VertexIterator v_iter =
                        query.template get_iterator<VertexIterator>(
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
            if (threadIdx.x == 0) {
                edge_link(EdgeHandle(patch_info.patch_id, e)) =
                    s_num_shared_one_ring;
            }
        }
    }
}
void link_condition(rxmesh::RXMeshDynamic&         rx,
                    rxmesh::EdgeAttribute<int8_t>* edge_link)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 384;

    edge_link->reset(0, DEVICE);

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::EV},
                         launch_box,
                         (void*)edge_link_condition<blockThreads>,
                         false,
                         false,
                         false,
                         false,
                         [&](uint32_t v, uint32_t e, uint32_t f) {
                             return 2 * detail::mask_num_bytes(v) +
                                    2 * ShmemAllocator::default_alignment;
                         });

    GPUTimer app_timer;
    app_timer.start();
    edge_link_condition<blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), *edge_link);
    app_timer.stop();

    RXMESH_INFO("Link Condition time {} (ms)", app_timer.elapsed_millis());
}