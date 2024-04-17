#pragma once
#include "rxmesh/cavity_manager.cuh"

template <typename T, uint32_t blockThreads>
__global__ static void sec(rxmesh::Context                   context,
                           rxmesh::VertexAttribute<T>        coords,
                           const CostHistogram<T>            histo,
                           const int                         reduce_threshold,
                           rxmesh::EdgeAttribute<EdgeStatus> edge_status,
                           rxmesh::EdgeAttribute<T>          e_attr,
                           int*                              d_num_cavities)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager<blockThreads, CavityOp::EV> cavity(
        block, context, shrd_alloc, true);

    const uint32_t pid = cavity.patch_id();

    if (pid == INVALID32) {
        return;
    }

    // we first use this mask to set the edge we want to collapse (and then
    // filter them). Then after cavity.prologue, we reuse this bitmask to mark
    // the newly added edges
    Bitmask edge_mask(cavity.patch_info().edges_capacity[0], shrd_alloc);

    // we use this bitmask to mark the other end of to-be-collapse edge during
    // checking for the link condition
    Bitmask v0_mask(cavity.patch_info().num_vertices[0], shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().num_vertices[0], shrd_alloc);


    // Precompute EV
    Query<blockThreads> ev_query(context, pid);
    ev_query.prologue<Op::EV>(block, shrd_alloc);
    block.sync();

    // 1) mark edge we want to collapse
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);

        if (edge_status(eh) != UNSEEN) {
            return;
        }
        const VertexIterator iter =
            ev_query.template get_iterator<VertexIterator>(eh.local_id());

        const VertexHandle v0 = iter[0];
        const VertexHandle v1 = iter[1];

        const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
        const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));

        T len2 = glm::distance2(p0, p1);

        if (histo.get_bin(len2) <= reduce_threshold) {
            //::atomicAdd(d_num_cavities + 1, 1);
            // cavity.create(eh);
            edge_mask.set(eh.local_id(), true);
        }
    });
    block.sync();


    // 2) check edge link condition. Here, for each edge marked in edge_mask,
    // all threads in the block collaborate to check the edge link condition of
    // this edge
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
                cavity.patch_info(),
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
    block.sync();

    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);
        if (edge_mask(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = OKAY;
        }
    });
    block.sync();

    ev_query.epilogue(block, shrd_alloc);

    // create the cavity
    if (cavity.prologue(block, shrd_alloc, coords, edge_status, e_attr)) {

        // if (threadIdx.x == 0) {
        //     uint16_t num_actual_cavities = 0;
        //     for (int i = 0; i < cavity.m_s_active_cavity_bitmask.size(); ++i)
        //     {
        //         if (cavity.m_s_active_cavity_bitmask(i)) {
        //             num_actual_cavities++;
        //         }
        //     }
        //     ::atomicAdd(d_num_cavities, num_actual_cavities);
        // }
        edge_mask.reset(block);
        block.sync();

        // fill in the cavities
        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            // TODO handle boundary edges

            VertexHandle v0, v1;

            cavity.get_vertices(src, v0, v1);

            const VertexHandle new_v = cavity.add_vertex();

            if (new_v.is_valid()) {

                coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) * 0.5;
                coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) * 0.5;
                coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) * 0.5;


                DEdgeHandle e0 =
                    cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));

                e_attr(e0.get_edge_handle())++;

                if (e0.is_valid()) {
                    edge_mask.set(e0.local_id(), true);

                    const DEdgeHandle e_init = e0;

                    for (uint16_t i = 0; i < size; ++i) {
                        const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                        const VertexHandle v_end =
                            cavity.get_cavity_vertex(c, (i + 1) % size);

                        const DEdgeHandle e1 =
                            (i == size - 1) ?
                                e_init.get_flip_dedge() :
                                cavity.add_edge(
                                    cavity.get_cavity_vertex(c, i + 1), new_v);

                        if (!e1.is_valid()) {
                            break;
                        }

                        if (i != size - 1) {
                            edge_mask.set(e1.local_id(), true);
                        }

                        const FaceHandle new_f = cavity.add_face(e0, e, e1);

                        if (!new_f.is_valid()) {
                            break;
                        }
                        e0 = e1.get_flip_dedge();
                    }
                }
            }
        });
    }


    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (edge_mask(eh.local_id())) {
                edge_status(eh) = ADDED;
            }
        });
    }
}


template <typename T, uint32_t blockThreads>
__global__ static void compute_min_max_cost(
    rxmesh::Context                  context,
    const rxmesh::VertexAttribute<T> coords,
    CostHistogram<T>                 histo)
{
    using namespace rxmesh;

    auto edge_len = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        const VertexHandle v0 = iter[0];
        const VertexHandle v1 = iter[1];

        const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
        const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));

        T len2 = glm::distance2(p0, p1);

        atomicMin(histo.min_value(), len2);
        atomicMax(histo.max_value(), len2);
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, edge_len);
}

template <typename T, uint32_t blockThreads>
__global__ static void populate_histogram(
    rxmesh::Context                  context,
    const rxmesh::VertexAttribute<T> coords,
    CostHistogram<T>                 histo)
{
    using namespace rxmesh;

    auto edge_len = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        const VertexHandle v0 = iter[0];
        const VertexHandle v1 = iter[1];

        const Vec3<T> p0(coords(v0, 0), coords(v0, 1), coords(v0, 2));
        const Vec3<T> p1(coords(v1, 0), coords(v1, 1), coords(v1, 2));

        T len2 = glm::distance2(p0, p1);

        histo.insert(len2);
    };

    auto block = cooperative_groups::this_thread_block();

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, edge_len);
}