#pragma once
#include <cuda_profiler_api.h>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

#include "util.cuh"


template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    edge_collapse(rxmesh::Context                   context,
                  const rxmesh::VertexAttribute<T>  coords,
                  rxmesh::EdgeAttribute<EdgeStatus> edge_status,
                  const T                           low_edge_len_sq,
                  const T                           high_edge_len_sq,
                  int*                              d_buffer)
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

    // a bitmask that indicates which edge we want to flip
    // we also use it to mark updated edges (for edge_status)
    Bitmask edge_mask(cavity.patch_info().edges_capacity[0], shrd_alloc);
    edge_mask.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // we use this bitmask to mark the other end of to-be-collapse edge during
    // checking for the link condition
    Bitmask v0_mask(cavity.patch_info().num_vertices[0], shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().num_vertices[0], shrd_alloc);

    // Precompute EV
    Query<blockThreads> query(context, pid);
    query.prologue<Op::EVDiamond>(block, shrd_alloc);
    block.sync();

    // 1. mark edge that we want to collapse based on the edge length
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);

        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices
        /*
            0
          / | \
         3  |  1
         \  |  /
            2
        */

        if (edge_status(eh) == UNSEEN /*&& edge_link(eh) == 2*/) {
            const VertexIterator iter =
                query.template get_iterator<VertexIterator>(eh.local_id());

            assert(iter.size() == 4);

            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[2];

            const VertexHandle v2 = iter[1];
            const VertexHandle v3 = iter[3];

            // don't collapse boundary edges
            if (v2.is_valid() && v3.is_valid()) {

                // degenerate cases
                if (v0 == v1 || v0 == v2 || v0 == v3 || v1 == v2 || v1 == v3 ||
                    v2 == v3) {
                    return;
                }
                const vec3<T> p0          = coords.to_glm<3>(v0);
                const vec3<T> p1          = coords.to_glm<3>(v1);
                const T       edge_len_sq = glm::distance2(p0, p1);

                if (edge_len_sq < low_edge_len_sq) {
                    edge_mask.set(eh.local_id(), true);
                }
            }
        }
    });
    block.sync();


    // 2. check link condition
    link_condition(
        block, cavity.patch_info(), query, edge_mask, v0_mask, v1_mask, 0, 2);
    block.sync();


    // 3. create cavity for the surviving edges
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (edge_mask(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = SKIP;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // create the cavity
    if (cavity.prologue(block, shrd_alloc, coords, edge_status)) {

        edge_mask.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            //::atomicAdd(&s_num_collapses, 1);
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            VertexHandle v0, v1;

            cavity.get_vertices(src, v0, v1);

            const vec3<T> p0 = coords.to_glm<3>(v0);
            const vec3<T> p1 = coords.to_glm<3>(v1);

            const vec3<T> new_p((p0[0] + p1[0]) * T(0.5),
                                (p0[1] + p1[1]) * T(0.5),
                                (p0[2] + p1[2]) * T(0.5));

            // check if we will create a long edge
            bool long_edge = false;

            for (uint16_t i = 0; i < size; ++i) {


                const VertexHandle vvv = cavity.get_cavity_vertex(c, i);

                const vec3<T> vp = coords.to_glm<3>(vvv);

                const T edge_len_sq = glm::distance2(vp, new_p);

                if (edge_len_sq >= low_edge_len_sq) {
                    long_edge = true;
                    break;
                }
            }

            if (long_edge) {
                // roll back
                cavity.recover(src);

                // mark this edge as SKIP because 1) if all cavities in this
                // patch are successful, then we want to indicate that this
                // edge is okay and should not be attempted again
                // 2) if we have to rollback all changes in this patch, we still
                // don't want to attempt this edge since we know that it creates
                // short edges
                edge_status(src) = SKIP;
            } else {

                const VertexHandle new_v = cavity.add_vertex();

                if (new_v.is_valid()) {

                    coords(new_v, 0) = new_p[0];
                    coords(new_v, 1) = new_p[1];
                    coords(new_v, 2) = new_p[2];

                    DEdgeHandle e0 =
                        cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));

                    if (e0.is_valid()) {
                        edge_mask.set(e0.local_id(), true);

                        const DEdgeHandle e_init = e0;

                        for (uint16_t i = 0; i < size; ++i) {
                            const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                            // edge_mask.set(e.local_id(), true);

                            const VertexHandle v_end =
                                cavity.get_cavity_vertex(c, (i + 1) % size);

                            const DEdgeHandle e1 =
                                (i == size - 1) ?
                                    e_init.get_flip_dedge() :
                                    cavity.add_edge(
                                        cavity.get_cavity_vertex(c, i + 1),
                                        new_v);

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
            }
        });
    }
    block.sync();

    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        // if (threadIdx.x == 0) {
        //    ::atomicAdd(d_buffer, s_num_collapses);
        //}
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (edge_mask(eh.local_id()) || cavity.is_recovered(eh)) {
                edge_status(eh) = ADDED;
            }
        });
    }
}

template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    edge_collapse_1(rxmesh::Context                   context,
                    const rxmesh::VertexAttribute<T>  coords,
                    rxmesh::EdgeAttribute<EdgeStatus> edge_status,
                    const T                           low_edge_len_sq,
                    const T                           high_edge_len_sq,
                    int*                              d_buffer)
{

    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager<blockThreads, CavityOp::EV> cavity(
        block, context, shrd_alloc, true);

    const uint32_t pid = cavity.patch_id();


    //__shared__ int s_num_collapses;
    // if (threadIdx.x == 0) {
    //    s_num_collapses = 0;
    //}

    if (pid == INVALID32) {
        return;
    }


    Bitmask is_updated(cavity.patch_info().edges_capacity[0], shrd_alloc);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();


    // for each edge we want to flip, we its id in one of its opposite vertices
    // along with the other opposite vertex
    uint16_t* v_info =
        shrd_alloc.alloc<uint16_t>(2 * cavity.patch_info().num_vertices[0]);
    fill_n<blockThreads>(
        v_info, 2 * cavity.patch_info().num_vertices[0], uint16_t(INVALID16));

    // a bitmask that indicates which edge we want to flip
    Bitmask e_collapse(cavity.patch_info().num_edges[0], shrd_alloc);
    e_collapse.reset(block);
    block.sync();

    auto should_collapse = [&](const EdgeHandle&     eh,
                               const VertexIterator& iter) {
        if (edge_status(eh) == UNSEEN) {

            assert(iter.size() == 4);

            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[2];

            const VertexHandle v2 = iter[1];
            const VertexHandle v3 = iter[3];

            // don't collapse boundary edges
            if (v0.is_valid() && v1.is_valid() && v2.is_valid() &&
                v3.is_valid()) {

                // degenerate cases
                if (v0 == v1 || v0 == v2 || v0 == v3 || v1 == v2 || v1 == v3 ||
                    v2 == v3) {
                    edge_status(eh) = SKIP;
                    return;
                }

                const vec3<T> p0 = coords.to_glm<3>(v0);
                const vec3<T> p1 = coords.to_glm<3>(v1);

                const T edge_len_sq = glm::distance2(p0, p1);

                if (edge_len_sq < low_edge_len_sq) {

                    const uint16_t c0(iter.local(0)), c1(iter.local(2));

                    uint16_t ret = ::atomicCAS(v_info + 2 * c0, INVALID16, c1);
                    if (ret == INVALID16) {
                        v_info[2 * c0 + 1] = eh.local_id();
                        e_collapse.set(eh.local_id(), true);
                    } else {
                        ret = ::atomicCAS(v_info + 2 * c1, INVALID16, c0);
                        if (ret == INVALID16) {
                            v_info[2 * c1 + 1] = eh.local_id();
                            e_collapse.set(eh.local_id(), true);
                        }
                    }
                }
            }
        }
    };

    // 1. mark edge that we want to collapse based on the edge lenght
    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_collapse);
    block.sync();


    auto check_edges = [&](const VertexHandle& vh, const VertexIterator& iter) {
        uint16_t opposite_v = v_info[2 * vh.local_id()];
        if (opposite_v != INVALID16) {
            int num_shared_v = 0;

            const VertexIterator opp_iter =
                query.template get_iterator<VertexIterator>(opposite_v);

            for (uint16_t v = 0; v < iter.size(); ++v) {

                for (uint16_t ov = 0; ov < opp_iter.size(); ++ov) {
                    if (iter.local(v) == opp_iter.local(ov)) {
                        num_shared_v++;
                        break;
                    }
                }
            }

            if (num_shared_v > 2) {
                e_collapse.reset(v_info[2 * vh.local_id() + 1], true);
            }
        }
    };
    // 2. make sure that the two vertices opposite to a flipped edge are not
    // connected
    query.dispatch<Op::VV>(
        block,
        shrd_alloc,
        check_edges,
        [](VertexHandle) { return true; },
        false,
        true);
    block.sync();


    // 3. create cavity for the surviving edges
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (e_collapse(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = SKIP;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // create the cavity
    if (cavity.prologue(block, shrd_alloc, coords, edge_status)) {

        is_updated.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            //::atomicAdd(&s_num_collapses, 1);
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);

            VertexHandle v0, v1;

            cavity.get_vertices(src, v0, v1);

            const vec3<T> p0 = coords.to_glm<3>(v0);
            const vec3<T> p1 = coords.to_glm<3>(v1);

            const vec3<T> new_p((p0[0] + p1[0]) * T(0.5),
                                (p0[1] + p1[1]) * T(0.5),
                                (p0[2] + p1[2]) * T(0.5));

            // check if we will create a long edge
            bool long_edge = false;

            for (uint16_t i = 0; i < size; ++i) {
                const VertexHandle vvv = cavity.get_cavity_vertex(c, i);

                const vec3<T> vp = coords.to_glm<3>(vvv);               

                const T edge_len_sq = glm::distance2(vp, new_p);
                if (edge_len_sq > high_edge_len_sq) {
                    long_edge = true;
                    break;
                }
            }

            if (long_edge) {
                // roll back
                cavity.recover(src);

                // mark this edge as SKIP because 1) if all cavities in this
                // patch are successful, then we want to indicate that this
                // edge is okay and should not be attempted again
                // 2) if we have to rollback all changes in this patch, we still
                // don't want to attempt this edge since we know that it creates
                // short edges
                edge_status(src) = SKIP;
            } else {

                const VertexHandle new_v = cavity.add_vertex();

                if (new_v.is_valid()) {

                    coords(new_v, 0) = new_p[0];
                    coords(new_v, 1) = new_p[1];
                    coords(new_v, 2) = new_p[2];

                    DEdgeHandle e0 =
                        cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));

                    if (e0.is_valid()) {
                        is_updated.set(e0.local_id(), true);

                        const DEdgeHandle e_init = e0;

                        for (uint16_t i = 0; i < size; ++i) {
                            const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                            // is_updated.set(e.local_id(), true);

                            const VertexHandle v_end =
                                cavity.get_cavity_vertex(c, (i + 1) % size);

                            const DEdgeHandle e1 =
                                (i == size - 1) ?
                                    e_init.get_flip_dedge() :
                                    cavity.add_edge(
                                        cavity.get_cavity_vertex(c, i + 1),
                                        new_v);

                            if (!e1.is_valid()) {
                                break;
                            }

                            if (i != size - 1) {
                                is_updated.set(e1.local_id(), true);
                            }

                            const FaceHandle new_f = cavity.add_face(e0, e, e1);

                            if (!new_f.is_valid()) {
                                break;
                            }
                            e0 = e1.get_flip_dedge();
                        }
                    }
                }
            }
        });
    }
    block.sync();

    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        // if (threadIdx.x == 0) {
        //     ::atomicAdd(d_buffer, s_num_collapses);
        // }
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id()) || cavity.is_recovered(eh)) {
                edge_status(eh) = ADDED;
            }
        });
    }
}


template <typename T>
inline void collapse_short_edges(rxmesh::RXMeshDynamic&             rx,
                                 rxmesh::VertexAttribute<T>*        coords,
                                 rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                                 rxmesh::EdgeAttribute<int8_t>*     edge_link,
                                 const T low_edge_len_sq,
                                 const T high_edge_len_sq,
                                 rxmesh::Timers<rxmesh::GPUTimer> timers,
                                 int*                             d_buffer)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();


    int num_outer_iter = 0;
    int num_inner_iter = 0;
    // int   num_collapses  = 0;

    timers.start("CollapseTotal");
    while (true) {
        num_outer_iter++;
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {
            // RXMESH_INFO(" Queue size = {}",
            //             rx.get_context().m_patch_scheduler.size());
            num_inner_iter++;

            // link_condition(rx, edge_link);

            LaunchBox<blockThreads> launch_box;
            rx.update_launch_box(
                {Op::EVDiamond, Op::VV},
                launch_box,
                (void*)edge_collapse_1<T, blockThreads>,
                true,
                false,
                false,
                false,
                [&](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * v * sizeof(uint16_t) +
                           2 * ShmemAllocator::default_alignment;
                    // 2 * detail::mask_num_bytes(v) +
                    // 3 * ShmemAllocator::default_alignment;
                });

            // CUDA_ERROR(cudaMemset(d_buffer, 0, sizeof(int)));

            timers.start("Collapse");
            edge_collapse_1<T, blockThreads>
                <<<launch_box.blocks,  // DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                *edge_status,
                                                //*edge_link,
                                                low_edge_len_sq,
                                                high_edge_len_sq,
                                                d_buffer);
            timers.stop("Collapse");

            timers.start("CollapseCleanup");
            rx.cleanup();
            timers.stop("CollapseCleanup");

            timers.start("CollapseSlice");
            rx.slice_patches(*coords, *edge_status /*, *edge_link */);
            timers.stop("CollapseSlice");

            timers.start("CollapseCleanup");
            rx.cleanup();
            timers.stop("CollapseCleanup");
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }
    timers.stop("CollapseTotal");

    // RXMESH_INFO("total num_collapses {}", num_collapses);
    RXMESH_INFO("num_outer_iter {}", num_outer_iter);
    RXMESH_INFO("num_inner_iter {}", num_inner_iter);
    RXMESH_INFO("Collapse total time {} (ms)",
                timers.elapsed_millis("CollapseTotal"));
    RXMESH_INFO("Collapse time {} (ms)", timers.elapsed_millis("Collapse"));
    RXMESH_INFO("Collapse slice time {} (ms)",
                timers.elapsed_millis("CollapseSlice"));
    RXMESH_INFO("Collapse cleanup time {} (ms)",
                timers.elapsed_millis("CollapseCleanup"));
}