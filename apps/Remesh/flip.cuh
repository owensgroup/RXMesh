#pragma once
#include <cuda_profiler_api.h>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_dynamic.h"

#include "util.cuh"


template <uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    compute_valence(rxmesh::Context                        context,
                    const rxmesh::VertexAttribute<uint8_t> v_valence)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    Query<blockThreads> query(context);
    query.compute_vertex_valence(block, shrd_alloc);
    block.sync();

    for_each_vertex(query.get_patch_info(), [&](VertexHandle vh) {
        v_valence(vh) = query.vertex_valence(vh);
    });
}


template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    edge_flip(rxmesh::Context                        context,
              const rxmesh::VertexAttribute<T>       coords,
              const rxmesh::VertexAttribute<uint8_t> v_valence,
              rxmesh::EdgeAttribute<EdgeStatus>      edge_status,
              int*                                   d_buffer)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, false, false);


    if (cavity.patch_id() == INVALID32) {
        return;
    }

    // a bitmask that indicates which edge we want to flip
    // we also used it to mark the new edges
    Bitmask edge_mask(cavity.patch_info().edges_capacity[0], shrd_alloc);
    edge_mask.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // we use this bitmask to mark the other end of to-be-collapse edge during
    // checking for the link condition
    Bitmask v0_mask(cavity.patch_info().num_vertices[0], shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().num_vertices[0], shrd_alloc);

    // precompute EVDiamond
    Query<blockThreads> query(context, cavity.patch_id());
    query.prologue<Op::EVDiamond>(block, shrd_alloc);
    block.sync();


    // 1. mark edge that we want to flip
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        assert(eh.local_id() < cavity.patch_info().num_edges[0]);

        const VertexIterator iter =
            query.template get_iterator<VertexIterator>(eh.local_id());

        // only if the edge is not seen before and its not a boundary edge
        if (edge_status(eh) == UNSEEN && iter[1].is_valid() &&
            iter[3].is_valid()) {

            if (iter[0] == iter[1] || iter[0] == iter[2] ||
                iter[0] == iter[3] || iter[1] == iter[2] ||
                iter[1] == iter[3] || iter[2] == iter[3]) {
                return;
            }

            // iter[0] and iter[2] are the edge two vertices
            // iter[1] and iter[3] are the two opposite vertices


            // since we only deal with closed meshes without boundaries
            constexpr int target_valence = 6;


            const int valence_a = v_valence(iter[0]);
            const int valence_b = v_valence(iter[2]);
            const int valence_c = v_valence(iter[1]);
            const int valence_d = v_valence(iter[3]);

            // clang-format off
                const int deviation_pre =
                    (valence_a - target_valence) * (valence_a - target_valence) +
                    (valence_b - target_valence) * (valence_b - target_valence) +
                    (valence_c - target_valence) * (valence_c - target_valence) +
                    (valence_d - target_valence) * (valence_d - target_valence);
            // clang-format on

            // clang-format off
                const int deviation_post =
                    (valence_a - 1 - target_valence)*(valence_a - 1 - target_valence) +
                    (valence_b - 1 - target_valence)*(valence_b - 1 - target_valence) +
                    (valence_c + 1 - target_valence)*(valence_c + 1 - target_valence) +
                    (valence_d + 1 - target_valence)*(valence_d + 1 - target_valence);
            // clang-format on

            if (deviation_pre > deviation_post) {
                edge_mask.set(eh.local_id(), true);
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

    if (cavity.prologue(block, shrd_alloc, coords, edge_status)) {

        edge_mask.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            DEdgeHandle new_edge = cavity.add_edge(
                cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));


            if (new_edge.is_valid()) {
                edge_mask.set(new_edge.local_id(), true);
                cavity.add_face(cavity.get_cavity_edge(c, 0),
                                new_edge,
                                cavity.get_cavity_edge(c, 3));


                cavity.add_face(cavity.get_cavity_edge(c, 1),
                                cavity.get_cavity_edge(c, 2),
                                new_edge.get_flip_dedge());
            }
        });
    }
    block.sync();

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
__global__ static void __launch_bounds__(blockThreads)
    edge_flip_1(rxmesh::Context                        context,
                const rxmesh::VertexAttribute<T>       coords,
                const rxmesh::VertexAttribute<uint8_t> v_valence,
                rxmesh::EdgeAttribute<EdgeStatus>      edge_status,
                int*                                   d_buffer)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, false, false);


    //__shared__ int s_num_flips;
    // if (threadIdx.x == 0) {
    //    s_num_flips = 0;
    //}
    if (cavity.patch_id() == INVALID32) {
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
    Bitmask e_flip(cavity.patch_info().num_edges[0], shrd_alloc);
    e_flip.reset(block);


    auto should_flip = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices


        // we use the local index since we are only interested in the
        // valence which computed on the local index space
        if (edge_status(eh) == UNSEEN) {
            if (iter[1].is_valid() && iter[3].is_valid() &&
                iter[0].is_valid() && iter[2].is_valid()) {

                if (iter[0] == iter[1] || iter[0] == iter[2] ||
                    iter[0] == iter[3] || iter[1] == iter[2] ||
                    iter[1] == iter[3] || iter[2] == iter[3]) {
                    edge_status(eh) = SKIP;
                    return;
                }

                // since we only deal with closed meshes without boundaries
                constexpr int target_valence = 6;


                const int valence_a = v_valence(iter[0]);
                const int valence_b = v_valence(iter[2]);
                const int valence_c = v_valence(iter[1]);
                const int valence_d = v_valence(iter[3]);

                // clang-format off
                const int deviation_pre =
                    (valence_a - target_valence) * (valence_a - target_valence) +
                    (valence_b - target_valence) * (valence_b - target_valence) +
                    (valence_c - target_valence) * (valence_c - target_valence) +
                    (valence_d - target_valence) * (valence_d - target_valence);
                // clang-format on

                // clang-format off
                const int deviation_post =
                    (valence_a - 1 - target_valence)*(valence_a - 1 - target_valence) +
                    (valence_b - 1 - target_valence)*(valence_b - 1 - target_valence) +
                    (valence_c + 1 - target_valence)*(valence_c + 1 - target_valence) +
                    (valence_d + 1 - target_valence)*(valence_d + 1 - target_valence);
                // clang-format on

                if (deviation_pre > deviation_post) {
                    uint16_t v_c(iter.local(1)), v_d(iter.local(3));

                    bool added = false;

                    if (iter[1].patch_id() == cavity.patch_id()) {
                        uint16_t ret =
                            ::atomicCAS(v_info + 2 * v_c, INVALID16, v_d);
                        if (ret == INVALID16) {
                            added               = true;
                            v_info[2 * v_c + 1] = eh.local_id();
                            e_flip.set(eh.local_id(), true);
                        }
                    }

                    if (iter[3].patch_id() == cavity.patch_id() && !added) {
                        uint16_t ret =
                            ::atomicCAS(v_info + 2 * v_d, INVALID16, v_c);
                        if (ret == INVALID16) {
                            v_info[2 * v_d + 1] = eh.local_id();
                            e_flip.set(eh.local_id(), true);
                        }
                    }
                }
            } else {
                edge_status(eh) = SKIP;
            }
        }
    };

    // 1. mark edge that we want to flip
    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_flip);
    block.sync();


    // 2. make sure that the two vertices opposite to a flipped edge are not
    // connected
    auto check_edges = [&](const VertexHandle& vh, const VertexIterator& iter) {
        uint16_t opposite_v = v_info[2 * vh.local_id()];
        if (opposite_v != INVALID16) {
            bool is_valid = true;
            for (uint16_t v = 0; v < iter.size(); ++v) {
                if (iter.local(v) == opposite_v) {
                    is_valid = false;
                    break;
                }
            }
            if (!is_valid) {
                e_flip.reset(v_info[2 * vh.local_id() + 1], true);
            }
        }
    };
    query.dispatch<Op::VV>(block, shrd_alloc, check_edges);
    block.sync();

    // 3. create cavity for the surviving edges
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (e_flip(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = SKIP;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block, shrd_alloc, coords, edge_status)) {

        is_updated.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            //::atomicAdd(&s_num_flips, 1);
            assert(size == 4);

            DEdgeHandle new_edge = cavity.add_edge(
                cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));


            if (new_edge.is_valid()) {
                is_updated.set(new_edge.local_id(), true);
                cavity.add_face(cavity.get_cavity_edge(c, 0),
                                new_edge,
                                cavity.get_cavity_edge(c, 3));


                cavity.add_face(cavity.get_cavity_edge(c, 1),
                                cavity.get_cavity_edge(c, 2),
                                new_edge.get_flip_dedge());
            }
        });
    }

    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        // if (threadIdx.x == 0) {
        //     ::atomicAdd(d_buffer, s_num_flips);
        // }
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id())) {
                edge_status(eh) = ADDED;
            }
        });
    }
}


template <typename T>
inline void equalize_valences(rxmesh::RXMeshDynamic&             rx,
                              rxmesh::VertexAttribute<T>*        coords,
                              rxmesh::VertexAttribute<uint8_t>*  v_valence,
                              rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                              rxmesh::EdgeAttribute<int8_t>*     edge_link,
                              rxmesh::Timers<rxmesh::GPUTimer>   timers,
                              int*                               d_buffer)
{

    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    // int   num_flips      = 0;
    int num_outer_iter = 0;
    int num_inner_iter = 0;

    timers.start("FlipTotal");
    while (true) {
        num_outer_iter++;
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {
            // RXMESH_INFO(" Queue size = {}",
            //             rx.get_context().m_patch_scheduler.size());
            num_inner_iter++;
            LaunchBox<blockThreads> launch_box;

            rx.update_launch_box({},
                                 launch_box,
                                 (void*)compute_valence<blockThreads>,
                                 false,
                                 false,
                                 true);

            timers.start("Flip");
            compute_valence<blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(), *v_valence);

            // link_condition(rx, edge_link);

            rx.update_launch_box(
                {Op::EVDiamond, Op::VV},
                launch_box,
                //(void*)edge_flip<T, blockThreads>,
                (void*)edge_flip_1<T, blockThreads>,
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

            edge_flip_1<T, blockThreads>
                <<<launch_box.blocks,  // DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                *v_valence,
                                                *edge_status,
                                                //*edge_link,
                                                d_buffer);
            timers.stop("Flip");

            timers.start("FlipCleanup");
            rx.cleanup();
            timers.stop("FlipCleanup");

            timers.start("FlipSlice");
            rx.slice_patches(*coords, *edge_status /*,edge_link*/);
            timers.stop("FlipSlice");

            timers.start("FlipCleanup");
            rx.cleanup();
            timers.stop("FlipCleanup");
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
        // RXMESH_INFO("num_flips {}, time {}",
        //             num_flips,
        //             app_time + slice_time + cleanup_time);
    }
    timers.stop("FlipTotal");

    // RXMESH_INFO("total num_flips {}", num_flips);
    RXMESH_INFO("num_outer_iter {}", num_outer_iter);
    RXMESH_INFO("num_inner_iter {}", num_inner_iter);
    RXMESH_INFO("Flip total time {} (ms)", timers.elapsed_millis("FlipTotal"));
    RXMESH_INFO("Flip time {} (ms)", timers.elapsed_millis("Flip"));
    RXMESH_INFO("Flip slice time {} (ms)", timers.elapsed_millis("FlipSlice"));
    RXMESH_INFO("Flip cleanup time {} (ms)",
                timers.elapsed_millis("FlipCleanup"));
}
