#pragma once
#include <cuda_profiler_api.h>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/rxmesh_dynamic.h"

#include "util.cuh"

template <typename T, uint32_t blockThreads>
__global__ static void edge_split(rxmesh::Context                   context,
                                  const rxmesh::VertexAttribute<T>  coords,
                                  rxmesh::EdgeAttribute<EdgeStatus> edge_status,
                                  rxmesh::VertexAttribute<bool>     v_boundary,
                                  const T high_edge_len_sq,
                                  const T low_edge_len_sq,
                                  int     iteration)
{
    // EV for calc edge len

    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();

    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, true);


    if (cavity.patch_id() == INVALID32) {
        return;
    }
    Bitmask is_updated(cavity.patch_info().edges_capacity, shrd_alloc);
    is_updated.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    auto should_split = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        // iter[0] and iter[2] are the edge two vertices
        // iter[1] and iter[3] are the two opposite vertices
        /*
            0
          / | \
         3  |  1
         \  |  /
            2
        */
        assert(iter.size() == 4);

        if (edge_status(eh) == UNSEEN) {
            const VertexHandle va = iter[0];
            const VertexHandle vb = iter[2];

            const VertexHandle vc = iter[1];
            const VertexHandle vd = iter[3];

            // don't split boundary edges
            if (!vc.is_valid() || !vd.is_valid() || !va.is_valid() ||
                !vb.is_valid()) {
                edge_status(eh) = SKIP;
                return;
            }

            // don't touch boundary vertices
            if (v_boundary(va) || v_boundary(vb) || v_boundary(vc) ||
                v_boundary(vd)) {
                return;
            }

            // degenerate cases
            if (va == vb || vb == vc || vc == va || va == vd || vb == vd ||
                vc == vd) {
                edge_status(eh) = SKIP;
                return;
            }

            const vec3<T> pa = coords.to_glm<3>(va);
            const vec3<T> pb = coords.to_glm<3>(vb);

            const T edge_len = glm::distance2(pa, pb);

            // if (edge_len > 1.01) {
            //     cavity.create(eh);
            // } else {
            //     edge_status(eh) = SKIP;
            // }


            if (edge_len > high_edge_len_sq) {

                vec3<T> p_new = (pa + pb) * T(0.5);

                vec3<T> pc = coords.to_glm<3>(vc);
                vec3<T> pd = coords.to_glm<3>(vd);

                T min_new_edge_len = std::numeric_limits<T>::max();

                min_new_edge_len =
                    std::min(min_new_edge_len, glm::distance2(p_new, pa));
                min_new_edge_len =
                    std::min(min_new_edge_len, glm::distance2(p_new, pb));
                min_new_edge_len =
                    std::min(min_new_edge_len, glm::distance2(p_new, pc));
                min_new_edge_len =
                    std::min(min_new_edge_len, glm::distance2(p_new, pd));

                if (min_new_edge_len >= low_edge_len_sq) {
                    cavity.create(eh);
                } else {
                    edge_status(eh) = SKIP;
                }
            } else {
                edge_status(eh) = SKIP;
            }
        }
    };

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_split);
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block, shrd_alloc, coords, edge_status, v_boundary)) {


        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);

            const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
            const VertexHandle v1 = cavity.get_cavity_vertex(c, 2);

            const VertexHandle new_v = cavity.add_vertex();

            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);
            // printf("\n src = %u, %u", src.patch_id(), src.local_id());

            if (new_v.is_valid()) {

                coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) * T(0.5);
                coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) * T(0.5);
                coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) * T(0.5);

#ifndef NDEBUG
                // sanity check: we don't introduce small edges

                const vec3<T> p_new = coords.to_glm<3>(new_v);
                for (int i = 0; i < 4; ++i) {

                    const VertexHandle v = cavity.get_cavity_vertex(c, i);

                    const vec3<T> p = coords.to_glm<3>(v);

                    assert(glm::distance2(p_new, p) >= low_edge_len_sq);
                }
#endif

                DEdgeHandle e0 =
                    cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));
                const DEdgeHandle e_init = e0;

                if (e0.is_valid()) {
                    is_updated.set(e0.local_id(), true);
                    //::atomicAdd(&s_num_splits, 1);
                    for (uint16_t i = 0; i < size; ++i) {
                        const DEdgeHandle e = cavity.get_cavity_edge(c, i);

                        // is_updated.set(e.local_id(), true);

                        const DEdgeHandle e1 =
                            (i == size - 1) ?
                                e_init.get_flip_dedge() :
                                cavity.add_edge(
                                    cavity.get_cavity_vertex(c, i + 1), new_v);
                        if (!e1.is_valid()) {
                            break;
                        }

                        is_updated.set(e1.local_id(), true);

                        const FaceHandle f = cavity.add_face(e0, e, e1);
                        if (!f.is_valid()) {
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
            if (is_updated(eh.local_id())) {
                edge_status(eh) = ADDED;
            }
        });
    }
}


template <typename T>
inline void split_long_edges(rxmesh::RXMeshDynamic&             rx,
                             rxmesh::VertexAttribute<T>*        coords,
                             rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                             rxmesh::VertexAttribute<bool>*     v_boundary,
                             const T                           high_edge_len_sq,
                             const T                           low_edge_len_sq,
                             rxmesh::Timers<rxmesh::GPUTimer>& timers,
                             int*                              d_buffer)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;


    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    int num_outer_iter = 0;
    int num_inner_iter = 0;

    LaunchBox<blockThreads> lb;
    rx.update_launch_box({Op::EVDiamond},
                         lb,
                         (void*)edge_split<T, blockThreads>,
                         true,
                         false,
                         false,
                         false,
                         [&](uint32_t v, uint32_t e, uint32_t f) {
                             return detail::mask_num_bytes(e) +
                                    ShmemAllocator::default_alignment;
                         });

    timers.start("SplitTotal");

    // float max_v_lf(0), min_v_lf(5000.0), max_e_lf(0), min_e_lf(5000.0),
    //     max_f_lf(0), min_f_lf(5000.0);

    float prv_time = 0;

    while (true) {
        num_outer_iter++;
        rx.reset_scheduler();

        while (!rx.is_queue_empty()) {
            num_inner_iter++;

            // RXMESH_INFO(" Queue size = {}",
            //             rx.get_context().m_patch_scheduler.size());

            timers.start("Split");
            edge_split<T, blockThreads>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(),
                    *coords,
                    *edge_status,
                    *v_boundary,
                    high_edge_len_sq,
                    low_edge_len_sq,
                    num_inner_iter % rx.get_num_colors());

            timers.stop("Split");
            // break;

            // RXMESH_INFO(" T = {}", timers.elapsed_millis("Split") -
            // prv_time);
            // prv_time = timers.elapsed_millis("Split");

            timers.start("SplitCleanup");
            rx.cleanup();
            timers.stop("SplitCleanup");

            timers.start("SplitSlice");
            rx.slice_patches(*coords, *edge_status, *v_boundary);
            timers.stop("SplitSlice");

            timers.start("SplitCleanup");
            rx.cleanup();
            timers.stop("SplitCleanup");

            bool show = false;
            if (show) {

                rx.update_host();
                EXPECT_TRUE(rx.validate());

                // float max_v(0), min_v(5000.0), max_e(0), min_e(5000.0),
                //     max_f(0), min_f(5000.0);
                // for (int p = 0; p < rx.get_num_patches(); ++p) {
                //     const PatchInfo& pi = rx.get_patch(p);
                //
                //     float lf_v = pi.lp_v.compute_load_factor();
                //     float lf_e = pi.lp_e.compute_load_factor();
                //     float lf_f = pi.lp_f.compute_load_factor();
                //
                //     max_v = std::max(max_v, lf_v);
                //     min_v = std::min(min_v, lf_v);
                //
                //     max_e = std::max(max_e, lf_e);
                //     min_e = std::min(min_e, lf_e);
                //
                //     max_f = std::max(max_f, lf_f);
                //     min_f = std::min(min_f, lf_f);
                // }
                //
                // RXMESH_INFO(
                //     "LF min_v={}, max_v={}, min_e={}, max_e={}, min_f={}, "
                //     "max_f={},",
                //     min_v,
                //     max_v,
                //     min_e,
                //     max_e,
                //     min_f,
                //     max_f);
                //
                // max_v_lf = std::max(max_v, max_v_lf);
                // min_v_lf = std::min(min_v, min_v_lf);
                //
                // max_e_lf = std::max(max_e, max_e_lf);
                // min_e_lf = std::min(min_e, min_e_lf);
                //
                // max_f_lf = std::max(max_f, max_f_lf);
                // min_f_lf = std::min(min_f, min_f_lf);


                // screen_shot(rx, coords, "Split");
                // stats(rx);

                // RXMESH_INFO(" ");
                // RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
                // RXMESH_INFO("#Edges {}", rx.get_num_edges());
                // RXMESH_INFO("#Faces {}", rx.get_num_faces());
                // RXMESH_INFO("#Patches {}", rx.get_num_patches());

                // coords->move(DEVICE, HOST);
                // edge_status->move(DEVICE, HOST);
                // rx.update_polyscope();
                // auto ps_mesh = rx.get_polyscope_mesh();
                // ps_mesh->updateVertexPositions(*coords);
                // ps_mesh->setEnabled(false);
                //
                // ps_mesh->addEdgeScalarQuantity("EdgeStatus", *edge_status);
                //
                // rx.render_vertex_patch();
                // rx.render_edge_patch();
                // rx.render_face_patch()->setEnabled(false);

                // polyscope::show();
            }
        }
        // break;
        // RXMESH_INFO("num_outer_iter {}", num_outer_iter);
        // RXMESH_INFO("num_inner_iter {}", num_inner_iter);


        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }
    timers.stop("SplitTotal");

    // RXMESH_INFO(
    //     "LF min_v_lf={}, max_v_lf={}, min_e_lf={}, max_e_lf={}, min_f_lf={},
    //     " "max_f_lf={},", min_v_lf, max_v_lf, min_e_lf, max_e_lf, min_f_lf,
    //     max_f_lf);

    // RXMESH_INFO("num_outer_iter {}", num_outer_iter);
    // RXMESH_INFO("num_inner_iter {}", num_inner_iter);
    // RXMESH_INFO("Split total time {} (ms)",
    //             timers.elapsed_millis("SplitTotal"));
    // RXMESH_INFO("Split time {} (ms)", timers.elapsed_millis("Split"));
    // RXMESH_INFO("Split slice time {} (ms)",
    //             timers.elapsed_millis("SplitSlice"));
    // RXMESH_INFO("Split cleanup time {} (ms)",
    //             timers.elapsed_millis("SplitCleanup"));
}
