#pragma once

#define G_EIGENVALUE_RANK_RATIO 0.03

#include "util.cuh"

#include "frame_stepper.h"
#include "rxmesh/rxmesh_dynamic.h"
#include "simulation.h"

#include "rxmesh/util/report.h"

#include "collapser.cuh"
#include "flipper.cuh"
#include "noise.h"
#include "smoother.cuh"
#include "splitter.cuh"
#include "tracking_kernels.cuh"

int* d_buffer;
int  total_num_iter;

rxmesh::Timers<rxmesh::GPUTimer> timers;

rxmesh::VertexAttribute<int>* v_err;

template <typename T, uint32_t blockThreads>
__global__ void ckeck_kernel_fv(const rxmesh::Context            context,
                                const rxmesh::VertexAttribute<T> position,
                                rxmesh::VertexAttribute<int>     v_attr)
{
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto ch = [&](FaceHandle vf, VertexIterator& iter) {
        assert(!isnan(position(iter[0], 0)));
        assert(!isnan(position(iter[0], 1)));
        assert(!isnan(position(iter[0], 2)));

        assert(!isnan(position(iter[1], 0)));
        assert(!isnan(position(iter[1], 1)));
        assert(!isnan(position(iter[1], 2)));

        assert(!isnan(position(iter[2], 0)));
        assert(!isnan(position(iter[2], 1)));
        assert(!isnan(position(iter[2], 2)));

        bool all_zero =
            position(iter[0], 0) == 0 && position(iter[0], 1) == 0 &&
            position(iter[0], 2) == 0 && position(iter[1], 0) == 0 &&
            position(iter[1], 1) == 0 && position(iter[1], 2) == 0 &&
            position(iter[2], 0) == 0 && position(iter[2], 1) == 0 &&
            position(iter[2], 2) == 0;

        if (all_zero) {
            printf("\n v0= %f, %f, %f, v1= %f, %f, %f, v2= %f, %f, %f",
                   position(iter[0], 0),
                   position(iter[0], 1),
                   position(iter[0], 2),
                   position(iter[1], 0),
                   position(iter[1], 1),
                   position(iter[1], 2),
                   position(iter[2], 0),
                   position(iter[2], 1),
                   position(iter[2], 2));

            v_attr(iter[0]) = 1;
            v_attr(iter[1]) = 1;
            v_attr(iter[2]) = 1;
        }

        // assert(!all_zero);
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, ch);
}


template <typename T, uint32_t blockThreads>
__global__ void ckeck_kernel_ev(const rxmesh::Context            context,
                                const rxmesh::VertexAttribute<T> position,
                                rxmesh::VertexAttribute<int>     v_attr)
{
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();

    auto ch = [&](EdgeHandle eh, VertexIterator& iter) {
        assert(!isnan(position(iter[0], 0)));
        assert(!isnan(position(iter[0], 1)));
        assert(!isnan(position(iter[0], 2)));

        assert(!isnan(position(iter[1], 0)));
        assert(!isnan(position(iter[1], 1)));
        assert(!isnan(position(iter[1], 2)));

        bool all_zero =
            position(iter[0], 0) == 0 && position(iter[0], 1) == 0 &&
            position(iter[0], 2) == 0 && position(iter[1], 0) == 0 &&
            position(iter[1], 1) == 0 && position(iter[1], 2) == 0;


        if (all_zero) {
            printf("\n v0= %f, %f, %f, v1= %f, %f, %f,",
                   position(iter[0], 0),
                   position(iter[0], 1),
                   position(iter[0], 2),
                   position(iter[1], 0),
                   position(iter[1], 1),
                   position(iter[1], 2));

            v_attr(iter[0]) = 1;
            v_attr(iter[1]) = 1;
        }

        // assert(!all_zero);
    };

    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, ch);
}


template <typename T>
void check(rxmesh::RXMeshDynamic& rx, rxmesh::VertexAttribute<T>& position)
{

    using namespace rxmesh;

    CUDA_ERROR(cudaDeviceSynchronize());

    auto viz = [&]() {
        rx.update_host();

        v_err->move(DEVICE, HOST);

        int err = 0;


        rx.for_each_vertex(
            HOST,
            [&](VertexHandle vh) {
                if ((*v_err)(vh) == 1) {
                    err = 1;
                }
            },
            NULL,
            false);

        if (err == 1) {

            position.move(DEVICE, HOST);

            rx.update_polyscope();

            rx.render_vertex_patch();
            rx.render_edge_patch();
            rx.render_face_patch();

            auto ps_mesh = rx.get_polyscope_mesh();
            ps_mesh->updateVertexPositions(position);
            ps_mesh->setEdgeWidth(1.0);
            ps_mesh->setEnabled(true);
            ps_mesh->addVertexScalarQuantity("vErr", *v_err);

            polyscope::show();
            ps_mesh->setEnabled(false);
        }
    };

    constexpr uint32_t blockThreads = 256;

    v_err->reset(0, DEVICE);

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box(
        {Op::FV}, launch_box, (void*)ckeck_kernel_fv<T, blockThreads>, false);

    ckeck_kernel_fv<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), position, *v_err);
    CUDA_ERROR(cudaDeviceSynchronize());
    viz();

    v_err->reset(0, DEVICE);
    rx.update_launch_box(
        {Op::EV}, launch_box, (void*)ckeck_kernel_ev<T, blockThreads>, false);
    ckeck_kernel_ev<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), position, *v_err);
    CUDA_ERROR(cudaDeviceSynchronize());
    viz();
}

template <typename T, typename U>
void check2(rxmesh::RXMeshDynamic&      rx,
            rxmesh::VertexAttribute<T>& v_attr,
            rxmesh::VertexAttribute<U>& position)
{
    using namespace rxmesh;

    rx.update_host();

    v_attr.move(DEVICE, HOST);

    int err = 0;
    rx.for_each_vertex(
        HOST,
        [&](VertexHandle vh) {
            if (v_attr(vh) == 1) {
                err = 1;
                printf("\n Error!!!");
            }
        },
        NULL,
        false);

    if (err == 1) {
        position.move(DEVICE, HOST);

        rx.export_obj("tracking_xx.obj", position);

        rx.update_polyscope();
        rx.render_vertex_patch();
        rx.render_edge_patch();
        rx.render_face_patch();

        auto ps_mesh = rx.get_polyscope_mesh();
        ps_mesh->updateVertexPositions(position);
        ps_mesh->setEdgeWidth(1.0);
        ps_mesh->setEnabled(true);
        ps_mesh->addVertexScalarQuantity("vErr", v_attr);

        polyscope::show();
        ps_mesh->setEnabled(false);
    }
}

template <typename T>
void update_polyscope(rxmesh::RXMeshDynamic&      rx,
                      rxmesh::VertexAttribute<T>& current_position,
                      rxmesh::VertexAttribute<T>& new_position)
{
#if USE_POLYSCOPE
    using namespace rxmesh;

    rx.update_host();
    current_position.move(DEVICE, HOST);
    new_position.move(DEVICE, HOST);

    rx.update_polyscope();

    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();

    // rx.export_obj("tracking.obj", current_position);

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(current_position);
    ps_mesh->setEdgeWidth(1.0);
    ps_mesh->setEnabled(true);

    polyscope::show();
    ps_mesh->setEnabled(false);
#endif
}

template <typename T>
void splitter(rxmesh::RXMeshDynamic&             rx,
              rxmesh::VertexAttribute<T>*        position,
              rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
              rxmesh::VertexAttribute<int8_t>*   is_vertex_bd)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    //=== long edges pass

    // RXMESH_INFO("Split long edges");

    LaunchBox<blockThreads> launch_box;

    rx.update_launch_box({Op::EVDiamond},
                         launch_box,
                         (void*)split_edges<T, blockThreads>,
                         true,
                         false,
                         false,
                         false,
                         [&](uint32_t v, uint32_t e, uint32_t f) {
                             return detail::mask_num_bytes(e) +
                                    ShmemAllocator::default_alignment;
                         });

    timers.start("SplitEdgeTotal");
    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    while (true) {
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {

            timers.start("SplitEdge");
            split_edges<T, blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *position,
                                                *edge_status,
                                                *is_vertex_bd,
                                                Arg.splitter_max_edge_length,
                                                Arg.min_triangle_area,
                                                Arg.min_triangle_angle,
                                                Arg.max_triangle_angle,
                                                EdgeSplitPredicate::Length);
            timers.stop("SplitEdge");

            timers.start("SplitEdgeCleanup");
            rx.cleanup();
            timers.stop("SplitEdgeCleanup");

            timers.start("SplitEdgeSlice");
            rx.slice_patches(*position, *edge_status, *is_vertex_bd);
            timers.stop("SplitEdgeSlice");

            timers.start("SplitEdgeCleanup");
            rx.cleanup();
            timers.stop("SplitEdgeCleanup");

            //{
            //    rx.update_host();
            //    rx.validate();
            //}
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }
    timers.stop("SplitEdgeTotal");


    //=== large angle pass

    // RXMESH_INFO("Split Angles");

    rx.update_launch_box({Op::EVDiamond},
                         launch_box,
                         (void*)split_edges<T, blockThreads>,
                         true,
                         false,
                         false,
                         false,
                         [&](uint32_t v, uint32_t e, uint32_t f) {
                             return detail::mask_num_bytes(e) +
                                    ShmemAllocator::default_alignment;
                         });

    timers.start("SplitAngTotal");
    edge_status->reset(UNSEEN, DEVICE);

    prv_remaining_work = rx.get_num_edges();

    while (true) {
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {

            timers.start("SplitAng");
            split_edges<T, blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *position,
                                                *edge_status,
                                                *is_vertex_bd,
                                                Arg.splitter_max_edge_length,
                                                Arg.min_triangle_area,
                                                Arg.min_triangle_angle,
                                                Arg.max_triangle_angle,
                                                EdgeSplitPredicate::Angle);
            timers.stop("SplitAng");

            timers.start("SplitAngCleanup");
            rx.cleanup();
            timers.stop("SplitAngCleanup");

            timers.start("SplitAngSlice");
            rx.slice_patches(*position, *edge_status, *is_vertex_bd);
            timers.stop("SplitAngSlice");

            timers.start("SplitAngCleanup");
            rx.cleanup();
            timers.stop("SplitAngCleanup");

            //{
            //    rx.update_host();
            //    rx.validate();
            //}
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }
    timers.stop("SplitAngTotal");
}

template <typename T>
void classify_vertices(rxmesh::RXMeshDynamic&                 rx,
                       rxmesh::VertexAttribute<T>*            position,
                       const rxmesh::VertexAttribute<int8_t>* is_vertex_bd,
                       rxmesh::VertexAttribute<int8_t>*       vertex_rank)
{
    using namespace rxmesh;

    RXMESH_INFO("Classify");

    //{
    //    rx.update_host();
    //    rx.validate();
    //}

    constexpr uint32_t blockThreads = 256;

    vertex_rank->reset(0, DEVICE);

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::VV},
                         launch_box,
                         (void*)classify_vertex<T, blockThreads>,
                         false,
                         true);

    classify_vertex<T, blockThreads><<<launch_box.blocks,
                                       launch_box.num_threads,
                                       launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *position, *is_vertex_bd, *vertex_rank);

    // check2(rx, *v_err, *position);
}

template <typename T>
void flipper(rxmesh::RXMeshDynamic&             rx,
             rxmesh::VertexAttribute<T>*        position,
             rxmesh::VertexAttribute<int8_t>*   vertex_rank,
             rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
             rxmesh::VertexAttribute<int8_t>*   is_vertex_bd)
{
    using namespace rxmesh;

    timers.start("FlipTotal");
    classify_vertices(rx, position, is_vertex_bd, vertex_rank);


    constexpr uint32_t blockThreads = 256;

    const uint32_t MAX_NUM_FLIP_PASSES = 5;

    uint32_t num_flip_passes = 0;

    LaunchBox<blockThreads> launch_box;

    rx.update_launch_box({Op::EVDiamond},
                         launch_box,
                         (void*)edge_flip<T, blockThreads>,
                         true,
                         false,
                         false,
                         false,
                         [&](uint32_t v, uint32_t e, uint32_t f) {
                             return detail::mask_num_bytes(e) +
                                    2 * detail::mask_num_bytes(v) +
                                    3 * ShmemAllocator::default_alignment;
                         });

    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    // while (num_flip_passes++ < MAX_NUM_FLIP_PASSES) {
    while (true) {
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {

            timers.start("Flip");
            edge_flip<T, blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *position,
                                                *vertex_rank,
                                                *edge_status,
                                                *is_vertex_bd,
                                                Arg.edge_flip_min_length_change,
                                                Arg.max_volume_change,
                                                Arg.min_triangle_area,
                                                Arg.min_triangle_angle,
                                                Arg.max_triangle_angle);
            timers.stop("Flip");

            timers.start("FlipCleanup");
            rx.cleanup();
            timers.stop("FlipCleanup");

            timers.start("FlipSlice");
            rx.slice_patches(
                *position, *vertex_rank, *edge_status, *is_vertex_bd);
            timers.stop("FlipSlice");

            timers.start("FlipCleanup");
            rx.cleanup();
            timers.stop("FlipCleanup");

            //{
            //    rx.update_host();
            //    rx.validate();
            //}
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }
    timers.stop("FlipTotal");
}


template <typename T>
void collapser(rxmesh::RXMeshDynamic&             rx,
               rxmesh::VertexAttribute<T>*        position,
               rxmesh::VertexAttribute<int8_t>*   vertex_rank,
               rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
               rxmesh::VertexAttribute<int8_t>*   is_vertex_bd)
{
    using namespace rxmesh;

    timers.start("CollapseTotal");

    classify_vertices(rx, position, is_vertex_bd, vertex_rank);

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> launch_box;

    rx.update_launch_box({Op::EVDiamond},
                         launch_box,
                         (void*)edge_collapse<T, blockThreads>,
                         true,
                         false,
                         false,
                         false,
                         [&](uint32_t v, uint32_t e, uint32_t f) {
                             return detail::mask_num_bytes(e) +
                                    2 * detail::mask_num_bytes(v) +
                                    3 * ShmemAllocator::default_alignment;
                         });


    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    while (true) {
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {

            timers.start("Collapse");
            edge_collapse<T, blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *position,
                                                *vertex_rank,
                                                *edge_status,
                                                *is_vertex_bd,
                                                Arg.collapser_min_edge_length,
                                                Arg.max_volume_change,
                                                Arg.min_triangle_area,
                                                Arg.min_triangle_angle,
                                                Arg.max_triangle_angle);
            timers.stop("Collapse");

            timers.start("CollapseCleanup");
            rx.cleanup();
            timers.stop("CollapseCleanup");

            timers.start("CollapseSlice");
            rx.slice_patches(
                *position, *vertex_rank, *edge_status, *is_vertex_bd);
            timers.stop("CollapseSlice");

            timers.start("CollapseCleanup");
            rx.cleanup();
            timers.stop("CollapseCleanup");

            //{
            //    rx.update_host();
            //    rx.validate();
            //}
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }


    timers.stop("CollapseTotal");
}


template <typename T>
void smoother(rxmesh::RXMeshDynamic&                 rx,
              const rxmesh::VertexAttribute<int8_t>* is_vertex_bd,
              rxmesh::VertexAttribute<T>*            current_position,
              rxmesh::VertexAttribute<T>*            new_position)
{
    using namespace rxmesh;

    //{
    //    rx.update_host();
    //    rx.validate();
    //}

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::VV},
                         launch_box,
                         (void*)null_space_smooth_vertex<T, blockThreads>,
                         false,
                         true);

    timers.start("SmoothTotal");
    null_space_smooth_vertex<T, blockThreads><<<launch_box.blocks,
                                                launch_box.num_threads,
                                                launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *is_vertex_bd, *current_position, *new_position);

    timers.stop("SmoothTotal");

    // check2(rx, *v_err, *new_position);
}


template <typename T>
void improve_mesh(rxmesh::RXMeshDynamic&             rx,
                  rxmesh::VertexAttribute<T>*        current_position,
                  rxmesh::VertexAttribute<T>*        new_position,
                  rxmesh::VertexAttribute<int8_t>*   vertex_rank,
                  rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                  rxmesh::VertexAttribute<int8_t>*   is_vertex_bd)
{
    // edge splitting
    // RXMESH_INFO("Splitter");
    splitter(rx, current_position, edge_status, is_vertex_bd);

    // update_polyscope(rx, *current_position, *new_position);

    // edge flipping
    // RXMESH_INFO("Flipper");
    flipper(rx, current_position, vertex_rank, edge_status, is_vertex_bd);
    // update_polyscope(rx, *current_position, *new_position);

    // edge collapsing
    // RXMESH_INFO("Collapser");
    collapser(rx, current_position, vertex_rank, edge_status, is_vertex_bd);
    // update_polyscope(rx, *current_position, *new_position);

    // null-space smoothing
    // RXMESH_INFO("Smoother");
    smoother(rx, is_vertex_bd, current_position, new_position);

    // update_polyscope(rx, *new_position, *current_position);
}

template <typename T>
void advance_sim(T                                  sim_dt,
                 Simulation<T>&                     sim,
                 FrameStepper<T>&                   frame_stepper,
                 const FlowNoise3<T>&               noise,
                 rxmesh::RXMeshDynamic&             rx,
                 rxmesh::VertexAttribute<T>*&       current_position,
                 rxmesh::VertexAttribute<T>*&       new_position,
                 rxmesh::VertexAttribute<int8_t>*   vertex_rank,
                 rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                 rxmesh::VertexAttribute<int8_t>*   is_vertex_bd)
{
    using namespace rxmesh;

    T accum_dt = 0;

    while ((accum_dt < 0.99 * sim_dt) &&
           (sim.m_curr_t + accum_dt < sim.m_max_t)) {
        total_num_iter++;


        RXMESH_INFO("total_num_iter {}", total_num_iter);

        timers.start("MeshImprove");
        // improve the mesh (also update new_position)
        improve_mesh(rx,
                     current_position,
                     new_position,
                     vertex_rank,
                     edge_status,
                     is_vertex_bd);
        timers.stop("MeshImprove");

        std::swap(current_position, new_position);

        T curr_dt = sim_dt - accum_dt;
        curr_dt   = std::min(curr_dt, sim.m_max_t - sim.m_curr_t - accum_dt);

        // move the mesh (update current_position)
        timers.start("Advect");
        curl_noise_predicate_new_position(
            rx, noise, *current_position, sim.m_curr_t + accum_dt, curr_dt);
        accum_dt += curr_dt;
        timers.stop("Advect");

        // CUDA_ERROR(cudaDeviceSynchronize());

        // update polyscope
        // update_polyscope(rx, *current_position, *new_position);
    }

    sim.m_curr_t += accum_dt;


    // check if max time is reached
    if (sim.done_simulation()) {
        sim.m_running = false;
    }
}

template <typename T>
void advance_frame(Simulation<T>&                     sim,
                   FrameStepper<T>&                   frame_stepper,
                   FlowNoise3<T>&                     noise,
                   rxmesh::RXMeshDynamic&             rx,
                   rxmesh::VertexAttribute<T>*&       current_position,
                   rxmesh::VertexAttribute<T>*&       new_position,
                   rxmesh::VertexAttribute<int8_t>*   vertex_rank,
                   rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                   rxmesh::VertexAttribute<int8_t>*   is_vertex_bd)
{
    if (!sim.m_currently_advancing_simulation) {
        sim.m_currently_advancing_simulation = true;


        // Advance frame
        while (!(frame_stepper.done_frame())) {
            T dt = frame_stepper.get_step_length(sim.m_dt);
            advance_sim(dt,
                        sim,
                        frame_stepper,
                        noise,
                        rx,
                        current_position,
                        new_position,
                        vertex_rank,
                        edge_status,
                        is_vertex_bd);
            frame_stepper.advance_step(dt);
        }

        // update frame stepper
        frame_stepper.next_frame();

        sim.m_currently_advancing_simulation = false;
    }
}

template <typename T>
void run_simulation(Simulation<T>&                     sim,
                    FrameStepper<T>&                   frame_stepper,
                    FlowNoise3<T>&                     noise,
                    rxmesh::RXMeshDynamic&             rx,
                    rxmesh::VertexAttribute<T>*        current_position,
                    rxmesh::VertexAttribute<T>*        new_position,
                    rxmesh::VertexAttribute<int8_t>*   vertex_rank,
                    rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                    rxmesh::VertexAttribute<int8_t>*   is_vertex_bd)
{
    sim.m_running = true;
    while (sim.m_running) {
        advance_frame(sim,
                      frame_stepper,
                      noise,
                      rx,
                      current_position,
                      new_position,
                      vertex_rank,
                      edge_status,
                      is_vertex_bd);
    }
    sim.m_running = false;
}


inline void tracking_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    EXPECT_TRUE(rx.validate());

    using namespace rxmesh;

    rxmesh::Report report("Tracking_RXMesh");
    report.command_line(Arg.argc, Arg.argv);
    report.device();
    report.system();
    report.model_data(Arg.plane_name + "_before", rx, "model_before");
    report.add_member("method", std::string("RXMesh"));

    report.add_member("n", Arg.n);
    report.add_member("frame_dt", Arg.frame_dt);
    report.add_member("sim_dt", Arg.sim_dt);
    report.add_member("end_sim_t", Arg.end_sim_t);
    report.add_member("max_volume_change", Arg.max_volume_change);
    report.add_member("min_edge_length", Arg.min_edge_length);
    report.add_member("max_edge_length", Arg.max_edge_length);
    report.add_member("min_triangle_area", Arg.min_triangle_area);
    report.add_member("min_triangle_angle", Arg.min_triangle_angle);
    report.add_member("max_triangle_angle", Arg.max_triangle_angle);

    auto current_position = rx.get_input_vertex_coordinates();

    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);

    auto new_position = rx.add_vertex_attribute<float>("NewPosition", 3);
    new_position->reset(0, LOCATION_ALL);

    auto vertex_rank = rx.add_vertex_attribute<int8_t>("vRank", 1);
    vertex_rank->reset(0, LOCATION_ALL);

    auto is_vertex_bd = rx.add_vertex_attribute<int8_t>("vBoundary", 1);
    is_vertex_bd->reset(0, LOCATION_ALL);


    FrameStepper<float> frame_stepper(Arg.frame_dt);

    Simulation<float> sim(Arg.sim_dt, Arg.end_sim_t);

    FlowNoise3<float> noise;

    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));

    // v_err = rx.add_vertex_attribute<int>("vError", 1).get();

    // compute avergae edge length
    float avg_edge_len = compute_avg_edge_length(rx, *current_position);

    Arg.max_volume_change *= avg_edge_len * avg_edge_len * avg_edge_len;

    Arg.collapser_min_edge_length = Arg.min_edge_length * avg_edge_len;
    Arg.collapser_min_edge_length *= Arg.collapser_min_edge_length;

    Arg.splitter_max_edge_length = Arg.max_edge_length * avg_edge_len;
    Arg.splitter_max_edge_length *= Arg.splitter_max_edge_length;

    // init boundary vertices and edges
    rx.get_boundary_vertices(*is_vertex_bd);


#if USE_POLYSCOPE
    polyscope::options::groundPlaneHeightFactor = 0.2;
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // polyscope::show();
#endif

    timers.add("Total");

    timers.add("SplitEdgeTotal");
    timers.add("SplitEdge");
    timers.add("SplitEdgeCleanup");
    timers.add("SplitEdgeSlice");

    timers.add("SplitAngTotal");
    timers.add("SplitAng");
    timers.add("SplitAngCleanup");
    timers.add("SplitAngSlice");

    timers.add("CollapseTotal");
    timers.add("Collapse");
    timers.add("CollapseCleanup");
    timers.add("CollapseSlice");

    timers.add("FlipTotal");
    timers.add("Flip");
    timers.add("FlipCleanup");
    timers.add("FlipSlice");

    timers.add("SmoothTotal");

    timers.add("MeshImprove");
    timers.add("Advect");

    total_num_iter = 0;

    CUDA_ERROR(cudaProfilerStart());

    timers.start("Total");

    run_simulation(sim,
                   frame_stepper,
                   noise,
                   rx,
                   current_position.get(),
                   new_position.get(),
                   vertex_rank.get(),
                   edge_status.get(),
                   is_vertex_bd.get());

    timers.stop("Total");

    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO(
        "tracking_rxmesh() RXMesh surface tracking took {} (ms), time/iter {} "
        "(ms)",
        timers.elapsed_millis("Total"),
        float(timers.elapsed_millis("Total")) / float(total_num_iter));

    RXMESH_INFO(
        "tracking_rxmesh() SplitEdgeTotal {} (ms), SplitEdge {} (ms), "
        "SplitEdgeCleanup {} (ms), SplitEdgeSlice {} (ms)",
        timers.elapsed_millis("SplitEdgeTotal"),
        timers.elapsed_millis("SplitEdge"),
        timers.elapsed_millis("SplitEdgeCleanup"),
        timers.elapsed_millis("SplitEdgeSlice"));

    RXMESH_INFO(
        "tracking_rxmesh() SplitAngTotal {} (ms), SplitAng {} (ms), "
        "SplitAngCleanup {} (ms), SplitAngSlice {} (ms)",
        timers.elapsed_millis("SplitAngTotal"),
        timers.elapsed_millis("SplitAng"),
        timers.elapsed_millis("SplitAngCleanup"),
        timers.elapsed_millis("SplitAngSlice"));


    RXMESH_INFO(
        "tracking_rxmesh() CollapseTotal {} (ms), Collapse {} (ms), "
        "CollapseCleanup {} (ms), CollapseSlice {} (ms)",
        timers.elapsed_millis("CollapseTotal"),
        timers.elapsed_millis("Collapse"),
        timers.elapsed_millis("CollapseCleanup"),
        timers.elapsed_millis("CollapseSlice"));

    RXMESH_INFO(
        "tracking_rxmesh() FlipTotal {} (ms), Flip {} (ms), "
        "FlipCleanup {} (ms), FlipSlice {} (ms)",
        timers.elapsed_millis("FlipTotal"),
        timers.elapsed_millis("Flip"),
        timers.elapsed_millis("FlipCleanup"),
        timers.elapsed_millis("FlipSlice"));

    RXMESH_INFO("tracking_rxmesh() SmoothTotal {} (ms)",
                timers.elapsed_millis("SmoothTotal"));

    RXMESH_INFO("tracking_rxmesh() MeshImprove {} (ms), Advect {} (ms)",
                timers.elapsed_millis("MeshImprove"),
                timers.elapsed_millis("Advect"));

    rx.update_host();

    report.add_member("total_tracking_time", timers.elapsed_millis("Total"));
    report.add_member("total_num_iter", total_num_iter);
    report.add_member(
        "time_per_iter",
        float(timers.elapsed_millis("Total")) / float(total_num_iter));
    report.model_data(Arg.plane_name + "_after", rx, "model_after");

    report.add_member(
        "attributes_memory_mg",
        current_position->get_memory_mg() + edge_status->get_memory_mg() +
            new_position->get_memory_mg() + vertex_rank->get_memory_mg() +
            is_vertex_bd->get_memory_mg());

    for (auto t : timers.m_total_time) {
        report.add_member(t.first, t.second);
    }

    update_polyscope(rx, *current_position, *new_position);

    report.write(Arg.output_folder + "/rxmesh_tracking",
                 "Tracking_RXMesh_" + extract_file_name(Arg.plane_name));

    noise.free();
    GPU_FREE(d_buffer);
}
