#pragma once

#define G_EIGENVALUE_RANK_RATIO 0.03

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include "frame_stepper.h"
#include "rxmesh/rxmesh_dynamic.h"
#include "simulation.h"

template <typename T>
using Vec3 = glm::vec<3, T, glm::defaultp>;

using EdgeStatus = int8_t;
enum : EdgeStatus
{
    UNSEEN = 0,  // means we have not tested it before for e.g., split/flip/col
    OKAY   = 1,  // means we have tested it and it is okay to skip
    UPDATE = 2,  // means we should update it i.e., we have tested it before
    ADDED  = 3,  // means it has been added to during the split/flip/collapse
};

int* d_buffer;

#include "collapser.cuh"
#include "flipper.cuh"
#include "noise.h"
#include "smoother.cuh"
#include "splitter.cuh"
#include "tracking_kernels.cuh"


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

    // rx.export_obj("tracking.obj", current_position);

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(current_position);
    ps_mesh->setEdgeWidth(1.0);
    ps_mesh->setEnabled(true);

    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    polyscope::show();
    ps_mesh->setEnabled(false);
#endif
}

int is_done(const rxmesh::RXMeshDynamic&             rx,
            const rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
            int*                                     d_buffer)
{
    using namespace rxmesh;

    // if there is at least one edge that is UNSEEN or UPDATE (i.e. newly
    // added), then we are not done yet
    CUDA_ERROR(cudaMemset(d_buffer, 0, sizeof(int)));

    rx.for_each_edge(
        DEVICE,
        [edge_status = *edge_status, d_buffer] __device__(const EdgeHandle eh) {
            if (edge_status(eh) == UNSEEN || edge_status(eh) == UPDATE) {
                ::atomicAdd(d_buffer, 1);
            }
        });

    CUDA_ERROR(cudaDeviceSynchronize());
    return d_buffer[0];
}


template <typename T>
void splitter(rxmesh::RXMeshDynamic&             rx,
              rxmesh::VertexAttribute<T>*        position,
              rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
              rxmesh::VertexAttribute<int8_t>*   is_vertex_bd,
              rxmesh::EdgeAttribute<int8_t>*     is_edge_bd)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 512;

    //=== long edges pass
    GPUTimer app_timer;
    app_timer.start();
    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    while (true) {
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {

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

            split_edges<T, blockThreads>
                <<<DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *position,
                                                *edge_status,
                                                *is_vertex_bd,
                                                *is_edge_bd,
                                                Arg.splitter_max_edge_length,
                                                Arg.min_triangle_area,
                                                Arg.min_triangle_angle,
                                                Arg.max_triangle_angle,
                                                EdgeSplitPredicate::Length);

            rx.cleanup();
            rx.slice_patches(
                *position, *edge_status, *is_vertex_bd, *is_edge_bd);
            rx.cleanup();
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }
    app_timer.stop();
    RXMESH_INFO("Splitter (long edges) time {} (ms)",
                app_timer.elapsed_millis());

    //=== large angle pass
    app_timer.start();
    edge_status->reset(UNSEEN, DEVICE);

    prv_remaining_work = rx.get_num_edges();

    while (true) {
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {

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

            split_edges<T, blockThreads>
                <<<DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *position,
                                                *edge_status,
                                                *is_vertex_bd,
                                                *is_edge_bd,
                                                Arg.splitter_max_edge_length,
                                                Arg.min_triangle_area,
                                                Arg.min_triangle_angle,
                                                Arg.max_triangle_angle,
                                                EdgeSplitPredicate::Angle);

            rx.cleanup();
            rx.slice_patches(
                *position, *edge_status, *is_vertex_bd, *is_edge_bd);
            rx.cleanup();
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }
    app_timer.stop();
    RXMESH_INFO("Splitter (large angles) time {} (ms)",
                app_timer.elapsed_millis());
}

template <typename T>
void classify_vertices(rxmesh::RXMeshDynamic&                 rx,
                       const rxmesh::VertexAttribute<T>*      position,
                       const rxmesh::VertexAttribute<int8_t>* is_vertex_bd,
                       rxmesh::VertexAttribute<int8_t>*       vertex_rank)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 384;

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
}

template <typename T>
void flipper(rxmesh::RXMeshDynamic&             rx,
             rxmesh::VertexAttribute<T>*        position,
             rxmesh::VertexAttribute<int8_t>*   vertex_rank,
             rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
             rxmesh::VertexAttribute<int8_t>*   is_vertex_bd,
             rxmesh::EdgeAttribute<int8_t>*     is_edge_bd)
{
    using namespace rxmesh;

    GPUTimer app_timer;
    app_timer.start();
    classify_vertices(rx, position, is_vertex_bd, vertex_rank);
    app_timer.stop();
    RXMESH_INFO("Flipper Classify Vertices time {} (ms)",
                app_timer.elapsed_millis());

    constexpr uint32_t blockThreads = 512;

    const uint32_t MAX_NUM_FLIP_PASSES = 5;

    uint32_t num_flip_passes = 0;

    app_timer.start();
    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    // while (num_flip_passes++ < MAX_NUM_FLIP_PASSES) {
    while (true) {
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {

            LaunchBox<blockThreads> launch_box;

            rx.update_launch_box(
                {Op::EVDiamond},
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

            edge_flip<T, blockThreads>
                <<<DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *position,
                                                *vertex_rank,
                                                *edge_status,
                                                *is_vertex_bd,
                                                *is_edge_bd,
                                                Arg.edge_flip_min_length_change,
                                                Arg.max_volume_change,
                                                Arg.min_triangle_area,
                                                Arg.min_triangle_angle,
                                                Arg.max_triangle_angle);

            rx.cleanup();
            rx.slice_patches(*position,
                             *vertex_rank,
                             *edge_status,
                             *is_vertex_bd,
                             *is_edge_bd);
            rx.cleanup();
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }
    app_timer.stop();
    RXMESH_INFO("Flipper time {} (ms)", app_timer.elapsed_millis());
}


template <typename T>
void collapser(rxmesh::RXMeshDynamic&             rx,
               rxmesh::VertexAttribute<T>*        position,
               rxmesh::VertexAttribute<int8_t>*   vertex_rank,
               rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
               rxmesh::VertexAttribute<int8_t>*   is_vertex_bd,
               rxmesh::EdgeAttribute<int8_t>*     is_edge_bd)
{
    using namespace rxmesh;

    GPUTimer app_timer;
    app_timer.start();
    classify_vertices(rx, position, is_vertex_bd, vertex_rank);
    app_timer.stop();
    RXMESH_INFO("Collapser Classify Vertices time {} (ms)",
                app_timer.elapsed_millis());

    constexpr uint32_t blockThreads = 512;

    app_timer.start();
    edge_status->reset(UNSEEN, DEVICE);

    int prv_remaining_work = rx.get_num_edges();

    while (true) {
        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {

            LaunchBox<blockThreads> launch_box;

            rx.update_launch_box(
                {Op::EVDiamond},
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

            edge_collapse<T, blockThreads>
                <<<DIVIDE_UP(launch_box.blocks, 8),
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *position,
                                                *vertex_rank,
                                                *edge_status,
                                                *is_vertex_bd,
                                                *is_edge_bd,
                                                Arg.collapser_min_edge_length,
                                                Arg.max_volume_change,
                                                Arg.min_triangle_area,
                                                Arg.min_triangle_angle,
                                                Arg.max_triangle_angle);

            rx.cleanup();
            rx.slice_patches(*position,
                             *vertex_rank,
                             *edge_status,
                             *is_vertex_bd,
                             *is_edge_bd);
            rx.cleanup();
        }

        int remaining_work = is_done(rx, edge_status, d_buffer);

        if (remaining_work == 0 || prv_remaining_work == remaining_work) {
            break;
        }
        prv_remaining_work = remaining_work;
    }
    app_timer.stop();
    RXMESH_INFO("Collapser time {} (ms)", app_timer.elapsed_millis());
}


template <typename T>
void smoother(rxmesh::RXMeshDynamic&                 rx,
              const rxmesh::VertexAttribute<int8_t>* is_vertex_bd,
              const rxmesh::VertexAttribute<T>*      current_position,
              rxmesh::VertexAttribute<T>*            new_position)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 384;

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::VV},
                         launch_box,
                         (void*)null_space_smooth_vertex<T, blockThreads>,
                         false,
                         true);

    GPUTimer app_timer;
    app_timer.start();
    null_space_smooth_vertex<T, blockThreads><<<launch_box.blocks,
                                                launch_box.num_threads,
                                                launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *is_vertex_bd, *current_position, *new_position);
    app_timer.stop();
    RXMESH_INFO("Smoother time {} (ms)", app_timer.elapsed_millis());
}


template <typename T>
void improve_mesh(rxmesh::RXMeshDynamic&             rx,
                  rxmesh::VertexAttribute<T>*        current_position,
                  rxmesh::VertexAttribute<T>*        new_position,
                  rxmesh::VertexAttribute<int8_t>*   vertex_rank,
                  rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
                  rxmesh::VertexAttribute<int8_t>*   is_vertex_bd,
                  rxmesh::EdgeAttribute<int8_t>*     is_edge_bd)
{
    // edge splitting
    splitter(rx, current_position, edge_status, is_vertex_bd, is_edge_bd);

    // edge flipping
    flipper(rx,
            current_position,
            vertex_rank,
            edge_status,
            is_vertex_bd,
            is_edge_bd);

    // edge collapsing
    collapser(rx,
              current_position,
              vertex_rank,
              edge_status,
              is_vertex_bd,
              is_edge_bd);

    // null-space smoothing
    smoother(rx, is_vertex_bd, current_position, new_position);
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
                 rxmesh::VertexAttribute<int8_t>*   is_vertex_bd,
                 rxmesh::EdgeAttribute<int8_t>*     is_edge_bd)
{
    using namespace rxmesh;

    T accum_dt = 0;

    while ((accum_dt < 0.99 * sim_dt) &&
           (sim.m_curr_t + accum_dt < sim.m_max_t)) {

        GPUTimer timer;
        timer.start();

        // improve the mesh (also update new_position)
        improve_mesh(rx,
                     current_position,
                     new_position,
                     vertex_rank,
                     edge_status,
                     is_vertex_bd,
                     is_edge_bd);
        std::swap(current_position, new_position);

        T curr_dt = sim_dt - accum_dt;
        curr_dt   = std::min(curr_dt, sim.m_max_t - sim.m_curr_t - accum_dt);

        // move the mesh (update current_position)
        curl_noise_predicate_new_position(
            rx, noise, *current_position, sim.m_curr_t + accum_dt, curr_dt);
        accum_dt += curr_dt;

        // CUDA_ERROR(cudaDeviceSynchronize());

        // update polyscope
        // update_polyscope(rx, *current_position, *new_position);
        timer.stop();
        RXMESH_INFO("** Step time {} (ms)", timer.elapsed_millis());
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
                   rxmesh::VertexAttribute<int8_t>*   is_vertex_bd,
                   rxmesh::EdgeAttribute<int8_t>*     is_edge_bd)
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
                        is_vertex_bd,
                        is_edge_bd);
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
                    rxmesh::VertexAttribute<int8_t>*   is_vertex_bd,
                    rxmesh::EdgeAttribute<int8_t>*     is_edge_bd)
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
                      is_vertex_bd,
                      is_edge_bd);
    }
    sim.m_running = false;
}


inline void tracking_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    EXPECT_TRUE(rx.validate());

    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    auto current_position = rx.get_input_vertex_coordinates();

    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);

    auto new_position = rx.add_vertex_attribute<float>("NewPosition", 3);
    new_position->reset(0, LOCALE_ALL);

    auto vertex_rank = rx.add_vertex_attribute<int8_t>("vRank", 1);
    vertex_rank->reset(0, LOCALE_ALL);

    auto is_vertex_bd = rx.add_vertex_attribute<int8_t>("vBoundary", 1);
    is_vertex_bd->reset(0, LOCALE_ALL);

    auto is_edge_bd = rx.add_edge_attribute<int8_t>("eBoundary", 1);
    is_edge_bd->reset(0, LOCALE_ALL);

    LaunchBox<blockThreads> launch_box;

    FrameStepper<float> frame_stepper(Arg.frame_dt);

    Simulation<float> sim(Arg.sim_dt, Arg.end_sim_t);

    FlowNoise3<float> noise;

    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));

    float total_time   = 0;
    float app_time     = 0;
    float slice_time   = 0;
    float cleanup_time = 0;


    // compute avergae edge length
    float avg_edge_len = compute_avg_edge_length(rx, *current_position);

    Arg.max_volume_change *= avg_edge_len * avg_edge_len * avg_edge_len;

    Arg.collapser_min_edge_length = Arg.min_edge_length * avg_edge_len;
    Arg.collapser_min_edge_length *= Arg.collapser_min_edge_length;

    Arg.splitter_max_edge_length = Arg.max_edge_length * avg_edge_len;
    Arg.splitter_max_edge_length *= Arg.splitter_max_edge_length;

    // init boundary vertices and edges
    init_boundary(rx, *is_vertex_bd, *is_edge_bd);

#if USE_POLYSCOPE
    polyscope::options::groundPlaneHeightFactor = 0.2;
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // polyscope::show();
#endif

    CUDA_ERROR(cudaProfilerStart());
    GPUTimer timer;
    timer.start();

    run_simulation(sim,
                   frame_stepper,
                   noise,
                   rx,
                   current_position.get(),
                   new_position.get(),
                   vertex_rank.get(),
                   edge_status.get(),
                   is_vertex_bd.get(),
                   is_edge_bd.get());

    timer.stop();
    total_time += timer.elapsed_millis();
    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("tracking_rxmesh() RXMesh surface tracking took {} (ms)",
                total_time);

    update_polyscope(rx, *current_position, *new_position);

    noise.free();
}