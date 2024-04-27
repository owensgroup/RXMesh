#pragma once

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

#include "improving_kernels.cuh"
#include "noise.h"
#include "tracking_kernels.cuh"


template <typename T>
void update_polyscope(rxmesh::RXMeshDynamic&      rx,
                      rxmesh::VertexAttribute<T>& coords)
{
#if USE_POLYSCOPE
    using namespace rxmesh;

    rx.update_host();
    coords.move(DEVICE, HOST);

    rx.update_polyscope();

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(coords);
    ps_mesh->setEnabled(true);

    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    polyscope::show();
    ps_mesh->setEnabled(false);
#endif
}

template <typename T>
void splitter(rxmesh::RXMeshDynamic& rx, rxmesh::VertexAttribute<T>* position)
{
}


template <typename T>
void flipper(rxmesh::RXMeshDynamic& rx, rxmesh::VertexAttribute<T>* position)
{
}


template <typename T>
void collapser(rxmesh::RXMeshDynamic& rx, rxmesh::VertexAttribute<T>* position)
{
}


template <typename T>
void smoother(rxmesh::RXMeshDynamic&            rx,
              const rxmesh::VertexAttribute<T>* current_position,
              rxmesh::VertexAttribute<T>*       new_position)
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
        rx.get_context(), *current_position, *new_position);
    app_timer.stop();
    RXMESH_INFO("Smoother time {} (ms)", app_timer.elapsed_millis());
}


template <typename T>
void improve_mesh(rxmesh::RXMeshDynamic&      rx,
                  rxmesh::VertexAttribute<T>* current_position,
                  rxmesh::VertexAttribute<T>* new_position)
{
    // edge splitting
    splitter(rx, current_position);

    // edge flippingl
    flipper(rx, current_position);

    // edge collapsing
    collapser(rx, current_position);

    // null-space smoothing
    smoother(rx, current_position, new_position);
}

template <typename T>
void advance_sim(T                           sim_dt,
                 Simulation<T>&              sim,
                 FrameStepper<T>&            frame_stepper,
                 const FlowNoise3<T>&        noise,
                 rxmesh::RXMeshDynamic&      rx,
                 rxmesh::VertexAttribute<T>*& current_position,
                 rxmesh::VertexAttribute<T>*& new_position)
{
    using namespace rxmesh;

    T accum_dt = 0;

    while ((accum_dt < 0.99 * sim_dt) &&
           (sim.m_curr_t + accum_dt < sim.m_max_t)) {

        // improve the mesh (also update new_position)
        improve_mesh(rx, current_position, new_position);
        std::swap(current_position, new_position);

        T curr_dt = sim_dt - accum_dt;
        curr_dt   = std::min(curr_dt, sim.m_max_t - sim.m_curr_t - accum_dt);

        // move the mesh (update current_position)
        curl_noise_predicate_new_position(
            rx, noise, *current_position, sim.m_curr_t + accum_dt, curr_dt);
        accum_dt += curr_dt;

        CUDA_ERROR(cudaDeviceSynchronize());

        // update polyscope
        update_polyscope(rx, *current_position);
    }

    sim.m_curr_t += accum_dt;


    // check if max time is reached
    if (sim.done_simulation()) {
        sim.m_running = false;
    }
}

template <typename T>
void advance_frame(Simulation<T>&              sim,
                   FrameStepper<T>&            frame_stepper,
                   FlowNoise3<T>&              noise,
                   rxmesh::RXMeshDynamic&      rx,
                   rxmesh::VertexAttribute<T>*& current_position,
                   rxmesh::VertexAttribute<T>*& new_position)
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
                        new_position);
            frame_stepper.advance_step(dt);
        }

        // update frame stepper
        frame_stepper.next_frame();

        sim.m_currently_advancing_simulation = false;
    }
}

template <typename T>
void run_simulation(Simulation<T>&              sim,
                    FrameStepper<T>&            frame_stepper,
                    FlowNoise3<T>&              noise,
                    rxmesh::RXMeshDynamic&      rx,
                    rxmesh::VertexAttribute<T>* current_position,
                    rxmesh::VertexAttribute<T>* new_position)
{
    sim.m_running = true;
    while (sim.m_running) {
        advance_frame(
            sim, frame_stepper, noise, rx, current_position, new_position);
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

    LaunchBox<blockThreads> launch_box;

    FrameStepper<float> frame_stepper(Arg.frame_dt);

    Simulation<float> sim(Arg.sim_dt, Arg.end_sim_t);

    FlowNoise3<float> noise;

    float total_time   = 0;
    float app_time     = 0;
    float slice_time   = 0;
    float cleanup_time = 0;

#if USE_POLYSCOPE
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
                   new_position.get());

    timer.stop();
    total_time += timer.elapsed_millis();
    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("tracking_rxmesh() RXMesh surface tracking took {} (ms)",
                total_time);

    update_polyscope(rx, *current_position);

    noise.free();
}