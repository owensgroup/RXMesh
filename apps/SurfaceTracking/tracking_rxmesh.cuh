#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>


#include "rxmesh/rxmesh_dynamic.h"

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

#include "tracking_kernels.cuh"


inline void tracking_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    EXPECT_TRUE(rx.validate());

    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;


    auto coords = rx.get_input_vertex_coordinates();

    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);

    LaunchBox<blockThreads> launch_box;

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

    bool validate = false;


    CUDA_ERROR(cudaProfilerStart());
    GPUTimer timer;
    timer.start();
    while (true) {

        // reset edge status
        edge_status->reset(UNSEEN, DEVICE);

        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {

            // RXMESH_INFO(" Queue size = {}",
            //             rx.get_context().m_patch_scheduler.size());

            // rx.update_launch_box(
            //     {Op::EV},
            //     launch_box,
            //     (void*)sec<float, blockThreads>,
            //     true,
            //     false,
            //     false,
            //     false,
            //     [&](uint32_t v, uint32_t e, uint32_t f) {
            //         return detail::mask_num_bytes(e) +
            //                2 * detail::mask_num_bytes(v) +
            //                3 * ShmemAllocator::default_alignment;
            //     });

            GPUTimer app_timer;
            app_timer.start();
            // sec<float, blockThreads>
            //     <<<launch_box.blocks,
            //        launch_box.num_threads,
            //        launch_box.smem_bytes_dyn>>>(rx.get_context(),
            //                                     *coords,
            //                                     histo,
            //                                     reduce_threshold,
            //                                     *edge_status);

            app_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords, *edge_status);
            slice_timer.stop();

            GPUTimer cleanup_timer2;
            cleanup_timer2.start();
            rx.cleanup();
            cleanup_timer2.stop();


            CUDA_ERROR(cudaDeviceSynchronize());
            CUDA_ERROR(cudaGetLastError());

            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();
            cleanup_time += cleanup_timer2.elapsed_millis();


            if (validate) {
                rx.update_host();
                EXPECT_TRUE(rx.validate());
            }
        }

        if (false) {

            RXMESH_INFO("#Vertices {}", rx.get_num_vertices(true));
            RXMESH_INFO("#Edges {}", rx.get_num_edges(true));
            RXMESH_INFO("#Faces {}", rx.get_num_faces(true));
            RXMESH_INFO("#Patches {}", rx.get_num_patches(true));

            if (false) {
                rx.update_host();
                coords->move(DEVICE, HOST);
                rx.update_polyscope();
                auto ps_mesh = rx.get_polyscope_mesh();
                ps_mesh->updateVertexPositions(*coords);
                ps_mesh->setEnabled(false);
                // ps_mesh->addEdgeScalarQuantity("eMark", *e_attr);
                // rx.render_vertex_patch();
                // rx.render_edge_patch();
                // rx.render_face_patch();

                polyscope::show();
            }
        }
    }
    timer.stop();
    total_time += timer.elapsed_millis();
    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("sec_rxmesh() RXMesh surface tracking took {} (ms)",
                total_time);
    RXMESH_INFO("sec_rxmesh() App time {} (ms)", app_time);
    RXMESH_INFO("sec_rxmesh() Slice timer {} (ms)", slice_time);
    RXMESH_INFO("sec_rxmesh() Cleanup timer {} (ms)", cleanup_time);

    if (!validate) {
        rx.update_host();
    }
    coords->move(DEVICE, HOST);

#if USE_POLYSCOPE
    rx.update_polyscope();

    auto ps_mesh = rx.get_polyscope_mesh();
    ps_mesh->updateVertexPositions(*coords);
    ps_mesh->setEnabled(false);

    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    polyscope::show();
#endif
    
}