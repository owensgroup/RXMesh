#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>


#include "rxmesh/query.cuh"
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

#include "histogram.cuh"
#include "sec_kernels.cuh"


inline void sec_rxmesh(rxmesh::RXMeshDynamic& rx,
                       const uint32_t         final_num_vertices)
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

    const int num_bins = 256;

    CostHistogram<float> histo(num_bins);

    auto e_attr = rx.add_edge_attribute<float>("eMark", 1);


#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // polyscope::show();
#endif

    bool validate = false;

    float reduce_ratio = 0.1;

    CUDA_ERROR(cudaProfilerStart());
    GPUTimer timer;
    timer.start();
    while (rx.get_num_vertices() > final_num_vertices) {

        // compute max-min histogram
        histo.init();

        rx.update_launch_box({Op::EV},
                             launch_box,
                             (void*)compute_min_max_cost<float, blockThreads>,
                             false);
        compute_min_max_cost<float, blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, histo);

        // compute histogram bins
        rx.update_launch_box({Op::EV},
                             launch_box,
                             (void*)populate_histogram<float, blockThreads>,
                             false);
        populate_histogram<float, blockThreads><<<launch_box.blocks,
                                                  launch_box.num_threads,
                                                  launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, histo, *e_attr);

        histo.scan();

#if USE_POLYSCOPE
        // e_attr->move(DEVICE, HOST);
        // auto ps_mesh = rx.get_polyscope_mesh();
        // ps_mesh->addEdgeScalarQuantity("eMark", *e_attr);
        // polyscope::show();
#endif

        // how much we can reduce the number of edge at each iterations
        
        // loop over the mesh, and try to collapse
        const int num_edges_before = int(rx.get_num_edges());

        const int reduce_threshold =
            std::max(1, int(reduce_ratio * float(num_edges_before)));

        // reset edge status
        edge_status->reset(UNSEEN, DEVICE);

        rx.reset_scheduler();
        while (!rx.is_queue_empty() &&
               rx.get_num_vertices() > final_num_vertices) {
            // RXMESH_INFO(" Queue size = {}",
            //             rx.get_context().m_patch_scheduler.size());

            rx.update_launch_box(
                {Op::EV},
                launch_box,
                (void*)sec<float, blockThreads>,
                true,
                false,
                false,
                false,
                [&](uint32_t v, uint32_t e, uint32_t f) {
                    return detail::mask_num_bytes(e) +
                           2 * detail::mask_num_bytes(v) +
                           3 * ShmemAllocator::default_alignment;
                });

            e_attr->reset(0, DEVICE);

            GPUTimer app_timer;
            app_timer.start();
            sec<float, blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                histo,
                                                reduce_threshold,
                                                *edge_status,
                                                *e_attr);

            app_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords, *edge_status, *e_attr);
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

        if (true) {
            coords->move(DEVICE, HOST);
            e_attr->move(DEVICE, HOST);
            rx.update_host();

            RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
            RXMESH_INFO("#Edges {}", rx.get_num_edges());
            RXMESH_INFO("#Faces {}", rx.get_num_faces());
            RXMESH_INFO("#Patches {}", rx.get_num_patches());
            RXMESH_INFO("request reduction = {}, achieved reduction= {}",
                        reduce_threshold,
                        num_edges_before - int(rx.get_num_edges()));

            if (num_edges_before == rx.get_num_edges()) {
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

    RXMESH_INFO("sec_rxmesh() RXMesh simplification took {} (ms)", total_time);
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

    histo.free();
}