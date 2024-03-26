#pragma once

#include <cuda_profiler_api.h>

#include "rxmesh/rxmesh_dynamic.h"

#include "simplification_kernels.cuh"

template <typename T>
inline void populate_bins(const rxmesh::RXMeshDynamic&   rx,
                          const rxmesh::EdgeAttribute<T> edge_cost,
                          const int                      num_bins,
                          const T*                       d_min_max_edge_cost,
                          int*                           d_bins)
{
    using namespace rxmesh;

    rx.for_each_edge(DEVICE,
                     [=](EdgeHandle eh) { float cost = edge_cost(eh); });
}

template <typename T>
void calc_edge_cost(const rxmesh::RXMeshDynamic&      rx,
                    const rxmesh::VertexAttribute<T>* coords,
                    rxmesh::VertexAttribute<T>*       vertex_quadrics,
                    rxmesh::EdgeAttribute<T>*         edge_cost,
                    rxmesh::EdgeAttribute<T>*         edge_col_coord,
                    T*                                d_min_max_edge_cost)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    float min_max_init[2] = {std::numeric_limits<float>::max(),
                             std::numeric_limits<float>::lowest()};
    CUDA_ERROR(cudaMemcpy(d_min_max_edge_cost,
                          min_max_init,
                          2 * sizeof(T),
                          cudaMemcpyHostToDevice));

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::FV},
                          launch_box,
                          (void*)compute_vertex_quadric_fv<T, blockThreads>,
                          false);
    compute_vertex_quadric_fv<T, blockThreads><<<launch_box.blocks,
                                                 launch_box.num_threads,
                                                 launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, *vertex_quadrics);


    rx.prepare_launch_box(
        {Op::EV}, launch_box, (void*)compute_edge_cost<T, blockThreads>, false);
    compute_edge_cost<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                        *coords,
                                        *vertex_quadrics,
                                        *edge_cost,
                                        *edge_col_coord,
                                        d_min_max_edge_cost);


    // CUDA_ERROR(cudaMemcpy(min_max_init,
    //                       d_min_max_edge_cost,
    //                       2 * sizeof(T),
    //                       cudaMemcpyDeviceToHost));
    // RXMESH_INFO("Min/max edge cost = {}, {}", min_max_init[0],
    // min_max_init[1]);
}

inline void simplification_rxmesh(rxmesh::RXMeshDynamic& rx,
                                  const uint32_t         final_num_faces)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    const uint32_t num_vertices = rx.get_num_vertices();
    const uint32_t num_edges    = rx.get_num_edges();
    const uint32_t num_faces    = rx.get_num_faces();


    auto coords = rx.get_input_vertex_coordinates();

    EXPECT_TRUE(rx.validate());

    auto vertex_quadrics = rx.add_vertex_attribute<float>("quadrics", 16);
    vertex_quadrics->reset(0, DEVICE);

    auto edge_cost = rx.add_edge_attribute<float>("cost", 1);
    edge_cost->reset(0, DEVICE);

    auto edge_col_coord = rx.add_edge_attribute<float>("eCoord", 3);
    edge_col_coord->reset(0, DEVICE);

    float* d_min_max_edge_cost = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_min_max_edge_cost, 2 * sizeof(float)));


    const int num_bins = 256;
    int*      d_bins   = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_bins, num_bins * sizeof(int)));

    float total_time   = 0;
    float app_time     = 0;
    float slice_time   = 0;
    float cleanup_time = 0;


#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    edge_cost->move(DEVICE, HOST);
    // rx.get_polyscope_mesh()->addEdgeScalarQuantity("ECost", *edge_cost);
    // for (uint32_t p = 0; p < rx.get_num_patches(); ++p) {
    //     rx.render_patch(p)->setEnabled(false);
    // }
    polyscope::show();
#endif


    bool validate = true;

    CUDA_ERROR(cudaProfilerStart());
    while (rx.get_num_faces() > final_num_faces) {

        calc_edge_cost(rx,
                       coords.get(),
                       vertex_quadrics.get(),
                       edge_cost.get(),
                       edge_col_coord.get(),
                       d_min_max_edge_cost);

        populate_bins(rx, edge_cost, num_bins, d_min_max_edge_cost, d_bins);

        rx.reset_scheduler();
        int inner_iter = 0;
        while (!rx.is_queue_empty() && rx.get_num_faces() > final_num_faces) {

            LaunchBox<blockThreads> launch_box;
            rx.prepare_launch_box({Op::VE},
                                  launch_box,
                                  (void*)simplify<float, blockThreads>,
                                  true,
                                  false,
                                  false,
                                  false,
                                  [](uint32_t v, uint32_t e, uint32_t f) {
                                      return e * sizeof(uint8_t);
                                  });

            GPUTimer timer;
            timer.start();

            GPUTimer app_timer;
            app_timer.start();
            simplify<float, blockThreads>
                <<<launch_box.blocks,
                   launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                *coords,
                                                *vertex_quadrics,
                                                *edge_cost,
                                                *edge_col_coord);
            app_timer.stop();

            GPUTimer slice_timer;
            slice_timer.start();
            rx.slice_patches(*coords, *vertex_quadrics);
            slice_timer.stop();

            GPUTimer cleanup_timer;
            cleanup_timer.start();
            rx.cleanup();
            cleanup_timer.stop();


            timer.stop();
            CUDA_ERROR(cudaDeviceSynchronize());
            CUDA_ERROR(cudaGetLastError());

            total_time += timer.elapsed_millis();
            app_time += app_timer.elapsed_millis();
            slice_time += slice_timer.elapsed_millis();
            cleanup_time += cleanup_timer.elapsed_millis();

            if (validate) {
                rx.update_host();
                EXPECT_TRUE(rx.validate());
            }
        }
    }

    CUDA_ERROR(cudaProfilerStop());

    RXMESH_INFO("simplification_rxmesh() RXMesh simplification took {} (ms)",
                total_time);
    RXMESH_INFO("simplification_rxmesh() App time {} (ms)", app_time);
    RXMESH_INFO("simplification_rxmesh() Slice timer {} (ms)", slice_time);
    RXMESH_INFO("simplification_rxmesh() Cleanup timer {} (ms)", cleanup_time);

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

    CUDA_ERROR(cudaFree(d_min_max_edge_cost));
    CUDA_ERROR(cudaFree(d_bins));
}