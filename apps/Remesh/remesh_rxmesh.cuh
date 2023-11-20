#include <cuda_profiler_api.h>

#include "rxmesh/rxmesh_dynamic.h"

#include "remesh_kernels.cuh"

template <typename T>
inline void split_long_edges(rxmesh::RXMeshDynamic&         rx,
                             rxmesh::VertexAttribute<T>*    coords,
                             rxmesh::EdgeAttribute<int8_t>* updated,
                             const T                        high_edge_len_sq)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;


    updated->reset(0, DEVICE);

    rx.reset_scheduler();
    while (!rx.is_queue_empty()) {
        // RXMESH_INFO("*** queue size= {}",
        //            rx.get_context().m_patch_scheduler.size());
        // rx.get_context().m_patch_scheduler.print_list();

        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({Op::EV},
                              launch_box,
                              (void*)edge_split<T, blockThreads>,
                              true,
                              false,
                              false,
                              false,
                              [&](uint32_t v, uint32_t e, uint32_t f) {
                                  return detail::mask_num_bytes(e) +
                                         ShmemAllocator::default_alignment;
                              });

        edge_split<T, blockThreads><<<DIVIDE_UP(launch_box.blocks, 2),
                                      launch_box.num_threads,
                                      launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *updated, high_edge_len_sq);

        rx.cleanup();
        rx.slice_patches(*coords, *updated);
        rx.cleanup();

        // RXMESH_TRACE(" ");
        // rx.update_host();
        // rx.validate();
        // RXMESH_INFO(" ");
        // RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
        // RXMESH_INFO("#Edges {}", rx.get_num_edges());
        // RXMESH_INFO("#Faces {}", rx.get_num_faces());
        // RXMESH_INFO("#Patches {}", rx.get_num_patches());
        //
        // bool show = false;
        // if (show) {
        //    coords->move(DEVICE, HOST);
        //    updated->move(DEVICE, HOST);
        //    rx.update_polyscope();
        //    auto ps_mesh = rx.get_polyscope_mesh();
        //    ps_mesh->updateVertexPositions(*coords);
        //    ps_mesh->setEnabled(false);
        //
        //    ps_mesh->addEdgeScalarQuantity("updated", *updated);
        //
        //    rx.render_vertex_patch();
        //    rx.render_edge_patch();
        //    rx.render_face_patch();
        //    polyscope::show();
        //}
    }
}

template <typename T>
inline void collapse_short_edges(rxmesh::RXMeshDynamic&         rx,
                                 rxmesh::VertexAttribute<T>*    coords,
                                 rxmesh::EdgeAttribute<int8_t>* updated,
                                 const T                        low_edge_len_sq,
                                 const T high_edge_len_sq)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    updated->reset(0, DEVICE);

    rx.reset_scheduler();
    while (!rx.is_queue_empty()) {
        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({Op::VE, Op::EV},
                              launch_box,
                              (void*)edge_collapse<T, blockThreads>,
                              true,
                              false,
                              true,
                              true,
                              [=](uint32_t v, uint32_t e, uint32_t f) {
                                  return detail::mask_num_bytes(v) +
                                         2 * detail::mask_num_bytes(e) +
                                         3 * ShmemAllocator::default_alignment;
                              });

        edge_collapse<T, blockThreads>
            <<<launch_box.blocks,
               launch_box.num_threads,
               launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                            *coords,
                                            *updated,
                                            low_edge_len_sq,
                                            high_edge_len_sq);

        rx.cleanup();
        rx.slice_patches(*coords, *updated);
        rx.cleanup();

        // RXMESH_TRACE(" ");
        // rx.update_host();
        // rx.validate();
        // RXMESH_INFO(" ");
        // RXMESH_INFO("#Vertices {}", rx.get_num_vertices());
        // RXMESH_INFO("#Edges {}", rx.get_num_edges());
        // RXMESH_INFO("#Faces {}", rx.get_num_faces());
        // RXMESH_INFO("#Patches {}", rx.get_num_patches());
        //
        // bool show = false;
        // if (show) {
        //    coords->move(DEVICE, HOST);
        //    updated->move(DEVICE, HOST);
        //    rx.update_polyscope();
        //    auto ps_mesh = rx.get_polyscope_mesh();
        //    ps_mesh->updateVertexPositions(*coords);
        //    ps_mesh->setEnabled(false);
        //
        //    ps_mesh->addEdgeScalarQuantity("updated", *updated);
        //
        //    rx.render_vertex_patch();
        //    rx.render_edge_patch();
        //    rx.render_face_patch();
        //    polyscope::show();
        //}
    }
}

template <typename T>
inline void equalize_valences(rxmesh::RXMeshDynamic&         rx,
                              rxmesh::VertexAttribute<T>*    coords,
                              rxmesh::EdgeAttribute<int8_t>* updated)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    updated->reset(0, DEVICE);

    rx.reset_scheduler();
    while (!rx.is_queue_empty()) {
        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({Op::EVDiamond},
                              launch_box,
                              (void*)edge_flip<T, blockThreads>,
                              true,
                              false,
                              true);

        edge_flip<T, blockThreads><<<launch_box.blocks,
                                     launch_box.num_threads,
                                     launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *updated);

        rx.cleanup();
        rx.slice_patches(*coords, *updated);
        rx.cleanup();
    }
}

template <typename T>
inline void tangential_relaxation(rxmesh::RXMeshDynamic&      rx,
                                  rxmesh::VertexAttribute<T>* coords,
                                  rxmesh::VertexAttribute<T>* new_coords)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::VV},
                          launch_box,
                          (void*)vertex_smoothing<T, blockThreads>,
                          false,
                          true);

    vertex_smoothing<T, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, *new_coords);
}

inline void remesh_rxmesh(rxmesh::RXMeshDynamic& rx)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

#if USE_POLYSCOPE
    rx.render_vertex_patch();
    rx.render_edge_patch();
    rx.render_face_patch();
    // polyscope::show();
#endif

    auto coords     = rx.get_input_vertex_coordinates();
    auto new_coords = rx.add_vertex_attribute<float>("newCoords", 3);
    auto updated    = rx.add_edge_attribute<int8_t>("Updated", 1);

    // compute average edge length
    float* average_edge_len;
    CUDA_ERROR(cudaMallocManaged((void**)&average_edge_len, sizeof(float)));

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box(
        {Op::EV},
        launch_box,
        (void*)compute_average_edge_length<float, blockThreads>,
        false);
    compute_average_edge_length<float, blockThreads>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, average_edge_len);
    CUDA_ERROR(cudaDeviceSynchronize());
    average_edge_len[0] /= rx.get_num_edges();

    // 4.0/5.0 * targe_edge_len
    const float low_edge_len =
        (4.f / 5.f) * Arg.relative_len * average_edge_len[0];
    const float low_edge_len_sq = low_edge_len * low_edge_len;

    // 4.0/3.0 * targe_edge_len
    const float high_edge_len =
        (4.f / 3.f) * Arg.relative_len * average_edge_len[0];
    const float high_edge_len_sq = high_edge_len * high_edge_len;

    GPUTimer timer;
    timer.start();

    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        split_long_edges(rx, coords.get(), updated.get(), high_edge_len_sq);
        collapse_short_edges(
            rx, coords.get(), updated.get(), low_edge_len_sq, high_edge_len_sq);
        equalize_valences(rx, coords.get(), updated.get());
        tangential_relaxation(rx, coords.get(), new_coords.get());
        std::swap(new_coords, coords);
    }
    timer.stop();
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaGetLastError());

    rx.update_host();
    new_coords->move(DEVICE, HOST);
    coords->move(DEVICE, HOST);
    coords->copy_from(*new_coords, HOST, HOST);

    EXPECT_TRUE(rx.validate());

    rx.export_obj("remesh.obj", *coords);
    RXMESH_INFO("remesh_rxmesh() took {} (ms)", timer.elapsed_millis());
    RXMESH_INFO("Output mesh #Vertices {}", rx.get_num_vertices());
    RXMESH_INFO("Output mesh #Edges {}", rx.get_num_edges());
    RXMESH_INFO("Output mesh #Faces {}", rx.get_num_faces());
    RXMESH_INFO("Output mesh #Patches {}", rx.get_num_patches());


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

    CUDA_ERROR(cudaFree(average_edge_len));
}