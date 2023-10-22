#include <cuda_profiler_api.h>

#include "rxmesh/rxmesh_dynamic.h"

#include "remesh_kernels.cuh"

template <typename T>
inline void split_long_edges(rxmesh::RXMeshDynamic&      rx,
                             rxmesh::VertexAttribute<T>* coords,
                             const T                     high_edge_len_sq)
{
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    rx.reset_scheduler();
    while (!rx.is_queue_empty()) {
        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box(
            {Op::EV}, launch_box, (void*)edge_split<T, blockThreads>);

        edge_split<float, blockThreads><<<launch_box.blocks,
                                          launch_box.num_threads,
                                          launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, high_edge_len_sq);

        rx.slice_patches(*coords);

        rx.cleanup();

        // rx.update_host();
        // EXPECT_TRUE(rx.validate());
    }
}

template <typename T>
inline void collapse_short_edges(rxmesh::RXMeshDynamic&      rx,
                                 rxmesh::VertexAttribute<T>* coords,
                                 const T                     low_edge_len_sq,
                                 const T                     high_edge_len_sq)
{
    // TODO
    using namespace rxmesh;

    constexpr uint32_t blockThreads = 256;

    rx.reset_scheduler();
    while (!rx.is_queue_empty()) {
        LaunchBox<blockThreads> launch_box;
        rx.prepare_launch_box({Op::VE, Op::EV},
                              launch_box,
                              (void*)edge_collapse<T, blockThreads>);

        edge_collapse<float, blockThreads><<<launch_box.blocks,
                                             launch_box.num_threads,
                                             launch_box.smem_bytes_dyn>>>(
            rx.get_context(), *coords, low_edge_len_sq, high_edge_len_sq);

        rx.slice_patches(*coords);

        rx.cleanup();

        // rx.update_host();
        // EXPECT_TRUE(rx.validate());
    }
}

template <typename T>
inline void equalize_valences(rxmesh::RXMeshDynamic&      rx,
                              rxmesh::VertexAttribute<T>* coords)
{
    // TODO
    rx.reset_scheduler();
    while (!rx.is_queue_empty()) {
    }
}

template <typename T>
inline void tangential_relaxation(rxmesh::RXMeshDynamic&      rx,
                                  rxmesh::VertexAttribute<T>* coords)
{
    // TODO
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

    auto coords = rx.get_input_vertex_coordinates();

    // compute average edge length
    float* average_edge_len;
    CUDA_ERROR(cudaMallocManaged((void**)&average_edge_len, sizeof(float)));

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box(
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

    bool validate = false;


    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        split_long_edges(rx, coords.get(), high_edge_len_sq);
        collapse_short_edges(
            rx, coords.get(), low_edge_len_sq, high_edge_len_sq);
        equalize_valences(rx, coords.get());
        tangential_relaxation(rx, coords.get());
    }

    rx.update_host();
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

    CUDA_ERROR(cudaFree(average_edge_len));
}